################## IMPORT ######################
import json
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
# from jax.experimental import optimizers
from jax.example_libraries import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score
from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)
# from statistics import mode
# from sympy import LM
# from torch import batch_norm_gather_stats_with_counts
import sys
MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8
import jraph
import src
import time
from jax.config import config
# from src import fgn, lnn
from src.graph import *
# from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp, GaussianNLL, initialize_mlp_gamma, forward_pass_gamma
from src.nve import NVEStates, nve, BrownianStates
from src.utils import *

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")

f32 = jnp.float32
f64 = jnp.float64

# N = 5
# dim = 2
# runs = 1
# kT = 1
# seed=42
# dt = 1.0e-3
# lr=1e-4
# batch_size=20
# epochs = 10000
# masses = jnp.ones(N)
# species = jnp.zeros(N, dtype=int)
# datapoints = None
# rname=True
# withdata = None

def main(N = 5, dim = 2, runs = 1, kT = 1, seed=42, dt = 1.0e-3, lr=1e-4, batch_size=20, epochs = 10000, datapoints = None, rname=True, withdata = None):
    masses = jnp.ones(N)
    species = jnp.zeros(N, dtype=int).reshape(-1,1)
    print("Configs: ")
    pprint(N, epochs, seed, rname, dt, lr, batch_size, namespace=locals())

    # randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"
    randfilename = '0' + f"_{datapoints}"
    
    PSYS = f"a-{N}-Spring-data-brownian_EM"
    TAG = f"2BNN"
    out_dir = f"../results"
    
    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename

    def displacement(a, b):
        return a - b

    def shift(R, dR):
        return R+dR

    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func

    loadmodel = OUT(src.models.loadmodel)
    savemodel = OUT(src.models.savemodel)

    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    save_ovito = OUT(src.io.save_ovito)


    
    ################################################
    ################## CONFIG ######################
    ################################################
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    try:
        dataset_states = loadfile(f"model_states_brownian.pkl", tag="data")[0]
    except:
        raise Exception("Generate dataset first.")

    model_states = dataset_states[0]

    # print(f"Total number of data points: {len(dataset_states)}x{model_states.position.shape[0]}")
    
    Rs = States_Brow().fromlist(dataset_states).get_array()
    
    # if datapoints is not None:
    #     Rs = Rs[:datapoints]
    
    Rs_in = Rs[:,:99,:,:].reshape(-1,5,2)
    Rs_out = Rs[:,1:100,:,:].reshape(-1,5,2)
    
    if datapoints is not None:
        indices = jax.random.choice(key, 9900, shape=(99*datapoints,), replace=False)
        Rs_in = Rs_in[indices,:,:]
        Rs_out = Rs_out[indices,:,:]
    
    print(f"Total number of data points: ", Rs_in.shape)
        
    ################################################
    ################### ML Model ###################
    ################################################
    # print("Creating Chain")
    x, _, senders, receivers = chain(N)

    hidden = 16
    nhidden = 2

    def get_layers(in_, out_):
        return [in_] + [hidden]*nhidden + [out_]

    def mlp(in_, out_, key, **kwargs):
        return initialize_mlp(get_layers(in_, out_), key, **kwargs)

    params = {"F_pos": mlp(N*dim, N*dim, key)}

    params["gamma"] = initialize_mlp_gamma([1,10,5,1], key)

    def nngamma(type, params):
        return forward_pass_gamma(params, type, activation_fn=models.SquarePlus)

    def gamma(type, params):
        return vmap(nngamma, in_axes=(0, None))(type.reshape(-1), params).reshape(-1, 1)
        # return nngamma(type.reshape(-1), params["gamma"])#.reshape(-1, 1)

    # ss = gamma(jax.nn.one_hot(species, 1),params["gamma"])


    def acceleration_node(x, params, **kwargs):
        n,dim = x.shape
        inp = x.flatten() #jnp.hstack([x.flatten(),v.flatten()])
        out = forward_pass(params, inp)
        return out.reshape(-1,dim)

    def _force_fn():    
        def apply(R, params):
            return acceleration_node(R, params)
        return apply

    def gamma_fn(species):    
        def fn(params):
            return gamma(jax.nn.one_hot(species, 1),params)    
        return fn

    apply_fn = _force_fn()
    gamma_fn = gamma_fn(species)

    def force_fn_model(x, params): return apply_fn(x, params['F_pos'])
    def gamma_fn_model(params): return gamma_fn(params["gamma"])
    
    # gamma_fn_model(params)
    

    def next_step_pos_gamma(force_fn_model, gamma_fn_model, shift, dt, kT, mass, runs, key):
        key, split = random.split(key)
        def fn(x, params):
            for i in range(runs):
                # calculate the force
                force = force_fn_model(x, params)
                _gamma = gamma_fn_model(params)
                xi = random.normal(split, x.shape, x.dtype)
                nu = f32(1) / lax.mul(mass.reshape(-1,1) , _gamma)
                x = x+ force * dt * nu + jnp.sqrt(f32(2) * kT * dt * nu) * xi
            return x, _gamma
        return fn



    rng_key = random.PRNGKey(0)
    rng_key, subkey = random.split(rng_key)

    next_step_pos_gamma_fn = next_step_pos_gamma(force_fn_model, gamma_fn_model, shift, dt, kT, masses, runs, subkey)
    v_next_step_pos_gamma_fn = vmap(next_step_pos_gamma_fn, in_axes=(0, None))
    v_v_next_step_pos_gamma_fn = vmap(v_next_step_pos_gamma_fn, in_axes=(0, None))
    

    @jit
    def loss_fn(params, Rs, Rs_1_ac,A=1, B=500): # A=4, B=996 wf=0.996):
        Rs_1_pred, gamma = v_next_step_pos_gamma_fn(Rs, params)
        var =1/gamma
        return GaussianNLL(var, Rs_1_pred, Rs_1_ac, A, B)

    def gloss(*args):
        return value_and_grad(loss_fn)(*args)

    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value

    @jit
    def step(i, ps, *args):
        return update(i, *ps, *args)

    opt_init, opt_update_, get_params = optimizers.adam(lr)

    @jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)

    def batching(*args, size=None):
        L = len(args[0])
        if size != None:
            nbatches1 = int((L - 0.5) // size) + 1
            nbatches2 = max(1, nbatches1 - 1)
            size1 = int(L/nbatches1)
            size2 = int(L/nbatches2)
            if size1*nbatches1 > size2*nbatches2:
                size = size1
                nbatches = nbatches1
            else:
                size = size2
                nbatches = nbatches2
        else:
            nbatches = 1
            size = L
        
        newargs = []
        for arg in args:
            newargs += [jnp.array([arg[i*size:(i+1)*size]
                                    for i in range(nbatches)])]
        return newargs

    bRs_in, bRs_out = batching(Rs_in, Rs_out, size=min(len(Rs_in), batch_size))

    print(f"training ...")

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []
    last_loss = 1000
    
    start = time.time()
    train_time_arr = []

    for epoch in range(epochs):
        l = 0.0
        count = 0
        for data in zip(bRs_in, bRs_out):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)
            l += l_
            count+=1
        # print("epoch,countttttt: ", epoch,count)
        # opt_state, params, l_ = step(optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)
        l = l/count
        larray += [l]
        # ltarray += [loss_fn(params, bRs_in, bVs_in, bRs_out)]
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}")#, test={ltarray[-1]}")
        if epoch % 100 == 0:
            print('gammaaaaa: ', gamma_fn_model(params))
            metadata = {
                "savedat": epoch,
                # "mpass": mpass,
                }
            savefile(f"fgnode_trained_model.dil",
                        params, metadata=metadata)
            # savefile(f"loss_array.dil", (larray, ltarray), metadata=metadata)
            savefile(f"loss_array.dil", larray, metadata=metadata)
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"fgnode_trained_model_low.dil",
                            params, metadata=metadata)
            fig, axs = panel(1, 1)
            # plt.semilogy(larray, label="Training")
            plt.plot(larray, label="Training")
            # plt.semilogy(ltarray, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(_filename(f"training_loss.png"))
            np.savetxt(_filename(f"training-time.txt"), train_time_arr, delimiter = "\n")
        
        now = time.time()
        train_time_arr.append((now - start))

    fig, axs = panel(1, 1)
    # plt.semilogy(larray, label="Training")
    plt.plot(larray, label="Training")
    # plt.semilogy(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss.png"))

    params = get_params(opt_state)
    savefile(f"fgnode_trained_model.dil", params, metadata=metadata)
    np.savetxt(_filename(f"training-time.txt"), train_time_arr, delimiter = "\n")
    # savefile(f"loss_array.dil", (larray, ltarray), metadata=metadata)
    
    if last_loss > larray[-1]:
        last_loss = larray[-1]
        savefile(f"fgnode_trained_model_low.dil", params, metadata=metadata)

fire.Fire(main)