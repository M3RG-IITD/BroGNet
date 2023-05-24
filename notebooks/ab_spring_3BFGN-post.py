
################## IMPORT ######################
import sys
from datetime import datetime
from functools import partial, wraps
import time

import fire
import jax
import jax.numpy as jnp
# import numpy as np
from jax import jit, random, value_and_grad, vmap
# from jax.experimental import optimizers
from jax.example_libraries import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score
# from sympy import LM
# from torch import batch_norm_gather_stats_with_counts

from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.fgn import cal_acceleration
# from src.lnn import acceleration, accelerationFull, accelerationTV, acceleration_GNODE
from src.md import *
from src.models import MSE, initialize_mlp, GaussianNLL, initialize_mlp_gamma, SquarePlus, forward_pass, forward_pass_gamma, ReLU
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
# useN=None
# dim = 2
# kT = 1
# seed=42
# dt = 1.0e-3
# spring_constant = 1.0
# length_constant = 1.0
# stride = 1
# runs=100
# rname=0
# withdata = None
# saveovito=0
# semilog=1
# plotthings = True
# maxtraj = 100
# redo=0

def main(N = 10, ratio=0.3, useN=None, dim = 2, kT = 1, seed=42, dt = 1.0e-3, spring_constant = 1.0, length_constant = 1.0, stride = 1, runs=100, rname=0, withdata = None, saveovito=0, semilog=1, plotthings = True, maxtraj = 100, redo=0):
    
    if useN is None:
        useN = N
        
    print("Configs: ")
    pprint(N, seed, rname, dt, namespace=locals())
    
    # PSYS = f"a-{N*kT+1}-Spring-data-brownian_EM"
    PSYS = f"a-{N}-AB-Spring-data-brownian_EM"
    TAG = f"3BFGN"
    out_dir = f"../results"
    randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    
    def _filename(name, tag=TAG, trained=None):
        if tag == "data":
            part = f"."
        else:
            part = f"."
        if trained is not None:
            psys = f"a-{trained}-{'-'.join(PSYS.split('-')[2:])}"
        else:
            psys = PSYS
        name = ".".join(name.split(".")[:-1]) + \
            part + name.split(".")[-1]
        rstring = randfilename if (rname and (tag != "data")) else (
            "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
        filename_prefix = f"{out_dir}/{psys}-{tag}/{rstring}/"
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
        def func(file, *args, tag=TAG, trained=None, **kwargs):
            return f(_filename(file, tag=tag, trained=trained), *args, **kwargs)
        return func
    
    def _fileexist(f):
        if redo:
            return False
        else:
            return os.path.isfile(f)
    
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
    
    masses = jnp.ones(N)
    species = np.where(np.arange(N) < int(N*ratio), 0, 1) #jnp.zeros(N, dtype=int)
    gamma_orig = jnp.where(jnp.arange(N) < int(N*ratio), 1.0, 2.0).reshape(-1,1)
    
    
    print("Creating Chain")
    x, v, senders, receivers = chain(N)
    
    ################################################
    ################### ML Model ###################
    ################################################
    
    
    def nngamma(type, params):
        return forward_pass_gamma(params, type, activation_fn=models.SquarePlus)
    
    def gamma(type, params):
        return vmap(nngamma, in_axes=(0, None))(type.reshape(-1), params).reshape(-1, 1)
        # return nngamma(type.reshape(-1), params["gamma"])#.reshape(-1, 1)
    
    R, _, senders, receivers = chain(N)
    
    def dist(*args):
        disp = displacement(*args)
        return jnp.sqrt(jnp.square(disp).sum())
    
    dij = vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])
    
    def acceleration_fn(params, graph):
        acc = cal_acceleration(params, graph, mpass=1)
        return acc
    
    def acc_fn(species):
        state_graph = jraph.GraphsTuple(nodes={
            "position": R,
            "type": species.reshape(-1,1)
        },
            edges={"dij": dij},
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders.shape[0]]),
            globals={})
        
        def apply(R, params):
            state_graph.nodes.update(position=R)
            state_graph.edges.update(dij=vmap(dist, in_axes=(0, 0))(R[senders], R[receivers]))
            return acceleration_fn(params, state_graph)
        return apply

    def gamma_fn(species):    
        def fn(params):
            return gamma(jax.nn.one_hot(species, 1),params)    
        return fn

    apply_fn = acc_fn(species)
    gamma_fn = gamma_fn(species)

    def force_fn_model(x, params): return apply_fn(x, params["F_pos"])
    def gamma_fn_model(params): return gamma_fn(params["gamma"])
        
        
    params = loadfile(f"fgnode_trained_model_low.dil", trained=useN)[0]
    
    def SPRING(x, stiffness=1.0, length=1.0):
        x_ = jnp.linalg.norm(x, keepdims=True)
        return 0.5*stiffness*(x_ - length)**2
    
    def pot_energy_orig(x):
        dr = x[senders, :] - x[receivers, :]
        return vmap(partial(SPRING, stiffness=spring_constant, length=length_constant))(dr).sum()
    
    def force_fn_orig(R, params):
        return -grad(pot_energy_orig)(R)
    
    
    def get_forward_sim(params = None, force_fn = None, gamma = None, runs=10):
            @jit
            def fn(R,key):
                return predition_brow(R, params, force_fn, shift, dt, kT, masses, gamma = gamma, stride=stride, runs=runs, key=key)
            return fn
    
    gamma_model = gamma_fn_model(params)
    
    sim_orig = get_forward_sim(params=None,force_fn=force_fn_orig, gamma=gamma_orig,runs=runs)
    sim_model = get_forward_sim(params=params,force_fn=force_fn_model, gamma=gamma_model,runs=runs)
    
    rng_key = random.PRNGKey(0)
    _gamma = gamma_fn_model(params)
    nexp = {
            "dz_actual": [],
            "dz_pred": [],
            "z_actual": [],
            "z_pred": [],
            "_gamma": [_gamma],
            "simulation_time":[],
            
            }
    
    trajectories = []
    for ind in range(maxtraj):
        print(f"Simulating trajectory {ind}/{maxtraj} ...")
        R, _ = chain(N)[:2]
        for rand in range(10):
            rng_key, subkey = random.split(rng_key)
            actual_traj = sim_orig(R,(ind+13)*subkey)
            rng_key, subkey = random.split(rng_key)
            
            start = time.time()
            pred_traj = sim_model(R, (ind+13)*subkey)
            end = time.time()
            
            nexp["simulation_time"] += [end-start]
            
            nexp["dz_actual"] += [actual_traj.position-R]
            nexp["dz_pred"] += [pred_traj.position-R]
            
            nexp["z_actual"] += [actual_traj.position]
            nexp["z_pred"] += [pred_traj.position]
            
            # trajectories += [(actual_traj, pred_traj)]
            
            if saveovito:
                if ind<1 and rand<1:
                    save_ovito(f"actual_{ind}_{rand}.xyz", [state for state in BrownianStates(actual_traj)], lattice="")
                    save_ovito(f"pred_{ind}_{rand}.xyz", [state for state in BrownianStates(pred_traj)], lattice="")
            
            # if ind%10==0:
            #     savefile("trajectories.pkl", trajectories)
    
    def KL_divergence(sigma0,mu0,sigma1,mu1, eps=1e-8):
        return jnp.log(sigma1/sigma0) + (jnp.square(sigma0)+jnp.square(mu0-mu1))/(2*jnp.square(sigma1)) - 0.5
    
    def get_kld(d_actual, d_pred):
        mu0 = jnp.mean(d_actual, axis=(0,2,3))
        std0 = jnp.std(d_actual, axis=(0,2,3))
        mu1 = jnp.mean(d_pred, axis=(0,2,3))
        std1 = jnp.std(d_pred, axis=(0,2,3))
        kld = []
        for i in range(len(std0)):
            kld.append(KL_divergence(std0[i],mu0[i],std1[i],mu1[i]))
        return jnp.array(kld)
    
    def get_std_rmse(d_actual, d_pred):
        std0 = jnp.std(d_actual, axis=(0,2,3))
        std1 = jnp.std(d_pred, axis=(0,2,3))
        return jnp.sqrt(jnp.square(std0 - std1))
    
    def get_dist_by_var(actual, pred, zeta):
        disp = displacement(actual, pred)
        dist_matrix = jnp.sqrt(jnp.square(disp).sum(-1))
        dist_mean = jnp.mean(dist_matrix, axis=(0,2))
        dist_by_zeta = dist_mean/zeta
        return dist_by_zeta
    
    nexp2 = {
            "kld": [],
            "std_rmse": [],
            "dist_by_var": [],
            }
    
    nexp2["kld"] = jnp.array(get_kld(jnp.array(nexp["dz_actual"]),jnp.array(nexp["dz_pred"])))
    
    nexp2["std_rmse"] = jnp.array(get_std_rmse(jnp.array(nexp["dz_actual"]),jnp.array(nexp["dz_pred"])))
    
    nexp2["dist_by_var"] = jnp.array(get_dist_by_var(jnp.array(nexp['z_actual']), jnp.array(nexp['z_pred']),1/(jnp.array(nexp['_gamma'])[0][0])))
    
    savefile(f"error_paramete_plot_a_b_c.pkl", nexp2)
    # savefile("trajectories.pkl", trajectories)

fire.Fire(main)