########################################
import jax
import jax.numpy as jnp
from jax import random, vmap, grad, jit, value_and_grad
from functools import partial, wraps
from shadow.plot import *
from jax.config import config
from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order, get_init, get_init_spring)

from jax_md import dataclasses, interpolate, quantity, simulate, space, util
NVEState = simulate.NVEState
import datetime

simulate.brownian
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


########################################
import sys
MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
# from src import lnn
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import NVEStates, nve, BrownianStates
from src.utils import *


################## CONFIG ######################
N = 5  # number of particles
dim = 2  # dimensions
runs = 100
saveat = 10
kT = 1 #1.380649e-23*T  # boltzmann constant*temperature
spring_constant = 1.0
length_constant = 1.0
nconfig=100
seed=42
dt = 1.0e-3 # time step
stride=1

# node_type = jnp.array([0,0,0,0,0])
masses = jnp.ones(N)
species = jnp.zeros(N, dtype=int)
gamma = jnp.ones(jnp.unique(species).shape)  # damping constant

traj = []

tag = f"a-{N}-non-linear-Spring-data-brownian_EM-data"
out_dir = f"../results"
rname = False
rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
filename_prefix = f"{out_dir}/{tag}/{rstring}/"

def _filename(name):
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
    def func(file, *args, **kwargs):
        return f(_filename(file), *args, **kwargs)
    return func

loadmodel = OUT(src.models.loadmodel)
savemodel = OUT(src.models.savemodel)

loadfile = OUT(src.io.loadfile)
savefile = OUT(src.io.savefile)
save_ovito = OUT(src.io.save_ovito)

########################################
np.random.seed(seed)
key = random.PRNGKey(seed)

init_confs = [chain(N)[:2] for i in range(nconfig)]

_, _, senders, receivers = chain(N)
R, V = init_confs[0]

# x, v, senders, receivers = chain(N)

################## SYSTEM ######################
def SPRING(x, stiffness=1.0, length=1.0):
    x_ = jnp.linalg.norm(x, keepdims=True)
    return 0.5*stiffness*(x_ - length)**4

def pot_energy_orig(x):
    dr = x[senders, :] - x[receivers, :]
    return vmap(partial(SPRING, stiffness=spring_constant, length=length_constant))(dr).sum()


def force_fn_orig(R, params):
    return -grad(pot_energy_orig)(R)

@jit
def forward_sim(R, key):
    return predition_brow(R, None, force_fn_orig, shift, dt, kT, masses, gamma = gamma, stride=stride, runs=runs, key=key)


############### DATA GENERATION ################
print("Data generation ...")
rng_key = random.PRNGKey(0)

ind = 0
dataset_states = []
for R, V in init_confs:
    ind += 1
    rng_key, subkey = random.split(ind*rng_key)
    print(f"{ind}/{len(init_confs)}", end='\r')
    model_states = forward_sim(R, subkey)
    dataset_states += [model_states]
    if ind % saveat == 0:
        print(f"{ind} / {len(init_confs)}")
        print("Saving datafile...")
        savefile(f"model_states_brownian.pkl", dataset_states)

print("Saving datafile...")
savefile(f"model_states_brownian.pkl", dataset_states)


def cal_energy(states):
    PE = vmap(pot_energy_orig)(states.position)
    return PE


print("plotting energy...")
ind = 0
for states in dataset_states:
    ind += 1
    PEs = cal_energy(states)
    fig, axs = panel(1, 1, figsize=(20, 5))
    plt.plot(PEs, label=["PE"], lw=6, alpha=0.5)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel("Energy")
    plt.xlabel("Time step")
    
    title = f"{N}-Spring random state {ind}"
    plt.title(title)
    plt.savefig(_filename(title.replace(" ", "_")+".png"), dpi=100)
    
    save_ovito(f"trajectory_{ind}.xyz", [state for state in BrownianStates(states)], lattice="")
    
    if ind > 5:
        break
