import importlib
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from jax import grad, jit, random, vmap
from jax_md import smap, space

from . import lnn, models


from sklearn.neighbors import KernelDensity
import numpy as np
def KDE(data, bins=100, bandwidth=6):
    Vecvalues = data[:, None]
    min_ = data.min()
    max_ = data.max()
    Vecpoints = np.linspace(min_, max_, bins)[:, None]
    kde = KernelDensity(kernel='gaussian',
                        bandwidth=bandwidth).fit(Vecvalues)
    f = lambda x: np.exp(kde.score_samples(x[:, None])) 
    h = np.exp(kde.score_samples(Vecpoints))
    return h, Vecpoints.flatten(), f 

def KDE_prob(data_, *args, **kwargs):
    y, x, f = KDE(data_)
    return f(data_)



def colnum(i, j, N):
    """Gives linear index for upper triangle matrix.
    """
    assert (j >= i), "j >= i, Upper Triangle indices."
    assert (i < N) and (j < N), "i<N & j<N where i and \
            j are atom type and N is number of species."
    return int(i*N - i*(i-1)/2 + j-i + 1)


def pair2mat(fn, displacement_or_metric, species, parameters,
             ignore_unused_parameters=True,
             reduce_axis=None,
             keepdims=False,
             use_onehot=False,
             **kwargs):
    kwargs, param_combinators = smap._split_params_and_combinators(kwargs)

    merge_dicts = partial(jax_md.util.merge_dicts,
                          ignore_unused_parameters=ignore_unused_parameters)
    d = lnn.t1(displacement=displacement_or_metric)
    if species is None:
        def fn_mapped(R: smap.Array, **dynamic_kwargs) -> smap.Array:
            _kwargs = merge_dicts(kwargs, dynamic_kwargs)
            _kwargs = smap._kwargs_to_parameters(
                None, _kwargs, param_combinators)
            dr = d(R)
            # NOTE(schsam): Currently we place a diagonal mask no matter what function
            # we are mapping. Should this be an option?
            return smap.high_precision_sum(fn(dr, **_kwargs),
                                           axis=reduce_axis, keepdims=keepdims) * smap.f32(0.5)

    elif jax_md.util.is_array(species):
        species = np.array(species)
        smap._check_species_dtype(species)
        species_count = int(np.max(species) + 1)
        if reduce_axis is not None or keepdims:
            raise ValueError

        def onehot(i, j, N):
            col = colnum(i, j, species_count)
            oneh = jnp.zeros(
                (N, colnum(species_count-1, species_count-1, species_count)))
            oneh = jax.ops.index_update(oneh, jnp.index_exp[:, int(col-1)], 1)
            return oneh

        def pot_pair_wise():
            if use_onehot:
                def func(i, j, dr, **s_kwargs):
                    dr = jnp.linalg.norm(dr, axis=1, keepdims=True)
                    ONEHOT = onehot(i, j, len(dr))
                    h = vmap(models.forward_pass, in_axes=(
                        None, 0))(parameters["ONEHOT"], ONEHOT)
                    dr = jnp.concatenate([h, dr], axis=1)
                    return smap.high_precision_sum(
                        fn(dr, params=parameters["PEF"], **s_kwargs))
                return func
            else:
                def func(i, j, dr, **s_kwargs):
                    return smap.high_precision_sum(
                        fn(dr, **parameters[i][j-i], **s_kwargs))
                return func

        pot_pair_wise_fn = pot_pair_wise()

        def fn_mapped(R, **dynamic_kwargs):
            U = smap.f32(0.0)
            for i in range(species_count):
                for j in range(i, species_count):
                    _kwargs = merge_dicts(kwargs, dynamic_kwargs)
                    s_kwargs = smap._kwargs_to_parameters(
                        (i, j), _kwargs, param_combinators)
                    Ra = R[species == i]
                    Rb = R[species == j]
                    if j == i:
                        dr = d(Ra)
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + smap.f32(0.5) * dU
                    else:
                        dr = vmap(vmap(displacement_or_metric, in_axes=(0, None)), in_axes=(
                            None, 0))(Ra, Rb).reshape(-1, Ra.shape[1])
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + dU
            return U
    return fn_mapped


def map_parameters(fn, displacement, species, parameters, **kwargs):
    mapped_fn = lnn.MAP(fn)

    def f(x, *args, **kwargs):
        out = mapped_fn(x, *args, **kwargs)
        return out
    return pair2mat(f, displacement, species, parameters, **kwargs)


class VV_unroll():
    def __init__(self, R, dt=1):
        self.R = R
        self.dt = dt
    
    def get_position(self):
        r = self.R[1:-1]
        return r

    def get_acceleration(self, dt=None):
        r = self.R[1:-1]
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus + r_minus - 2*r)/dt**2
        else:
            return (r_plus + r_minus - 2*r)/self.dt**2
    
    def get_velocity(self, dt=None):
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus - r_minus)/2/dt
        else:
            return (r_plus - r_minus)/2/self.dt
    
    def get_all(self, dt=None):
        return self.get_position(), self.get_velocity(dt=dt), self.get_acceleration(dt=dt)


class States:
    def __init__(self, state=None, const_size=True):
        if state is None:
            self.isarrays = False
            self.const_size = const_size
            self.position = []
            self.velocity = []
            self.force = []
            if self.const_size:
                self.mass = None
            else:
                self.mass = []
        else:
            self.position = [state.position]
            self.velocity = [state.velocity]
            self.force = [state.force]
            if self.const_size:
                self.mass = state.mass
            else:
                self.mass = [state.mass]

    def add(self, state):
        self.position += [state.position]
        self.velocity += [state.velocity]
        self.force += [state.force]
        if self.const_size:
            if self.mass is None:
                self.mass = state.mass
        else:
            self.mass += [state.mass]

    def fromlist(self, states, const_size=True):
        out = States(const_size=const_size)
        for state in states:
            out.add(state)
        return out

    def makearrays(self):
        if not(self.isarrays):
            self.position = jnp.array(self.position)
            self.velocity = jnp.array(self.velocity)
            self.force = jnp.array(self.force)
            self.mass = jnp.array([self.mass])
            self.isarrays = True

    def get_array(self):
        self.makearrays()
        return self.position, self.velocity, self.force

    def get_mass(self):
        self.makearrays()
        return self.mass

    def get_kin(self):
        self.makearrays()
        if self.const_size:
            acceleration = self.force/self.mass.reshape(1, self.mass.shape)
        else:
            acceleration = self.force/self.mass
        return self.position, self.velocity, acceleration

class States_H:
    def __init__(self, state=None, const_size=True):
        if state is None:
            self.isarrays = False
            self.const_size = const_size
            self.position = []
            self.velocity = []
            self.constraint_force = []
            self.force = []
            if self.const_size:
                self.mass = None
            else:
                self.mass = []
        else:
            self.position = [state.position]
            self.velocity = [state.velocity]
            self.constraint_force = [state.constraint_force]
            self.force = [state.force]
            if self.const_size:
                self.mass = state.mass
            else:
                self.mass = [state.mass]

    def add(self, state):
        self.position += [state.position]
        self.velocity += [state.velocity]
        self.constraint_force += [state.constraint_force]
        self.force += [state.force]
        if self.const_size:
            if self.mass is None:
                self.mass = state.mass
        else:
            self.mass += [state.mass]

    def fromlist(self, states, const_size=True):
        out = States(const_size=const_size)
        for state in states:
            out.add(state)
        return out

    def makearrays(self):
        if not(self.isarrays):
            self.position = jnp.array(self.position)
            self.velocity = jnp.array(self.velocity)
            self.constraint_force = jnp.array(self.constraint_force)
            self.force = jnp.array(self.force)
            self.mass = jnp.array([self.mass])
            self.isarrays = True

    def get_array(self):
        self.makearrays()
        return self.position, self.velocity, self.constraint_force, self.force
    
    def get_mass(self):
        self.makearrays()
        return self.mass
    
    def get_kin(self):
        self.makearrays()
        if self.const_size:
            acceleration = self.force/self.mass.reshape(1, self.mass.shape)
        else:
            acceleration = self.force/self.mass
        return self.position, self.velocity, self.constraint_force, acceleration

    def __repr__(self) -> str:
        return "position="+self.position.__repr__() + ", velocity="+self.velocity.__repr__() + ", constraint_force="+self.constraint_force.__repr__() + ", force=" + self.force.__repr__()

def reload(list_of_modules):
    for module in list_of_modules:
        try:
            print("Reload: ", module.__name__)
            importlib.reload(module)
        except:
            print("Reimports failed.")


def timeit(stmt, setup="", number=5):
    from timeit import timeit
    return timeit(stmt=stmt, setup=setup, number=number)


def factorial(n):
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)


def nCk(n, k):
    return factorial(n)//factorial(k)//factorial(n-k)


def Range(*args, **kwargs): return jnp.array(range(*args, **kwargs))

def make_graph(R, disp_fn, species=None, atoms=None, V=None, A=None, mass=None, cutoff=2.5):
    """Make graph from position of atoms.

    :param R: position
    :type R: jnp.ndarray
    :param atoms: types of atoms
    :type atoms: dict
    :param disp_fn: displacement function
    :type disp_fn: Function
    :param V: velocity, defaults to None
    :type V: jnp.ndarray, optional
    :param A: acceleration, defaults to None
    :type A: jnp.ndarray, optional
    :param cutoff: cutoff for neighborhood, defaults to 2.0
    :type cutoff: float, optional
    :return: graph
    :rtype: dict
    """
    
    if species is None:
        species = jnp.hstack([ind*jnp.ones(v, dtype=int).flatten()
                              for ind, v in enumerate(atoms.values())])

    species = species.astype(int)
    
    # a != b
    bond_mask = vmap(vmap(lambda a, b: True, in_axes=(
        0, None)), in_axes=(None, 0))(species, species)
    
    if mass is not None:
        mass = jnp.array(mass, dtype=float)[species]
    
    nodes = {
        "position": R,
        "velocity": V,
        "acceleration": A,
        "type": species,
        "mass": mass,
    }
    
    dd = (vmap(vmap(disp_fn, in_axes=(0, None)),
          in_axes=(None, 0))(R, R)**2).sum(axis=-1)**0.5
    mask = dd < cutoff
    # mask = mask*bond_mask
    mask = mask.at[Range(len(R)), Range(len(R))].set(False)
    # jax.ops.index_update(
        # mask, jax.ops.index[Range(len(R)), Range(len(R))], False)
    
    inds = Range(len(R))
    NUMBER = jnp.zeros(mask.shape, dtype=int)
    # NUMBER = jax.ops.index_update(
    #     NUMBER, jax.ops.index[mask], Range(mask.sum()))
    NUMBER = NUMBER.at[mask].set(Range(mask.sum()))
    edge_order = NUMBER.T[mask]
    
    ROWS = vmap(lambda row, ind: row*ind, in_axes=(0, 0))(mask, inds)
    # edges = jnp.vstack([ROWS[mask], ROWS.T[mask]]).T
    senders = ROWS[mask]
    receivers = ROWS.T[mask]
    edges = {}
    
    graph = dict(nodes=nodes, edges=edges, e_order=edge_order,
                 receivers=receivers, senders=senders, n_node=jnp.array([len(R)], dtype=int), n_edge=jnp.array([len(senders)], dtype=int), atoms=atoms)
    return graph

def create_G(R, Disp_Vec_fn, pair_cutoffs, V=None, A= None, mass=None, species=None, atoms = None):
    """
    R: Node Positions
    Disp_vec_fn: Calculates distance between atoms considering periodic boundaries
    species: Node type info 0 and 1
    cutoffs: pair cutoffs (N,N) shape
    sigma  : pair sigma   (N,N)
        """
    
    if species is None:
        species = jnp.hstack([ind*jnp.ones(v, dtype=int).flatten()
                              for ind, v in enumerate(atoms.values())])
    species = species.astype(int)
    
    if mass is not None:
        mass = jnp.array(mass, dtype=float)[species]
    
    #1: Calculate pair distances
    R_pair = Disp_Vec_fn(R, R)
    dr_pair =jax_md.space.distance(R_pair)
    
    #3: Creating neigh_list and senders and receivers    
    n_list=(dr_pair<pair_cutoffs).astype(int)
    n_list=n_list.at[jnp.diag_indices_from(n_list)].set(0)
    (senders,receivers)=jnp.where(n_list==1)
    
    #4: edge order
    mask = dr_pair<pair_cutoffs
    mask = mask.at[Range(len(R)), Range(len(R))].set(False)
    NUMBER = jnp.zeros(mask.shape, dtype=int)
    NUMBER = NUMBER.at[mask].set(Range(mask.sum()))
    edge_order = NUMBER.T[mask]
    
    #5: Node features
    Node_feats = {
        "position": R,
        "velocity": V,
        "acceleration": A,
        "type": species,
        "mass": mass,
        }
    
    #6: Edge Features
    # dist_2d=R_pair[senders,receivers,:]
    # Edge_feats=dist_2d
    Edge_feats ={}
    
    #7: atoms type        
    # atoms = {
    #     "A": sum(species == 0),
    #     "B": sum(species == 1)
    #     }

    G=dict(nodes=Node_feats, edges=Edge_feats, e_order = edge_order,
            receivers=receivers, senders=senders, n_node=jnp.array([len(R)], dtype=int),
            n_edge=jnp.array([len(senders)], dtype=int), atoms=atoms)
                                
    return G , n_list

def fileloc(f, fileloc, TAG=""):
    """Append filename with fileloc and TAG.

    :param f: function that takes first argument as filename
    :type f: function
    :param fileloc: file location
    :type fileloc: string
    :param TAG: appended before filename, defaults to ""
    :type TAG: str, optional
    """
    def func(file, *args, tag=None, **kwargs):
        if tag is None:
            tag = TAG
        filename = f"{fileloc}/{tag}_{file}"
        filename = filename.replace("/_", "/")
        filename = filename.replace("//", "/")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return f(filename, *args, **kwargs)
    return func

def periodic_image(R, L=None, image=1, **kwargs):
    dim = R.shape[-1]
    if isinstance(image, int):
        image = [image]*dim
    if dim != len(image):
        raise Exception(
            f"Lase array dimension {dim} and image len({image}) not matching.")

    def nimage(image, ind=0, info={}):
        out = []
        if len(image) > 1:
            for i in range(image[0]):
                info.update({ind: i})
                out += nimage(image[1:], ind=ind+1, info=info)
            return out
        else:
            for i in range(image[0]):
                info.update({ind: i})
                out += [info.copy()]
            return out
    OUT = []
    for img in nimage(image):
        R_ = 1*R
        for ind, n in img.items():
            R_[:, ind] += L*n
        OUT += [R_]

    KW = {k: np.concatenate([v]*len(OUT), axis=0) for k, v in kwargs.items()}

    return np.concatenate(OUT, axis=-2), KW

class States_Brow:
    def __init__(self, state=None, const_size=True):
        if state is None:
            self.isarrays = False
            self.const_size = const_size
            self.position = []
            if self.const_size:
                self.mass = None
            else:
                self.mass = []
        else:
            self.position = [state.position]
            if self.const_size:
                self.mass = state.mass
            else:
                self.mass = [state.mass]
    
    def add(self, state):
        self.position += [state.position]
        if self.const_size:
            if self.mass is None:
                self.mass = state.mass
        else:
            self.mass += [state.mass]
    
    def fromlist(self, states, const_size=True):
        out = States_Brow(const_size=const_size)
        for state in states:
            out.add(state)
        return out
    
    def makearrays(self):
        if not(self.isarrays):
            self.position = jnp.array(self.position)
            self.mass = jnp.array([self.mass])
            self.isarrays = True
    
    def get_array(self):
        self.makearrays()
        return self.position
    
    def get_mass(self):
        self.makearrays()
        return self.mass
    
