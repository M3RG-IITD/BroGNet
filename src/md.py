from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, value_and_grad, random
# from jax.experimental import optimizers
from jax.example_libraries import optimizers
from jax_md import simulate
from .nve import nve

# ===============================

# ===============================


def dynamics_generator(ensemble, force_fn, shift_fn, params, dt, mass,):
    func = partial(force_fn, mass=mass)
    init, apply = ensemble(lambda R, V: func(R, V, params), shift_fn, dt)
    
    def f(state, runs=100, stride=10):
        return solve_dynamics(
            state, apply, runs=runs, stride=stride)

    return init, f


def prediction(R, V, params, force_fn, shift_fn, dt, mass,  runs=1000, stride=10):
    func = partial(force_fn, mass=mass)
    init, apply = nve(lambda R, V: func(R, V, params), shift_fn, dt)
    state = init(R, V, mass)
    states = solve_dynamics(state, apply, runs=runs, stride=stride)
    return states

def predition_brow(R, params, force_fn, shift_fn, dt, kT, mass, gamma=1.0, runs=1000, stride=10, key=random.PRNGKey(0)):
    init, apply = simulate.brownian(lambda R: force_fn(R, params), shift_fn, dt, kT, gamma)
    state = init(key, R, mass=mass)
    states = solve_dynamics(state, apply, runs, stride)
    return states

def predition_brow_LJ(R, params, force_fn, shift_fn, dt, kT, mass, gamma=1.0, runs=1000, stride=10, key=random.PRNGKey(0)):
    init, apply = simulate.brownian(lambda R: force_fn(R), shift_fn, dt, kT, gamma)
    state = init(key, R, mass=mass)
    states = solve_dynamics(state, apply, runs, stride)
    return states


def predition_brow_learning(R, params, force_fn, shift_fn, dt, kT, mass, gamma_fn, runs=1000, stride=10, key=random.PRNGKey(0)):
    init, apply = simulate.brownian(lambda R: force_fn(R, params), shift_fn, dt, kT, jnp.abs(gamma_fn(R, params)))
    state = init(key, R, mass=mass)
    states = solve_dynamics(state, apply, runs, stride)
    return states


# def prediction(R, V, params, force_fn, shift_fn, dt, mass,  runs=1000, stride=10):
#     func = partial(force_fn, mass=mass)
#     init, apply = nve(lambda R, V: func(R, V, params), shift_fn, dt)
#     state = init(R, V, mass)
#     states = solve_dynamics(state, apply, runs=runs, stride=stride)
#     return states


def solve_dynamics(init_state, apply, runs=100, stride=10):
    step = jit(lambda i, state: apply(state))
    
    def f(state):
        y = jax.lax.fori_loop(0, stride, step, state)
        return y, y
    
    def func(state, i): return f(state)
    
    @jit
    def scan(init_state):
        return jax.lax.scan(func, init_state, jnp.array(range(runs)))
    
    final_state, traj = scan(init_state)
    return traj


# def solve_dynamics(state, apply, runs=100, stride=10):
#     step = jit(lambda i, state: apply(state))
#     states = [state]
#     for i in range(runs):
#         state = lax.fori_loop(0, stride, step, state)
#         states += [state]
#     return states


# def solve_dynamics(state, apply, runs=100, stride=10):
#     step = jit(lambda i, state: apply(state))
#     states = [state]
#     for i in range(runs):
#         state = lax.fori_loop(0, stride, step, state)
#         states += [state]
#     return states

def minimize(R, params, shift, pot_energy_fn, steps=10, stride=None, gtol=1.0e-7, lr=1.0e-3):

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(R)

    def gloss2(R):
        return value_and_grad(lambda R: pot_energy_fn(R, params))(R)

    print(f"Step\tPot. Eng.\t\tTolerance")

    def step_fn(i, init_val):
        R, opt_state, v, _tol = init_val
        v, grads_ = gloss2(R)
        grads = jnp.clip(jnp.nan_to_num(grads_), a_min=-1.0, a_max=1.0)
        opt_state = opt_update(i, grads, opt_state)
        R_ = get_params(opt_state)
        dR = R_ - R
        R, _ = shift(R, dR, R)
        opt_state = opt_state._replace(packed_state=(
            [R, *opt_state.packed_state[0][-2:]],))
        _tol = jnp.square(grads).sum()
        return R, opt_state, v, _tol

    if stride is None:
        stride = 1
        if steps // 10 != 0:
            stride = 10
        if steps // 100 != 0:
            stride = 100

    runs = steps // stride

    states = []

    for i in range(runs):
        v = 0.0
        _tol = 0.0

        R, opt_state, v, _tol = jax.lax.fori_loop(
            i*stride, (i+1)*stride, step_fn, (R, opt_state, v, _tol))

        states += [src.nve.NVEState(R, 0*R, 0*R, jnp.ones(len(R)))]

        print(f"{i*stride}\t{v}\t\t{_tol}")
        if _tol < gtol:
            print(f"gtol reached: {_tol} which is < {gtol}")
            break

    return R, states


def _reflective(R, dR, V, _min=0.0, _max=4.0):
    V_ = V
    R_ = R
    dR_ = jnp.maximum(jnp.minimum(dR, (_max-_min)/2), -(_max-_min)/2)
    V_ = jnp.where(R + dR_ < _min, -V, V)
    V_ = jnp.where(R + dR_ > _max, -V, V_)
    R_ = jnp.where(R + dR_ < _min, 2*_min - (R+dR_), R+dR_)
    R_ = jnp.where(R + dR_ > _max, 2*_max - (R+dR_), R_)
    return R_, V_


def _periodic(R, dR, V, _min=0.0, _max=4.0):
    V_ = V
    R_ = R
    dR_ = jnp.maximum(jnp.minimum(dR, (_max-_min)/2), -(_max-_min)/2)
    R_ = jnp.where(R + dR_ < _min, _max - _min + (R+dR_), R+dR_)
    R_ = jnp.where(R + dR_ > _max, _min - _max + (R+dR_), R_)
    return R_, V_


def _open(R, dR, V):
    """R -> R + dR
    V -> V

    :param R: Position
    :type R: array
    :param dR: Displacement
    :type dR: array
    :param V: Velocity
    :type V: array
    :return: (R+dR, V)
    :rtype: tuple
    """
    return R+dR, V


shift = _open


def displacement(a, b):
    """A - B

    :param a: Vector A
    :type a: array
    :param b: Vector B
    :type b: array
    :return: a-b
    :rtype: array
    """
    return a-b
