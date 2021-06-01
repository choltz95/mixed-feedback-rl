from jax import numpy as jnp
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Tanh, Flatten, LogSoftmax, Softmax, Dropout

import sys, os, math, time, pickle, itertools
import numpy as np


def get_network(name,outshape=4, mode=None):
    active_policy = pwl(
        Dense(128), Relu,
        Dense(1),
    )
    
    gate_small = pwl(
        Dense(16), Relu,
        Dropout(0.2, mode=mode),
        Dense(outshape),
        Softmax
    )

    agent_small = pwl(
        Dense(16), Relu,
        Dense(outshape),
    )
    
    agent_medium = pwl(
        Dense(64), Tanh,
        Dense(outshape), Tanh
    )

    lander_expert = pwl(
        Dense(16), Relu,
        Dense(16), Relu,
        Dense(16), Relu,
        Dense(4), Softmax
    )
    
    cts_lander = pwl(
        Dense(400), Relu,
        Dense(300), Relu,
        Dense(2), Tanh
    )
    
    walker_expert = pwl(
        Dense(256), Tanh,
        Dense(256), Tanh,
        Dense(6)
    )
    
    hopper_expert = pwl(
        Dense(64), Tanh,
        Dense(64), Tanh,
        Dense(6)
    )
    
    cheetah_expert1 = pwl(
        Dense(64), Tanh,
        Dense(64), Tanh,
        Dense(6)
    )
    
    cheetah_expert2 = pwl(
        Dense(256), Tanh,
        Dense(256), Tanh,
        Dense(6)
    )

    cartpole_expert = pwl(
        Dense(64), Tanh,
        Dense(64), Tanh,
        Dense(2)
    )
    
    networks = {
        'active-policy':active_policy,
        'agent-small':agent_small,
        'agent-medium':agent_medium,
        'lander-expert':lander_expert,
        'lander-cts-expert':cts_lander,
        'cartpole-expert':cartpole_expert,
        'hopper-expert':hopper_expert,
        'cheetah-expert':cheetah_expert1,
        'cheetah-expert2':cheetah_expert2,
        'walker-expert':walker_expert,
        'gate-small':gate_small
    }
    
    return networks[name]

def pwl(*layers):
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)
    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params
    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs
    def walk_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        profiles = []
        activations = []
        for i, (fun, param, rng) in enumerate(zip(apply_funs, params, rngs)):
            l = len(param)
            if ((len(param) == 0) and (i != 0) and (i!=4) and i != len(apply_funs)):
                activations.append(inputs)
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs, activations
    return init_fun, apply_fun, walk_fun
