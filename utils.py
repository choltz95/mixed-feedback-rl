from jax import numpy as jnp
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
import sys, os, math, time, pickle, itertools
import numpy as np

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