import os,sys, argparse, pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from jax import numpy as jnp
import numpy as np
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

import gym
from gym import wrappers
from tqdm import tqdm
from utils import pwl

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--envname', default='LunarLander-v2', type=str, help='environment name')
parser.add_argument('--it', default=1000, type=int, help='iterations')
parser.add_argument('--mpath', default='./4_dagger_model.h5', type=str, help='model path')
args = parser.parse_args()


net_init, net_apply, net_walk = pwl(
    Dense(1024), Relu,
    Dense(4),
)

params = pickle.load(open(args.mpath,'rb'))

env = gym.make(args.envname)
env = wrappers.Monitor(env, "./gym-results", force=True)

observation = env.reset()
for t in tqdm(range(args.it)):
    env.render()
    action = net_apply(params, observation[None, :])
    action = np.argmax( action )
    observation, reward, done, info = env.step(action)
    if done:
        break
