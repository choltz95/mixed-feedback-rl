from jax import numpy as jnp
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
import sys, os, math, time, pickle, itertools
import numpy as np

import gym
from gym import wrappers
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, axis

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

class Policy:
    def __init__(self, rng, net, opt):
        self.observations = []
        self.rewards = [] # mean rewards per rollout
        self.num_data = []
        
        self.X = []
        self.y = []
        self.active_mask = []
        
        self.env
        self.params
        
        self.rng = rng
        self.net_init, self.net_apply, self.net_walk = net
        self.init_params
        self.in_shape
        self.out_shape
        self.opt_init, self.opt_update, self.get_params = opt
        self.opt_state
        self.loss
        
    def reset_policy(self):
        self.out_shape, self.init_params = self.net_init(self.rng, self.in_shape)
        self.opt_init, self.opt_update, selfget_params = optimizers.adam(step_size=0.001)
        self.opt_state = self.opt_init(self.init_params)
        
    @jit
    def step(i, opt_state, batch):
        x1, y1 = batch
        p = self.get_params(opt_state)
        g = grad(self.loss)(p, x1, y1)
        return opt_update(i, g, opt_state)
         
    def fit_policy(batch_size, epochs):
        X_tr = self.X
        y_tr = self.y
        num_train = X_tr.shape[0]

        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        def data_stream():
            while True:
                perm = self.rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield X_tr[batch_idx], y_tr[batch_idx]
        batches = data_stream()

        itercount = itertools.count()

        for epoch in tqdm(range(epochs),desc='tr policy', position=2, leave=False):
            for _ in range(num_batches):
                ii = next(itercount)
                self.opt_state = step(ii, self.opt_state, next(batches))

        params = self.get_params(self.opt_state)
        self.params = params
        
    def aggregate_data(self, data):
        new_observations, new_actions = data
        self.X = np.concatenate((self.X, np.array(new_observations)))
        new_actions = np.array(new_actions)
        self.y = np.concatenate((self.y, np.array(new_actions.reshape(new_actions.shape[0], new_actions.shape[1]))))


    def to_file(self, folder, this_run_file):
        pass

def make_plot(policies, nonpolicies):
    fig = figure(figsize=(30, 5))
    num_policies = len(policies)
    #num_epochs = np.arange(len(policies[0].num_data)-1)
    
    a=fig.add_subplot(1,2,1)
    a.set_title('mean reward/episode')
    for i,policy in enumerate(policies):
        means = [np.mean(r) for r in policy.rewards]
        stds = [np.std(r) for r in policy.rewards]
        a.plot(np.array(means),label=policy.name)
        plt.errorbar(np.arange(len(policy.num_data)-1), np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')
        handles, labels = a.get_legend_handles_labels()
        fig.legend(handles, labels)
        
    for i,policy in enumerate(nonpolicies):
        means = [np.mean(r) for r in policy[1]]
        stds = [np.std(r) for r in policy[1]]
        a.plot(np.array(means),label=policy[0])
        plt.errorbar(np.arange(len(policy[1])), np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')

        handles, labels = a.get_legend_handles_labels()
        fig.legend(handles, labels)        
        
    a=fig.add_subplot(1,2,2)
    a.set_title('mean reward/#data')
    for i,policy in enumerate(policies):
        means = [np.mean(r) for r in policy.rewards]
        stds = [np.std(r) for r in policy.rewards]
        num_data = policy.num_data[1:]
        if policy.name=='expert':
            num_data = policy.num_data
            means = [means[0]] + means
            stds = [stds[0]] + stds
            a.plot(num_data,np.array(means),label=policy.name)
            plt.errorbar(num_data, np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')
        else:
            a.plot(num_data,np.array(means),label=policy.name)
            plt.errorbar(num_data, np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')

        handles, labels = a.get_legend_handles_labels()
        fig.legend(handles, labels)
        
    for i,policy in enumerate(nonpolicies):
        means = [np.mean(r) for r in policy[1]]
        stds = [np.std(r) for r in policy[1]]
        num_data = policy[2][1:]
        if policy[0] == 'noisy expert':
            num_data = policy[2]
            means = [means[0]] + means
            stds = [stds[0]] + stds
            a.plot(num_data,np.array(means),label=policy[0])
            plt.errorbar(num_data, np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')
        else:
            a.plot(num_data,np.array(means),label=policy[0])
            plt.errorbar(num_data, np.array(means), yerr=np.array(stds)/2, ls = 'none',ecolor='black')
        
        
        handles, labels = a.get_legend_handles_labels()
        fig.legend(handles, labels) 
        
def plot_and_evaluate():
    expert.num_data = [0] + expert.num_data
    expert.rewards = [expert.rewards[0]] + expert.rewards
    noisy_expert_rewards = [noisy_expert_rewards[0]] + noisy_expert_rewards
    make_plot([dagger,expert,active_dagger, aggrevate, active_aggrevate],[('noisy expert',noisy_expert_rewards, expert.num_data)])
    constrained_exp = expert.apply_policy(constrained_obs)
    
    for policy in [dagger,expert,active_dagger, aggrevate, active_aggrevate]:
        constrained = policy.apply_policy(constrained_obs)
        print(policy.name,np.mean(policy.rewards[-1]), np.linalg.norm(constrained - constrained_exp))
        
def make_envs(envname='LunarLander-v2'):
    env = gym.make(envname)
    env_expert = deepcopy(env)
    env_noisy_expert = deepcopy(env)
    env_dagger = deepcopy(env)
    env_active_dagger = deepcopy(env)
    env_aggrevate = deepcopy(env)
    env_active_aggrevate = deepcopy(env)
    
    return env, (env_expert, env_noisy_expert, env_dagger, env_active_dagger, env_aggrevate, env_active_aggrevate)
