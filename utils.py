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

def make_plot(policies, nonpolicies):
    fig = figure(figsize=(30, 5))
    num_policies = len(policies)
    
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
        
def make_envs(envname='LunarLander-v2', numenvs=0):
    env = gym.make(envname)
    envs = []
    for _ in range(numenvs):
        envs.append(deepcopy(env))
    return env, envs
