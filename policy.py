from jax import numpy as jnp
from jax import lax
from jax import device_put
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
import sys, os, math, time, pickle, itertools
import numpy as np
from functools import partial
from tqdm.notebook import tqdm


class Policy:
    def __init__(self, rng, in_shape, net, env=None, name='policy', losstype='mse', discrete=False, trainable=True, cost_sensitive=False, constrained=False, active=False):
        self.active = active # is this an active learning policy?
        self.discrete = discrete # is this a discrete environment? 
        self.trainable = trainable # is this a trainable policy?
        self.training=False
        self.done = False
        
        self.env = env
        self.cur_obs=None 
        self.Vs = [jnp.array([1])]
        self.r_t = [1]
        self.dsct_fctr = 0.95
        
        self.expert_observations = []
        self.expert_actions = []
        self.self_observations = []
        self.self_actions = []
        
        self.episode_rewards = []
        self.rollout_rewards = [] # rewards for current rollout
        self.rewards = [] # mean rewards per set of rollouts
        self.num_data = [] # size of datasets
        self.reward_accum = 0.0
                
        self.X = jnp.array([])
        self.y = jnp.array([])

        self.X_expert = jnp.array([])
        self.y_expert = jnp.array([])
        self.X_self = jnp.array([])
        self.y_self = jnp.array([])
        
        self.self_X = jnp.array([])
        self.self_y = jnp.array([])
        
        self.active_mask = jnp.array([],dtype=jnp.bool_) # active learning mask
        self.cur_active_mask = []
        self.self_mask = jnp.array([],dtype=jnp.bool_) # active learning mask
        self.cur_self_mask = []
        self.importance_weights = jnp.array([])  # aggrevate weights
        self.cost_sensitive = cost_sensitive
        
        self.confs = []
        
        self.init_params = ()
        self.params = ()
        
        self.std_mean = 0.0 ## only for stanford gaussian policy experts
        self.std_std = 1.0
        
        self.rng = rng
        
        self.net_init, self.net_apply, self.net_walk = net
        
        self.in_shape = in_shape
        self.out_shape = ()
        
        self.opt_init = None
        self.opt_update = None
        self.get_params = None
        self.opt_state = None
        self.loss = None
        
        self.name = name
        self.constrained = constrained
                
        if losstype=='mse':
            self.loss = self.mseloss
        else:
            self.loss = self.celoss    
            
        self.g1 = []
        self.g2 = []
     
    """
    Initialize policy for epoch
    """
    def init_policy(self):
        if self.trainable:
            self.out_shape, self.init_params = self.net_init(self.rng, self.in_shape)
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=0.001)
            self.opt_state = self.opt_init(self.init_params)
            self.params = self.init_params
            self.training=True
            self.cur_active_mask = []
            self.cur_self_mask = []
        self.rewards.append([])
        self.confs.append([])
        self.expert_observations = []
        self.expert_actions = []
        self.self_observations = []
        self.self_actions = []

    """
    Reset environment before rollout
    """
    def reset_env(self):
        self.cur_obs = self.env.reset()
        self.done=False
        self.reward_accum = 0.0
        self.r_t = [1]
   
    @partial(jit, static_argnums=(0,))
    def value_fn(self, d_i, d_i_til):
        V = (jnp.triu(jnp.power(self.dsct_fctr, d_i_til - d_i_til.T))*(-jnp.array(self.r_t))).sum(axis=1)
        V = (V - V.mean())/V.std() + 1 
        return V
        
    """
    Take single env step
    """
    def step(self, d_i):
        logits, action, conf = self.take_action(self.cur_obs)
        if not self.discrete:
            action = logits
        obs, r, done, info = self.env.step(np.array(action))
        if done:
            d_i_ar = jnp.arange(0,d_i)
            d_i_til = jnp.tile(d_i_ar,(d_i,1))
            V = self.value_fn(d_i, d_i_til)
            self.Vs.append(V)
            self.done=True
        self.cur_obs = obs
        self.reward_accum += r
        return obs, r, done
    
    @partial(jit, static_argnums=(0,))
    def _take_action(self, params, obs):
        logits = self.net_apply(params, obs)
        softm_logits = stax.softmax(logits.flatten())
        sorted_logits = jnp.sort(softm_logits)
        conf = sorted_logits[-1] - sorted_logits[-2]
        action = jnp.argmax(logits)
        return logits,action,conf  
    
    """
    Apply policy and return logits, action, confidence
    """
    def take_action(self, obs):
        #obs = (obs - self.std_mean)/(self.std_std+1e-6)
        return self._take_action(self.params, obs)
    
    @partial(jit, static_argnums=(0,))
    def tr_step(self, params, i, opt_state, batch):
        x1, y1, w = batch
        p = params
        g = grad(self.loss)(p, x1, y1, w)
        return self.opt_update(i, g, opt_state)

    @partial(jit, static_argnums=(0,))
    def tr_constraint_step(self, params, i, opt_state, inputs, cons):
        x, y, w = inputs
        cx, cy, w = cons
        p = params
        def reg(p,x,cx,cy):
            return self.m_reg(p,x,cx)
        g = grad(reg)(p, x, cx, cy)
        return self.opt_update(i, g, opt_state)    
        
    """
    Train policy on current data
    """
    def fit_policy(self,constraints=(),batch_size=64, epochs=30):
        assert self.trainable
        if self.active:
            #X_tr = jnp.concatenate([self.X_expert[self.active_mask,:],self.X_self[~self.active_mask,:]])
            #y_tr = jnp.concatenate([self.y_expert[self.active_mask,:],self.y_self[~self.active_mask,:]])
            X_tr = jnp.concatenate([self.X_expert[self.active_mask,:],self.X_self[self.self_mask]])
            y_tr = jnp.concatenate([self.y_expert[self.active_mask,:],self.y_self[self.self_mask]])
            w = jnp.concatenate([self.importance_weights[self.active_mask],self.importance_weights[self.self_mask]])
        else:
            X_tr = self.X_expert[self.active_mask,:]
            y_tr = self.y_expert[self.active_mask,:]
            w = self.importance_weights[self.active_mask]
                    
        num_train = X_tr.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        
        def data_stream(X,y):
            while True:
                rng = np.random.RandomState(0)
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield X[batch_idx], y[batch_idx], w[batch_idx]
                    
        batches = data_stream(X_tr,y_tr)
        batches_constraints = data_stream(self.X_expert, self.y_expert)
        itercount = itertools.count()

        for epoch in tqdm(range(epochs),desc='tr policy', position=2, leave=False):
            for _ in range(num_batches):
                ii = next(itercount)
                b = next(batches)
                c = next(batches_constraints)
                p = self.get_params(self.opt_state)
                self.opt_state = self.tr_step(p, ii, self.opt_state, b)
                if self.constrained:
                    self.opt_state = self.tr_constraint_step(p, ii, self.opt_state, b, c)

        params = self.get_params(self.opt_state)
        self.params = params

    """
    Aggregate data
    """
    def aggregate(self, data=None, weights=None, mask=None):        
        if data:
            expert_observations, expert_actions = data#[0]
            self_observations, self_actions = data#[1]
            if weights is None:
                weights = jnp.ones(len(expert_observations))
            if mask is None:
                mask = jnp.ones(len(expert_observations), dtype=jnp.bool_)
                self_mask = jnp.ones(len(expert_observations), dtype=jnp.bool_)
        else:
            expert_observations, expert_actions = (self.expert_observations, self.expert_actions)
            self_observations, self_actions = (self.self_observations, self.self_actions)
            weights = jnp.concatenate(self.Vs)
            mask = jnp.array(self.cur_active_mask, dtype=jnp.bool_)
            self_mask = jnp.array(self.cur_self_mask, dtype=jnp.bool_)
        
        if not self.active:
            mask = jnp.ones(len(expert_observations), dtype=jnp.bool_)
        if not self.cost_sensitive:
            weights = jnp.ones_like(weights)

        if len(self.num_data) == 0:
            self.X_expert = jnp.array(expert_observations)
            self.y_expert = jnp.array(expert_actions)
            self.X_self = jnp.array(self_observations)
            self.y_self = jnp.array(self_actions)
            self.importance_weights = weights
            self.active_mask  = mask
            self.self_mask = self_mask
        else:
            self.X_expert = jnp.concatenate((self.X_expert, jnp.array(expert_observations)))
            self.y_expert = jnp.concatenate((self.y_expert, jnp.array(expert_actions)))
            if len(self_observations) > 0:
                self.X_self = jnp.concatenate((self.X_self, jnp.array(self_observations)))
                self.y_self = jnp.concatenate((self.y_self, jnp.array(self_actions)))
            self.importance_weights = jnp.concatenate((self.importance_weights, jnp.array(weights)))
            self.active_mask = jnp.concatenate((self.active_mask, jnp.array(mask)))
            self.self_mask = jnp.concatenate((self.self_mask, jnp.array(self_mask)))
            
        self.num_data.append(self.active_mask.sum().item())
        
    def aggregate_old(self, data, weights=None, mask=None):
        new_observations, new_actions = data
        if weights is None:
            weights = jnp.ones(new_observations.shape[0])
        if mask is None or len(mask) == 0:
            mask = jnp.ones(new_observations.shape[0], dtype=jnp.bool_)

        if len(self.num_data) == 0:
            self.X = jnp.array(new_observations)
            self.y = jnp.array(new_actions)
            self.importance_weights = weights
            self.active_mask  = mask
        else:
            self.X = jnp.concatenate((self.X, jnp.array(new_observations)))
            self.y = jnp.concatenate((self.y, jnp.array(new_actions)))
            self.importance_weights = jnp.concatenate((self.importance_weights, jnp.array(weights)))
            self.active_mask = jnp.concatenate((self.active_mask, jnp.array(mask)))
            
        self.num_data.append(self.active_mask.sum().item())
            
    # need to separate these somehow
    def mseloss(self, params, inputs, targets, weights):
        predictions = self.net_apply(params, inputs)
        loss = jnp.mean(jnp.linalg.norm(targets - predictions)**2 * weights)
        return loss
        
    def barrier(self, params, inputs, targets):
        predictions = self.net_apply(params, inputs)
        #loss = jnp.clip(jnp.log(jnp.linalg.norm(targets - predictions)),a_min=-1e10)
        loss = jnp.mean((targets - predictions)**2)
        return loss
    
    def m_reg(self, params, inputs, constraints):
        """
        calculate pairwise euclidean distance
        between all pair-rows of A and B
        """
        def fastdiff(A,B):
            return jnp.sum((A[:, None, :] - B[None, :, :])**2, axis=-1)
        N_b = inputs.shape[0]
        N_c = constraints.shape[0]
        predictions_c = self.net_apply(params, inputs)
        predictions_b = self.net_apply(params, constraints)
        xijdiff = fastdiff(constraints,inputs) + 1
        sigma = 100.0
        #xijdiff = np.exp(-np.divide(xijdiff,2*np.square(sigma)))
        fijdiff = fastdiff(predictions_c,predictions_b)
        nn = jnp.divide(fijdiff, xijdiff.T)
        reg = jnp.divide(jnp.sum(nn),N_b * N_c)

        return reg 

    def celoss(self, params, inputs, targets, weights):
        logits = self.net_apply(params, inputs)
        logits = stax.logsoftmax(logits)  # log normalize
        loss = -jnp.mean(jnp.sum(logits * targets, axis=1) * weights)  # cross entropy loss
        return loss
    
    def save(self, folder, fname):
        pickle.dump(self.params, open('{}{}.h5'.format(folder,fname), "wb" ))

    def deserialize(self,folder,fname):
        tmp_dict = Pickle.load(open('{}{}.pkl'.format(folder,fname), 'rb'))
        self.__dict__.update(tmp_dict) 
        
    def serialize(self, folder, fname):
        pickle.dump(self.__dict__, open('{}{}.pkl'.format(folder,fname), "wb" ))