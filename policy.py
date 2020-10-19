from jax import numpy as jnp
from jax import random, grad, vmap, jit, tree_multimap
from jax.experimental import stax, optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
import sys, os, math, time, pickle, itertools
import numpy as np

from tqdm.notebook import tqdm


class Policy:
    def __init__(self, rng, in_shape, net, env=None, name='policy', losstype='mse', discrete=False, trainable=True, cost_sensitive=False, constrained=False, active=False):
        self.trainable=trainable
        self.training=False
        
        self.env = env
        self.cur_obs=None
        self.discrete = discrete
        
        self.expert_observations = []
        self.expert_actions = []
        
        self.episode_rewards = []
        self.rollout_rewards = [] # rewards for current rollout
        self.rewards = [] # mean rewards per set of rollouts
        self.num_data = [] # size of datasets
        self.reward_accum = 0.0
                
        self.X = jnp.array([])
        self.y = jnp.array([])
        
        self.self_X = jnp.array([])
        self.self_y = jnp.array([])
        
        self.active_mask = jnp.array([],dtype=jnp.bool_) # active learning mask
        self.importance_weights = jnp.array([])  # aggrevate weights
        self.cost_sensitive = cost_sensitive
        self.active = active
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
        self.rewards.append([])
        self.confs.append([])

    """
    Reset environment before rollout
    """
    def reset_env(self):
        self.cur_obs = self.env.reset()
        self.reward_accum = 0.0
        
    """
    Take single env step
    """
    def step(self):
        logits, action, conf = self.take_action(self.cur_obs)
        if not self.discrete:
            action = logits
        obs, r, done, info = self.env.step(np.array(action))
        self.cur_obs = obs
        self.reward_accum += r
        return obs, r, done

    """
    Apply policy net to an observation
    """
    def apply_policy(self,obs):
        #obs = (obs - self.std_mean)/(self.std_std+1e-6)
        return self.net_apply(self.params, obs)
            
    """
    Apply policy and return logits, action, confidence
    """
    def take_action(self, obs):
        #obs = (obs - self.std_mean)/(self.std_std+1e-6)
        logits = self.net_apply(self.params, obs)
        softm_logits = stax.softmax(logits.flatten())
        sorted_logits = np.sort(softm_logits)
        conf = sorted_logits[-1] - sorted_logits[-2]
        action = np.argmax(logits)
        return logits, action, conf
         
    """
    Train policy on current data
    """
    def fit_policy(self,constraints=(),batch_size=64, epochs=30):
        assert self.trainable
        @jit
        def step(i, opt_state, batch):
            x1, y1, w = batch
            p = self.get_params(opt_state)
            g = grad(self.loss)(p, x1, y1, w)
            #self.g1.append(g)
            return self.opt_update(i, g, opt_state)

        @jit
        def constraint_step(i, opt_state, inputs, constraints):
            x, y, w = inputs
            cx,cy = constraints
            p = self.get_params(opt_state)
            def reg(p,x,cx,cy):
                #return self.barrier(p,cx,cy) + 10.0*self.m_reg(p,x,cx)
                return self.m_reg(p,x,cx)
            g = grad(reg)(p, x, cx, cy)
            #self.g2.append(g)
            return self.opt_update(i, g, opt_state)
        X_tr = self.X[self.active_mask,:]
        y_tr = self.y[self.active_mask,:]
        w = self.importance_weights[self.active_mask]
                    
        num_train = X_tr.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        
        def data_stream():
            while True:
                rng = np.random.RandomState(0)
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield X_tr[batch_idx], y_tr[batch_idx], w[batch_idx]
        batches = data_stream()
        itercount = itertools.count()

        for epoch in tqdm(range(epochs),desc='tr policy', position=2, leave=False):
            for _ in range(num_batches):
                ii = next(itercount)
                b = next(batches)
                self.opt_state = step(ii, self.opt_state, b)
                if self.constrained and len(constraints)>0:
                    self.opt_state = constraint_step(ii, self.opt_state, b, constraints)

        params = self.get_params(self.opt_state)
        self.params = params

    """
    Aggregate data
    """
    def aggregate(self, data, weights=None, mask=None):
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