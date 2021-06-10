#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@The SAC code was adapt from: https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/sac
"""
import numpy as np
import tensorflow as tf
import gym
import time

import os
import random
import gurobipy as gp
from gurobipy import GRB
import mujoco_py

import core
from core import get_vars
#from spinup.utils.logx import EpochLogger
tf.compat.v1.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'

SAVE_FILE = True
random_seed = 0
hidden_layer = [400, 300]

before_action = []
after_action = []

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


#############################  Projection  ####################################

def Projection(action,state):
    with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as half_m:
                neta1=action[0]
                neta2=action[1]
                neta3=action[2]
                neta4=action[3]
                neta5=action[4]
                neta6=action[5]
                w1=state[11]
                w2=state[12]
                w3=state[13]
                w4=state[14]
                w5=state[15]
                w6=state[16]
                a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
                a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
                a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
                a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
                a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
                a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
                v = half_m.addVar(ub=20, name="v")
                u1 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u1")
                u2 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u2")
                u3 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u3")
                u4 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u4")
                u5 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u5")
                u6 = half_m.addVar(lb=-gp.GRB.INFINITY, name="u6")
                abs_u1 = half_m.addVar(ub=20, name="abs_u1")
                abs_u2 = half_m.addVar(ub=20, name="abs_u2")
                abs_u3 = half_m.addVar(ub=20, name="abs_u3")
                abs_u4 = half_m.addVar(ub=20, name="abs_u4")
                abs_u5 = half_m.addVar(ub=20, name="abs_u5")
                abs_u6 = half_m.addVar(ub=20, name="abs_u6")
                obj= (a1-neta1)**2+ (a2-neta2)**2 + (a3-neta3)**2+ (a4-neta4)**2 + (a5-neta5)**2+ (a6-neta6)**2
                half_m.setObjective(obj,GRB.MINIMIZE)
                
                
                half_m.addConstr(u1==a1*w1)   
                half_m.addConstr(u2==a2*w2)
                half_m.addConstr(u3==a3*w3)   
                half_m.addConstr(u4==a4*w4)
                half_m.addConstr(u5==a5*w5)   
                half_m.addConstr(u6==a6*w6)
                half_m.addConstr(abs_u1==(gp.abs_(u1)))
                half_m.addConstr(abs_u2==(gp.abs_(u2)))
                half_m.addConstr(abs_u3==(gp.abs_(u3)))
                half_m.addConstr(abs_u4==(gp.abs_(u4)))
                half_m.addConstr(abs_u5==(gp.abs_(u5)))
                half_m.addConstr(abs_u6==(gp.abs_(u6)))
                half_m.addConstr((abs_u1 + abs_u2 + abs_u3 + abs_u4 + abs_u5 + abs_u6 )== v)
    
    
                half_m.optimize()
                return half_m.X[0:6]
            
##################################################################################


def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        eval_freq=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=0.0005, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    max_ep_len = env._max_episode_steps

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    ######################## Seed ###############################
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    ############################################################

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.compat.v1.variable_scope('main'):
        mu, pi, logp_pi, q1, q2 = actor_critic(x_ph, a_ph, **ac_kwargs)

    with tf.compat.v1.variable_scope('main', reuse=True):
        # compose q with pi, for pi-learning
        _, _, _, q1_pi, q2_pi = actor_critic(x_ph, pi, **ac_kwargs)

        # get actions and log probs of actions for next states, for Q-learning
        _, pi_next, logp_pi_next, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.compat.v1.variable_scope('target'):
        # target q values, using actions from *current* policy
        _, _, _, q1_targ, q2_targ  = actor_critic(x2_ph, pi_next, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_targ = tf.minimum(q1_targ, q2_targ)

    # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_targ - alpha * logp_pi_next))

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(input_tensor=alpha * logp_pi - min_q_pi)
    q1_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean(input_tensor=(q_backup - q2)**2)
    value_loss = q1_loss + q2_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.compat.v1.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.compat.v1.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
  

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent():
        test_env.seed(random_seed + 100)
        avgreward = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                # a = get_action(o, True)
                a = get_action(o, False)
                a = Projection(a, o)
                action_feasibility = (abs(a[0]*o[11]) + abs(a[1]*o[12]) + abs(a[2]*o[13]) + abs(a[3]*o[14]) + abs(a[4]*o[15]) + abs(a[5]*o[16]))
                assert action_feasibility - 20 <= 1e-6, 'Unfeasible!!! result is: %.6f' % action_feasibility
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1

            print('Episode {}\tReward: {} \t AvgReward'.format(j, ep_ret))
            avgreward += ep_ret
        avgreward = avgreward / num_test_episodes
        print("------------------------------------------------")
        print("Evaluation average reward :",avgreward)
        print("------------------------------------------------")
        return avgreward
       

    store_ewma = []
    store_eval_reward = []
    store_before_action = []
    store_after_action = []
    store_state = []

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    total_steps = max_ep_len * epochs
    episode_num = 0
    ewma=0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        if t >= start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()
        store_before_action.append(a)
        before_action = a # debug
        
        a = Projection(a, o)
        action_feasibility = (abs(a[0]*o[11]) + abs(a[1]*o[12]) + abs(a[2]*o[13]) + abs(a[3]*o[14]) + abs(a[4]*o[15]) + abs(a[5]*o[16]))
        assert action_feasibility - 20 <= 1e-6, 'Unfeasible!!! result is: %.6f' % action_feasibility
        store_after_action.append(a)
        store_state.append(o)
        after_action = a # debug
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)


        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # epoch = (t+1) // steps_per_epoch
            ewma=ewma*0.95+ep_ret*0.05
            print('Episode time: ', time.time() - start_time)
            print("t:", t+1, "episode:", episode_num+1,"Reward:",ep_ret,"ewma:",ewma)
            start_time = time.time()

            store_ewma.append(ewma)
            if SAVE_FILE:
                np.save(
                    filepath+'/cheetah_con2_seed{}_ewma_reward'.format(seed), np.array(store_ewma))
                np.save(
                    filepath+'/cheetah_con2_seed{}_after_action'.format(seed), np.array(store_after_action))
                np.save(
                    filepath+'/cheetah_con2_seed{}_before_action'.format(seed), np.array(store_before_action))
                np.save(
                    filepath+'/cheetah_con2_seed{}_state'.format(seed), np.array(store_state))
            
            # np.save(
            #     filepath+'/cheetah_con2_seed{}_ewma_reward'.format(seed), np.array(store_ewma))
            # np.save(
            #     filepath+'/cheetah_con2_seed{}_after_action'.format(seed), np.array(store_after_action))
            # np.save(
            #     filepath+'/cheetah_con2_seed{}_before_action'.format(seed), np.array(store_before_action))
            o, ep_ret, ep_len = env.reset(), 0, 0
            episode_num += 1

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                outs = sess.run(step_ops, feed_dict)
              
        # End of epoch wrap-up
        if (t+1) % eval_freq == 0:
            # Test the performance of the deterministic version of the agent.
            store_eval_reward.append(test_agent())
            if SAVE_FILE:
                np.save(filepath+'/cheetah_con2_seed{}_eval_reward'.format(seed), np.array(store_eval_reward))

    # np.save(
    #     filepath+'/cheetah_con2_seed{}_ewma_reward'.format(seed), np.array(store_ewma))
    # np.save(
    #     filepath+'/cheetah_con2_seed{}_after_action'.format(seed), np.array(store_after_action))
    # np.save(
    #     filepath+'/cheetah_con2_seed{}_before_action'.format(seed), np.array(store_before_action))

           

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=random_seed)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    filepath = "./results_cheetah_con2_sac"
    if not os.path.exists(filepath):
        os.makedirs(filepath)


    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=hidden_layer),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=0)