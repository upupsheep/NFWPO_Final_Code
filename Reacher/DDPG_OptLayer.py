#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:31 2021

@author: Yang-Shang-Hsuan
@The DDPG code was adapt from: https://github.com/sfujim/TD3/blob/master/DDPG.py
"""
# -*- coding: utf-8 -*-
import gym
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import pickle
import traceback
import time
from torch.autograd import Function, Variable
from qpth.qp import QPFunction, QPSolvers
import gurobipy as gp
from gurobipy import GRB

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

IS_TRAINING = 0
IS_EVAL = 1
IS_SELECTING = 2
IS_OTHER = 3

#####################  hyper parameters  ####################

MAX_EPISODES = 500000
MAX_EP_STEPS = 1000
TOTAL_STEPS = 1e+6
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.001     # soft replacement
MEMORY_CAPACITY = 10000
c = 0.1
BATCH_SIZE = 16
RENDER = False
start_training = 1000
random_seed = 4

arg_env = 'Reacher-v2'

EVAL = True
eval_freq = 5000

SAVE_FILE = False
filepath = "./results_reacher_gaussian"
if not os.path.exists(filepath):
    os.makedirs(filepath)

SAVE_MODEL = False
LOAD_MODEL = False
file_name = "reacher_gaussian_seed{}".format(random_seed)

before_opt = []
after_opt = []
after_gaussian = []
eval_before_opt = []
eval_after_opt = []
#############################################################

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    eval_action = []

    if LOAD_MODEL:
        eval_action = list(
            np.load(filepath+'/reacher_seed{}_eval_action.npy'.format(random_seed)))

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(
                np.array(state), None, training=IS_EVAL)
            eval_action.append(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    if SAVE_FILE:
        np.save(filepath+'/reacher_seed{}_eval_action'.format(random_seed),
                np.array(eval_action))
    return avg_reward


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def OptLayer_function(action, random_sample=False):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as reacher_m:

            batch_size = action.shape[0]
            if random_sample:
                rtn_action = np.array([])
            else:
                rtn_action = torch.tensor([])

            for i in range(batch_size):
                batch_action = action[i]
                neta1 = batch_action[0]
                neta2 = batch_action[1]

                a1 = reacher_m.addVar(
                    lb=-1, ub=1, name='a1', vtype=GRB.CONTINUOUS)
                a2 = reacher_m.addVar(
                    lb=-1, ub=1, name='a2', vtype=GRB.CONTINUOUS)

                obj = (a1-neta1)**2 + (a2-neta2)**2
                reacher_m.setObjective(obj, GRB.MINIMIZE)

                reacher_m.addConstr(a1+a2 <= 0.1)
                reacher_m.addConstr(-0.1 <= a1+a2)
                reacher_m.addConstr((a1**2+a2**2) <= 0.02)

                reacher_m.optimize()

                if random_sample:
                    rtn_action = np.append(rtn_action, [a1.X, a2.X])
                else:
                    batch_opt_a = torch.tensor([a1.X, a2.X])
                    rtn_action = torch.cat((rtn_action, batch_opt_a))

            if random_sample:
                return np.reshape(rtn_action, (batch_size, action.shape[1]))
            else:
                return torch.reshape(rtn_action, (batch_size, action.shape[1])).float()


class OptLayer(torch.nn.Module):
    def __init__(self, D_in, D_out, a_bound):
        super(OptLayer, self).__init__()

        cons1_size = 1
        cons2_size = 1

        # Q_sqrt = cp.Parameter((D_in, D_in))
        q = cp.Parameter(D_in)  # input: atilde => w@q+b

        # linear constraint
        G = cp.Parameter((cons1_size, D_in))
        h = cp.Parameter(cons1_size)

        # Ax >= 0
        A = cp.Parameter((cons2_size, D_in))
        b = cp.Parameter(cons2_size)

        x = cp.Variable(D_out)  # output
        obj = cp.Minimize(0.5*cp.sum_squares(x) - q.T @ x)
        cons = [A @ x >= b, G @ x <= h, cp.sum_squares(x) <= 0.02]

        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(
            prob, parameters=[q, A, b, G, h], variables=[x])

    def forward(self, x):
        Gval = torch.tensor([[1, 1]], dtype=torch.float32,
                            requires_grad=False).to(device)
        hval = torch.tensor([0.1], dtype=torch.float32,
                            requires_grad=False).to(device)
        Aval = torch.tensor([[1, 1]], dtype=torch.float32,
                            requires_grad=False).to(device)
        bval = torch.tensor([-0.1], dtype=torch.float32,
                            requires_grad=False).to(device)

        return self.layer(x, Aval, bval, Gval, hval)[0]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.opt_layer = OptLayer(action_dim, action_dim, max_action)

        self.max_action = max_action

    def forward(self, state, training):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        scaled_a = self.max_action * torch.tanh(self.l3(a))

        if training == IS_TRAINING:
            opt_a = self.opt_layer(scaled_a)
        elif training == IS_SELECTING:
            noise_a = scaled_a + \
                torch.normal(0, 0.1, size=scaled_a.shape).to(device)
            opt_a = OptLayer_function(noise_a)

            before_opt.append(scaled_a.detach().cpu().numpy())
            after_gaussian.append(noise_a.detach().cpu().numpy())
        elif training == IS_EVAL:
            opt_a = OptLayer_function(scaled_a)

            eval_before_opt.append(scaled_a.detach().cpu().numpy())
            eval_after_opt.append(opt_a.detach().cpu().numpy())
        elif training == IS_OTHER:
            opt_a = OptLayer_function(scaled_a).to(device)
        else:
            print("Forward parameter error: traing =", training)
            exit(0)
        return opt_a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_perturbed = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau

    def select_action(self, state, para, training):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state, training=training).cpu().data.numpy().flatten()

    def train(self, replay_buffer, t, batch_size=64):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(
            next_state, self.actor_target(next_state, training=IS_OTHER))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the actor
        if (t + 1) % 50 == 0:
            # Compute actor loss
            actor_loss = - \
                self.critic(state, self.actor(
                    state, training=IS_TRAINING)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if SAVE_MODEL and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(arg_env)

    # Set seeds
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": GAMMA,
        "tau": TAU,
    }

    # # Initialize policy
    policy = DDPG(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    ewma_r = 0
    episode_timesteps = 0
    episode_num = 0

    evaluations = []
    store_ewma = []
    store_action = []
    store_reward = []
    store_state = []

    if LOAD_MODEL:
        print("=======LOAD")
        policy_file = file_name
        policy.load(f"./models/{policy_file}")
        with open('replay_buffer_reacher_gaussian_seed{}.pkl'.format(random_seed), 'rb') as input:
            replay_buffer = pickle.load(input)
        print("==========end load")
        print("r size: ", replay_buffer.size)

        store_reward = list(
            np.load(filepath+'/reacher_seed{}_episode_reward.npy'.format(random_seed)))
        store_ewma = list(
            np.load(filepath+'/reacher_seed{}_ewma_reward.npy'.format(random_seed)))
        store_action = list(
            np.load(filepath+'/reacher_seed{}_action.npy'.format(random_seed)))
        before_opt = list(np.load(
            filepath+'/reacher_seed{}_before_opt.npy'.format(random_seed), allow_pickle=True))
        after_opt = list(np.load(
            filepath+'/reacher_seed{}_after_opt.npy'.format(random_seed), allow_pickle=True))
        evaluations = list(
            np.load(filepath+'/reacher_seed{}_eval_reward.npy'.format(random_seed)))

    param_noise = None
    try:
        start_t = 0
        for t in range(int(MAX_EPISODES)):
            with torch.no_grad():
                episode_timesteps += 1
                noise_counter = 0
                if param_noise is not None:
                    policy.perturb_actor_parameters(param_noise)

                if t < 1000:
                    action_env = np.reshape(
                        env.action_space.sample(), (1, action_dim))
                    action = OptLayer_function(action_env, random_sample=True)
                    before_opt.append(action_env)
                    after_gaussian.append(action)
                    action = action[0]
                else:
                    action = policy.select_action(
                        state, param_noise, training=IS_SELECTING)

                store_action.append(action)
                # store_state.append(state)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(
                done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            noise_counter += 1

            if t % 50 == 0:
                print("t: ", t)
                # save_object(replay_buffer, 'replay_buffer_reacher_gaussian.pkl')

            if t >= start_training:  # 10000:
                policy.train(replay_buffer, t, BATCH_SIZE)

                if SAVE_MODEL:
                    policy.save(f"./models/{file_name}")

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                ewma_r = 0.05 * episode_reward + (1 - 0.05) * ewma_r
                print('Episode time: ', time.time() - start_t)
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} EWMA: {ewma_r:.3f}")
                start_t = time.time()

                # save results
                store_reward.append(episode_reward)
                store_ewma.append(ewma_r)
                if SAVE_FILE:
                    np.save(
                        filepath+'/reacher_seed{}_ewma_reward'.format(random_seed), np.array(store_ewma))
                    np.save(
                        filepath+'/reacher_seed{}_action'.format(random_seed), np.array(store_action))
                    np.save(
                        filepath+'/reacher_seed{}_before_opt'.format(random_seed), before_opt)
                    np.save(
                        filepath+'/reacher_seed{}_after_gaussian'.format(random_seed), after_gaussian)
                    np.save(
                        filepath+'/reacher_seed{}_eval_before_opt'.format(random_seed), eval_before_opt)
                    np.save(
                        filepath+'/reacher_seed{}_eval_after_opt'.format(random_seed), eval_after_opt)

                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                evaluations.append(eval_policy(policy, arg_env, random_seed))
                if SAVE_FILE:
                    np.save(
                        filepath+'/reacher_seed{}_eval_reward'.format(random_seed), np.array(evaluations))

    except:
        print("SAVING...")
        if SAVE_MODEL:
            policy.save(f"./models/{file_name}")
        save_object(
            replay_buffer, './replay_buffer_reacher_gaussian_seed{}.pkl'.format(random_seed))
        traceback.print_exc()
