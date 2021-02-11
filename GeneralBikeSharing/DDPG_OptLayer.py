import gym
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from cvxpylayers.torch import CvxpyLayer
#import cvxpy as cp
import gym_BSS
import pickle
import traceback
import time
from torch.autograd import Function, Variable
from qpth.qp import QPFunction, QPSolvers
import gurobipy as gp
from gurobipy import GRB

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEBICES"] = "2, 3"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

IS_TRAINING = 0
IS_EVAL = 1
IS_SELECTING = 2
IS_OTHER = 3

before_tanh = 0

#####################  hyper parameters  ####################

MAX_EPISODES = 500000 # 5000
MAX_EP_STEPS = 100
TOTAL_STEPS = 1e+6
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.001  # 0.01      # soft replacement
MEMORY_CAPACITY = 10000  # 10000
c = 0.1  # 0.1
BATCH_SIZE = 16  # 32
RENDER = False
start_training = 10000
random_seed = 4
# ENV_NAME = 'Pendulum-v0'
arg_env = 'BSSEnvTest-v0'

EVAL = True
eval_freq = 5000

SAVE_FILE = True
filepath = "./results_5zone_gaussian"
if not os.path.exists(filepath):
    os.makedirs(filepath)

SAVE_MODEL = False
LOAD_MODEL = False
file_name = "5zone_gaussian_seed{}".format(random_seed)

before_opt = []
after_opt = []
after_gaussian = []
eval_before_opt = []
eval_after_opt = []

scaled_grad = []
opt_grad = []
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
    eval_state = []

    if LOAD_MODEL:
        eval_action = list(np.load(filepath+'/5zone_seed{}_eval_action.npy'.format(random_seed)))

    for _ in range(eval_episodes):
        state, done = np.round(eval_env.reset()), False
        while not done:
            action_float = policy.select_action(np.array(state), None, training=IS_EVAL)

            # Switch to INT
            '''
            action = np.round(action_float)
            diff = abs(np.sum(action) - env.nbikes)
            if np.sum(action) < env.nbikes:
                min_idx = np.argmin(action)
                action[min_idx] += diff
            elif np.sum(action) > env.nbikes:
                max_idx = np.argmax(action)
                action[max_idx] -= diff
            '''
            action = action_float

            eval_action.append(action)
            eval_state.append(state)
            # print('EVAL action: ', action.shape)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    if SAVE_FILE:
        np.save(filepath+'/5zone_seed{}_eval_action'.format(random_seed), np.array(eval_action))
        np.save(filepath+'/5zone_seed{}_eval_state'.format(random_seed), np.array(eval_state))
    return avg_reward


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        with gp.Model(env=env) as half_m:
            
            # print('action: ', action)
            batch_size = action.shape[0]
            if random_sample:
                rtn_action = np.array([])
            else:
                rtn_action = torch.tensor([])

            for i in range(batch_size):
                batch_action = action[i]
                joint1_a1 = batch_action[0]
                joint1_a2 = batch_action[1]
                joint2_a1 = batch_action[2]
                joint2_a2 = batch_action[3]
                joint3_a1 = batch_action[4]

                a1 = half_m.addVar(lb=0, ub=35, name='a1', vtype=GRB.INTEGER)
                a2 = half_m.addVar(lb=0, ub=35, name='a2', vtype=GRB.INTEGER)
                a3 = half_m.addVar(lb=0, ub=35, name='a3', vtype=GRB.INTEGER)
                a4 = half_m.addVar(lb=0, ub=35, name='a4', vtype=GRB.INTEGER)
                a5 = half_m.addVar(lb=0, ub=35, name='a5', vtype=GRB.INTEGER)

                obj = (a1-joint1_a1)**2 + (a2-joint1_a2)**2 + (a3-joint2_a1)**2 + (a4-joint2_a2)**2 + (a5-joint3_a1)**2
                half_m.setObjective(obj, GRB.MINIMIZE)

                half_m.addConstr((a1+a2+a3+a4+a5)==150)

                half_m.optimize()

                # return a1.X, a2.X, a3.X, a4.X, a5.X
                # print('a1.x: ', a1.x)
                if random_sample:
                    rtn_action = np.append(rtn_action, [a1.X, a2.X, a3.X, a4.X, a5.X])
                else:
                    batch_opt_a = torch.tensor([a1.X, a2.X, a3.X, a4.X, a5.X])
                    rtn_action = torch.cat((rtn_action, batch_opt_a))
            # print('Return shape: ', rtn_action.shape)
            # if batch_size == 1:
            #     return rtn_action.float()
            # else:
            if random_sample:
                return np.reshape(rtn_action, (batch_size, action.shape[1]))
            else:
                return torch.reshape(rtn_action, (batch_size, action.shape[1])).float()

class OptLayer(torch.nn.Module):
    def __init__(self, D_in, D_out, a_bound):
        super(OptLayer, self).__init__()
        
        self.Q = Variable(torch.eye(D_in, dtype=torch.float64).to(device))
        # Gx <= h
        self.G = Variable(torch.tensor([[1, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 1],
                                        [-1, 0, 0, 0, 0],
                                        [0, -1, 0, 0, 0],
                                        [0, 0, -1, 0, 0],
                                        [0, 0, 0, -1, 0],
                                        [0, 0, 0, 0, -1]], dtype=torch.float64).to(device))
        self.h = Variable(torch.tensor([35, 35, 35, 35, 35, 0, 0, 0, 0, 0], dtype=torch.float64).to(device))
        # Ax == b 
        self.A = Variable(torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float64).to(device))
        self.b = Variable(torch.tensor([150], dtype=torch.float64).to(device))               
            


    def forward(self, x):
        # print('Q: ', self.Q.shape)
        batch_size = x.shape[0]
        # x = x.view(batch_size, -1)
        # print('x.shape: ', x.shape)
        Q = self.Q.repeat(batch_size, 1, 1)
        G = self.G.repeat(batch_size, 1, 1)
        h = self.h.repeat(batch_size, 1)
        A = self.A.repeat(batch_size, 1, 1)
        b = self.b.repeat(batch_size, 1)
        # print('Q size: ', Q.shape)
        # print('G size: ', G.shape)
        # print('h size: ', h.shape)
        # print('A size: ', A.shape)
        # print('b size: ', b.shape)
        return QPFunction(verbose=-1, notImprovedLim=1, eps=1e-6)(Q, x.double(), G, h, A, b).float()
        

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.opt_layer = OptLayer(action_dim, action_dim, max_action)
        
        self.max_action = max_action

    
    def forward(self, state, training):
        '''
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        '''
        a = F.leaky_relu(self.l1(state), negative_slope=0.2)
        a = F.leaky_relu(self.l2(a), negative_slope=0.2)
        
        # scaled_a = self.max_action * torch.tanh(self.l3(a))
        # scaled_a = F.relu(self.l3(a))
        # scaled_a = torch.add(torch.tanh(self.l3(a)), 9/5) * 25/2
        scaled_a = F.leaky_relu(self.l3(a), negative_slope=0.2)

        # scaled_a = (torch.tanh(self.l3(a)) + 1) / 2 * self.max_action

        
        # print('before: ', scaled_a)
        if training == IS_TRAINING:
            a_start = time.time()
            self.scaled_a = scaled_a
            opt_a = self.opt_layer(scaled_a)
            self.opt_a = opt_a
            # print('QP time: ', time.time() - a_start)
        elif training == IS_SELECTING:
            noise_a = scaled_a + torch.normal(0, 5, size=scaled_a.shape).to(device)
            opt_a = OptLayer_function(noise_a)
            
            before_opt.append(scaled_a.detach().cpu().numpy())
            after_gaussian.append(noise_a.detach().cpu().numpy())
        elif training == IS_EVAL:
            opt_a = OptLayer_function(scaled_a)
            
            eval_before_opt.append(scaled_a.detach().cpu().numpy())
            eval_after_opt.append(opt_a.detach().cpu().numpy())
        elif training == IS_OTHER:
            opt_a = OptLayer_function(scaled_a)
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
        '''
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        '''
        q = F.leaky_relu(self.l1(state), negative_slope=0.2)
        q = F.leaky_relu(self.l2(torch.cat([q, action], 1)), negative_slope=0.2)
        return self.l3(q)



class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_perturbed = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau


    def select_action(self, state, para, training):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            if para is not None:
                return self.actor_perturbed(state, training=training).cpu().data.numpy().flatten()
            else:
                return self.actor(state, training=training).cpu().data.numpy().flatten()


    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            random = torch.randn(param.shape).to(device)
            # if use_cuda:
                # random = random.cuda()
            param += random * param_noise.current_stddev


    def train(self, replay_buffer, t, batch_size=64):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state, training=IS_OTHER).to(device))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        c_time = time.time()
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # print('-> critic time: ', time.time() - c_time)
        
        # Optimize the actor 
        if (t + 1) % 50 == 0:
            # Compute actor loss
            print("====================")
            actor_loss = -self.critic(state, self.actor(state, training=IS_TRAINING)).mean()
            
            a_time = time.time()
            self.actor_optimizer.zero_grad()
            #self.actor.scaled_a.register_hook(print)
            #self.actor.opt_a.register_hook(print)
            self.actor.scaled_a.retain_grad()
            self.actor.opt_a.retain_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # print("====================")
            print('actor loss: ', actor_loss)
            
            grad_of_scaled = self.actor.scaled_a.grad.norm().detach().cpu().numpy()
            grad_of_opt = self.actor.opt_a.grad.norm().detach().cpu().numpy()
            
            print('scaled_a: ', grad_of_scaled)
            print('opt_a: ', grad_of_opt)
            scaled_grad.append(grad_of_scaled)
            opt_grad.append(grad_of_opt)
            # print(self.actor.opt_a.grad, self.actor.scaled_a.grad)
            # for p in self.actor.parameters():
            #     print(p.grad.norm())
            # print('-> actor time: ', time.time() - a_time)

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_perturbed = copy.deepcopy(self.actor)


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if SAVE_MODEL and not os.path.exists("./models"):
        os.makedirs("./models")
    print("BEFORE GYM...")
    env = gym.make(arg_env)
    print("AFTER GYM...")

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

    valid=np.zeros((20176, 5))
    counter = 0
    while (counter < 20176):
        for i in range(20, 36):
            for j in range(20, 36):
                for k in range(20, 36):
                    for l in range(20,36):
                        for m_ in range(20,36):
                                if (i + j + k + l + m_  == 150):
                                    valid[counter][0] = i
                                    valid[counter][1] = j
                                    valid[counter][2] = k
                                    valid[counter][3] = l
                                    valid[counter][4] = m_
                                    counter = counter+1

            

    # # Initialize policy
    # if args.policy == "TD3":
    # 	# Target policy smoothing is scaled wrt the action scale
    # 	kwargs["policy_noise"] = args.policy_noise * max_action
    # 	kwargs["noise_clip"] = args.noise_clip * max_action
    # 	kwargs["policy_freq"] = args.policy_freq
    # 	policy = TD3.TD3(**kwargs)
    # elif args.policy == "OurDDPG":
    # 	policy = OurDDPG.DDPG(**kwargs)
    # elif args.policy == "DDPG":
    # 	policy = DDPG.DDPG(**kwargs)
    policy = DDPG(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    
    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, arg_env, random_seed)]

    state, done = np.round(env.reset()), False
    episode_reward = 0
    ewma_r = 0
    episode_timesteps = 0
    episode_num = 0

    evaluations = []
    store_ewma = []
    # eva_reward = []
    store_action = []
    store_reward = []
    store_state = []

    if LOAD_MODEL:
        print("=======LOAD")
        policy_file = file_name
        policy.load(f"./models/{policy_file}")
        with open('replay_buffer_5zone_gaussian_seed{}.pkl'.format(random_seed), 'rb') as input:
            replay_buffer = pickle.load(input)
        print("==========end load")
        print("r size: ", replay_buffer.size)

        store_reward = list(np.load(filepath+'/5zone_seed{}_episode_reward.npy'.format(random_seed)))
        store_ewma = list(np.load(filepath+'/5zone_seed{}_ewma_reward.npy'.format(random_seed)))
        store_action = list(np.load(filepath+'/5zone_seed{}_action.npy'.format(random_seed)))
        before_opt = list(np.load(filepath+'/5zone_seed{}_before_opt.npy'.format(random_seed), allow_pickle = True))
        after_opt = list(np.load(filepath+'/5zone_seed{}_after_opt.npy'.format(random_seed), allow_pickle = True))
        evaluations = list(np.load(filepath+'/5zone_seed{}_eval_reward.npy'.format(random_seed)))

    # param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)
    param_noise = None
    print("HERE!!!!!")
    try:
        start_t = 0
        for t in range(int(MAX_EPISODES)):
            with torch.no_grad():
                episode_timesteps += 1
                noise_counter = 0
                if param_noise is not None:
                    policy.perturb_actor_parameters(param_noise)

                sele_time = time.time()
                if t < 10000:
                    # action_env = env.action_space.sample()
                    action_env = np.random.uniform(20.0, 36.0, size=(1, action_dim))
                    action_float = OptLayer_function(action_env, random_sample=True)
                    before_opt.append(action_env)
                    after_gaussian.append(action_float)
                    action_float = action_float[0]
                    '''
                    chose=np.random.randint(0, 20176)
                    action_float=[valid[chose][0], valid[chose][1], valid[chose][2],valid[chose][3],valid[chose][4]]
                    before_opt.append(action_float)
                    after_gaussian.append(action_float)
                    '''
                else:
                    action_float = policy.select_action(state, param_noise, training=IS_SELECTING)
                # print('selecting time: ', time.time() - sele_time)
        
                # Switch to INT
                '''
                action = np.round(action_float)
                diff = abs(np.sum(action) - env.nbikes)
                if np.sum(action) < env.nbikes:
                    min_idx = np.argmin(action)
                    action[min_idx] += diff
                elif np.sum(action) > env.nbikes:
                    max_idx = np.argmax(action)
                    action[max_idx] -= diff
                '''
                action = action_float

                store_action.append(action)
                # store_state.append(state)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            done_bool = float(done) if episode_timesteps < MAX_EP_STEPS else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward
            noise_counter += 1

            if t % 50 == 0:
                print("t: ", t)
                # save_object(replay_buffer, 'replay_buffer_5zone_gaussian.pkl')

            # Train agent after collecting sufficient data
            # if t >= c*MEMORY_CAPACITY:
            if t >= start_training: # 10000:
                # print("before train...")
                # exit(0)
                # while True:
                    # try:
                # start_t = time.time()
                policy.train(replay_buffer, t, BATCH_SIZE)
                # print('training time: ', time.time() - start_t)
                    # except:
                        # print("Retry!!!")
                        # print("Now r size: ", replay_buffer.size)
                        # continue
                    # break
                # print("==============")
                # print("TRAIN!!!")
                # print("==============")
                if SAVE_MODEL: policy.save(f"./models/{file_name}")

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                ewma_r = 0.05 * episode_reward + (1 - 0.05) * ewma_r
                print('Episode time: ', time.time() - start_t)
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} EWMA: {ewma_r:.3f}")
                start_t = time.time()

                # save results
                # store_reward.append(episode_reward)
                store_ewma.append(ewma_r)
                if SAVE_FILE:
                    # np.save(filepath+'/5zone_seed{}_episode_reward'.format(random_seed), np.array(store_reward))
                    np.save(filepath+'/5zone_seed{}_ewma_reward'.format(random_seed), np.array(store_ewma))
                    np.save(filepath+'/5zone_seed{}_action'.format(random_seed), np.array(store_action))
                    np.save(filepath+'/5zone_seed{}_before_opt'.format(random_seed), before_opt)
                    np.save(filepath+'/5zone_seed{}_after_gaussian'.format(random_seed), after_gaussian)
                    np.save(filepath+'/5zone_seed{}_eval_before_opt'.format(random_seed), eval_before_opt)
                    np.save(filepath+'/5zone_seed{}_eval_after_opt'.format(random_seed), eval_after_opt)
                    np.save(filepath+'/5zone_seed{}_scaled_grad'.format(random_seed), scaled_grad)
                    np.save(filepath+'/5zone_seed{}_opt_grad'.format(random_seed), opt_grad)
                    # np.save(filepath+'/5zone_seed{}_state'.format(random_seed), store_state)

                #if SAVE_MODEL: policy.save(f"./models/{file_name}")

                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                evaluations.append(eval_policy(policy, arg_env, random_seed))
                if SAVE_FILE:
                    np.save(filepath+'/5zone_seed{}_eval_reward'.format(random_seed), np.array(evaluations))
                # np.save(f"./results/{file_name}", evaluations)
                # if SAVE_MODEL: policy.save(f"./models/{file_name}")

            # if SAVE_FILE:
            #     np.save(filepath+'/5zone_seed{}_episode_reward'.format(random_seed), np.array(store_reward))
            #     np.save(filepath+'/5zone_seed{}_ewma_reward'.format(random_seed), np.array(store_ewma))
            #     np.save(filepath+'/5zone_seed{}_action'.format(random_seed), np.array(store_action))
            #     np.save(filepath+'/5zone_seed{}_before_opt'.format(random_seed), before_opt)
            #     np.save(filepath+'/5zone_seed{}_after_opt'.format(random_seed), after_opt)
    
    except:
        print("SAVING...")
        if SAVE_MODEL: policy.save(f"./models/{file_name}")
        save_object(replay_buffer, './replay_buffer_5zone_gaussian_seed{}.pkl'.format(random_seed))
        traceback.print_exc()