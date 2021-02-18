# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:49:19 2021

@The DDPG code was adapt from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""
import matplotlib.pyplot as plt
import os.path
import sys
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import gym
import numpy as np
import tensorflow as tf
import gym_BSS  # noqa: F401
from scipy.optimize import linprog, minimize
from numpy import linalg as LA
import math
import random
np.set_printoptions(precision=8)
env_name = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(env_name)  # gym.Env

arg_seed = 0
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#####################  hyper parameters  ####################
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
episode = 10000
arraysize = int(((episode*100)-MEMORY_CAPACITY)/100)
###############################  DDPG  ####################################
tf.compat.v1.reset_default_graph()
np.random.seed(arg_seed)
env.seed(arg_seed)
env.action_space.np_random.seed(arg_seed)
random.seed(arg_seed)
############################################################################
store_before_action = []
store_after_action = []
store_testing_before_action = []
store_testing_after_action = []
eval_freq = 5000
eva_reward = []


def evaluation(env_name, seed, ddpg, eval_episode=5):
    avgreward = 0
    avg = []
    eval_env = gym.make(env_name)
    eval_env.seed(seed+100)
    for eptest in range(eval_episode):
        running_reward = 0
        done = False
        s = eval_env.reset()
        while not done:
            a1, a2 = ddpg.DDPG_policy(s)
            a1 = np.rint(a1)
            a2 = np.rint(a2)
            action = [a1, a2, env.nbikes - a1 - a2]
            store_testing_before_action.append(action)
            action, loss = Projection(action)
            store_testing_after_action.append(action)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward = avgreward+running_reward
        avg.append(running_reward)
    avgreward = avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :", avgreward)
    print("------------------------------------------------")

    return avgreward


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()
        tf.random.set_seed(arg_seed)
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        self.a = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'a')
        self.a_ = tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'a_')

        with tf.compat.v1.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(
                self.S_, self.a_, scope='target', trainable=False)
        self.Qvalue = q
        self.ce_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.ct_params, self.ce_params)]
        q_target = self.R + GAMMA * q_
        td_error = tf.compat.v1.losses.mean_squared_error(
            labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(
            LR_C).minimize(td_error, var_list=self.ce_params)

        self.action_grad = tf.gradients(q, self.a)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def critic_learn(self):
        # soft target replacement
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        na1 = targettable[(np.rint(bs_[:, 3])-index_0).astype(int),
                          (np.rint(bs_[:, 4])-index_0).astype(int), 0]
        na2 = targettable[(np.rint(bs_[:, 3])-index_0).astype(int),
                          (np.rint(bs_[:, 4])-index_0).astype(int), 1]
        next_action = [na1, na2, env.nbikes-na1-na2]

        next_action = np.asarray(next_action).T
        self.sess.run(self.ctrain, {
                      self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.a_: next_action})

    def get_gradient(self, s, a):
        return self.sess.run(self.action_grad, {self.S: s, self.a: a})

    def get_Q_value(self, s, a):
        return self.sess.run(self.Qvalue, {self.S: s, self.a: a})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.compat.v1.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable(
                'b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # Q(s,a)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)


################ DDPG_update#####################
all_q_values = np.zeros((arraysize, 21, 21), dtype=float)  # by time pass
gradient_q = np.zeros((arraysize, 21, 21), dtype=float)


class DDPG_action(object):
    def DDPG_policy(self, s):
        return table[int(round(s[3]-index_0)), int(round(s[4]-index_0))][0], table[int(round(s[3]-index_0)), int(round(s[4]-index_0))][1]

    def update(self, ddpg, lr):
        for s1 in range(16):
            for s2 in range(16):
                a = table[s1, s2]
                gradient = ddpg.get_gradient(
                    [[s1, s1, s1, s1+20, s2+20, env.nbikes-s1-s2-40, s1]], [a])
                gradient = np.squeeze(gradient)

                table[s1, s2] = table[s1, s2]+gradient*lr


#############################  Projection  ####################################

def Projection(action):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as opt_m:
            net_a1 = action[0]
            net_a2 = action[1]
            net_a3 = action[2]

            a1 = opt_m.addVar(lb=20, ub=35, name="a1", vtype=GRB.INTEGER)
            a2 = opt_m.addVar(lb=20, ub=35, name="a2", vtype=GRB.INTEGER)
            a3 = opt_m.addVar(lb=20, ub=35, name="a3", vtype=GRB.INTEGER)

            obj = (a1-net_a1)**2 + (a2-net_a2)**2+(a3-net_a3)**2
            opt_m.setObjective(obj, GRB.MINIMIZE)

            opt_m.addConstr(a1+a2+a3 == 90)  # x+y+z==90

            opt_m.optimize()

            return (a1.X, a2.X, a3.X), opt_m.objVal


##################################################################################
Rs = []
ewma = []
nploss = []
# 2*ZONE+1 ZONE's Demand,zone's number of resource on zone K (dS_) +time
s_dim = env.observation_space.shape[0]
# equal to get_observe function in env
a_dim = env.action_space.shape[0]
print(a_dim, "YEEEEEEE")
# print(env.action_space.low,"low")
a_bound = env.action_space.high  # bound , in txt file
demand_range = 20
DDPGaction = DDPG_action()

# initial table
# calculate table num
bikebound = 15
index_0 = 20

table = np.zeros((bikebound+1, bikebound+1, a_dim))  # (16,16,2)
table[:, :, 0] = 20
table[:, :, 1] = 35
table[:, :, 2] = 35
ddpg = DDPG(a_dim, s_dim, a_bound)
targettable = table
i = 0
learning_rate = 0.001
ewma_r = 0
norm = []
counter = 0
valid = np.zeros((136, 3))
for i in range(20, 36):
    for j in range(20, 36):
        for k in range(20, 36):
            if (i + j + k == 90):
                valid[counter][0] = i
                valid[counter][1] = j
                valid[counter][2] = k
                counter = counter+1

exploration = 0.1
for ep in range(episode):
    R = 0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    s = env.reset()  # [0,0,0,0,8,7,8,8,0]
    while not done:
        i = i+1
        if np.random.random() >= exploration:
            # here need to change to tabuler
            a1, a2 = DDPGaction.DDPG_policy(s)
            a1 = np.rint(a1)
            a2 = np.rint(a2)
            action = [a1, a2, env.nbikes - a1 - a2]
            store_before_action.append(action)
            action, loss = Projection(action)
            store_after_action.append(action)
        else:
            chose = np.random.randint(0, 136)
            action = [valid[chose][0], valid[chose][1], valid[chose][2]]
        nploss.append(loss)
        s_, r, done, info = env.step(action)
        r_shape = r-(np.sqrt(loss))
        ddpg.store_transition(s, action, r_shape, s_)
        if ddpg.pointer > MEMORY_CAPACITY:
            if(ddpg.pointer % 25 == 0):
                ddpg.critic_learn()
            if(ddpg.pointer % 50 == 0):
                DDPGaction.update(ddpg, learning_rate)

            if(ddpg.pointer % 100 == 0):
                targettable = table
        if (ddpg.pointer+1) % eval_freq == 0:
            norm_1 = 0
            for i in range(20, 36):
                for j in range(20, 36):
                    for k in range(20, 36):
                        if (i+j+k == 90):
                            a1 = np.rint(table[i-20][j-20][0])
                            a2 = np.rint(table[i-20][j-20][1])
                            a3 = 90-a1-a2
                            norm_1 = norm_1 + \
                                np.sqrt(abs(a1-i)**2+abs(a2-j)**2+abs(a3-k)**2)
            norm.append(norm_1)
            eva_reward.append(evaluation(env_name, arg_seed, DDPGaction))

        s = s_
        R += r
        ld_pickup += info["lost_demand_pickup"]
        ld_dropoff += info["lost_demand_dropoff"]
        revenue += info["revenue"]
        scenario = info["scenario"]
    ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
    print({
        'episode': ep,
        'reward': R,
        'emma reward': ewma_r,
        'lost_demand_pickup': ld_pickup,
        "lost_demand_dropoff": ld_dropoff,
        "revenue": revenue,
        "scenario": scenario

    })
    Rs.append(R)
    ewma.append(ewma_r)
print('')
print('---------------------------')
print('Average reward per episode:', np.average(Rs))

np.save("Bike_{}_RewardShaping_Tabular_Reward".format(arg_seed), Rs)
np.save("Bike_{}_RewardShaping_Tabular_before_Action".format(
    arg_seed), store_before_action)
np.save("Bike_{}_RewardShaping_Tabular_After_Action".format(
    arg_seed), store_after_action)
np.save("Bike_{}_RewardShaping_Tabular_eval_reward".format(arg_seed), eva_reward)
np.save("Bike_{}_RewardShaping_Tabular_eval_before_Action".format(
    arg_seed), store_testing_before_action)
np.save("Bike_{}_RewardShaping_Tabular_eval_after_Action".format(
    arg_seed), store_testing_after_action)
np.save("Bike_{}_RewardShaping_Tabular_table".format(arg_seed), table)
np.save("Bike_{}_RewardShaping_Tabular_loss".format(arg_seed), loss)
np.save("Bike_{}_RewardShaping_Tabular_norm".format(arg_seed), norm)
