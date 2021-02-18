# -*- coding: utf-8 -*-
"""
@author: Lin-Jyun-Li
@The DDPG code was adapt from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

"""

import os.path
import sys
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import gym
import numpy as np
import tensorflow as tf
import gym_BSS  # noqa: F401
from scipy.optimize import linprog,minimize
from numpy import linalg as LA
import math
np.set_printoptions(precision=8) 
import matplotlib.pyplot as plt
env_name = sys.argv[1] if len(sys.argv) > 1 else 'BSSEnvTest-v0'
env = gym.make(env_name)  # gym.Env

arg_seed = 4
# print(env.observation_space, env.action_space)
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

#####################  hyper parameters  ####################
LR_C = 0.001
LR_A = 0.0001
GAMMA = 0.99 
TAU = 0.001
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
episode=100000
###############################  DDPG  ####################################
tf.compat.v1.reset_default_graph()
np.random.seed(arg_seed)
env.seed(arg_seed)
env.action_space.np_random.seed(arg_seed)
############################################################################
store_before_action =[]
store_after_action =[]
store_testing_before_action =[]
store_testing_after_action =[]
eval_freq = 5000
eva_reward =[]
def evaluation(env_name,seed,ddpg,eval_episode=10):
    avgreward=0
    avg=[]
    eval_env=gym.make(env_name)
    eval_env.seed(seed+100)
    for eptest in range(eval_episode):
        running_reward =0
        done=False
        s=eval_env.reset()
        while not done:     
            action = ddpg.choose_action(s)
            store_testing_before_action.append(action)
            action ,loss= OptLayer_function(action)
            store_testing_after_action.append(action)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward=avgreward+running_reward
        avg.append(running_reward)
    avgreward=avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :",avgreward)
    print("------------------------------------------------")

    return avgreward
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1+1), dtype=np.float32)
        self.pointer = 0
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        tf.random.set_seed(arg_seed)
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.Done=tf.compat.v1.placeholder(tf.float32, [None, 1], 'done')

        
        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            
        with tf.compat.v1.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_,a_ , scope='target', trainable=False)
        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        q_target = tf.stop_gradient(self.R + (1-self.Done)*GAMMA * q_)
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
        self.a_loss = - tf.reduce_mean(input_tensor=self.q)    # maximize the q
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)
        
        self.action_grad=tf.gradients(self.q,self.a)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        buffer_size= min(ddpg.pointer+1,MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br=  bt[:,self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        bs_ = bt[:,self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+1+self.s_dim]
        bd = bt[:,-1:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.Done:bd})
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]     
    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            net1 = tf.compat.v1.layers.dense(net, 300, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            a = tf.compat.v1.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            a = tf.multiply(tf.add(a,9/5), 25/2, name='scaled_a')
            return a
    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.leaky_relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable('w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, 300], trainable=trainable)
            b2= tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2= tf.nn.leaky_relu(tf.matmul(a,w2_a)+tf.matmul(net,w2_net)+b2)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
#############################  optlayer  ####################################

def OptLayer_function(action):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as opt_m:
            net_a1=action[0]
            net_a2=action[1]
            net_a3=action[2]
            net_a4=action[3]
            net_a5=action[4]
           
            a1 = opt_m.addVar(lb=0,ub=35, name="a1",vtype=GRB.INTEGER)
            a2 = opt_m.addVar(lb=0,ub=35, name="a2",vtype=GRB.INTEGER)
            a3 = opt_m.addVar(lb=0,ub=35, name="a3",vtype=GRB.INTEGER)
            a4 = opt_m.addVar(lb=0,ub=35, name="a4",vtype=GRB.INTEGER)
            a5 = opt_m.addVar(lb=0,ub=35, name="a5",vtype=GRB.INTEGER)

            
            obj= (a1-net_a1)**2+ (a2-net_a2)**2+(a3-net_a3)**2+(a4-net_a4)**2+(a5-net_a5)**2
            opt_m.setObjective(obj,GRB.MINIMIZE)
            
        
            opt_m.addConstr(a1+a2+a3+a4+a5==150)   #x+y+z==90
          
            opt_m.optimize()
        
            return (a1.X,a2.X,a3.X,a4.X,a5.X),opt_m.objVal
            
##################################################################################
Rs = []
ewma = []
s_dim = env.observation_space.shape[0]  # 2*ZONE+1 ZONE's Demand,zone's number of resource on zone K (dS_) +time
a_dim = env.action_space.shape[0]
print(a_dim,"YEEEEEEE")
a_bound = env.action_space.high   #bound , in txt file
demand_range = 20

#initial table
#calculate table num
bikebound = 15
index_0 = 20

ewma=[]
eva_reward=[]
store_before_action=[]
store_after_action=[]
store_before_action_and_gaussain=[]
ddpg = DDPG(a_dim, s_dim, a_bound)
i = 0
learning_rate = 0.001
ewma_r = 0

counter=0
valid=np.zeros((20176, 5))
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
                
exploration = 0.1
for ep in range(episode):
    R = 0
    step=0
    ld_pickup = 0
    ld_dropoff = 0
    revenue = 0
    scenario = None
    done = False
    s = env.reset()  # [0,0,0,0,8,7,8,8,0]
    while not done:
        step=step+1
        if ddpg.pointer<10000:
            chose=np.random.randint(0, 20176)
            action=[valid[chose][0], valid[chose][1], valid[chose][2],valid[chose][3],valid[chose][4]]
            store_before_action.append(action)
            store_before_action_and_gaussain.append(action)
            loss=0
            store_after_action.append(action)
        else :
            action=ddpg.choose_action(s)
            store_before_action_and_gaussain.append(action)
            action=action+np.random.normal(0,5,a_dim)
            
            store_before_action.append(action)
            
            action,loss=OptLayer_function(action)
            store_after_action.append(action)
        s_, r, done, info = env.step(action)
        done_bool = False if step==100 else done 

        loss=max(0,loss)
        r_shape=r-np.sqrt(loss)*4
        
        ddpg.store_transition(s, action, r_shape , s_,done_bool)
        if ddpg.pointer >= 10000:
            ddpg.learn()
        if (ddpg.pointer+1)% eval_freq==0:
            eva_reward.append(evaluation(env_name,arg_seed,ddpg))
        #print(s)
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
        'emma reward' : ewma_r,
        'lost_demand_pickup': ld_pickup,
        "lost_demand_dropoff": ld_dropoff,
        "revenue": revenue,
        "scenario": scenario
        
    })
    Rs.append(R)
    ewma.append(ewma_r)
    if(ddpg.pointer>=500000):
        print("done training")
        break

np.save("Bike_{}_DDPGRewardShaping_Reward".format(arg_seed),Rs)
np.save("Bike_{}_DDPGRewardShaping_before_Action".format(arg_seed),store_before_action)
np.save("Bike_{}_DDPGRewardShaping_after_Action".format(arg_seed),store_after_action)
np.save("Bike_{}_DDPGRewardShaping_eval_reward".format(arg_seed),eva_reward)
np.save("Bike_{}_DDPGRewardShaping_eval_before_Action".format(arg_seed),store_testing_before_action)
np.save("Bike_{}_DDPGRewardShaping_eval_after_Action".format(arg_seed),store_testing_after_action)
np.save("Bike_{}_DDPGRewardShaping_before_Action_Gaussian".format(arg_seed),store_before_action_and_gaussain)


