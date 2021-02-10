#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:07:02 2021

@author: Lin-Jyun-Li
@The DDPG code was adapt from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""
# -*- coding: utf-8 -*-

import os.path
import gurobipy as gp
from gurobipy import GRB
import scipy
import numpy
import gym
import mujoco_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random

tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'

env_name='HalfCheetah-v2'
env=gym.make(env_name)
s_dim = env.observation_space.shape[0] #111
a_dim = env.action_space.shape[0]               #8
a_bound=env.action_space.high
ewma_r=0
arg_seed=0
#########################seed##############################
tf.compat.v1.reset_default_graph()
random.seed(arg_seed)
np.random.seed(arg_seed)
env.seed(arg_seed)
env.action_space.np_random.seed(arg_seed)
#####################  hyper parameters  ####################
exploration =0.1
LR_C=0.001
LR_A=0.0001
GAMMA=0.99
TAU=0.001
eval_freq=5000
MEMORY_CAPACITY=1000000
BATCH_SIZE=64
store_testing_before_action =[]
store_testing_after_action =[]
####################testing part#################################
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
            action= ddpg.choose_action(s)
            store_testing_before_action.append(action)
            action,loss=Projection(action,s)
            store_testing_after_action.append(action)
            s_,r,done,info=eval_env.step(action)
            s=s_
            running_reward=running_reward+r
        print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
        avgreward=avgreward+running_reward
        avg.append(running_reward)
    avgreward=avgreward/eval_episode
    print("------------------------------------------------")
    print("Evaluation average reward :",avgreward)
    print("------------------------------------------------")

    return avgreward
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
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
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + (1-self.Done)*GAMMA * q_
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
       
        a_loss = - tf.reduce_mean(input_tensor=self.q)    # maximize the q
        
        
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())
   
    def choose_action(self, s):#
       
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

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

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.compat.v1.layers.dense(net,300, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.compat.v1.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable('w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, 300], trainable=trainable)
            b2= tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2= tf.nn.relu(tf.matmul(a,w2_a)+tf.matmul(net,w2_net)+b2)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
        
        
ddpg = DDPG(a_dim,s_dim,a_bound)    
#############################  optlayer  ####################################

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
                return (half_m.X[0:6]),half_m.objVal
            
##################################################################################

Net_action=np.zeros((1000000,a_dim+2))   
ewma = []
eva_reward=[]
store_before_action_and_gaussain=[]
store_before_action=[]
store_after_action =[]
max_action = float(env.action_space.high[0])
lossnp=[]
reward=[]
var=3
step=0
for ep in range(1000000):
    #env.render()
    R=0
    done=False
    s=env.reset()
    while not done:
        step=step+1
        #use gaussian exploration by TD3 (0,0.1) to each eaction
        if ddpg.pointer<10000:
            action=env.action_space.sample()
            store_before_action.append(action)
        else :
            action=ddpg.choose_action(s)
            store_before_action_and_gaussain.append(action)
            action=(action+np.random.normal(0,0.1,a_dim)).clip(-max_action,max_action)
            store_before_action.append(action)
        
        action,loss=Projection(action,s)
        store_after_action.append(action)
        assert (abs(s[11:]*action)).sum()<=1e-6+20
        
        
        s_,r,done,info=env.step(action)
        loss=max(0,loss)
        loss=np.sqrt(loss)*3
        lossnp.append(loss)
        r_shape=r-loss
        done_bool = False if step==1000 else done 
        ddpg.store_transition(s,action,r_shape,s_,done_bool)
        
        if ddpg.pointer>=10000:
            ddpg.learn()
        if (ddpg.pointer+1)% eval_freq==0:
            eva_reward.append(evaluation(env_name,arg_seed,ddpg))
        s= s_
        R += r
    reward.append(R)
    ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
    if(ddpg.pointer>10000):
        print("start_training,ddpgpointer:{}".format(ddpg.pointer))
    print({
        'episode': ep,
        'reward' :R,
        'ewma_reward' :ewma_r
    })
    ewma.append(ewma_r)
    if(ddpg.pointer>=1000000):
        print("done training")
        break
a=[] 
for i in range(ep+1):
    a.append(i)
plt.plot(a,ewma)
plt.title("ewma reward, final ewma={}".format(ewma[ep]))  


np.save("Halfcheetah_{}_RewardShaping_Reward_state_relate".format(arg_seed),reward)
np.save("Halfcheetah_{}_RewardShaping_before_Action_state_relate".format(arg_seed),store_before_action)
np.save("Halfcheetah_{}_RewardShaping_after_Action_state_relate".format(arg_seed),store_after_action)
np.save("Halfcheetah_{}_RewardShaping_eval_reward_state_relate".format(arg_seed),eva_reward)
np.save("Halfcheetah_{}_RewardShaping_before_Action_Gaussian_state_relate".format(arg_seed),store_before_action_and_gaussain)
np.save("Halfcheetah_{}_RewardShaping_eval_before_Action_state_relate".format(arg_seed),store_testing_before_action)
np.save("Halfcheetah_{}_RewardShaping_eval_After_Action_state_relate".format(arg_seed),store_testing_after_action)
np.save("Halfcheetah_{}_RewardShaping_loss_state_relate".format(arg_seed),lossnp)
