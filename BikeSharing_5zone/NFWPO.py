#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lin-Jyun-Li
@The DDPG code was adapt from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os.path
import gurobipy as gp
from gurobipy import GRB
import scipy
import numpy
import gym
import numpy as np
import gym_BSS  # noqa: F401

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
import random
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1,2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #make render not lag

env_name='BSSEnvTest-v0'
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
BATCH_SIZE=16
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
            action=Projection_function(action,GRB.INTEGER)
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
fw=np.zeros((BATCH_SIZE,a_dim))
fwmat=np.zeros((BATCH_SIZE,a_dim))

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1+1), dtype=np.float32)
        self.pointer = 0
        configuration = tf.compat.v1.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=configuration)
        #self.sess = tf.compat.v1.Session()
        tf.random.set_seed(arg_seed)
        
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.Done=tf.compat.v1.placeholder(tf.float32, [None, 1], 'done')
        
        self.tf_table=tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'table')
        
        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
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
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
        action_td_error = tf.compat.v1.losses.mean_squared_error(labels=self.tf_table, predictions=self.a)
        self.update = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(action_td_error, var_list=self.ae_params)
        
        self.action_grad=tf.gradients(q,self.a)
        self.para_grad=tf.gradients(self.a,self.ae_params)
        self.sess.run(tf.compat.v1.global_variables_initializer())
    def get_par_grad(self,s):
        return self.sess.run(self.para_grad,{self.S:s})
    def choose_action(self, s):
       
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
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.Done:bd})

    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            net2 = tf.compat.v1.layers.dense(net,300, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            a = tf.compat.v1.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a
    
    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.leaky_relu, name='cl1', trainable=trainable)     
            w2_net = tf.compat.v1.get_variable('w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, 300], trainable=trainable)
            b2= tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2= tf.nn.leaky_relu(tf.matmul(a,w2_a)+tf.matmul(net,w2_net)+b2)
            
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
   
    def get_gradient(self,s,a):
        return self.sess.run(self.action_grad,{self.S: s, self.a: a})
    def fw_update(self):
        lr=0.05
        buffer_size= min(ddpg.pointer+1,MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba=self.sess.run(self.a,{self.S:bs})
        #######let ba in constraint######
        for i in range(BATCH_SIZE) :
            ba[i]=Projection_function(ba[i],GRB.CONTINUOUS)
                
        ########let ba in constraint #####
       
        grad=self.get_gradient(bs,ba)
        grad=np.squeeze(grad)
        action_table=np.zeros((BATCH_SIZE,self.a_dim))   #(16,16,2
        for i in range(BATCH_SIZE):
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model(env=env) as fw_m:
                    a1 = fw_m.addVar(lb=0,ub=35, name="a1",vtype=GRB.CONTINUOUS)
                    a2 = fw_m.addVar(lb=0,ub=35, name="a2",vtype=GRB.CONTINUOUS)
                    a3 = fw_m.addVar(lb=0,ub=35, name="a3",vtype=GRB.CONTINUOUS)
                    a4 = fw_m.addVar(lb=0,ub=35, name="a4",vtype=GRB.CONTINUOUS)
                    a5 = fw_m.addVar(lb=0,ub=35, name="a5",vtype=GRB.CONTINUOUS)
                    
                    
                    obj= a1*grad[i][0]+a2*grad[i][1]+a3*grad[i][2]+a4*grad[i][3]+a5*grad[i][4]
                    fw_m.setObjective(obj,GRB.MAXIMIZE)
                    fw_m.addConstr((a1+a2+a3+a4+a5)==150)
                    fw_m.optimize()
                    action_table[i][0]=a1.X
                    action_table[i][1]=a2.X
                    action_table[i][2]=a3.X
                    action_table[i][3]=a4.X
                    action_table[i][4]=a5.X
                    fw_m.reset()
                fw[i]=action_table[i]
                action_table[i]=action_table[i]*lr+ba[i] *(1-lr)
    
        self.sess.run(self.update,{self.S:bs,self.tf_table:action_table})
ddpg = DDPG(a_dim,s_dim,a_bound)    
#############################  Projection  ####################################

def Projection_function(action,arg):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as half_m:
            joint1_a1=action[0]
            joint1_a2=action[1]
            joint2_a1=action[2]
            joint2_a2=action[3]
            joint3_a1=action[4]
           
           
            a1 = half_m.addVar(lb=0,ub=35, name="a1",vtype=arg)
            a2 = half_m.addVar(lb=0,ub=35, name="a2",vtype=arg)
            a3 = half_m.addVar(lb=0,ub=35, name="a3",vtype=arg)
            a4 = half_m.addVar(lb=0,ub=35, name="a4",vtype=arg)
            a5 = half_m.addVar(lb=0,ub=35, name="a5",vtype=arg)
            
            obj= (a1-joint1_a1)**2+ (a2-joint1_a2)**2+(a3-joint2_a1)**2+(a4-joint2_a2)**2+(a5-joint3_a1)**2
            half_m.setObjective(obj,GRB.MINIMIZE)
            
            half_m.addConstr((a1+a2+a3+a4+a5)==150)



            half_m.optimize()
        
            return a1.X,a2.X,a3.X,a4.X,a5.X            
##################################################################################

Net_action=np.zeros((1000000,a_dim+2))   
ewma = []
eva_reward=[]
store_before_action_and_gaussain=[]
store_before_action=[]
store_after_action =[]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
counter=0
valid=np.zeros((23751, 5))
for i in range(10, 36):
    for j in range(10, 36):
        for k in range(10, 36):
            for l in range(10,36):
                for m_ in range(10,36):
                        if (i + j + k + l + m_  == 150):
                            valid[counter][0] = i
                            valid[counter][1] = j
                            valid[counter][2] = k
                            valid[counter][3] = l
                            valid[counter][4] = m_
                            counter = counter+1
reward=[]
var=3
step=0
for ep in range(1000000):
    R=0
    step=0
    done=False
    s=env.reset()
    while not done:
        step=step+1
        #use gaussian exploration by TD3 (0,0.1) to each eaction
        if  ddpg.pointer<10000:
            chose=np.random.randint(0, 23751)
            action = [valid[chose][0], valid[chose][1], valid[chose][2],valid[chose][3],valid[chose][4]]
            store_before_action.append(action)
        else :
            action=ddpg.choose_action(s)
            store_before_action_and_gaussain.append(action)

            action=(action+np.random.normal(0,5,a_dim)).clip(0,max_action)
            store_before_action.append(action)
        action=Projection_function(action,GRB.INTEGER)
        store_after_action.append(action)

        
        s_,r,done,info=env.step(action)
        done_bool = False if step==100 else done 
        ddpg.store_transition(s,action,r,s_,done_bool)

        if ddpg.pointer>=10000:
            ddpg.learn()
            ddpg.fw_update()
           
        if (ddpg.pointer+1)% eval_freq==0:
            eva_reward.append(evaluation(env_name,arg_seed,ddpg))
        s= s_
        R += r
    reward.append(R)
    ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
    if(ddpg.pointer>=10000):
        print("start_training,ddpgpointer:{}".format(ddpg.pointer))
    print({
        'episode': ep,
        'reward' :R,
        'ewma_reward' :ewma_r
    })
    ewma.append(ewma_r)
    if(ddpg.pointer>=500000):
        print("done training")
        break
a=[]
for i in range(ep+1):
    a.append(i)
plt.plot(a,ewma)
plt.title("ewma reward, final ewma={}".format(ewma[ep]))  

np.save("Bike_{}_DDPGFW_Reward".format(arg_seed),reward)
np.save("Bike_{}_DDPGFW_before_Action".format(arg_seed),store_before_action)
np.save("Bike_{}_DDPGFW_after_Action".format(arg_seed),store_after_action)
np.save("Bike_{}_DDPGFW_eval_reward".format(arg_seed),eva_reward)
np.save("Bike_{}_DDPGFW_before_Action_Gaussian".format(arg_seed),store_before_action_and_gaussain)
np.save("Bike_{}_DDPGFW_eval_before_Action".format(arg_seed),store_testing_before_action)
np.save("Bike_{}_DDPGFW_eval_After_Action".format(arg_seed),store_testing_after_action)
