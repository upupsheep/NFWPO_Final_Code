

import os.path
import gurobipy as gp
from gurobipy import GRB
import scipy
import numpy
import gym
#import mujoco_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
#import network_sim
import NSFnet_multiV2

tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3' #make render not lag
os.environ['TF_DETERMINISTIC_OPS'] = '1'
env_name='PccNs_uti_NSFnet_multiV2-v0'
env=gym.make(env_name)
s_dim = env.observation_space.shape[0] #40
a_dim = env.action_space.shape[0]               #4
a_bound=env.action_space.high
ewma_r=0
arg_seed=0
#########################seed##############################
tf.compat.v1.reset_default_graph()
np.random.seed(arg_seed)
env.seed(arg_seed)
env.action_space.np_random.seed(arg_seed)
#####################  hyper parameters  ####################
exploration =0.1
LR_C=0.001
LR_A=0.0001
GAMMA=0.99
TAU=0.001

MEMORY_CAPACITY=50000
BATCH_SIZE=16
eval_freq = 10000
####################testing part#################################
def evaluation(env_name,seed,ddpg,eval_episode=5):
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
            action = OptLayer_function(action)
            s_, r, done, info = eval_env.step(action)
            s = s_
            running_reward = running_reward + r
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
        #self.sess = tf.compat.v1.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.Done=tf.compat.v1.placeholder(tf.float32, [None, 1], 'done')
        self.grad = []
        self.tf_table=tf.compat.v1.placeholder(tf.float32, [None, a_dim], 'table')

    
        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
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
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        
       
        action_td_error = tf.compat.v1.losses.mean_squared_error(labels=self.tf_table, predictions=self.a)
        self.update = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(action_td_error, var_list=self.ae_params)
        
        self.action_grad=tf.gradients(self.q,self.a) 
        
                 

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
       
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        buffer_size = min(ddpg.pointer+1,MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br=  bt[:,self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        #br = bt[:, -self.s_dim - 1-1: -self.s_dim-1]
        bs_ = bt[:,self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+1+self.s_dim]
        #bs_ = bt[:, -self.s_dim-1:-self.s_dim] 
        bd = bt[:,-1:]
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.Done:bd})
        #print(self.sess.run(self.a_loss,{self.S:bs}))
    def store_transition(self, s, a, r, s_,done):
        transition = np.hstack((s, a, [r], s_,done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
    '''
    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a
            #return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
    '''

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            net2 = tf.compat.v1.layers.dense(net,300, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            #a = tf.compat.v1.layers.dense(net2, self.a_dim, activation=self.new_relu10000, name='a', trainable=trainable)
            a = tf.compat.v1.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            #a = tf.multiply(tf.add(a,1), 50, name='scaled_a')
            #return  tf.multiply(a, self.a_bound, name='scaled_a')
            return a
            #return tf.multiply(a, self.a_bound, name='scaled_a')
        #tf.multiply(a, self.a_bound, name='scaled_a')
        #
    
    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu, name='cl1', trainable=trainable)
            #net2 = tf.compat.v1.layers.dense(tf.concat[net,a], 300, activation=tf.nn.relu, name='cl1', trainable=trainable)
            w2_net = tf.compat.v1.get_variable('w2_net', [400, 300], trainable=trainable)
            w2_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, 300], trainable=trainable)
            b2= tf.compat.v1.get_variable('b1', [1, 300], trainable=trainable)
            net2= tf.nn.relu(tf.matmul(a,w2_a)+tf.matmul(net,w2_net)+b2)
            return tf.compat.v1.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)
    def get_gradient(self,s,a):
        return self.sess.run(self.action_grad,{self.S: s, self.a: a})
    def fw_update(self):
        lr=0.05
        grads = []
        #lr= 2/((self.pointer-10000)%50+2)
        buffer_size = min(ddpg.pointer+1,MEMORY_CAPACITY)
        indices = np.random.choice(buffer_size, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        #ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        ba=self.sess.run(self.a, {self.S:bs})
        for i in range(BATCH_SIZE) :
            ba[i]=OptLayer_function(ba[i])
        for i in range(BATCH_SIZE) :
            grad=self.get_gradient([bs[i]],[ba[i]])
            grads.append(grad[0][0])
        grads=np.squeeze(grads)
        action_table=np.zeros((BATCH_SIZE, 9))   #(16,16,2
        #print(grads)
        for i in range(BATCH_SIZE):
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                with gp.Model(env=env) as fw_m:
                    a1 = fw_m.addVar(lb=0, ub=50, name="a1", vtype=GRB.CONTINUOUS)
                    a2 = fw_m.addVar(lb=0, ub=50, name="a2", vtype=GRB.CONTINUOUS)
                    a3 = fw_m.addVar(lb=0, ub=50, name="a3", vtype=GRB.CONTINUOUS)
                    a4 = fw_m.addVar(lb=0, ub=50, name="a4", vtype=GRB.CONTINUOUS)
                    a5 = fw_m.addVar(lb=0, ub=50, name="a5", vtype=GRB.CONTINUOUS) 
                    a6 = fw_m.addVar(lb=0, ub=50, name="a6", vtype=GRB.CONTINUOUS)
                    a7 = fw_m.addVar(lb=0, ub=50, name="a7", vtype=GRB.CONTINUOUS)
                    a8 = fw_m.addVar(lb=0, ub=50, name="a8", vtype=GRB.CONTINUOUS)
                    a9 = fw_m.addVar(lb=0, ub=50, name="a9", vtype=GRB.CONTINUOUS)
                    obj= a1*grads[i][0]+a2*grads[i][1]+a3*grads[i][2]+a4*grads[i][3]+a5*grads[i][4]+\
                        a6*grads[i][5]+a7*grads[i][6]+a8*grads[i][7]+a9*grads[i][8]
                    fw_m.setObjective(obj,GRB.MAXIMIZE)
                    fw_m.addConstr(a1+a2+a4+a5<=50) #16
                    fw_m.addConstr(a3+a6<=50) #17
                    fw_m.addConstr(a1+a2+a4<=50) #19 21
                    fw_m.addConstr(a3+a5+a6<=50) #22
                    fw_m.addConstr(a2+a4+a7+a9<=50) #23
                    fw_m.addConstr(a1+a8<=50) #24
                    fw_m.addConstr(a2+a4+a9<=50) #25
                    fw_m.addConstr(a2+a3+a9<=50) #27 30
                    fw_m.optimize()
                    action_table[i][0]=a1.X
                    action_table[i][1]=a2.X
                    action_table[i][2]=a3.X
                    action_table[i][3]=a4.X
                    action_table[i][4]=a5.X
                    action_table[i][5]=a6.X
                    action_table[i][6]=a7.X
                    action_table[i][7]=a8.X
                    action_table[i][8]=a9.X
                    fw_m.reset()
                action_table[i]=action_table[i]*lr+ba[i] *(1-lr)
        #print(action_table)
        #print("----------------")
        #tf_action=tf.convert_to_tensor(action_table)
        #action_td_error = tf.compat.v1.losses.mean_squared_error(labels=tf_table, predictions=self.a)
        #update = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(action_td_error)#)
        self.sess.run(self.update, {self.S:bs, self.tf_table:action_table})
        
ddpg = DDPG(a_dim,s_dim,a_bound)    

def OptLayer_function(action):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as net_m:
            neta1=action[0]
            neta2=action[1]
            neta3=action[2]
            neta4=action[3]
            neta5=action[4]
            neta6=action[5]
            neta7=action[6]
            neta8=action[7]
            neta9=action[8]
          
            a1 = net_m.addVar(lb=0, ub=50, name="a1", vtype=GRB.CONTINUOUS)
            a2 = net_m.addVar(lb=0, ub=50, name="a2", vtype=GRB.CONTINUOUS)
            a3 = net_m.addVar(lb=0, ub=50, name="a3", vtype=GRB.CONTINUOUS)
            a4 = net_m.addVar(lb=0, ub=50, name="a4", vtype=GRB.CONTINUOUS)
            a5 = net_m.addVar(lb=0, ub=50, name="a5", vtype=GRB.CONTINUOUS) 
            a6 = net_m.addVar(lb=0, ub=50, name="a6", vtype=GRB.CONTINUOUS)
            a7 = net_m.addVar(lb=0, ub=50, name="a7", vtype=GRB.CONTINUOUS)
            a8 = net_m.addVar(lb=0, ub=50, name="a8", vtype=GRB.CONTINUOUS)
            a9 = net_m.addVar(lb=0, ub=50, name="a9", vtype=GRB.CONTINUOUS)

            obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + (a4-neta4)**2 + (a5-neta5)**2 + \
                (a6-neta6)**2 + (a7-neta7)**2 + (a8-neta8)**2 + (a9-neta9)**2 
            net_m.setObjective(obj,GRB.MINIMIZE)
            
        
            net_m.addConstr(a1+a2+a4+a5<=50) #16
            net_m.addConstr(a3+a6<=50) #17
            net_m.addConstr(a1+a2+a4<=50) #19 21
            net_m.addConstr(a3+a5+a6<=50) #22
            net_m.addConstr(a2+a4+a7+a9<=50) #23
            net_m.addConstr(a1+a8<=50) #24
            net_m.addConstr(a2+a4+a9<=50) #25
            net_m.addConstr(a2+a3+a9<=50) #27 30
            net_m.optimize()
        
            return a1.X,a2.X,a3.X,a4.X,a5.X,a6.X,a7.X,a8.X,a9.X


        
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
ewma = []
all_action=[]
Op_action=[]
eva_reward=[]
store_before_action=[]
store_after_action=[]
store_network_output_action=[]
reward=[]
var=3
i=0
for ep in range(500):
    #env.render()
    R=0
    step=0
    done=False
    s=env.reset()
    while not done:
        #env.render()
        '''
        if ddpg.pointer<10000:
            action=env.action_space.sample()
        else:
            if np.random.random() <= exploration:
                action = env.action_space.sample()
            else:    
                action = ddpg.choose_action(s)
        '''
        #if ddpg.pointer %10 ==0:
        #    exploration=exploration*0.9999
        #use gaussian exploration by TD3 (0,0.1) to each eaction
        
        if ddpg.pointer<1000:
            action=env.action_space.sample()
            store_network_output_action.append(action)
            store_before_action.append(action)
            action=OptLayer_function(action)
            store_after_action.append(action)
        else :
            #action=(ddpg.choose_action(s)+1)/2 *100
            #action=action+np.random.normal(0,3,a_dim)
            a_temp = ddpg.choose_action(s)
            action=(a_temp+np.random.normal(0,3,a_dim)).clip(min_action,max_action)
            store_network_output_action.append(a_temp)
            if ddpg.pointer %250==0 :
                all_action.append(action)
                print(action)
            #action=action*a_bound
            store_before_action.append(action)
            action=OptLayer_function(action)
            store_after_action.append(action)
        step=step+1
        if ddpg.pointer %250==0 :
            Op_action.append(action)
            print("AfterOPt",action)
            
        #action=np.clip(action,0,100)
        #if ddpg.pointer %250==0 :
        #    print("afterclip",action)
        s_,r,done,info=env.step(action)
        done_bool = False if step==1000 else done    
        ddpg.store_transition(s,action,r,s_,done_bool)
        if ddpg.pointer>1000:
            if ddpg.pointer%50==0:
                ddpg.learn()
                ddpg.fw_update()
        if (ddpg.pointer+1)% eval_freq==0:
            eva_reward.append(evaluation(env_name,arg_seed,ddpg))
        s= s_
        R += r
        
    ewma_r = 0.05 * R + (1 - 0.05) * ewma_r
    if(ddpg.pointer>10000):
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
for i in range(500):
    a.append(1000*i)
plt.plot(a,ewma)
plt.title("ewma reward, lr=0.05 fix, final ewma={}".format(ewma[499])) 
#mask = np.isin(Net_action[:,2], -1)
#violate_index=np.where(mask)    
np.save("Network_{}_DDPGFW_NSFnet_multi_new_Reward".format(arg_seed),ewma)
np.save("Network_{}_DDPGFW_NSFnet_multi_new_store_network_output_action".format(arg_seed),store_network_output_action)
np.save("Network_{}_DDPGFW_NSFnet_multi_new_before_Action".format(arg_seed),store_before_action)
np.save("Network_{}_DDPGFW_NSFnet_multi_new_after_Action".format(arg_seed),store_after_action)
np.save("Network_{}_DDPGFW_NSFnet_multi_new_eval_reward".format(arg_seed),eva_reward)

    
