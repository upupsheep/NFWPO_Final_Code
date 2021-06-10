import numpy as np
import torch
from utils import torch_to_numpy
import gurobipy as gp
from gurobipy import GRB

class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL with GAE
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, obs_dim, act_dim, batch_size, max_eps_len):

        # Hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len

        # Batch buffer
        self.obs_buf = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cadv_buf = np.zeros((batch_size, 1), dtype=np.float32)

        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.eps_len = 0
        self.not_terminal = 1


        # Pointer
        self.ptr = 0
    def proj_function(self,action):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as half_m:
                joint1_a1=action[0]
                joint1_a2=action[1]
                joint2_a1=action[2]
                joint2_a2=action[3]
                joint3_a1=action[4]
                joint3_a2=action[5]
               
               
                a1 = half_m.addVar(lb=-1,ub=1, name="a1",vtype=GRB.CONTINUOUS)
                a2 = half_m.addVar(lb=-1,ub=1, name="a2",vtype=GRB.CONTINUOUS)
                a3 = half_m.addVar(lb=-1,ub=1, name="a3",vtype=GRB.CONTINUOUS)
                a4 = half_m.addVar(lb=-1,ub=1, name="a4",vtype=GRB.CONTINUOUS)
                a5 = half_m.addVar(lb=-1,ub=1, name="a5",vtype=GRB.CONTINUOUS)
                a6 = half_m.addVar(lb=-1,ub=1, name="a6",vtype=GRB.CONTINUOUS)
                
                obj= (a1-joint1_a1)**2+ (a2-joint1_a2)**2+(a3-joint2_a1)**2+(a4-joint2_a2)**2+(a5-joint3_a1)**2+(a6-joint3_a2)**2
                half_m.setObjective(obj,GRB.MINIMIZE)
                
                half_m.addConstr((a1**2+a2**2+a3**2)<=1)
                half_m.addConstr((a4**2+a5**2+a6**2)<=1)
    
    
                #reacher_m.addConstr((a1**2+a2**2)<=0.02) non-linear constranit 
    
                half_m.optimize()
                return [a1.X,a2.X,a3.X,a4.X,a5.X,a6.X],half_m.objVal
                #print('Obj: %g' % bike_m.objVal)
    def projection_state(self, action, state):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as half_m:
                neta1 = action[0]
                neta2 = action[1]
                neta3 = action[2]
                neta4 = action[3]
                neta5 = action[4]
                neta6 = action[5]
                w1 = state[11]
                w2 = state[12]
                w3 = state[13]
                w4 = state[14]
                w5 = state[15]
                w6 = state[16]
                su = np.abs(neta1 * w1) + np.abs(neta2 * w2) + np.abs(neta3 * w3) + np.abs(neta4 * w4) + np.abs(neta5 * w5) + np.abs(neta6 * w6)
                a1 = half_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
                a2 = half_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
                a3 = half_m.addVar(lb=-1, ub=1, name="a3", vtype=GRB.CONTINUOUS)
                a4 = half_m.addVar(lb=-1, ub=1, name="a4", vtype=GRB.CONTINUOUS)
                a5 = half_m.addVar(lb=-1, ub=1, name="a5", vtype=GRB.CONTINUOUS)
                a6 = half_m.addVar(lb=-1, ub=1, name="a6", vtype=GRB.CONTINUOUS)
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
                obj = (a1-neta1)**2 + (a2-neta2)**2 + (a3-neta3)**2 + \
                    (a4-neta4)**2 + (a5-neta5)**2 + (a6-neta6)**2
                half_m.setObjective(obj, GRB.MINIMIZE)

                half_m.addConstr(u1 == a1*w1)
                half_m.addConstr(u2 == a2*w2)
                half_m.addConstr(u3 == a3*w3)
                half_m.addConstr(u4 == a4*w4)
                half_m.addConstr(u5 == a5*w5)
                half_m.addConstr(u6 == a6*w6)
                half_m.addConstr(abs_u1 == (gp.abs_(u1)))
                half_m.addConstr(abs_u2 == (gp.abs_(u2)))
                half_m.addConstr(abs_u3 == (gp.abs_(u3)))
                half_m.addConstr(abs_u4 == (gp.abs_(u4)))
                half_m.addConstr(abs_u5 == (gp.abs_(u5)))
                half_m.addConstr(abs_u6 == (gp.abs_(u6)))
                half_m.addConstr((abs_u1 + abs_u2 + abs_u3 +
                                  abs_u4 + abs_u5 + abs_u6) == v)

                half_m.optimize()
                return (half_m.X[0:6]), su
    def reacher_proj(self,action):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as reacher_m:
                neta1 = action[0]
                neta2 = action[1]
                su = neta1 ** 2 + neta2 ** 2
                a1 = reacher_m.addVar(lb=-1, ub=1, name="a1", vtype=GRB.CONTINUOUS)
                a2 = reacher_m.addVar(lb=-1, ub=1, name="a2", vtype=GRB.CONTINUOUS)
                obj = (a1-neta1)**2 + (a2-neta2)**2
                reacher_m.setObjective(obj, GRB.MINIMIZE)
    
                reacher_m.addConstr((a1**2+a2**2) <= 0.02)
    
                reacher_m.optimize()
    
                return (a1.X, a2.X), su

    def run_traj(self, env, policy, value_net, cvalue_net, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint):

        batch_idx = 0

        cost_ret_hist = []

        avg_eps_len = 0
        num_eps = 0
        network_actions = []
        network_states = []
        while batch_idx < self.batch_size:
            obs = env.reset()

            ret_eps = 0
            cost_ret_eps = 0

            for t in range(self.max_eps_len):
                act = policy.get_act(torch.Tensor(obs).to(dtype).to(device))
                act = torch_to_numpy(act).squeeze()
                network_actions.append(act)
                network_states.append(obs)
                act , cost = self.projection_state(act,obs)
                next_obs, rew, done, info = env.step(act)

                if constraint == 'gurobi':
                    cost = cost
                elif constraint == 'state':
                    cost = cost
                elif constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                elif constraint == 'circle':
                    cost = info['cost']

                ret_eps += rew
                cost_ret_eps += (c_gamma ** t) * cost


                # Store in episode buffer
                self.obs_eps[t] = obs
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_obs
                self.rew_eps[t] = rew
                self.cost_eps[t] = cost

                obs = next_obs

                self.eps_len += 1
                batch_idx += 1

                # Terminal state if done or reach maximum episode length
                self.not_terminal = 0 if done or t == self.max_eps_len - 1 else 1


                # Store return for score if only episode is terminal
                if self.not_terminal == 0:
                    score_queue.append(ret_eps)
                    cscore_queue.append(cost_ret_eps)
                    cost_ret_hist.append(cost_ret_eps)

                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                if done or batch_idx == self.batch_size:
                    break

            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]
            self.cost_eps = self.cost_eps[:self.eps_len]


            # Calculate advantage
            adv_eps, vtarg_eps = self.get_advantage(value_net, gamma, gae_lam, dtype, device, mode='reward')
            cadv_eps, cvtarg_eps = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost')


            # Update batch buffer
            start_idx, end_idx = self.ptr, self.ptr + self.eps_len
            self.obs_buf[start_idx: end_idx], self.act_buf[start_idx: end_idx] = self.obs_eps, self.act_eps
            self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps, adv_eps
            self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx] = cvtarg_eps, cadv_eps


            # Update pointer
            self.ptr = end_idx

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.eps_len = 0
            self.not_terminal = 1

        avg_cost = np.mean(cost_ret_hist)
        std_cost = np.std(cost_ret_hist)


        # Normalize advantage functions
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)


        return {'states':self.obs_buf, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': self.adv_buf,
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf,
                'avg_cost': avg_cost, 'std_cost': std_cost, 'avg_eps_len': avg_eps_len,
                'network_actions': network_actions, 'network_states': network_states}

    def evaluation(self, env, policy, value_net, cvalue_net, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint, eval_episode = 10):
        avgreward=0
        for eptest in range(eval_episode):
            obs = env.reset()
            ret_eps=0
            cost_ret_eps=0
            for t in range(self.max_eps_len):
                    act = policy.get_act(torch.Tensor(obs).to(dtype).to(device))
                    act = torch_to_numpy(act).squeeze()
                    act , cost = self.projection_state(act,obs)
                    next_obs, rew, done, info = env.step(act)
    
                    if constraint == 'gurobi':
                        cost = cost
                    elif constraint == 'state':
                        cost = cost
                    elif constraint == 'velocity':
                        if 'y_velocity' not in info:
                            cost = np.abs(info['x_velocity'])
                        else:
                            cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                    elif constraint == 'circle':
                        cost = info['cost']
    
                    ret_eps += rew
                    cost_ret_eps += (c_gamma ** t) * cost
    
    
                    # Store in episode buffer
                    self.obs_eps[t] = obs
                    self.act_eps[t] = act
                    self.next_obs_eps[t] = next_obs
                    self.rew_eps[t] = rew
                    self.cost_eps[t] = cost
    
                    obs = next_obs
    
                    self.eps_len += 1
                    
    
                    # Terminal state if done or reach maximum episode length
                    self.not_terminal = 0 if done or t == self.max_eps_len - 1 else 1
    
    
                    # Store return for score if only episode is terminal
                    if self.not_terminal == 0:
                        score_queue.append(ret_eps)
                        cscore_queue.append(cost_ret_eps)

    
                    if done:
                        break

            print('Episode {}\tReward: {} \t AvgReward'.format(eptest, ret_eps))
            avgreward=avgreward+ret_eps
           
        avgreward=avgreward/eval_episode
        print("------------------------------------------------")
        print("Evaluation average reward :",avgreward)
        print("------------------------------------------------")
    
        return avgreward
    def get_advantage(self, value_net, gamma, gae_lam, dtype, device, mode='reward'):
        gae_delta = np.zeros((self.eps_len, 1))
        adv_eps =  np.zeros((self.eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((self.eps_len, 1))
        status[-1] = self.not_terminal
        prev_adv = 0

        for t in reversed(range(self.eps_len)):
            # Get value for current and next state
            obs_tensor = torch.Tensor(self.obs_eps[t]).to(dtype).to(device)
            next_obs_tensor = torch.Tensor(self.next_obs_eps[t]).to(dtype).to(device)
            current_val, next_val = torch_to_numpy(value_net(obs_tensor), value_net(next_obs_tensor))

            # Calculate delta and advantage
            if mode == 'reward':
                gae_delta[t] = self.rew_eps[t] + gamma * next_val * status[t] - current_val
            elif mode =='cost':
                gae_delta[t] = self.cost_eps[t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv

            # Update previous advantage
            prev_adv = adv_eps[t]

        # Get target for value function
        obs_eps_tensor = torch.Tensor(self.obs_eps).to(dtype).to(device)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor)) + adv_eps



        return adv_eps, vtarg_eps
