"""
@This basline code was adapt from: https://github.com/pfnet-research/capg

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import logging
import os
import gurobipy as gp
from gurobipy import GRB
import chainer
from chainer import functions as F
import gym
#gym.undo_logger_setup()
import gym.wrappers
import numpy as np

import chainerrl

from CAPG_TRPO_Halfcheetah_bound_constraints import ClippedGaussianPolicy

from call_render import CallRender
from clip_action import ClipAction

reward=[]
class ObsNormalizedModel(chainerrl.agents.a3c.A3CSeparateModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, policy, vf, obs_size):
        super().__init__(policy, vf)
        with self.init_scope():
            self.obs_filter = chainerrl.links.EmpiricalNormalization(
                shape=obs_size
            )

    def __call__(self, obs):
        obs = F.clip(self.obs_filter(obs, update=False),
                     -5.0, 5.0)
        return super().__call__(obs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2',
                        help='Gym Env ID')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total time steps for training.')
    parser.add_argument('--eval-interval', type=int, default=100000,
                        help='Interval between evaluation phases in steps.')
    parser.add_argument('--eval-n-runs', type=int, default=100,
                        help='Number of episodes ran in an evaluation phase')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training')
    parser.add_argument('--load', type=str, default='',
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')
    parser.add_argument('--ppo-update-interval', type=int, default=2048,
                        help='Interval steps of PPO iterations.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--use-clipped-gaussian', action='store_true',
                        help='Use ClippedGaussian instead of Gaussian')
    parser.add_argument('--n-hidden-channels', type=int, default=64,
                        help='Number of hidden channels.')
    parser.add_argument('--adam-lr', type=float, default=3e-4)
    parser.add_argument('--label', type=str, default='')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

           
                    
    def evaluation(env_name,seed,agent,eval_episode=10):
        avgreward=0
        r=0
        avg=[]
        eval_env=gym.make(env_name)
        eval_env.seed(seed+100)
        for eptest in range(eval_episode):
            running_reward =0
            r=0
            done=False
            s=eval_env.reset()
            while not done:     
                action= agent.act_and_train(s,r)
                s_,r,done,info=eval_env.step(action)
                s=s_
                running_reward=running_reward+r
            agent.stop_episode_and_train(s,r,done)
            print('Episode {}\tReward: {} \t AvgReward'.format(eptest, running_reward))
            avgreward=avgreward+running_reward
            avg.append(running_reward)
        avgreward=avgreward/eval_episode
        print("------------------------------------------------")
        print("Evaluation average reward :",avgreward)
        print("------------------------------------------------")
    
        return avgreward

    def make_env(test):
        env = gym.make(args.env)
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        assert 0 <= env_seed < 2 ** 32
        env.seed(env_seed)
        mode = 'evaluation' if test else 'training'
        env = gym.wrappers.Monitor(
            env,
            args.outdir,
            mode=mode,
            video_callable=False,
            uid=mode,
        )
        if args.render:
            env = CallRender(env)
        env = ClipAction(env)
        return env

    env = make_env(test=False)
    #print(env.spec.max_episode_steps)
    timestep_limit = env.spec.max_episode_steps
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    if not isinstance(obs_space, gym.spaces.Box):
        print("""\
This example only supports gym.spaces.Box observation spaces. To apply it to
other observation spaces, use a custom phi function that convert an observation
to numpy.ndarray of numpy.float32.""")  # NOQA
        return

    # Parameterize log std
    def var_func(x): return F.exp(x) ** 2

    assert isinstance(action_space, gym.spaces.Box)
    # Use a Gaussian policy for continuous action spaces
    
    args.use_clipped_gaussian=False
   
    if args.use_clipped_gaussian:
        policy = \
            ClippedGaussianPolicy(
                obs_space.low.size,
                action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
                var_func=var_func,
                var_param_init=0,  # log std = 0 => std = 1
                min_action=action_space.low.astype(np.float32),
                max_action=action_space.high.astype(np.float32),
            )
    else:
        policy = \
            chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_space.low.size,
                action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
                var_func=var_func,
                var_param_init=0,  # log std = 0 => std = 1
            )

    # Use a value function to reduce variance
    vf = chainerrl.v_functions.FCVFunction(
        obs_space.low.size,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=2,
        last_wscale=0.01,
        nonlinearity=F.tanh,
    )

    model = ObsNormalizedModel(policy, vf, obs_space.low.size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    opt = chainer.optimizers.Adam(args.adam_lr)
    opt.setup(model)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    chainerrl.misc.draw_computational_graph(
        [model(fake_obs)], os.path.join(args.outdir, 'model'))

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    agent = chainerrl.agents.PPO(
        model=model,
        optimizer=opt,
        phi=lambda x: x.astype(np.float32, copy=False),
        update_interval=args.ppo_update_interval,
        gamma=0.995,
        lambd=0.97,
        standardize_advantages=True,
        entropy_coef=0,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(test=True)
        eval_stats = chainerrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        eval_freq=5000
        eva_reward=[]
        action_before=[]
        action_after=[]
        s= env.reset()
        R=0
        r=0
        ewma_r=0
        done=False
        steps=0
        for ep in range(args.steps):
            s= env.reset()
            R=0
            r=0
            done=False
            while not done :
               steps=steps+1
               action= agent.act_and_train(s,r)
               action_before.append(action)
               action_after.append(action)
               s_,r,done,info=env.step(action)
               s=s_
               R=R+r
               if (steps)% eval_freq==0:
                   eva_reward.append(evaluation(args.env,args.seed,agent))
               else :
                   agent.stop_episode_and_train(s,r,done)
            reward.append(R)
            ewma_r=0.05*R+(1-0.05)*ewma_r
            print({
                'episode:':ep,
                'Reward':R,
                'ewma':ewma_r,
                'stpes':steps
                })
            if steps>=1000000:
                break
        agent.save('final_agent')
        np.save("Halfcheetah_{}_PPO_withoutCAPG_Evaluation_Reward_Lower_upper".format(args.seed),eva_reward)
        np.save("Halfcheetah_{}_PPO_withoutCAPG_Action_before_Lower_upper".format(args.seed),action_before)
        np.save("Halfcheetah_{}_PPO_withoutCAPG_Action_After_Lower_upper".format(args.seed),action_after)
        np.save("Halfcheetah_{}_PPO_withoutCAPG_Training_Reward_Lower_upper".format(args.seed),reward)

        
            
        
        '''
        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
        )
        '''
        


if __name__ == '__main__':
    main()
