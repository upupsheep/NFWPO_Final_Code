# Revisiting Action-Constrained RL via Frank-Wolfe
This repo contains code accompaning the paper, Escaping from Zero Gradient: Revisiting Action-Constrained ReinforcementLearning via Frank-Wolfe Optimization (UAI 2021). It includes code for running the NFWPO algorithm presented in the paper, and other baseline methods such as DDPG+OptLayer, DDPG+Projection, DDPG+Reward Shaping, SAC+Projection, PPO+Projection, TRPO+Projection, FOCOPS.

## Dependencies
This code requires the following:

- python 3.\*
- TensorFlow v2.0+
- pytorch+cuda
- [mujoco-py](https://github.com/openai/mujoco-py)
- [Gurobi](https://www.gurobi.com/)
- [CVXPY](https://www.cvxpy.org/)
- [CvxpyLayer](https://locuslab.github.io/2019-10-28-cvxpylayers/)
- [QPTH](https://github.com/locuslab/qpth)

## Usage
To run the code, enter the directory of the corresponding environment, and run the following command:
(Change `ALGORITHM_NAME` to the corresponding algorithm, which includes `NFWPO`, `DDPG_Projection`, `DDPG_RewardShaping`, `DDPG_OptLayer`, `SAC_Projection`)

```=bash
python3 [ALGORITHM_NAME].py
```
(To run other baselines such as PPO+Projection, TRPO+Projection, and FOCOPS, please refer to the description below.)

Following are the examples for running the experiments in Ubuntu.

### BIKE SHARING SYSTEMS
To run the experiments metioned in Secion 4.1, please follow the instructions below:
#### A. Evaluating  FWPO  with  tabular  parameterization:
1. Enter the directory `BSS-3`:
```
cd BSS-3
```
2. Set the random seed `arg_seed` between 0-4 in Line 25 of `NFWPO.py`.
3. Use the following command to train NFWPO:
```
python3 NFWPO.py
```
4. To run other baseline methods, set the random seed `arg_seed` between 0-4 in `DDPG_Projection.py`, `DDPG_RewardShaping.py`, and run the corresponding command:
```
python3 DDPG_Projection.py
```
```
python3 DDPG_RewardShaping.py
```
5. The result is shown in Figure 1.

#### B. Evaluating  NFWPO:
1. Enter the directory `BSS-5`:
```
cd BSS-5
```
2. Set the random seed `arg_seed` between 0-4 in `NFWPO.py`.
3. Use the following command to train NFWPO:
```
python3 NFWPO.py
```
4. To run other baseline methods, set the random seed `arg_seed` between 0-4 in `DDPG_Projection.py`, `DDPG_RewardShaping.py`, and set `random_seed` in `DDPG_OptLayer`. Then run the corresponding command:
```
python3 DDPG_Projection.py
```
```
python3 DDPG_RewardShaping.py
```
```
python3 DDPG_OptLayer.py
```
5. The result is shown in Figure 2.

###   UTILITY MAXIMIZATION OFCOMMUNICATION NETWORKS
1. Enter the directory `NSFnet/src/gym`:
```
cd NSFnet/src/gym
```
2. Set the random seed `arg_seed` between 0-4 in `NFWPO.py`.
3. Use the following command to train NFWPO:
```
python3 NFWPO.py
```
4. To run other baseline methods, set the random seed `arg_seed` between 0-4 in `DDPG_Projection.py`, `DDPG_RewardShaping.py`, and set `random_seed` in `DDPG_OptLayer`. Then run the corresponding command:
```
python3 DDPG_Projection.py
```
```
python3 DDPG_RewardShaping.py
```
```
python3 DDPG_OptLayer.py
```
5. The result is shown in Figure 3.

###    MUJOCO CONTINUOUS CONTROL TASKS
To run the experiments metioned in Secion 4.3, please first enter the directory `Reacher` for **Reacher with nonlinear constraints**, and enter `Halfcheetah-State` for **Halfcheetah  with  state-dependent  constraints**.
```
cd Reacher
```
```
cd Halfcheetah-State
```
<!-- Then the remaining steps are same as the previous experiment. -->
To run NFWPO, DDPG+Projection, DDPG+Reward Shaping, DDPG+OptLayer, please refer to the description in the previous experiment.

To run SAC+Projection, use the following command:
```
python3 SAC+Projection
```

To run TRPO+Projection, PPO+Projection:
```
# For Halfcheetah-state task
python3 PPO_TRPO_Projection/PPO_Projection_Halfcheetah_State_Relate_gym.py 
python3 PPO_TRPO_Projection/TRPO_Projection_Halfcheetah_State_Relate_gym.py

# For Reacher task
python3 PPO_TRPO_Projection/PPO_Projection_Reacher_State_Relate_gym.py 
python3 PPO_TRPO_Projection/TRPO_Projection_Reahcer_State_Relate_gym.py
```

To run FOCOPS:
```
# For Halfcheetah-state task
python3 FOCOPS/focops_main_cheetah.py

# For Reacher task
python3 python3 FOCOPS/focops_main_reacher.py
```
The result is shown in Figure 4-5.

###  ADDITIONAL EXPERIMENT
To run the experiments metioned in Appendix D.3, please first enter the directory `Halfcheetah-CAPG`.
```
cd Halfcheetah-CAPG
```
Then run the following commands for corresponding baselines:
```
# CAPG+PPO
python3 CAPG_PPO_Halfcheetah_bound_constraints.py

# CAPG+TRPO
python3 CAPG_TRPO_Halfcheetah_bound_constraints.py
```
The result is shown in Figure 6.