# Revisiting Action-Constrained RL via Frank-Wolfe
This repo contains code accompaning the paper, Escaping from Zero Gradient: Revisiting Action-Constrained ReinforcementLearning via Frank-Wolfe Optimization (UAI 2021). It includes code for running the NFWPO algorithm presented in the paper, and other baseline methods such as DDPG+OptLayer, DDPG+Projection, DDPG+Reward Shaping.

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
(Change `ALGORITHM_NAME` to the corresponding algorithm, which includes `NFWPO`, `DDPG_Projection`, `DDPG_RewardShaping`, `DDPG_OptLayer`)

```=bash
python3 [ALGORITHM_NAME].py
```
Following are the examples for running the experiments in Ubuntu.

### BIKE SHARING SYSTEMS
To run the experiments metioned in Secion 4.1, please follow the instructions below:
#### A. Evaluating  FWPO  with  tabular  parameterization:
1. Enter the directory `BSS-3`.
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
1. Enter the directory `BSS-5`.
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
1. Enter the directory `NSFnet/src/gym`.
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
To run the experiments metioned in Secion 4.3, please first enter the directory `Reacher` for **Reacher with linear and nonlinear constraints**, and enter `Halfcheetah-State` for **Halfcheetah  with  state-dependent  constraints**
Then the remaining steps are same as the previous experiment.
The result is shown in Figure 4-5.

###  ADDITIONAL EXPERIMENT
To run the experiments metioned in Appendix C.3, please first enter the directory `Halfcheetah-Quad`.
Then the remaining steps are same as the previous experiment.
The result is shown in Figure 6.