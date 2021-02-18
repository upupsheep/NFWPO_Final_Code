# Revisiting Action-Constrained RL via Frank-Wolfe

This repo contains code accompaning the paper, Escaping from Zero Gradient: Revisiting Action-Constrained ReinforcementLearning via Frank-Wolfe Optimization (Lin et al., UAI 2021). It includes code for running the NFWPO algorithm presented in the paper, and other base line methods such as DDPG+OptLayer, DDPG+Projection, DDPG+Reward Shaping.

## Dependencies

This code requires the following:

- python 3.\*
- TensorFlow v2.0+
- pytorch+cuda
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

Note that to run the code of NSFnet, you must first enter `NSFnet/src/gym/`.
