Implementation for First Order Constrained Optimization in Policy Space (FOCOPS).

Link to paper https://arxiv.org/abs/2002.06506

### Requirements
[python](https://www.python.org/) (tested on 3.6.8) <br>
[pytorch](https://pytorch.org/) (tested on 1.3.1) <br>
[gym](https://github.com/openai/gym) (tested on 0.15.3) <br>
[MuJoCo v2.0](http://www.mujoco.org/) <br>
[mujoco-py](https://github.com/openai/mujoco-py) (tested on 1.50.1.0) <br>
For the circle experiments, please also install circle environments at
https://github.com/ymzhang01/mujoco-circle.

### Implementation
Example: Humanoid task in the robots with speed 
limits experiments (using the default parameters)
```
python focops_main.py --env-id='Humanoid-v3' --constraint='velocity'
```



