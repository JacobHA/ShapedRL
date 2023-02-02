# configuration:
from ShapedDQN import ShapedDQN
from ShapedSAC import ShapedSAC
from ShapedTD3 import ShapedTD3

ENTITY = "reward-shapers"

# Hyperparams were based on https://openreview.net/pdf?id=HJjvxl-Cb
sac_mujoco_experiment = {"ALGO": ShapedSAC, "PROJ": "SAC-mujoco",
                         "GAMMA": 0.99, "ENT_COEF": 1/100, "ENV": "Reacher-v2"}

td3_mujoco_experiment = {"ALGO": ShapedTD3, "PROJ": "TD3-mujoco",
                         "GAMMA": 0.99, "ENV": "Hopper-v3"}


# set the sweep configuration
experiment = td3_mujoco_experiment
