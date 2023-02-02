# configuration:
import os
from ShapedDQN import ShapedDQN
from ShapedSAC import ShapedSAC
from ShapedTD3 import ShapedTD3

ENTITY = "reward-shapers"

# Hyperparams were based on https://openreview.net/pdf?id=HJjvxl-Cb
sac_mujoco_experiment = {"ALGO": ShapedSAC, "PROJ": "SAC-mujoco",
                         "GAMMA": 0.99, "ENT_COEF": 1/100, "ENV": "Reacher-v2",
                         "SWEEPID": "g3tvvhat", "COUNT": 1, 'TRAIN_STEPS': 50_000}

td3_mujoco_experiment = {"ALGO": ShapedTD3, "PROJ": "TD3-mujoco",
                         "GAMMA": 0.99, "ENV": "Hopper-v3",
                         "SWEEPID": "nnlahd9g", "COUNT": 1, 'TRAIN_STEPS': 100_000}


# set the sweep configuration
experiment = td3_mujoco_experiment

if experiment['wandb_link'] is None:
    # Run the sweep_generator.py script to generate a new sweep id in cmd line:
    os.system("python sweep_generator.py")
    # in the future, automatically populate the sweepid in the experiment config
