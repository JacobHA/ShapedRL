# configuration:
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

td3_mujoco_finetuned0 = {"ALGO": ShapedTD3, "SHAPED": 1, "PROJ": "TD3-mujoco",
                         "GAMMA": 0.99, "ENV": "Hopper-v3",
                         "SWEEPID": None, "COUNT": 1, 'TRAIN_STEPS': 100_000,
                         "LEARNING_RATE": 4.5e-5, "BUFFER_SIZE": 3e6,
                         "BATCH_SIZE": 1.5e5, "LEARNING_STARTS": 200,
                         "TAU": 0.0062}

td3_mujoco_finetuned1 = {"ALGO": ShapedTD3, "SHAPED": 0, "PROJ": "TD3-mujoco",
                         "GAMMA": 0.99, "ENV": "Hopper-v3",
                         "SWEEPID": None, "COUNT": 1, 'TRAIN_STEPS': 100_000,
                         "LEARNING_RATE": 5e-5, "BUFFER_SIZE": 1.8e7,
                         "BATCH_SIZE": 1.5e6, "LEARNING_STARTS": 170,
                         "TAU": 0.006}
# set the sweep configuration
experiment = td3_mujoco_finetuned0
