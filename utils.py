""" Utilities for running sweep experiments with wandb. """
import wandb
import yaml
from config import ENTITY, experiment
from ShapedDQN import ShapedDQN
from ShapedSAC import ShapedSAC
from ShapedTD3 import ShapedTD3

PROJ = experiment['PROJ']
env = experiment['ENV']
gamma = experiment['GAMMA']
algo = experiment['ALGO']

try:
    ent_coef = experiment['ENT_COEF']
except KeyError:
    print("No entropy coefficient specified.")
    if algo == ShapedSAC:
        raise ValueError(
            "Entropy coefficient (or 'auto') must be specified for SAC.")


if algo == ShapedDQN:
    algo_str = "ShapedDQN"
elif algo == ShapedSAC:
    algo_str = "ShapedSAC"
elif algo == ShapedTD3:
    algo_str = "ShapedTD3"


def make_new_sweep():
    """ Generate a new sweep id based on config file."""
    # Is there a better way to do it?:
    algo_short = algo_str.split("Shaped")[1].lower()
    yaml_file = f'sweep_configs/{algo_short}_sweep.yml'
    with open(yaml_file, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project=PROJ, entity=ENTITY)
    return sweep_id


def configured_model(config):
    """ Return a wandb configured model."""
    if algo_str == "ShapedDQN":
        model = ShapedDQN("MlpPolicy", env, verbose=0,
                          shaped=config.apply_shaping,
                          learning_rate=config.learning_rate,
                          batch_size=int(config.batch_size),
                          buffer_size=int(config.buffer_size),
                          exploration_fraction=config.exploration_fraction,
                          exploration_final_eps=config.exploration_final_eps,
                          gamma=gamma,
                          shaped=config.shaped,
                          tensorboard_log="./logs/")

    elif algo_str == "ShapedSAC":
        model = ShapedSAC("MlpPolicy", env, verbose=0,
                          learning_rate=config.learning_rate,
                          gamma=gamma,
                          buffer_size=int(config.buffer_size),
                          ent_coef=ent_coef,
                          batch_size=int(config.batch_size),
                          learning_starts=int(config.learning_starts),
                          tau=config.tau,
                          shaped=config.shaped,
                          tensorboard_log="./logs/",)

    elif algo_str == "ShapedTD3":
        model = ShapedTD3("MlpPolicy", env, verbose=0,
                          learning_rate=config.learning_rate,
                          gamma=gamma,
                          buffer_size=int(config.buffer_size),
                          batch_size=int(config.batch_size),
                          learning_starts=int(config.learning_starts),
                          tau=config.tau,
                          shaped=config.shaped,
                          tensorboard_log="./logs/",)
    else:
        raise ValueError(f"Invalid algorithm name: {algo_str}")
    return model
