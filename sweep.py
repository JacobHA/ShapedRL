import argparse
import json
import wandb
import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecMonitor

from algos import ShapedDQN, ShapedTD3, ShapedSAC


ENTITY = "qcoolers"
DEFALUT_SWEEP_CONFIG = "sweep_config.json"
env_name = None
algo = None
n_envs = 1


wrapper_to_class = {
    "stable_baselines3.common.atari_wrappers.AtariWrapper": AtariWrapper,
}


same_dqn_atari_hparams = {
    "AsteroidsNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
}


def get_hparams(**hyperparams):
    envname = "AlmostEverythingWithNoFrameskip-v4" if env_name in same_dqn_atari_hparams else env_name
    with open("hparams.json") as f:
        config = json.load(f)[algo][envname]
    config.update(hyperparams)
    return config


def _make_env(hparams):
    def _init():
        env = gym.make(env_name)
        for wrapper in hparams["env_wrapper"]:
            env = wrapper_to_class[wrapper](env)
        return env
    return _init


def make_env(hparams):
    env = SubprocVecEnv([_make_env(hparams) for _ in range(n_envs)])
    env = VecMonitor(env)
    if "frame_stack" in hparams:
        env = VecFrameStack(env, hparams["frame_stack"])
    if "normalize" in hparams and hparams["normalize"]:
        env = VecNormalize(env, **hparams["normalize"])
    return env


def make_model(env, hparams):
    del [hparams["frame_stack"], hparams["env_wrapper"], hparams["n_timesteps"], hparams["normalize"]]
    return ShapedDQN(env=env, **hparams)


def wandb_atari():
    with wandb.init(sync_tensorboard=True) as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        hparams = get_hparams(**dict_cfg)
        n_time_steps = hparams["n_timesteps"]
        env = make_env(hparams)
        model = make_model(env, hparams)
        model.learn(n_time_steps, log_interval=10, tb_log_name="runs")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="bs-rl")
    parser.add_argument("-et", "--entity", type=str, default=ENTITY)
    parser.add_argument("-s", "--sweepid", type=str, help="sweep id", default=None)
    parser.add_argument("-n", "--number", type=int, help="number of runs", default=1)
    parser.add_argument("-e", "--env", type=str, help="gym environment name", required=True)
    parser.add_argument("-a", "--algo", type=str, help="algorithm name", default="dqn")
    parser.add_argument("--nenvs", type=int, help="number of parallel envs", default=1)
    args = parser.parse_args()
    if not args.sweepid:
        with open(DEFALUT_SWEEP_CONFIG) as f:
            config = json.load(f)
        sweep_id = wandb.sweep(config, project=args.project)
    else:
        sweep_id = args.sweepid
    env_name = args.env
    algo = args.algo
    n_envs = args.nenvs
    wandb.agent(sweep_id, function=wandb_atari, count=args.number, project=args.project)