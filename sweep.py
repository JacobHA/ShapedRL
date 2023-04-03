import argparse
import json
from algos import ShapedDQN, ShapedTD3, ShapedSAC
from stable_baselines3.common.callbacks import EvalCallback
import gym
import wandb


ENTITY = "qcoolers"
DEFAULT_SWEEP_CONFIG = "sweep_config.json"
env_name = None
algo = None

algo_to_class = {"dqn": ShapedDQN,
                 "td3": ShapedTD3,
                 "sac": ShapedSAC}

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

def sample_config():
    with open(DEFAULT_SWEEP_CONFIG) as f:
        sweep_config = json.load(f)

    return sweep_config

def wandb_atari():
    with wandb.init(sync_tensorboard=True) as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        hparams = get_hparams(algo, **dict_cfg)
        env = gym.make(env_name)
        eval_env = gym.make(env_name)

        model = algo_to_class[algo](env, **hparams)
        eval_callback = EvalCallback(eval_env, n_eval_episodes=1,
                                     log_path=f'./runs/{run.id}',
                                     eval_freq=15_000,
                                     deterministic=True,
                                     best_model_save_path=f'./best_model/{run.id}')
        
        model.learn(cfg.n_timesteps, log_interval=10, tb_log_name="runs", callback=eval_callback)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="bs-rl")
    parser.add_argument("-s", "--sweepid", type=str, help="sweep id", default=None)
    parser.add_argument("-n", "--number", type=int, help="number of runs", default=1)
    parser.add_argument("-e", "--env", type=str, help="gym environment name", required=True)
    parser.add_argument("-a", "--algo", type=str, help="algorithm name (dqn, sac, td3)", default="dqn")
    args = parser.parse_args()
    if not args.sweepid:
        with open(DEFAULT_SWEEP_CONFIG) as f:
            config = json.load(f)
        sweep_id = wandb.sweep(config, project=args.project)
    else:
        sweep_id = args.sweepid
    env_name = args.env
    algo = args.algo
    wandb.agent(sweep_id, function=wandb_atari, count=args.number, project=args.project)
