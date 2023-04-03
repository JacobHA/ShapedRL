import argparse
import json
import wandb
import gym

from ShapedDQN import ShapedDQN


ENTITY = "qcoolers"
DEFALUT_SWEEP_CONFIG = "sweep_config.json"


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


def get_hparams(algo, env_name, **hyperparams):
    env_name = "AlmostEverythingWithNoFrameskip-v4" if env_name in same_dqn_atari_hparams else env_name
    with open("hparams.json") as f:
        config = json.load(f)[algo][env_name]
    config.update(hyperparams)
    return config


def wandb_atari():
    with wandb.init(sync_tensorboard=True) as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        for env_name in same_dqn_atari_hparams:
            hparams = get_hparams("dqn", **dict_cfg)
            env = gym.make(env_name)
            model = ShapedDQN(env, **hparams)
            model.learn(cfg.n_timesteps, log_interval=10, tb_log_name="runs")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="bs-rl")
    parser.add_argument("-s", "--sweepid", type=str, help="sweep id", default=None)
    parser.add_argument("-n", "--number", type=int, help="number of runs", default=1)
    args = parser.parse_args()
    if not args.sweepid:
        with open(DEFALUT_SWEEP_CONFIG) as f:
            config = json.load(f)
        sweep_id = wandb.sweep(config, project=args.project)
    else:
        sweep_id = args.sweepid
    wandb.agent(sweep_id, function=wandb_atari, count=args.number, project=args.project)