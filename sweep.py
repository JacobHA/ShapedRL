import argparse
import wandb

from ShapedDQN import ShapedDQN


ENTITY = "qcoolers"


sweep_ant = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/tot_ep_rew_mean'
    },
    "parameters": {
        "shape": {
            "values": [True, False],
        },
        "minus_v": {
            "values": [True, False],
        },
    },
}

envs = [
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
]


def wandb_atari():
    run = wandb.init(config=sweep_ant, sync_tensorboard=True)
    cfg = run.config
    model = ShapedDQN(cfg.map_name, cfg.clip_method, wandb_run=run)
    model.learn(cfg.n_timesteps, log_interval=10, tb_log_name="runs")
    wandb.finish()


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="bs-rl")
    args = parser.parse_args()
    sweep_id = wandb.sweep(config, project=args.project)
    wandb.agent(sweep_id, function=wandb_atari, count=40, project=args.project)