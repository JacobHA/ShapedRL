import json

import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

from wrappers import FrameStack, PermuteAtariObs

from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.dqn import DQN

import argparse


atari_envs = {"AdventureNoFrameskip-v4", "AirRaidNoFrameskip-v4", "AlienNoFrameskip-v4", "AmidarNoFrameskip-v4", "AssaultNoFrameskip-v4", "AsterixNoFrameskip-v4", "AsteroidsNoFrameskip-v4", "AtlantisNoFrameskip-v4", "Atlantis2NoFrameskip-v4", "BackgammonNoFrameskip-v4", "BankHeistNoFrameskip-v4", "BasicMathNoFrameskip-v4", "BattleZoneNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "BerzerkNoFrameskip-v4", "BlackjackNoFrameskip-v4", "BowlingNoFrameskip-v4", "BoxingNoFrameskip-v4", "BreakoutNoFrameskip-v4", "CarnivalNoFrameskip-v4", "CasinoNoFrameskip-v4", "CentipedeNoFrameskip-v4", "ChopperCommandNoFrameskip-v4", "CrazyClimberNoFrameskip-v4", "CrossbowNoFrameskip-v4", "DarkchambersNoFrameskip-v4", "DefenderNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "DonkeyKongNoFrameskip-v4", "DoubleDunkNoFrameskip-v4", "EarthworldNoFrameskip-v4", "ElevatorActionNoFrameskip-v4", "EnduroNoFrameskip-v4", "EntombedNoFrameskip-v4", "EtNoFrameskip-v4", "FishingDerbyNoFrameskip-v4", "FlagCaptureNoFrameskip-v4", "FreewayNoFrameskip-v4", "FroggerNoFrameskip-v4", "FrostbiteNoFrameskip-v4", "GalaxianNoFrameskip-v4", "GopherNoFrameskip-v4", "GravitarNoFrameskip-v4", "HangmanNoFrameskip-v4", "HauntedHouseNoFrameskip-v4", "HeroNoFrameskip-v4", "HumanCannonballNoFrameskip-v4", "IceHockeyNoFrameskip-v4", "JamesbondNoFrameskip-v4", "JourneyEscapeNoFrameskip-v4", "KaboomNoFrameskip-v4", "KangarooNoFrameskip-v4", "KeystoneKapersNoFrameskip-v4", "KingKongNoFrameskip-v4", "KlaxNoFrameskip-v4", "KoolaidNoFrameskip-v4", "KrullNoFrameskip-v4", "KungFuMasterNoFrameskip-v4", "LaserGatesNoFrameskip-v4", "LostLuggageNoFrameskip-v4", "MarioBrosNoFrameskip-v4", "MiniatureGolfNoFrameskip-v4", "MontezumaRevengeNoFrameskip-v4", "MrDoNoFrameskip-v4", "MsPacmanNoFrameskip-v4", "NameThisGameNoFrameskip-v4", "OthelloNoFrameskip-v4", "PacmanNoFrameskip-v4", "PhoenixNoFrameskip-v4", "PitfallNoFrameskip-v4", "Pitfall2NoFrameskip-v4", "PongNoFrameskip-v4", "PooyanNoFrameskip-v4", "PrivateEyeNoFrameskip-v4", "QbertNoFrameskip-v4", "RiverraidNoFrameskip-v4", "RoadRunnerNoFrameskip-v4", "RobotankNoFrameskip-v4", "SeaquestNoFrameskip-v4", "SirLancelotNoFrameskip-v4", "SkiingNoFrameskip-v4", "SolarisNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", "SpaceWarNoFrameskip-v4", "StarGunnerNoFrameskip-v4", "SupermanNoFrameskip-v4", "SurroundNoFrameskip-v4", "TennisNoFrameskip-v4", "TetrisNoFrameskip-v4", "TicTacToe3DNoFrameskip-v4", "TimePilotNoFrameskip-v4", "TrondeadNoFrameskip-v4", "TurmoilNoFrameskip-v4", "TutankhamNoFrameskip-v4", "UpNDownNoFrameskip-v4", "VentureNoFrameskip-v4", "VideoCheckersNoFrameskip-v4", "VideoChessNoFrameskip-v4", "VideoCubeNoFrameskip-v4", "VideoPinballNoFrameskip-v4", "WizardOfWorNoFrameskip-v4", "WordZapperNoFrameskip-v4", "YarsRevengeNoFrameskip-v4", "ZaxxonNoFrameskip-v4"}
EVAL_FREQ = 50_000


def run(env_str, hparams, total_timesteps, log_freq, device='cuda', log_dir="./runs"):
    env_kwargs = {}
    det=False
    if env_str == "FrozenLake-v1":
        env_kwargs['is_slippery'] = False
        det=True

    if env_str in atari_envs:
        env = make_atari_env(env_str, n_envs=1, seed=42)
        # Stack 4 frames
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        # eval
        eval_env = make_atari_env(env_str, n_envs=1, seed=0)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        eval_env = VecTransposeImage(eval_env)
        policy = "CnnPolicy"
    else:
        env = gym.make(env_str, **env_kwargs)
        eval_env = gym.make(env_str, **env_kwargs)
        policy = "MlpPolicy"

    eval_callback = EvalCallback(eval_env, n_eval_episodes=20,
                    log_path=f'./runs/',
                    eval_freq=EVAL_FREQ,
                    deterministic=True,
                    verbose=1,)

    model = ShapedDQN(policy, env,
        verbose=4, **hparams, device=device, tensorboard_log=log_dir
    )
    model.learn(total_timesteps, log_interval=log_freq,
        callback=eval_callback, tb_log_name=str(model)+env_str+f'-det={det}'
    )
    # model = DQN(policy, env, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
    # model.learn(total_timesteps, log_interval=10, callback=eval_callback, tb_log_name="DQN"+env_str+f'-det={det}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=bool, default=True)
    parser.add_argument("-t", type=bool, default=False,
                        help="If true, the done mask is not applied to the potential function")
    parser.add_argument("-e", "--env", type=str, default="PongNoFrameskip-v4")  # "FrozenLake-v1" "PongNoFrameskip-v4"
    args = parser.parse_args()
    do_shape = args.s
    no_done_mask = args.t
    env_str = args.env
    with open(f"configs/{env_str}.json", 'r') as f:
        hparams = json.load(f)
    total_timesteps = hparams.pop('total_timesteps')
    log_freq = hparams.pop('log_freq')
    hparams['do_shape'] = do_shape
    hparams['no_done_mask'] = no_done_mask
    run(env_str, hparams, total_timesteps, log_freq, 'cuda')