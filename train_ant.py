from stable_baselines3 import SAC
from CustomAnt import DirectedAnt


direction = [1.0, 0.0]
env = DirectedAnt()

model = SAC("MlpPolicy", env, verbose=1,
            learning_starts=10_000, 
)

model.learn(total_timesteps=100_000, log_interval=4)

model.save("sac_ant_x")