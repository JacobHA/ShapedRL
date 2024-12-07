## Define a custom ant environment where the goal is to run in a target direction
## The reward is the dot product of the velocity and the target direction


import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.ant_v4 import AntEnv

class DirectedAnt(AntEnv):
    def __init__(self, target_direction=None):
        super(DirectedAnt, self).__init__()
        if target_direction is None:
            target_direction = np.array([1, 0])
        else:
            assert len(target_direction) == 2
            if not isinstance(target_direction, np.ndarray):
                target_direction = np.array(target_direction)

        target_direction = target_direction.astype(np.float64)
        # Normalize the target direction
        target_direction /= np.linalg.norm(target_direction, ord=2)
        self.target_direction = target_direction

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        desired_velocity = np.dot(xy_velocity, self.target_direction)

        forward_reward = desired_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info
