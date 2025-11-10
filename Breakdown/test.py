import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
import numpy as np
 	


def find_rocket(obs):
    rocket_color = [200, 72, 72]  # RGB color of the rocket
    hand_raws = obs[-19, :,:]
    # match the color
    return np.where((hand_raws == rocket_color).all(1))[0][0]


def find_ball(obs_0, obs):
    """
        obs_0: previous observation
       obs: current observation
       #does not take into account bricks being hit
    """
    diff = np.maximum(0, obs.astype(np.int16) - obs_0.astype(np.int16))
    diff_sum = np.sum(diff, axis=2)
    x_indices, y_indices = np.where(diff_sum > 0)
    if len(x_indices) == 0:
        return None, None
    return x_indices[0], y_indices[0]


gym.register_envs(ale_py)
#env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.make("BreakoutNoFrameskip-v0", render_mode="rgb_array")


play(env)


obs, info = env.reset()
#obs.shape = 210 160 3


obs, reward, terminated, truncated, info = env.step(1)
env.close()
plt.imshow(obs)
plt.axis('off')
plt.show()

