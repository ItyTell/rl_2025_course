import gymnasium as gym
import ale_py
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
 	

def find_rocket(obs):
    # Example function to find the rocket in the observation
    # This is a placeholder implementation
    rocket_color = [200, 72, 72]  # RGB color of the rocket
    mask = (obs[:, :, 0] == rocket_color[0]) & (obs[:, :, 1] == rocket_color[1]) & (obs[:, :, 2] == rocket_color[2])
    return mask

def find_ball(obs):
    # Example function to find the ball in the observation
    # This is a placeholder implementation
    ball_color = [236, 236, 236]  # RGB color of the ball
    mask = (obs[:, :, 0] == ball_color[0]) & (obs[:, :, 1] == ball_color[1]) & (obs[:, :, 2] == ball_color[2])
    return mask




gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

#play(env)


obs, info = env.reset()
#obs.shape = 210 160 3


obs, reward, terminated, truncated, info = env.step(1)
env.close()
plt.imshow(obs)
plt.axis('off')
plt.show()

