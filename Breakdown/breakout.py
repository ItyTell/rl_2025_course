import collections
import gymnasium as gym
import numpy as np
import cv2
from PIL import Image
import torch
import ale_py


gym.register_envs(ale_py)

class NotSoDqnBreakout(gym.Wrapper):

    def __init__(self, render_mode='rgb_array', repeat = 4, device=torch.device("cuda")):
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)
        super(NotSoDqnBreakout, self).__init__(env)
        self.repeat = repeat
        self.lives = env.unwrapped.ale.lives()
        self.frame_buffer = []
        self.device = device
        self.life_worth = -1 #penalty for losing a life


    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)

            current_lives = info['lives']
            if current_lives < self.lives:
                total_reward += self.life_worth
                self.lives = current_lives


            total_reward += reward
            self.frame_buffer.append(obs)
            done = terminated #or truncated
            if done:
                break

        frames = self.frame_buffer[-3:]
        data = self.process_frame(frames)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1).float()
        done = done.to(self.device)

        return data, total_reward, done, info
    
    def process_frame(self, frames):
        data = tile_matrix(frames[-1])
        rocket = find_rocket(frames[-1])
        ball_x, ball_y = find_ball(frames[-2], frames[-1])
        ball_x_prev, ball_y_prev = find_ball(frames[-3], frames[-2])
        combined = np.concatenate(([rocket, ball_x, ball_y, ball_x_prev, ball_y_prev], data))
        combined = torch.tensor(combined).float()
        combined = combined.to(self.device)
        return combined

    def reset(self):
        self.frame_buffer = []

        obs, _ = self.env.reset()

        self.lives = self.env.unwrapped.ale.lives()

        return self.process_frame([obs, obs, obs])










def find_rocket(obs):
    rocket_color = [200, 72, 72]  # RGB color of the rocket
    hand_raws = obs[-19, :,:]
    # match the color
    return np.where((hand_raws == rocket_color).all(1))[0][0] / 150


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
        return -1, -1
    return x_indices[0] / 150, y_indices[0] / 200



def tile_matrix(obs):
    """ Returns a matrix representing the presence of bricks in the breakout game."""
    h = 6
    w = 8
    matrix = np.zeros((6, 18))
    for i in range(6):
        for j in range(18):
            tile_center = obs[57 + i*h + int(h / 2), 8 + j*w + int(w / 2)]
            if (tile_center == [0, 0, 0]).all():
                matrix[i, j] = 0
            else:
                matrix[i, j] = 1
    matrix = matrix.flatten()
    return matrix

