
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from time import sleep
import random
from gymnasium.wrappers import RecordVideo
import pygame


env = gym.make('MountainCar-v0', render_mode='human')
env.reset()
for i in range(1000):
    next_state, reward, done, _, _ = env.step(1)
    print(next_state)
