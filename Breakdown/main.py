
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from breakout import (NotSoDqnBreakout)
from model import BreakoutInator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BreakoutInator().to(device)


if __name__ == "__main__":

    environment = NotSoDqnBreakout(render_mode='human', device=device)

    state = environment.reset()

    model.forward(state)


    for _ in range(100):
        action = environment.action_space.sample()
        next_state, reward, done, info = environment.step(action)

        if done:
            state = environment.reset()
