
import torch
from breakout import (NotSoDqnBreakout)
from model import BreakoutInator
from agent import Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = BreakoutInator().to(device)

model.load_model()

agent = Agent(model=model,
               device=device,
               epsilon=0,
               nb_warnup=5000,
               nb_actions=4,
               learning_rate=0.00001,
               memory_capacity=1000000,
               batch_size=64)


if __name__ == "__main__":

    environment = NotSoDqnBreakout(device=device, render_mode='human')

    agent.test(env = environment)
