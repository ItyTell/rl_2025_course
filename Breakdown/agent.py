import random
import torch
import copy
import torch.nn.functional as F
from plot import LivePlot
import numpy as np


class ReplayMemory:
    
    def __init__(self, capacity, device = 'cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0
    
    def insert(self, transition):
        transition = [ item.to(self.device) for item in transition ]
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else: 
            self.memory.remove(self.memory[0])
        
    def sample(self, batch_size=32):
        assert self.can_sample(batch_size), "Not enough elements in memory to sample"

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)

        return [ torch.cat(items).to(self.device) for items in batch ]
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10


    def __len__(self):
        return len(self.memory)


class Agent:
    

    def __init__(self, model, device, epsilon = 1, min_epsilon = 0.1, nb_warnup = 10000, nb_actions = None, memory_capacity = 10000,
                 batch_size = 32, learning_rate = 0.00025, gamma = 0.99):
        self.memory = ReplayMemory(memory_capacity, device)
        self.model = model
        self.model = self.model.to(device)
        self.target_model = copy.deepcopy(model)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - ((epsilon - min_epsilon) / nb_warnup ) * 2
        self.nb_actions = nb_actions
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"Startign epsilon: {self.epsilon}, min_epsilon: {self.min_epsilon}, epsilon_decay: {self.epsilon_decay}")


    def get_action(self, state):
        if random.random() < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)
    
    def train(self, env, epochs):
        stats = {"Returns": [], "avgReturns": [], "Epsilon": []}

        plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)

                self.memory.insert((state, action, reward, next_state, done))

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, next_state_b, done_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=1, keepdim=True)[0]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.nse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                state = next_state
                ep_return += reward.item()

            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:
                self.mdoel.save_the_model()
                print(" ")
                average_returns = np.mean(stats["Returns"][-100:])

                stats["avgReturns"].append(average_returns)
                stats["Epsilon"].append(self.epsilon)

                if (len(stats["Returns"]) > 100):
                    print(f"Epoch: {epoch} - Average Return {average_returns} - Epsilon: {self.epsilon}")
                else:
                    print(f"Epoch: {epoch} - Average Return {np.mean(stats["Returns"][-1:])} - Epsilon: {self.epsilon}")
            
            if epoch % 100 == 0:
                self.target_model.load_state.dict(self.model.state_dict())
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iteration{epoch}.pt")
        
        return stats















