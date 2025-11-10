
import torch
import torch.nn as nn


class BreakoutInator(nn.Module):

    def __init__(self, nb_action = 4):
        super(BreakoutInator, self).__init__()
        self.relu = nn.ReLU()
        self.ac_fc1 = nn.Linear(113, 512)
        self.ac_fc2 = nn.Linear(512, 512)
        self.ac_fc3 = nn.Linear(512, nb_action)

        self.st_fc1 = nn.Linear(113, 512)
        self.st_fc2 = nn.Linear(512, 512)
        self.st_fc3 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = torch.tensor(x)

        action_val = self.relu(self.ac_fc1(x))
        action_val = self.relu(self.ac_fc2(action_val))
        action_val = self.ac_fc3(action_val)

        state_val = self.relu(self.st_fc1(x))
        state_val = self.dropout(state_val)
        state_val = self.relu(self.st_fc2(state_val))
        state_val = self.dropout(state_val)
        state_val = self.st_fc3(state_val)

        output = state_val + (action_val - action_val.mean())

        return output
    

    def save_model(self, path="models/breakout_inator.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="models/breakout_inator.pth"):
        try:
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")