import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class MLP(nn.Module):
    """
    Multilayer perception
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        h_dim = 128

        self.linear1 = nn.Linear(in_features=input_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQN(nn.Module):
    """
    Deep Q-Network (DQN)
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output_dim = output_dim
        self.q_network = MLP(input_dim=input_dim, output_dim=output_dim)
        self.target_q_network = MLP(input_dim=input_dim, output_dim=output_dim)
        self.update_target_network()
        self.target_q_network.eval()

        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=5e-5)
        self.gamma = 0.99  # add this line to set the discount factor


    def forward(self, x):
        return self.q_network(x)

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:  # exploration
            action_id = random.randint(0, self.output_dim - 1)
        else:  # exploitation
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.forward(state)
            action_id = q_values.argmax(dim=1).item()

        return action_id

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
