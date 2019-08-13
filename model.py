import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        fc_unit_input_size = state_size
        self.fc = nn.ModuleList([])
        for fc_unit_size in fc_units:
            self.fc.append(nn.Linear(fc_unit_input_size, fc_unit_size))
            fc_unit_input_size = fc_unit_size
        self.fc.append(nn.Linear(fc_unit_input_size, action_size))
        #self.fc1 = nn.Linear(state_size, fc1_units)
        #self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc_net in self.fc:
            y = fc_net(x)
            x = F.relu(y)
        #x = F.relu(self.fc1(state))
        #x = F.relu(self.fc2(x))
        return y

