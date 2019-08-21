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
            fc_units list[(int)]: List of Number of nodes in the hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        fc_unit_input_size = state_size
        self.fc = nn.ModuleList([])
        for fc_unit_size in fc_units:
            self.fc.append(nn.Linear(fc_unit_input_size, fc_unit_size))
            fc_unit_input_size = fc_unit_size
        self.fc.append(nn.Linear(fc_unit_input_size, action_size))


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc_net in self.fc:
            y = fc_net(x)
            x = F.relu(y)
        return y

