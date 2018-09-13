import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_layers = [128, 64]

class QNet(nn.Module):
    """Neural network for approximating action value (Q) function"""

    def __init__(self, state_size, action_size, seed, 
                 hidden_layers=hidden_layers):
        """
        Instantiate modules
        
        # Parameters
            state_size (int): Size of state dimension
            action_size (int): Size of action dimension
            seed (int): Random seed
            hidden_layers (list, int): Units in each hidden layer
        """
        
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        # First hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, 
                                                      hidden_layers[0])])
        
        # Additional hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(s0, s1) for s0, s1 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Define network structure"""
        # ReLU activation for each hidden layer
        x = state 
        for linear in self.hidden_layers:
            x = F.relu(linear(x))

        return self.output(x)
    
class QNet_Duel(nn.Module):
    """Dueling network for approximating action value (Q) function"""

    def __init__(self, state_size, action_size, seed, 
                 hidden_layers=hidden_layers):
        """
        Instantiate modules
        
        # Parameters
            state_size (int): Size of state dimension
            action_size (int): Size of action dimension
            seed (int): Random seed
            hidden_layers (list, int): Units in each hidden layer
        """
        
        super(QNet_Duel, self).__init__()
        self.seed = torch.manual_seed(seed)

        # First hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, 
                                                      hidden_layers[0])])
        
        # Additional hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(s0, s1) for s0, s1 in layer_sizes])
        
        # Advantage stream output
        self.output_a = nn.Linear(hidden_layers[-1], action_size)
        
        # State value stream output
        self.output_v = nn.Linear(hidden_layers[-1], 1)

    def forward(self, state):
        """Define network structure"""
        # ReLU activation for each hidden layer
        x = state 
        for linear in self.hidden_layers[:-1]:
            x = F.relu(linear(x))

        # Advantage stream
        x_a = F.relu(self.hidden_layers[-1](x))
        x_a = self.output_a(x_a)
        
        # State value stream
        x_v = F.relu(self.hidden_layers[-1](x))
        x_v = self.output_v(x_v)

        # Aggregating module
        x_q = x_v + (x_a - torch.mean(x_a, dim=1, keepdim=True))

        return x_q