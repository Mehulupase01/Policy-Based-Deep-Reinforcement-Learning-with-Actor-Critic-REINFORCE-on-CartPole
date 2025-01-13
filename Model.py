import torch
from torch import nn
import numpy as np

def argmax_with_random_tie_breaking(x):
    """
    Custom implementation of np.argmax with random tie breaking.
    """
    try:
        max_indices = torch.where(x == torch.max(x))[0]
        return torch.tensor(np.random.choice(max_indices))
    except:
        return torch.argmax(x)

class MLP(nn.Module):
    """
    Simple multi-layer perceptron.
    Can be used as a policy or value network.
    """
    def __init__(self, input_dim, output_dim, value=False, hidden_layers=[32,16,32,16], noise_std=0.1):
        super().__init__()
        self.value = value
        self.noise_std = noise_std
        self.hidden_layers = nn.ModuleList()

        prev_dim = input_dim
        for dim in hidden_layers:
            self.hidden_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim

        if value:
            self.output_layer = nn.Sequential(
                nn.Linear(prev_dim, 1),
                nn.ReLU()
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=1)
            )

    def forward(self, x, device):
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        x = x + torch.randn_like(x) * self.noise_std * 1

        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)