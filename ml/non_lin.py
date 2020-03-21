import torch
import torch.nn as nn


class Tanh(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.tanh(self.linear(x))
