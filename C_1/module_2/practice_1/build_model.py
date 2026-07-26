import torch 
import torch.nn as nn 

torch.manual_seed(42)

class SimpleMNISTDNN(nn.Module):
    def __init__(self):
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self, input_):
        x = self.flatten(input_)
        x = self.layers(x)

        return x 
