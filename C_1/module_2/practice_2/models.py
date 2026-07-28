import torch
import torch.nn as nn 

torch.manual_seed()

class SimpleMNISTLetterDetective(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.Flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(156,26)
        )

    def forward(self, input_):
        X = self.Flatten(input_)
        X = self.layers(X)

        return X