import torch
from torch import nn

class NumberGeneration(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x): 
        x = torch.tensor(nn.functional.one_hot(x, num_classes=10), dtype=torch.float32)
        return self.model(x)
