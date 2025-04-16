import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=200, num_classes=3):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def compute_loss(self, predictions, targets):
        return F.cross_entropy(predictions, targets)

    def backward_step(self, loss):
        loss.backward()
