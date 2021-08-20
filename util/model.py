import torch
from torch import nn
from util.utils import seeder

class FFNN(nn.Module):
    """The desired 2 hidden layer feed forward neural network."""
    def __init__(self, random_state=0):
        super(FFNN, self).__init__()

        # Seeding the random weight initialization of the network.
        seeder(random_state)

        # Layer Initialization
        self.hidden1 = torch.nn.Linear(2, 20)
        self.hidden2 = torch.nn.Linear(20, 10)
        self.out = torch.nn.Linear(10,2)

    def Z(self, x):
        """
        Returns:
            Logit values on forward pass which need to be passed to the softmax for classification.
        """
        z = self.hidden1(x)
        z = torch.relu(z)
        z = self.hidden2(z)
        z = torch.relu(z)
        return self.out(z)
        
    def forward(self, x):
        """
        Returns:
            Log-softmax values on forward pass to the network.
        """
        logit = self.Z(x)
        return torch.log_softmax(logit, dim=1)
