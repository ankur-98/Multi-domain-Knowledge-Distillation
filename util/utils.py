import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seeder(state):
    """Seeding for reproducibility purposes."""
    random.seed(state)
    np.random.seed(state)
    torch.manual_seed(state)

def accuracy(pred, target):
    """Calculates the accuracy."""
    assert pred.shape[0] == target.shape[0]
    if pred.dim != 1: pred = torch.argmax(pred, dim=1)
    return (pred == target).sum() / target.shape[0]

def plotter(train, valid, ylabel=None):
    """Plots a graph for train and/or valid data."""
    plt.figure(figsize=(8, 8))
    plt.plot(train)
    if len(valid) > 0: plt.plot(valid)

    plt.title(f"Model {ylabel}.")
    plt.ylabel(f"{ylabel}")
    plt.xlabel(f"Epoch")

    legend = ['train']
    if len(valid) > 0: legend.append('valid')
    plt.legend(legend)
    plt.show()
    