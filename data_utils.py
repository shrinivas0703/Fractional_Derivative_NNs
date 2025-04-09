import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_fashion_mnist_loaders(data_dir="data/", batch_size=64, seed=42):
    torch.manual_seed(seed)

    transform = transforms.ToTensor()
    os.makedirs(data_dir, exist_ok=True)

    # Training dataset
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Test dataset
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
