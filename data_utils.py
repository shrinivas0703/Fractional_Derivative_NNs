import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import random


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


def get_cifar10_loaders_grayscale(data_dir="data/", batch_size=64, seed=42):
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale values
        ]
    )

    os.makedirs(data_dir, exist_ok=True)

    # Training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar100_loaders_grayscale(data_dir="data/", batch_size=64, seed=42):
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale values
        ]
    )

    os.makedirs(data_dir, exist_ok=True)

    # Training dataset
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Test dataset
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
