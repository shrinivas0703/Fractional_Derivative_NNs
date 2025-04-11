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


def get_cifar100_loaders(data_dir="data/", batch_size=64, seed=42):
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
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


def get_noisy_fashion_mnist_loaders(
    data_dir="data/", batch_size=64, seed=42, subset_ratio=0.1, noise_ratio=0.2
):
    torch.manual_seed(seed)
    random.seed(seed)

    transform = transforms.ToTensor()
    os.makedirs(data_dir, exist_ok=True)

    # Load full training dataset
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Subsample 10% of the data
    num_total = len(full_train_dataset)
    num_subset = int(subset_ratio * num_total)
    subset_indices = random.sample(range(num_total), num_subset)
    subset_dataset = Subset(full_train_dataset, subset_indices)

    # Inject label noise into 20% of subset
    noisy_targets = full_train_dataset.targets[subset_indices].clone()
    num_noisy = int(noise_ratio * num_subset)
    noisy_indices = random.sample(range(num_subset), num_noisy)
    for idx in noisy_indices:
        true_label = noisy_targets[idx].item()
        new_label = random.choice([l for l in range(10) if l != true_label])
        noisy_targets[idx] = new_label

    # Manually override targets in subset
    subset_dataset.dataset.targets = full_train_dataset.targets.clone()
    for i, subset_idx in enumerate(subset_indices):
        subset_dataset.dataset.targets[subset_idx] = noisy_targets[i]

    # Test dataset (clean)
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
