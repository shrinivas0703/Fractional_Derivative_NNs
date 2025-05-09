import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.datasets import make_blobs
import openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def get_fashion_mnist_loaders(data_dir="data/", batch_size=64, seed=42, val_split=0.1):
    torch.manual_seed(seed)
    transform = transforms.ToTensor()
    os.makedirs(data_dir, exist_ok=True)

    full_train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cifar10_loaders(data_dir="data/", batch_size=64, seed=42, val_split=0.1):
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )
    os.makedirs(data_dir, exist_ok=True)

    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cifar100_loaders(data_dir="data/", batch_size=64, seed=42, val_split=0.1):
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

    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_kmnist_loaders(data_dir="data/", batch_size=64, seed=42, val_split=0.1):
    """
    Returns train, val, and test DataLoaders for the KMNIST dataset.
    """
    torch.manual_seed(seed)
    transform = transforms.ToTensor()
    os.makedirs(data_dir, exist_ok=True)

    full_train_dataset = datasets.KMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.KMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_helena_loaders(batch_size=128, seed=42, val_split=0.1):
    torch.manual_seed(seed)

    dataset = openml.datasets.get_dataset("Helena", version=1)
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )

    # Encode labels as integers
    y = pd.factorize(y)[0]

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    full_dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X.shape[1], len(set(y))
