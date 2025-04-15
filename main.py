#!/usr/bin/env python3

import torch
from torch import optim
from ConvolutionNeuralNetwork import ResNet50
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from FracGrad import FractionalSGD
import argparse
import copy
import torch.nn as nn

DATA_DIR = "data/"
PLOTS_DIR = "plots/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def train(
    model, train_loader, val_loader, optimizer, device, max_epochs=50, patience=5
):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = model.compute_loss(outputs, labels)
            model.backward_step(loss)
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = model.compute_loss(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: validation loss = {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def validation(model, device, validation_loader, quiet=False):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    if not quiet:
        print(
            "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                validation_loss,
                correct,
                len(validation_loader.dataset),
                100.0 * correct / len(validation_loader.dataset),
            )
        )

    return 100.0 * correct / len(validation_loader.dataset)


def plot_flatness(
    model, test_loader, device, initial_params, final_params, model_id, flag, dataset
):
    os.makedirs(os.path.join(PLOTS_DIR, dataset), exist_ok=True)
    v_err = []
    alpha_v = []
    model.to(device)
    for alpha in torch.linspace(0, 1.5, steps=10):
        alpha = alpha.to(device)
        for name, param in model.named_parameters():
            param.data = (1.0 - alpha) * initial_params[
                name
            ].data + alpha * final_params[name].data
        v_err.append(100.0 - validation(model, device, test_loader, quiet=True))
        alpha_v.append(alpha.to("cpu"))
        print(f" alpha = {alpha:.2f} has validation error {v_err[-1]:.2f}%")

    plt.plot(alpha_v, v_err)
    plt.xlabel("Interpolation coefficient (alpha)")
    plt.ylabel("Validation error (%)")
    plt.title(f"Loss Flatness for ResNet-50 Model #{model_id} on {dataset}")
    plt.grid(True)
    fname = (
        f"flatness_plot_resnet50_model_{model_id}.png"
        if not flag
        else f"flatness_plot_fractional_resnet50_model_{model_id}.png"
    )
    plt.savefig(os.path.join(PLOTS_DIR, dataset, fname))
    plt.close()


def compute_sharpness(model, loss_fn, data_loader, device, rho=1e-2, num_batches=1):
    model.eval()
    original_state = copy.deepcopy(model.state_dict())

    batch_iter = iter(data_loader)
    total_original_loss = 0.0
    total_perturbed_loss = 0.0

    for _ in range(num_batches):
        inputs, targets = next(batch_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute original loss
        outputs = model(inputs)
        original_loss = loss_fn(outputs, targets)
        total_original_loss += original_loss.item()

        # Apply perturbation to weights
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.empty_like(param).uniform_(-rho, rho)
                    param.add_(noise)

        # Compute perturbed loss
        with torch.no_grad():
            outputs_perturbed = model(inputs)
            perturbed_loss = loss_fn(outputs_perturbed, targets)
            total_perturbed_loss += perturbed_loss.item()

        # Restore original model parameters
        model.load_state_dict(original_state)

    avg_orig = total_original_loss / num_batches
    avg_pert = total_perturbed_loss / num_batches
    sharpness = ((avg_pert - avg_orig) / (1 + avg_orig)) * 100
    return sharpness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-fractional",
        action="store_true",
        help="Use FractionalSGD instead of standard SGD",
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["fashion-mnist", "cifar10", "cifar100"],
        default="fashion-mnist",
        help="Dataset to use: fashion-mnist, cifar10, or cifar100",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "fashion-mnist":
        from data_utils import get_fashion_mnist_loaders

        get_loader_fn = get_fashion_mnist_loaders
        in_channels = 1
        num_classes = 10
    elif args.data == "cifar10":
        from data_utils import get_cifar10_loaders

        get_loader_fn = get_cifar10_loaders
        in_channels = 3
        num_classes = 10
    elif args.data == "cifar100":
        from data_utils import get_cifar100_loaders

        get_loader_fn = get_cifar100_loaders
        in_channels = 3
        num_classes = 100

    for seed in range(3):
        print(
            f"Training {'fractional' if args.use_fractional else 'vanilla'} model #{seed}"
        )
        train_loader, val_loader, test_loader = get_loader_fn(
            data_dir=DATA_DIR, seed=seed
        )

        model = ResNet50(num_classes=num_classes, in_channels=in_channels).to(device)

        optimizer = (
            FractionalSGD(model.parameters(), lr=0.01, alpha=1.5)
            if args.use_fractional
            else optim.SGD(model.parameters(), lr=0.01)
        )

        initial_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        max_epochs = 100
        train(model, train_loader, val_loader, optimizer, device, max_epochs=max_epochs)

        final_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        acc = test(model, test_loader, device)
        print(f"Model #{seed} test accuracy: {acc:.2f}%")

        plot_flatness(
            model,
            test_loader,
            device,
            initial_params,
            final_params,
            model_id=seed,
            flag=args.use_fractional,
            dataset=args.data,
        )

        sharpness = compute_sharpness(model, nn.CrossEntropyLoss(), test_loader, device)
        print(f"Sharpness: {sharpness:.2f}%")
