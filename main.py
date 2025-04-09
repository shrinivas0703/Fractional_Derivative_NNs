import torch
from torch import optim
from NeuralNetwork import NeuralNetwork
from data_utils import get_fashion_mnist_loaders
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from FracGrad import FractionalSGD
import argparse

DATA_DIR = 'data/'
PLOTS_DIR = 'plots/'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def train(model, train_loader, optimizer, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.compute_loss(outputs, labels)
        model.backward_step(loss)
        optimizer.step()

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
            validation_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    if not quiet:
        print("\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            validation_loss,
            correct,
            len(validation_loader.dataset),
            100.0 * correct / len(validation_loader.dataset),
        ))

    return 100.0 * correct / len(validation_loader.dataset)

def plot_flatness(model, test_loader, device, initial_params, final_params, model_id, flag):
    v_err = []
    alpha_v = []
    model.to(device)
    for alpha in torch.linspace(0, 1.5, steps=10):
        alpha = alpha.to(device)
        for name, param in model.named_parameters():
            param.data = (1.0 - alpha) * initial_params[name].data + alpha * final_params[name].data
        v_err.append(100.0 - validation(model, device, test_loader, quiet=True))
        alpha_v.append(alpha.to("cpu"))
        print(f" alpha = {alpha:.2f} has validation error {v_err[-1]:.2f}%")

    plt.plot(alpha_v, v_err)
    plt.xlabel("Interpolation coefficient (alpha)")
    plt.ylabel("Validation error (%)")
    plt.title(f"Loss Flatness for Model #{model_id}")
    plt.grid(True)
    if flag:
        plt.savefig(os.path.join(PLOTS_DIR, f"flatness_plot_fractional_model_{model_id}.png"))
    else:
        plt.savefig(os.path.join(PLOTS_DIR, f"flatness_plot_model_{model_id}.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-fractional', action='store_true', help='Use FractionalSGD instead of standard SGD')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in range(3):
        if args.use_fractional:
            print(f"Training fractional model #{seed}")
        else:
            print(f"Training vanilla model #{seed}")
        

        train_loader, test_loader = get_fashion_mnist_loaders(data_dir=DATA_DIR, seed=seed)
        model = NeuralNetwork().to(device)
        flag = False
        if args.use_fractional:
            optimizer = FractionalSGD(model.parameters(), lr=0.01)
            flag = True
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Save initial parameters
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        train(model, train_loader, optimizer, device)

        # Save final parameters
        final_params = {name: param.clone().detach() for name, param in model.named_parameters()}

        acc = test(model, test_loader, device)
        print(f"Model #{seed} test accuracy: {acc:.2f}%")

        # Plot flatness
        plot_flatness(model, test_loader, device, initial_params, final_params, model_id=seed, flag=flag)
