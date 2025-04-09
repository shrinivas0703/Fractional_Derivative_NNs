import torch
from torch.optim import Optimizer


class FractionalSGD(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, h=0.8, max_history=10):
        """
        Implements fractional-order SGD using Grünwald-Letnikov derivative.

        Args:
            params (iterable): Model parameters to optimize.
            lr (float): Learning rate.
            alpha (float): Fractional order (> 0).
            h (float): Step size.
            max_history (int): Number of past gradients to use.
        """
        defaults = dict(lr=lr, alpha=alpha, h=h, max_history=max_history)
        super(FractionalSGD, self).__init__(params, defaults)

    def _gl_coeffs(self, alpha, N, device):
        """Compute Grünwald-Letnikov coefficients."""
        coeffs = [1.0]
        for k in range(1, N + 1):
            coeffs.append(coeffs[-1] * (alpha - k + 1) / k)
        return torch.tensor(coeffs, device=device)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            h = group["h"]
            max_history = group["max_history"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Initialize per-parameter history
                state = self.state[p]
                if "grad_history" not in state:
                    state["grad_history"] = []

                hist = state["grad_history"]
                hist.append(grad.clone())

                if len(hist) > max_history:
                    hist.pop(0)

                # Not enough history? Fall back to vanilla SGD
                if len(hist) < 2:
                    p.data -= lr * grad
                    continue

                grads = torch.stack(hist)  # shape: (T, ...)
                coeffs = self._gl_coeffs(alpha, len(grads) - 1, grad.device)

                # Reshape coeffs for broadcasting
                for _ in range(grad.dim()):
                    coeffs = coeffs.unsqueeze(-1)

                # Fractional gradient
                d_alpha = torch.sum(coeffs * grads.flip(0), dim=0) / (h**alpha)
                p.data -= lr * d_alpha

        return loss
