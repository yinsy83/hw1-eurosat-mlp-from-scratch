from __future__ import annotations

from typing import Iterable

import numpy as np

from autograd import Tensor


def cross_entropy_with_logits(logits: Tensor, targets: np.ndarray) -> Tensor:
    if logits.data.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape (batch_size, num_classes).")

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    batch_size = targets.shape[0]
    loss_value = -np.log(probs[np.arange(batch_size), targets] + 1e-12).mean()

    out = Tensor(loss_value, requires_grad=logits.requires_grad, parents=(logits,), op="cross_entropy")

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), targets] -= 1.0
        grad_logits /= batch_size
        grad_logits *= out.grad
        logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits

    out._backward = _backward
    return out


def l2_regularization(parameters: Iterable[Tensor], weight_decay: float, include_bias: bool = False) -> Tensor:
    penalty = Tensor(0.0, requires_grad=False)
    for parameter in parameters:
        if not include_bias and parameter.data.ndim == 2:
            penalty = penalty + (parameter * parameter).sum()
        elif include_bias:
            penalty = penalty + (parameter * parameter).sum()
    return penalty * (0.5 * weight_decay)
