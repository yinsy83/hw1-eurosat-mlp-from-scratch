from __future__ import annotations

from typing import Iterable, List

import numpy as np

from autograd import Parameter


class SGD:
    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        self.parameters: List[Parameter] = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data -= self.lr * parameter.grad.astype(np.float32)
