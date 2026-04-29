from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from autograd import Parameter, Tensor


class Module:
    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, name: str) -> None:
        # Xavier-style initialization is more stable across ReLU/Tanh for this NumPy MLP.
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(
            rng.normal(0.0, scale, size=(in_features, out_features)).astype(np.float32),
            name=f"{name}.weight",
        )
        self.bias = Parameter(np.zeros((1, out_features), dtype=np.float32), name=f"{name}.bias")

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer.")
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.activation = activation.lower()
        rng = np.random.default_rng(seed)

        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.layers = [
            Linear(dims[idx], dims[idx + 1], rng=rng, name=f"layers.{idx}") for idx in range(len(dims) - 1)
        ]

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        if self.activation == "tanh":
            return x.tanh()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self._activate(layer(x))
        return self.layers[-1](x)

    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            state[f"layer{idx}_weight"] = layer.weight.data.copy()
            state[f"layer{idx}_bias"] = layer.bias.data.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for idx, layer in enumerate(self.layers):
            layer.weight.data[...] = state[f"layer{idx}_weight"]
            layer.bias.data[...] = state[f"layer{idx}_bias"]

    def named_parameters(self) -> Iterable[tuple[str, Parameter]]:
        for idx, layer in enumerate(self.layers):
            yield f"layer{idx}_weight", layer.weight
            yield f"layer{idx}_bias", layer.bias
