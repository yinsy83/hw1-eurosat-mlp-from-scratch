from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple, Union

import numpy as np


ArrayLike = Union[float, int, np.ndarray, Sequence[float]]


def _ensure_array(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float32, copy=False)
    return np.array(data, dtype=np.float32)


def _sum_to_shape(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if grad.shape == shape:
        return grad

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        parents: Tuple["Tensor", ...] = (),
        op: str = "",
        name: Optional[str] = None,
    ) -> None:
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.parents = parents
        self.op = op
        self.name = name
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self.op})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradient must be provided for non-scalar tensors.")
            grad = np.ones_like(self.data, dtype=np.float32)
        else:
            grad = _ensure_array(grad)

        topo = []
        visited: Set[int] = set()

        def build(node: "Tensor") -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = grad

        for node in reversed(topo):
            node._backward()

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = _sum_to_shape(out.grad, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = _sum_to_shape(out.grad, other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -Tensor(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) - self

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, parents=(self,), op="neg")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = -out.grad
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = _sum_to_shape(out.grad * other.data, self.data.shape)
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = _sum_to_shape(out.grad * self.data, other.data.shape)
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) / self

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self.grad = grad_self if self.grad is None else self.grad + grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other.grad = grad_other if other.grad is None else other.grad + grad_other

        out._backward = _backward
        return out

    def pow(self, exponent: float) -> "Tensor":
        out = Tensor(
            np.power(self.data, exponent),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="pow",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * exponent * np.power(self.data, exponent - 1.0)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad
            if axis is None:
                grad_self = np.broadcast_to(grad_self, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                if not keepdims:
                    for ax in sorted((ax if ax >= 0 else ax + self.data.ndim) for ax in axes):
                        grad_self = np.expand_dims(grad_self, axis=ax)
                grad_self = np.broadcast_to(grad_self, self.data.shape)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = float(self.data.size)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = 1.0
            for ax in axes:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad.reshape(self.data.shape)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if axes is None:
                grad_self = out.grad.transpose()
            else:
                inverse_axes = np.argsort(axes)
                grad_self = out.grad.transpose(tuple(inverse_axes))
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * (self.data > 0.0)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sigma = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sigma, requires_grad=self.requires_grad, parents=(self,), op="sigmoid")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * sigma * (1.0 - sigma)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_x = np.tanh(self.data)
        out = Tensor(tanh_x, requires_grad=self.requires_grad, parents=(self,), op="tanh")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * (1.0 - tanh_x**2)
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        exp_x = np.exp(self.data)
        out = Tensor(exp_x, requires_grad=self.requires_grad, parents=(self,), op="exp")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad * exp_x
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out

    def log(self, eps: float = 1e-12) -> "Tensor":
        safe = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe), requires_grad=self.requires_grad, parents=(self,), op="log")

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad_self = out.grad / safe
            self.grad = grad_self if self.grad is None else self.grad + grad_self

        out._backward = _backward
        return out


class Parameter(Tensor):
    def __init__(self, data: ArrayLike, name: Optional[str] = None) -> None:
        super().__init__(data=data, requires_grad=True, name=name)


def tensor(data: ArrayLike, requires_grad: bool = False, name: Optional[str] = None) -> Tensor:
    return Tensor(data=data, requires_grad=requires_grad, name=name)
