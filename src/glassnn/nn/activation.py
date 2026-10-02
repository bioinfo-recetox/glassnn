"""Activation functions as modules (each calls :mod:`glassnn.functional`)."""

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class ReLU(Module):
    """Rectified linear unit; see :func:`glassnn.functional.relu`."""

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.relu(input)


class LeakyReLU(Module):
    """Leaky ReLU; see :func:`glassnn.functional.leaky_relu`."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Store the slope for negative inputs."""
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.leaky_relu(input, self.negative_slope)

    def extra_repr(self) -> str:
        """Show the slope."""
        return f"negative_slope={self.negative_slope}"


class GELU(Module):
    """Gaussian error linear unit; see :func:`glassnn.functional.gelu`."""

    def __init__(self, approximate: str = "none") -> None:
        """Choose the exact form (``"none"``) or the tanh approximation."""
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        """Show the approximation."""
        return f"approximate={self.approximate!r}"


class Tanh(Module):
    """Hyperbolic tangent; see :func:`glassnn.functional.tanh`."""

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.tanh(input)


class Sigmoid(Module):
    """Logistic sigmoid; see :func:`glassnn.functional.sigmoid`."""

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.sigmoid(input)


class Softplus(Module):
    """Softplus; see :func:`glassnn.functional.softplus`."""

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        """Store the sharpness and the linear threshold."""
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        """Show the settings."""
        return f"beta={self.beta}, threshold={self.threshold}"


class Softmax(Module):
    """Softmax along ``dim``; see :func:`glassnn.functional.softmax`."""

    def __init__(self, dim: int) -> None:
        """Store the dimension along which the outputs sum to 1."""
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        """Show the dimension."""
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """Log-softmax along ``dim``; see :func:`glassnn.functional.log_softmax`."""

    def __init__(self, dim: int) -> None:
        """Store the dimension of the classes."""
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.log_softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        """Show the dimension."""
        return f"dim={self.dim}"
