"""Activation functions as modules (each calls :mod:`glassnn.functional`).

Every module applies its function elementwise (``Softmax`` and
``LogSoftmax`` along one dimension); input and output have the same shape.
The derivatives are given with the functions in :mod:`glassnn.functional`.

Book chapter: :book:`Backpropagation in a multilayer perceptron
<chapters/03-mlp.html>`.
"""

from glassnn import functional as F
from glassnn.nn.module import Module
from glassnn.tensor import Tensor


class ReLU(Module):
    r"""Rectified linear unit, :math:`z = \max(a, 0)`.

    See :func:`glassnn.functional.relu`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.ReLU()(Tensor([-1.0, 2.0]))
        Tensor([0., 2.])
    """

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.relu(input)


class LeakyReLU(Module):
    r"""Leaky ReLU, :math:`z = \max(a, 0) + s \min(a, 0)` with slope :math:`s`.

    See :func:`glassnn.functional.leaky_relu`.

    Args:
        negative_slope: The slope :math:`s` for negative inputs.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.LeakyReLU(0.1)(Tensor([-1.0, 2.0]))
        Tensor([-0.1,  2. ])
    """

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
    r"""Gaussian error linear unit :cite:p:`hendrycks2016gaussian`.

    :math:`z = a\, \Phi(a)`, with :math:`\Phi` the standard normal
    distribution function. See
    :func:`glassnn.functional.gelu`.

    Args:
        approximate: ``"none"`` for the exact form with ``erf``, ``"tanh"``
            for the tanh approximation of :math:`\Phi`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.GELU()(Tensor([0.0, 1.0]))
        Tensor([0.       , 0.8413447])
    """

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
    r"""Hyperbolic tangent, :math:`z = \tanh a`.

    See :func:`glassnn.functional.tanh`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Tanh()(Tensor([0.0]))
        Tensor([0.])
    """

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.tanh(input)


class Sigmoid(Module):
    r"""Logistic sigmoid, :math:`z = 1 / (1 + e^{-a})`.

    See :func:`glassnn.functional.sigmoid`.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Sigmoid()(Tensor([0.0]))
        Tensor([0.5])
    """

    def forward(self, input: Tensor) -> Tensor:
        """Apply the function."""
        return F.sigmoid(input)


class Softplus(Module):
    r"""Softplus, :math:`z = \frac{1}{\beta} \ln(1 + e^{\beta a})`.

    See :func:`glassnn.functional.softplus`.

    Args:
        beta: The sharpness :math:`\beta`.
        threshold: Above this value of :math:`\beta a` the output is ``a``.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Softplus()(Tensor([0.0]))
        Tensor([0.6931472])
    """

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
    r"""Softmax along ``dim``, :math:`z_j = e^{a_j} / \sum_k e^{a_k}`.

    See :func:`glassnn.functional.softmax`.

    Args:
        dim: The dimension along which the outputs sum to 1.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.Softmax(dim=1)(Tensor([[0.0, 0.0]]))
        Tensor([[0.5, 0.5]])
    """

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
    r"""Log-softmax along ``dim``, :math:`z_j = a_j - \ln \sum_k e^{a_k}`.

    See :func:`glassnn.functional.log_softmax`.

    Args:
        dim: The dimension of the classes.

    Example:
        >>> from glassnn import Tensor, nn
        >>> nn.LogSoftmax(dim=1)(Tensor([[0.0, 0.0]]))
        Tensor([[-0.6931472, -0.6931472]])
    """

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
