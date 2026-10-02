"""``Module``, the base class of all layers and models, and ``Sequential``."""

from collections.abc import Callable, Iterator
from typing import Any

from glassnn import backend
from glassnn.nn.parameter import Parameter
from glassnn.tensor import Tensor


class Module:
    """Base class of layers and models.

    A subclass creates its parameters and sub-modules in ``__init__`` (after
    calling ``super().__init__()``) and computes its output in ``forward``.
    Calling the module, ``model(x)``, calls ``model.forward(x)``.

    Assigning a :class:`Parameter` or a ``Module`` to an attribute registers
    it (see :meth:`__setattr__`), so that :meth:`parameters` can collect all
    learnable tensors of a model, at any depth, for the optimizer.

    Attributes:
        training: ``True`` in training mode, ``False`` after :meth:`eval`.
            Layers such as dropout behave differently in the two modes.

    Example:
        >>> from glassnn import Tensor, nn
        >>> class Affine(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.scale = nn.Parameter([2.0])
        ...         self.shift = nn.Parameter([1.0])
        ...     def forward(self, x):
        ...         return self.scale * x + self.shift
        >>> model = Affine()
        >>> [name for name, _ in model.named_parameters()]
        ['scale', 'shift']
        >>> model(Tensor([3.0]))
        Tensor([7.], op='add')
    """

    _parameters: dict[str, Parameter]
    _modules: dict[str, "Module"]
    training: bool

    def __init__(self) -> None:
        """Create the registries; subclasses must call this first."""
        # object.__setattr__ bypasses our own __setattr__ below.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, registering parameters and sub-modules.

        ``self.weight = Parameter(...)`` stores the parameter under the name
        ``"weight"``; ``self.layer = Linear(...)`` stores the sub-module.
        Assigning anything else (e.g. ``None``) under a registered name
        removes the registration.

        Raises:
            AttributeError: If ``super().__init__()`` was not called first.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                f"Call super().__init__() at the start of "
                f"{type(self).__name__}.__init__, before assigning attributes."
            )
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the output; every subclass must define it.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run :meth:`forward`."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameters and sub-modules
    # ------------------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield ``(name, parameter)`` for this module and all sub-modules.

        Names are dotted paths such as ``"0.weight"``. Own parameters come
        first, then those of each sub-module in the order of assignment. A
        parameter shared by several modules is yielded once.
        """
        seen: set[int] = set()
        for name, parameter in self._named_parameters_with_duplicates(prefix):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter

    def _named_parameters_with_duplicates(
        self, prefix: str
    ) -> Iterator[tuple[str, Parameter]]:
        for name, parameter in self._parameters.items():
            yield prefix + name, parameter
        for name, module in self._modules.items():
            yield from module._named_parameters_with_duplicates(prefix + name + ".")

    def parameters(self) -> Iterator[Parameter]:
        """Yield all parameters (see :meth:`named_parameters`)."""
        for _, parameter in self.named_parameters():
            yield parameter

    def modules(self) -> Iterator["Module"]:
        """Yield this module and all sub-modules, depth first."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def apply(self, fn: Callable[["Module"], Any]) -> "Module":
        """Call ``fn`` on every sub-module, then on this module.

        Typical use: custom initialization, e.g. ``model.apply(init_weights)``.

        Returns:
            This module.
        """
        for module in self._modules.values():
            module.apply(fn)
        fn(self)
        return self

    # ------------------------------------------------------------------
    # Modes and gradients
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "Module":
        """Set training mode (``mode=True``) or evaluation mode, recursively.

        Returns:
            This module.
        """
        for module in self.modules():
            object.__setattr__(module, "training", mode)
        return self

    def eval(self) -> "Module":
        """Set evaluation mode; the same as ``train(False)``."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Reset the gradients of all parameters to ``None``.

        Gradients accumulate across :meth:`~glassnn.tensor.Tensor.backward`
        calls, so a training loop resets them before every step.
        """
        for parameter in self.parameters():
            parameter.grad = None

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Tensor]:
        """Return a copy of all parameters, keyed by their dotted names.

        Note:
            Differences from PyTorch: the values are copies, not tensors
            that share memory with the parameters.
        """
        return {
            name: Tensor(backend.xp.array(parameter.data), dtype=parameter.dtype)
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Copy values from ``state_dict`` into the parameters.

        Args:
            state_dict: Names mapped to tensors or arrays, as returned by
                :meth:`state_dict`.
            strict: If ``True``, the names must match exactly.

        Raises:
            KeyError: If ``strict`` and names are missing or unexpected (the
                message lists them).
            ValueError: If a value has the wrong shape (the message names the
                parameter and both shapes).
        """
        own = dict(self.named_parameters())
        missing = [name for name in own if name not in state_dict]
        unexpected = [name for name in state_dict if name not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"missing keys: {missing}; unexpected keys: {unexpected}")
        for name, parameter in own.items():
            if name not in state_dict:
                continue
            value = state_dict[name]
            data = value.data if isinstance(value, Tensor) else value
            new = backend.xp.array(data, dtype=parameter.dtype)
            if tuple(new.shape) != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for {name!r}: the parameter has shape "
                    f"{parameter.shape}, the state_dict has {tuple(new.shape)}."
                )
            parameter.data = new

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """The settings shown between the parentheses of ``repr``."""
        return ""

    def __repr__(self) -> str:
        """Show the module and its sub-modules, one per line."""
        name = type(self).__name__
        if not self._modules:
            return f"{name}({self.extra_repr()})"
        lines = [f"{name}("]
        for child_name, child in self._modules.items():
            child_repr = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({child_name}): {child_repr}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """A chain of modules, applied one after the other.

    Args:
        *modules: The modules, registered under the names ``"0"``, ``"1"``...

    Example:
        >>> from glassnn import nn
        >>> model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
        >>> len(model), type(model[1]).__name__
        (3, 'ReLU')
    """

    def __init__(self, *modules: Module) -> None:
        """Register the modules in order."""
        super().__init__()
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def forward(self, input: Any) -> Any:
        """Apply the modules in order."""
        for module in self._modules.values():
            input = module(input)
        return input

    def __len__(self) -> int:
        """The number of modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over the modules."""
        return iter(self._modules.values())

    def __getitem__(self, index: int | slice) -> Module:
        """Return one module, or a ``Sequential`` of a slice of them."""
        modules = list(self._modules.values())
        if isinstance(index, slice):
            return Sequential(*modules[index])
        return modules[index]
