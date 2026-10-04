"""``Module``, the base class of all layers and models, and ``Sequential``."""

from collections.abc import Callable, Iterable, Iterator
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
    Tensors that are part of the state of a model but are not learned, such
    as the running statistics of batch normalization, are registered with
    :meth:`register_buffer`.

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
    _buffers: dict[str, Tensor | None]
    _modules: dict[str, "Module"]
    training: bool

    def __init__(self) -> None:
        """Create the registries; subclasses must call this first."""
        # object.__setattr__ bypasses our own __setattr__ below.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, registering parameters and sub-modules.

        ``self.weight = Parameter(...)`` stores the parameter under the name
        ``"weight"``; ``self.layer = Linear(...)`` stores the sub-module.
        Assigning anything else (e.g. ``None``) under a registered name
        removes the registration. Assigning a ``Tensor`` (or ``None``) to the
        name of a buffer replaces the buffer.

        Raises:
            AttributeError: If ``super().__init__()`` was not called first.
            TypeError: If a buffer is replaced by something else than a
                ``Tensor`` or ``None``.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                f"Call super().__init__() at the start of "
                f"{type(self).__name__}.__init__, before assigning attributes."
            )
        if name in self._buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(
                    f"Cannot assign a {type(value).__name__} to the buffer "
                    f"{name!r}; assign a Tensor or None."
                )
            self._buffers[name] = value
            object.__setattr__(self, name, value)
            return
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """Register a tensor that is part of the state but is not learned.

        A buffer is saved by :meth:`state_dict` and restored by
        :meth:`load_state_dict`, but :meth:`parameters` does not yield it, so
        an optimizer never changes it. ``None`` registers the name without a
        value (it is then skipped by :meth:`state_dict`).

        Args:
            name: The attribute name, e.g. ``"running_mean"``.
            tensor: The value, or ``None``.

        Raises:
            KeyError: If ``name`` is already an attribute.
            TypeError: If ``tensor`` is neither a ``Tensor`` nor ``None``.

        Note:
            Differences from PyTorch: there is no ``persistent`` argument;
            every buffer is saved.

        Example:
            >>> from glassnn import Tensor, nn
            >>> module = nn.Module()
            >>> module.register_buffer("count", Tensor(0.0))
            >>> [name for name, _ in module.named_buffers()]
            ['count']
        """
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(
                f"Cannot register the buffer {name!r}: the attribute exists."
            )
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                f"A buffer must be a Tensor or None, got {type(tensor).__name__}."
            )
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

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

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Yield ``(name, buffer)`` for this module and all sub-modules.

        Names and order are as in :meth:`named_parameters`; buffers that are
        ``None`` are skipped.
        """
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield prefix + name, buffer
        for name, module in self._modules.items():
            yield from module.named_buffers(prefix + name + ".")

    def buffers(self) -> Iterator[Tensor]:
        """Yield all buffers (see :meth:`named_buffers`)."""
        for _, buffer in self.named_buffers():
            yield buffer

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
        """Return a copy of all parameters and buffers, keyed by dotted names.

        Parameters come first, then buffers.

        Note:
            Differences from PyTorch: the values are copies, not tensors
            that share memory with the parameters; PyTorch lists the
            parameters and buffers of each module together.
        """
        return {
            name: Tensor(backend.xp.array(tensor.data), dtype=tensor.dtype)
            for name, tensor in self._named_state()
        }

    def _named_state(self) -> Iterator[tuple[str, Tensor]]:
        yield from self.named_parameters()
        yield from self.named_buffers()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Copy values from ``state_dict`` into the parameters and buffers.

        Args:
            state_dict: Names mapped to tensors or arrays, as returned by
                :meth:`state_dict`.
            strict: If ``True``, the names must match exactly.

        Raises:
            KeyError: If ``strict`` and names are missing or unexpected (the
                message lists them).
            ValueError: If a value has the wrong shape (the message names the
                parameter or buffer and both shapes).
        """
        own = dict(self._named_state())
        missing = [name for name in own if name not in state_dict]
        unexpected = [name for name in state_dict if name not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"missing keys: {missing}; unexpected keys: {unexpected}")
        for name, tensor in own.items():
            if name not in state_dict:
                continue
            value = state_dict[name]
            data = value.data if isinstance(value, Tensor) else value
            new = backend.xp.array(data, dtype=tensor.dtype)
            if tuple(new.shape) != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name!r}: the model has shape "
                    f"{tensor.shape}, the state_dict has {tuple(new.shape)}."
                )
            tensor.data = new

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


class ModuleList(Module):
    """A list of modules, registered so that their parameters are found.

    Unlike :class:`Sequential`, a ``ModuleList`` has no ``forward``: the
    model that owns it decides how to use the modules, e.g. in a loop.

    Args:
        modules: The initial modules (registered as ``"0"``, ``"1"``...).

    Example:
        >>> from glassnn import nn
        >>> layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        >>> len(layers), len(list(layers.parameters()))
        (3, 6)
    """

    def __init__(self, modules: Iterable[Module] = ()) -> None:
        """Register the modules in order."""
        super().__init__()
        self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        """Add one module at the end."""
        setattr(self, str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        """Add several modules at the end."""
        for module in modules:
            self.append(module)
        return self

    def __len__(self) -> int:
        """The number of modules."""
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over the modules."""
        return iter(self._modules.values())

    def __getitem__(self, index: int) -> Module:
        """Return one module."""
        return list(self._modules.values())[index]
