"""The API reference is complete: every public symbol follows the docstring
template of PLAN.md, section 6.1, as far as it applies to its kind.

- Every public module links to its book chapter (``:book:`` role).
- Every public function and class has a summary, ``Args:`` (if it takes
  arguments), ``Returns:`` (functions that return something) and a doctested
  example.
- Operations, layers, losses, initializers, optimizers and schedulers state
  their mathematics.
- Methods from the literature cite it.

Exemptions are listed by name, each with its reason (PLAN, decision 46).
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import glassnn


def _public_modules():
    names = [glassnn.__name__]
    names += [m.name for m in pkgutil.walk_packages(glassnn.__path__, "glassnn.")]
    return [importlib.import_module(name) for name in names]


def _public_symbols():
    """Public functions and classes, under the name of their defining module."""
    symbols = {}
    for module in _public_modules():
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if obj.__module__ == module.__name__:
                symbols[f"{module.__name__}.{name}"] = obj
    return symbols


MODULES = {module.__name__: module for module in _public_modules()}
SYMBOLS = _public_symbols()

# Modules whose public symbols compute something with a formula.
MATH_MODULES = (
    "glassnn.functional",
    "glassnn.nn.activation",
    "glassnn.nn.attention",
    "glassnn.nn.conv",
    "glassnn.nn.dropout",
    "glassnn.nn.embedding",
    "glassnn.nn.init",
    "glassnn.nn.linear",
    "glassnn.nn.loss",
    "glassnn.nn.normalization",
    "glassnn.nn.pooling",
    "glassnn.nn.transformer",
    "glassnn.optim.",
)

NO_EXAMPLE = {
    "glassnn.gradcheck.GradcheckError": "an exception; gradcheck shows it",
    "glassnn.analysis.LayerStats": "a result type; activation_stats shows it",
    "glassnn.optim.optimizer.Optimizer": "abstract base class",
    "glassnn.optim.lr_scheduler.LRScheduler": "abstract base class",
}

NO_MATH = {
    "glassnn.nn.pooling.Flatten": "a reshape, no formula",
    "glassnn.nn.init.zeros_": "fills with a constant",
    "glassnn.nn.init.ones_": "fills with a constant",
    "glassnn.optim.optimizer.Optimizer": "abstract base class",
    "glassnn.optim.lr_scheduler.LRScheduler": "abstract base class",
}

# Methods from the literature: their docstrings must cite it.
NEEDS_REFERENCE = [
    "glassnn.tensor.Tensor",
    "glassnn.gradcheck.gradcheck",
    "glassnn.backend.set_backend",
    "glassnn.backend.from_dlpack",
    "glassnn.backend.to_dlpack",
    "glassnn.functional.gelu",
    "glassnn.functional.dropout",
    "glassnn.functional.batch_norm",
    "glassnn.functional.layer_norm",
    "glassnn.functional.conv1d",
    "glassnn.functional.conv2d",
    "glassnn.functional.cross_entropy",
    "glassnn.functional.scaled_dot_product_attention",
    "glassnn.functional.log_softmax",
    "glassnn.functional.logsumexp",
    "glassnn.nn.activation.GELU",
    "glassnn.nn.dropout.Dropout",
    "glassnn.nn.normalization.BatchNorm1d",
    "glassnn.nn.normalization.LayerNorm",
    "glassnn.nn.conv.Conv1d",
    "glassnn.nn.conv.Conv2d",
    "glassnn.nn.linear.Linear",
    "glassnn.nn.attention.MultiheadAttention",
    "glassnn.nn.embedding.PositionalEncoding",
    "glassnn.nn.transformer.TransformerEncoderLayer",
    "glassnn.nn.transformer.TransformerEncoder",
    "glassnn.nn.init.xavier_uniform_",
    "glassnn.nn.init.xavier_normal_",
    "glassnn.nn.init.kaiming_uniform_",
    "glassnn.nn.init.kaiming_normal_",
    "glassnn.nn.loss.CrossEntropyLoss",
    "glassnn.optim.sgd.SGD",
    "glassnn.optim.adam.Adam",
    "glassnn.optim.adam.AdamW",
    "glassnn.optim.rmsprop.RMSprop",
    "glassnn.optim.lr_scheduler.CosineAnnealingLR",
    "glassnn.optim.lr_scheduler.LinearWarmup",
    "glassnn.optim.clip_grad.clip_grad_norm_",
    "glassnn.analysis.empirical_ntk",
]


def _doc(name):
    return inspect.getdoc(SYMBOLS[name]) or ""


def _parameters(obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        parameters = inspect.signature(target).parameters
    except ValueError:  # classes without a Python __init__
        return []
    return [p for p in parameters if p not in ("self", "args", "kwargs")]


def _returns_something(obj):
    if inspect.isclass(obj):
        return False
    annotation = inspect.signature(obj).return_annotation
    return annotation not in (None, "None", inspect.Signature.empty)


def test_the_symbol_list_is_not_empty():
    assert len(SYMBOLS) > 90


@pytest.mark.parametrize("name", sorted(MODULES))
def test_module_links_to_the_book(name):
    assert ":book:`" in (MODULES[name].__doc__ or ""), name


@pytest.mark.parametrize("name", sorted(SYMBOLS))
def test_summary(name):
    assert _doc(name).strip(), f"{name} has no docstring"


@pytest.mark.parametrize("name", sorted(SYMBOLS))
def test_args_section(name):
    if _parameters(SYMBOLS[name]):
        assert "Args:" in _doc(name), f"{name} documents no arguments"


@pytest.mark.parametrize("name", sorted(SYMBOLS))
def test_returns_section(name):
    if _returns_something(SYMBOLS[name]):
        assert "Returns:" in _doc(name), f"{name} documents no return value"


@pytest.mark.parametrize("name", sorted(set(SYMBOLS) - set(NO_EXAMPLE)))
def test_example(name):
    assert ">>>" in _doc(name), f"{name} has no doctested example"


MATH_SYMBOLS = sorted(
    name for name in SYMBOLS if name.startswith(MATH_MODULES) and name not in NO_MATH
)


@pytest.mark.parametrize("name", MATH_SYMBOLS)
def test_math(name):
    doc = _doc(name)
    assert ".. math::" in doc or ":math:" in doc, f"{name} states no formula"


@pytest.mark.parametrize("name", NEEDS_REFERENCE)
def test_reference(name):
    assert ":cite:" in _doc(name), f"{name} cites no reference"


def test_exemptions_name_existing_symbols():
    listed = set(NO_EXAMPLE) | set(NO_MATH) | set(NEEDS_REFERENCE)
    assert listed <= set(SYMBOLS), sorted(listed - set(SYMBOLS))


DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize("name", sorted(MODULES))
def test_module_has_an_api_page(name):
    assert (DOCS / f"{name}.rst").is_file(), f"docs/{name}.rst is missing"
