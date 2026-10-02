import subprocess
import sys

import numpy as np
import pytest

from glassnn import backend


def test_fresh_import_defaults():
    code = (
        "import numpy as np\n"
        "from glassnn import backend\n"
        "assert backend.get_backend() == 'numpy'\n"
        "assert backend.xp is np\n"
        "assert backend.get_default_dtype() == np.float32\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_set_backend_numpy_is_accepted():
    backend.set_backend("numpy")
    assert backend.get_backend() == "numpy"
    assert backend.xp is np


def test_set_backend_cupy_is_not_available_yet():
    with pytest.raises(NotImplementedError, match="M6"):
        backend.set_backend("cupy")
    assert backend.get_backend() == "numpy"


def test_set_backend_rejects_unknown_names():
    with pytest.raises(ValueError, match="'jax'"):
        backend.set_backend("jax")


@pytest.mark.parametrize("name", ["float32", "float64", np.float32, np.float64])
def test_set_default_dtype_accepts_float32_and_float64(name):
    backend.set_default_dtype(name)
    assert backend.get_default_dtype() == np.dtype(name)


@pytest.mark.parametrize("name", ["float16", "int64", "complex128"])
def test_set_default_dtype_rejects_other_dtypes(name):
    with pytest.raises(ValueError, match="float32 or float64"):
        backend.set_default_dtype(name)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_asarray_casts_floating_input_to_default_dtype(dtype):
    backend.set_default_dtype(dtype)
    for data in (1.5, [1.0, 2.0], np.ones(3, dtype=np.float16), np.ones(3)):
        assert backend.asarray(data).dtype == np.dtype(dtype)


def test_asarray_keeps_integer_and_boolean_input():
    assert backend.asarray([1, 2]).dtype.kind == "i"
    assert backend.asarray(np.array([1, 2], dtype=np.int32)).dtype == np.int32
    assert backend.asarray([True, False]).dtype == np.bool_


def test_asarray_explicit_dtype_wins():
    assert backend.asarray([1, 2], dtype="float32").dtype == np.float32
    assert backend.asarray(np.ones(2), dtype=np.float32).dtype == np.float32


@pytest.mark.parametrize("data", [["a", "b"], [1 + 2j], [object()]])
def test_asarray_rejects_non_real_input(data):
    with pytest.raises(TypeError, match="real numbers"):
        backend.asarray(data)


def test_to_numpy_returns_a_numpy_array():
    x = backend.asarray([1.0, 2.0])
    out = backend.to_numpy(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_manual_seed_makes_the_global_generator_reproducible():
    backend.manual_seed(7)
    first = backend.get_generator().random(5)
    backend.manual_seed(7)
    np.testing.assert_array_equal(backend.get_generator().random(5), first)


def test_manual_seed_returns_the_new_generator():
    generator = backend.manual_seed(3)
    assert generator is backend.get_generator()
    assert isinstance(generator, np.random.Generator)


def test_glassnn_manual_seed_is_the_backend_function():
    import glassnn

    assert glassnn.manual_seed is backend.manual_seed


def test_the_generator_is_unseeded_before_manual_seed():
    code = (
        "from glassnn import backend\n"
        "print(backend.get_generator().integers(0, 2**62))\n"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        ).stdout
        for _ in range(2)
    }
    assert len(runs) == 2


def test_erf_matches_scipy():
    from scipy import special

    x = np.linspace(-4, 4, 17)
    np.testing.assert_allclose(backend.erf(x), special.erf(x))
