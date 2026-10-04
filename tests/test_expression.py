"""Tests of the expression simulator in examples/expression.py."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "examples" / "expression.py"
_spec = importlib.util.spec_from_file_location("expression", _PATH)
expression = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(expression)


def test_shapes_types_and_latent_trajectory():
    X, z, cell_type = expression.make_expression_data(300, n_genes=50, seed=1)
    assert X.shape == (300, 50) and z.shape == (300, 2)
    assert set(np.unique(cell_type)) == {0, 1, 2}
    assert (X >= 0).all()  # log1p of positive expression
    # Progenitors lie on the segment from (0, 0) to (1, 0); branch A curves
    # up and branch B down.
    assert np.abs(z[cell_type == 0, 1]).max() < 0.2
    assert z[cell_type == 1, 1].mean() > 0.2 > -0.2 > z[cell_type == 2, 1].mean()


def test_samples_with_different_seeds_share_the_genes():
    first, _, _ = expression.make_expression_data(2000, seed=0)
    second, _, _ = expression.make_expression_data(2000, seed=1)
    other_genes, _, _ = expression.make_expression_data(2000, seed=1, gene_seed=5)
    same = np.abs(first.mean(axis=0) - second.mean(axis=0)).max()
    different = np.abs(first.mean(axis=0) - other_genes.mean(axis=0)).max()
    assert same < 0.1 < different


def test_reproducible():
    a = expression.make_expression_data(10, seed=3)
    b = expression.make_expression_data(10, seed=3)
    np.testing.assert_array_equal(a[0], b[0])
