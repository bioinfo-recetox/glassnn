"""Simulated expression-like data with a low-dimensional latent structure (example 08).

Cells lie on a branching trajectory in a 2-D latent space: a progenitor
segment from (0, 0) to (1, 0), then two curved branches, (1 + t, t^2) and
(1 + t, -t^2) for t in [0, 1], with a little latent noise. Each gene's mean
expression is a nonlinear function of the latent position (a random
two-layer map ending in a softplus), and the observed expression has
multiplicative (log-normal) noise. The returned matrix is log1p of the
expression, as is usual for single-cell data.

Written for GlassNN (PLAN.md, decision 37).
"""

import numpy as np

CELL_TYPES = ("progenitor", "branch A", "branch B")


def latent_trajectory(
    n_cells: int, rng: np.random.Generator, latent_noise: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """Latent positions ``(n_cells, 2)`` and cell types (0, 1, 2)."""
    cell_type = rng.integers(0, 3, size=n_cells)
    t = rng.uniform(0, 1, size=n_cells)
    sign = np.where(cell_type == 1, 1.0, -1.0)
    branch = np.stack([1 + t, sign * t**2], axis=1)
    stem = np.stack([t, np.zeros(n_cells)], axis=1)
    z = np.where((cell_type == 0)[:, None], stem, branch)
    return z + latent_noise * rng.normal(size=z.shape), cell_type


def gene_means(z: np.ndarray, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    """Mean expression ``(n_cells, n_genes)``: softplus of a random 2-layer map."""
    hidden = 16
    W1 = rng.normal(scale=2.0, size=(2, hidden))
    b1 = rng.normal(size=hidden)
    W2 = rng.normal(scale=1.5 / np.sqrt(hidden), size=(hidden, n_genes))
    b2 = rng.normal(size=n_genes)
    activation = np.tanh(z @ W1 + b1)
    return 5.0 * np.logaddexp(0.0, activation @ W2 + b2)  # 5 softplus(...)


def make_expression_data(
    n_cells: int = 1000,
    n_genes: int = 200,
    noise: float = 0.15,
    seed: int = 0,
    gene_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-expression ``X`` ``(n_cells, n_genes)``, latent ``z`` and cell types.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        noise: Standard deviation of the log-normal noise.
        seed: Seed of the cells (latent positions, types and noise).
        gene_seed: Seed of the gene map; keep it fixed so that different
            samples of cells (e.g. training and test) share the same genes.

    Returns:
        ``(X, z, cell_type)``.
    """
    rng = np.random.default_rng(seed)
    z, cell_type = latent_trajectory(n_cells, rng)
    means = gene_means(z, n_genes, np.random.default_rng(gene_seed))
    expression = means * np.exp(noise * rng.normal(size=means.shape))
    return np.log1p(expression), z, cell_type
