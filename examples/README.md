# Examples

Reference notebooks for lectures 10 and 11 of *Advanced Machine Learning*. The book
chapters use the same experiments; `tests/test_examples.py` asserts each
expected finding at a smaller scale.

| Notebook | Topic | Book chapter |
|---|---|---|
| `01_autodiff` | forward and reverse mode; backprop of a two-layer MLP by hand | 2, 3 |
| `02_init_and_depth` | activation statistics versus depth for several initializations | 4 |
| `03_double_descent` | random ReLU features with the minimum-norm readout | 7 |
| `04_ntk_lazy_training` | empirical NTK at initialization and during training, versus width | 8 |
| `05_memorizing_noise` | random labels, label corruption, weight decay, early stopping | 6, 7 |
| `06_cnn_dna_motif` | a 1-D CNN finds a planted motif; comparison with the spectrum kernel | 9 |

`dna_motifs.py` generates the synthetic DNA data of notebooks 06 and 07
(provisional, see `PLAN.md`, decision 28).

The notebooks are stored without outputs. Run them in Jupyter
(`uv sync --extra examples`, then `uv run jupyter lab` if Jupyter is
installed), or execute all of them as CI does:

```
uv run python examples/_execute.py
```
