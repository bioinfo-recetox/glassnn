"""Synthetic DNA sequences with a planted motif, for examples 06 and 07.

Positive sequences contain one copy of a fixed motif (optionally with point
mutations) at a random position; negative sequences are uniformly random.
Every base is drawn independently and uniformly from ``ACGT``.

``kmer_counts`` and ``mismatch_counts`` give the feature vectors of the
spectrum kernel (Leslie, Eskin and Noble 2002) and of the mismatch kernel
(Leslie et al. 2004): a linear kernel on these features is the string kernel.

Written for GlassNN (PLAN.md, decision 28).
"""

import numpy as np

ALPHABET = "ACGT"


def random_sequence(length: int, rng: np.random.Generator) -> str:
    """A uniformly random DNA sequence."""
    return "".join(rng.choice(list(ALPHABET), size=length))


def mutate(motif: str, rate: float, rng: np.random.Generator) -> str:
    """Replace each base, with probability ``rate``, by one of the other three."""
    bases = []
    for base in motif:
        if rng.uniform() < rate:
            base = rng.choice([b for b in ALPHABET if b != base])
        bases.append(base)
    return "".join(bases)


def make_motif_dataset(
    n: int,
    length: int = 60,
    motif: str = "TGACGTCA",
    positive_fraction: float = 0.5,
    mutation_rate: float = 0.0,
    seed: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sequences, labels (1 = contains the motif) and motif positions.

    Args:
        n: Number of sequences.
        length: Length of every sequence.
        motif: The planted motif (default: the palindromic CRE site).
        positive_fraction: Expected fraction of positive sequences.
        mutation_rate: Probability of a point mutation per motif base, drawn
            anew for every positive sequence.
        seed: Seed of the random generator.

    Returns:
        ``(sequences, labels, positions)``; ``positions`` is -1 for
        negative sequences.
    """
    rng = np.random.default_rng(seed)
    sequences, labels, positions = [], [], []
    for _ in range(n):
        sequence = random_sequence(length, rng)
        position = -1
        if rng.uniform() < positive_fraction:
            position = int(rng.integers(0, length - len(motif) + 1))
            copy = mutate(motif, mutation_rate, rng)
            sequence = sequence[:position] + copy + sequence[position + len(copy) :]
        sequences.append(sequence)
        labels.append(int(position >= 0))
        positions.append(position)
    return sequences, np.array(labels), np.array(positions)


def one_hot(sequences: list[str]) -> np.ndarray:
    """One-hot encoding of shape ``(N, 4, L)``: channel ``c`` is base ``ACGT[c]``."""
    index = np.array([[ALPHABET.index(b) for b in s] for s in sequences])
    return (index[:, None, :] == np.arange(4)[None, :, None]).astype(np.float64)


def kmer_counts(sequences: list[str], k: int) -> np.ndarray:
    """Counts of all 4**k k-mers in each sequence, shape ``(N, 4**k)``.

    The linear kernel on these counts is the spectrum kernel.
    """
    counts = np.zeros((len(sequences), 4**k))
    for row, sequence in enumerate(sequences):
        for start in range(len(sequence) - k + 1):
            code = 0
            for base in sequence[start : start + k]:
                code = 4 * code + ALPHABET.index(base)
            counts[row, code] += 1
    return counts


def mismatch_counts(sequences: list[str], k: int, m: int) -> np.ndarray:
    """Features of the (k, m)-mismatch kernel, shape ``(N, 4**k)``.

    Feature ``b`` counts the k-mers of a sequence that differ from the k-mer
    ``b`` in at most ``m`` positions; for ``m = 0`` these are the k-mer
    counts. Computed as the k-mer counts times the 0/1 matrix of all pairs
    of k-mers within Hamming distance ``m`` (fine for ``k <= 6``).
    """
    codes = np.arange(4**k)
    # digits[a, j] is the j-th base (0..3) of the k-mer with code a.
    digits = (codes[:, None] // 4 ** np.arange(k - 1, -1, -1)) % 4
    hamming = (digits[:, None, :] != digits[None, :, :]).sum(axis=2)
    return kmer_counts(sequences, k) @ (hamming <= m).astype(np.float64)


def kmer_tokens(sequences: list[str], k: int) -> np.ndarray:
    """Overlapping k-mer token ids, shape ``(N, L - k + 1)``, values in ``[0, 4**k)``.

    Token ``i`` is the code of the k-mer starting at position ``i`` (base
    ``ACGT[c]`` has digit ``c``, most significant first), as in
    :func:`kmer_counts`.
    """
    index = np.array([[ALPHABET.index(b) for b in s] for s in sequences])
    length = index.shape[1] - k + 1
    codes = np.zeros((len(sequences), length), dtype=np.int64)
    for offset in range(k):
        codes = 4 * codes + index[:, offset : offset + length]
    return codes
