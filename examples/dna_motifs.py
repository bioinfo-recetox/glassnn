"""Synthetic DNA sequences with a planted motif, for examples 06 and 07.

Positive sequences contain one copy of a fixed motif (optionally with point
mutations) at a random position; negative sequences are uniformly random.
Every base is drawn independently and uniformly from ``ACGT``.

Provisional (PLAN.md, decision 28): written for GlassNN, to be checked
against the generator of lecture 5 of the course.
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
