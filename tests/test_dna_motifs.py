"""Tests of the motif generator in examples/dna_motifs.py."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "examples" / "dna_motifs.py"
_spec = importlib.util.spec_from_file_location("dna_motifs", _PATH)
dna_motifs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dna_motifs)


def test_positives_contain_the_motif_at_the_reported_position():
    sequences, labels, positions = dna_motifs.make_motif_dataset(200, seed=3)
    assert len(sequences) == 200 and all(len(s) == 60 for s in sequences)
    assert 0.35 < labels.mean() < 0.65
    for sequence, label, position in zip(sequences, labels, positions, strict=True):
        if label == 1:
            assert sequence[position : position + 8] == "TGACGTCA"
        else:
            assert position == -1


def test_mutation_rate_changes_the_expected_fraction_of_bases():
    rng = np.random.default_rng(0)
    copies = [dna_motifs.mutate("ACGTACGTAC", 0.2, rng) for _ in range(2000)]
    changed = np.mean(
        [a != b for c in copies for a, b in zip(c, "ACGTACGTAC", strict=True)]
    )
    assert abs(changed - 0.2) < 0.01
    assert dna_motifs.mutate("ACGT", 0.0, rng) == "ACGT"


def test_generator_is_reproducible():
    first = dna_motifs.make_motif_dataset(20, seed=5)
    second = dna_motifs.make_motif_dataset(20, seed=5)
    assert first[0] == second[0]


def test_one_hot_and_kmer_counts():
    encoded = dna_motifs.one_hot(["ACGT", "TTAA"])
    assert encoded.shape == (2, 4, 4)
    np.testing.assert_array_equal(encoded[0], np.eye(4))
    np.testing.assert_array_equal(encoded.sum(axis=1), 1.0)
    counts = dna_motifs.kmer_counts(["AAAC"], k=2)
    assert counts.shape == (1, 16)
    assert counts[0, 0] == 2 and counts[0, 1] == 1  # AA twice, AC once
    assert dna_motifs.kmer_counts(["ACGTACGT"], 3).sum() == 6


def test_mismatch_counts():
    sequences = ["ACGTTGCA", "AAAAAAAA"]
    np.testing.assert_array_equal(
        dna_motifs.mismatch_counts(sequences, 3, 0),
        dna_motifs.kmer_counts(sequences, 3),
    )
    features = dna_motifs.mismatch_counts(["AC"], k=2, m=1)
    # AC itself and the 3 + 3 2-mers that differ from it in one position.
    assert features.shape == (1, 16) and features.sum() == 7
    assert features[0, 1] == 1  # AC
    assert features[0, 0] == 1 and features[0, 5] == 1  # AA, CC
    assert features[0, 15] == 0  # TT differs in both positions


def test_kmer_tokens():
    tokens = dna_motifs.kmer_tokens(["ACGTA", "TTTTT"], k=2)
    assert tokens.shape == (2, 4) and tokens.dtype.kind == "i"
    np.testing.assert_array_equal(tokens[0], [1, 6, 11, 12])  # AC, CG, GT, TA
    np.testing.assert_array_equal(tokens[1], [15] * 4)
    counts = dna_motifs.kmer_counts(["ACGTA"], 2)[0]
    np.testing.assert_array_equal(np.bincount(tokens[0], minlength=16), counts)
