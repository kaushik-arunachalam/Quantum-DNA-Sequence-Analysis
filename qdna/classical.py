"""Classical baselines to compare the quantum results against."""
from __future__ import annotations

import numpy as np

from .encoding import kmer_frequency


def find_mutations(reference: str, sample: str) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(reference, sample)) if a != b]


def find_motif(sequence: str, motif: str) -> list[int]:
    k = len(motif)
    return [i for i in range(len(sequence) - k + 1) if sequence[i:i + k] == motif]


def cosine_similarity_sq(seq_a: str, seq_b: str, k: int = 2) -> float:
    u, v = kmer_frequency(seq_a, k), kmer_frequency(seq_b, k)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))) ** 2
