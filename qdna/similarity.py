"""Quantum sequence similarity via the swap test.

Two sequences -> k-mer frequency vectors -> amplitude-encoded states |a>, |b>.
Swap test:  P(ancilla = 0) = 1/2 + |<a|b>|^2 / 2
so  |<a|b>|^2 = 2 P(0) - 1  (= squared cosine similarity of the k-mer profiles).
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

from .encoding import kmer_frequency

_SIM = AerSimulator()


def swap_test_circuit(u: np.ndarray, v: np.ndarray) -> QuantumCircuit:
    n = int(math.log2(len(u)))
    qc = QuantumCircuit(1 + 2 * n, 1)
    a = list(range(1, 1 + n))
    b = list(range(1 + n, 1 + 2 * n))
    qc.append(StatePreparation(u / np.linalg.norm(u)), a)
    qc.append(StatePreparation(v / np.linalg.norm(v)), b)
    qc.h(0)
    for qa, qb in zip(a, b):
        qc.cswap(0, qa, qb)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def quantum_similarity(seq_a: str, seq_b: str, k: int = 2, shots: int = 8192,
                       simulator=_SIM) -> float:
    """Estimated squared cosine similarity of the two sequences' k-mer profiles."""
    u, v = kmer_frequency(seq_a, k), kmer_frequency(seq_b, k)
    qc = transpile(swap_test_circuit(u, v), simulator)
    counts = simulator.run(qc, shots=shots).result().get_counts()
    p0 = counts.get("0", 0) / shots
    return max(0.0, 2 * p0 - 1)
