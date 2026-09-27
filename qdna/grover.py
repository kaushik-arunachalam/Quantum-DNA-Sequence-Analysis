"""Grover search driver.

Two modes:
  * run_distribution(): fixed iteration count, returns the measured histogram
    (shows amplitude amplification).
  * find_all(): BBHT randomized Grover (unknown number of solutions). Each
    candidate is checked classically in O(1), then excluded from the oracle
    so the next round finds a new one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

_SIM = AerSimulator()


def diffuser(qc: QuantumCircuit, qubits) -> None:
    """Inversion about the mean on the index register."""
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def grover_circuit(oracle: QuantumCircuit, n_index: int, iterations: int,
                   boxed: bool = False) -> QuantumCircuit:
    """boxed=True keeps the oracle/diffuser as labelled blocks (for drawing)."""
    total = oracle.num_qubits
    idx = list(range(n_index))
    anc = total - 1
    qc = QuantumCircuit(total, n_index)
    qc.h(idx)
    qc.x(anc)
    qc.h(anc)  # |-> for phase kickback
    if boxed:
        diff = QuantumCircuit(n_index, name="Diffuser")
        diffuser(diff, list(range(n_index)))
        oracle_gate, diff_gate = oracle.to_gate(label="Oracle"), diff.to_gate(label="Diffuser")
    for _ in range(iterations):
        if boxed:
            qc.append(oracle_gate, range(total))
            qc.append(diff_gate, idx)
        else:
            qc.compose(oracle, inplace=True)
            diffuser(qc, idx)
    qc.measure(idx, range(n_index))
    return qc


def optimal_iterations(n_index: int, n_solutions: int) -> int:
    if n_solutions <= 0:
        return 0
    return max(0, math.floor(math.pi / 4 * math.sqrt(2**n_index / n_solutions)))


def run_distribution(oracle, n_index, iterations, shots=2048, simulator=_SIM) -> dict[int, int]:
    qc = transpile(grover_circuit(oracle, n_index, iterations), simulator)
    counts = simulator.run(qc, shots=shots).result().get_counts()
    return {int(k, 2): v for k, v in counts.items()}


@dataclass
class SearchStats:
    grover_iterations: int = 0  # oracle calls on the quantum side
    circuits_run: int = 0
    classical_checks: int = 0
    found: list[int] = field(default_factory=list)


def bbht_search(
    oracle: QuantumCircuit,
    n_index: int,
    verify: Callable[[int], bool],
    rng: np.random.Generator,
    stats: SearchStats,
    simulator=_SIM,
) -> int | None:
    """Boyer-Brassard-Hoyer-Tapp search: finds one solution without knowing how many exist.

    Returns None once the query budget is spent (=> no solutions, w.h.p.).
    """
    N = 2**n_index
    budget = math.ceil(9 * math.sqrt(N))
    m, lam, spent = 1.0, 6 / 5, 0
    while spent < budget:
        j = int(rng.integers(0, math.ceil(m)))
        qc = transpile(grover_circuit(oracle, n_index, j), simulator)
        bitstring = next(iter(simulator.run(qc, shots=1).result().get_counts()))
        candidate = int(bitstring, 2)
        stats.grover_iterations += j
        stats.circuits_run += 1
        stats.classical_checks += 1
        spent += max(j, 1)
        if verify(candidate):
            return candidate
        m = min(lam * m, math.sqrt(N))
    return None


def find_all(
    make_oracle: Callable[[set[int]], QuantumCircuit],
    n_index: int,
    verify: Callable[[int], bool],
    rng: np.random.Generator,
) -> SearchStats:
    """Repeatedly find a solution, then exclude it from the oracle, until none remain."""
    stats = SearchStats()
    excluded: set[int] = set()
    while True:
        hit = bbht_search(make_oracle(excluded), n_index,
                          lambda i: i not in excluded and verify(i), rng, stats)
        if hit is None:
            break
        excluded.add(hit)
        stats.found.append(hit)
    stats.found.sort()
    return stats
