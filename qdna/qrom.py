"""Quantum Read-Only Memory (QROM).

Loads classical data into a quantum register indexed by a superposed address:
    |i>|0>  ->  |i>|table[i]>
Every write is an XOR, so applying load() a second time uncomputes it.
"""
from __future__ import annotations

from qiskit import QuantumCircuit


def load(qc: QuantumCircuit, index_qubits, data_qubits, table) -> None:
    """XOR table[i] into data_qubits, controlled on the index register == i.

    index_qubits[j] holds bit j of the address, data_qubits[b] holds bit b.
    """
    n = len(index_qubits)
    for i, value in enumerate(table):
        if value == 0:
            continue
        # Turn "index == i" into "all controls are 1".
        zeros = [index_qubits[j] for j in range(n) if not (i >> j) & 1]
        if zeros:
            qc.x(zeros)
        for b, target in enumerate(data_qubits):
            if (value >> b) & 1:
                qc.mcx(list(index_qubits), target)
        if zeros:
            qc.x(zeros)
