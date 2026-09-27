"""Grover oracles for genomic search.

Qubit layout for every oracle:  [ index (n) | work registers | phase ancilla ]
The phase ancilla (last qubit) is prepared in |-> by the Grover driver, so an
MCX onto it flips the phase of the marked index states (phase kickback).
Work registers are always uncomputed, leaving only the phase behind.
"""
from __future__ import annotations

import math

from qiskit import QuantumCircuit

from .encoding import encode_base, encode_kmer
from .qrom import load


def index_bits(n_items: int) -> int:
    return max(1, math.ceil(math.log2(n_items)))


def mutation_oracle(reference: str, sample: str, excluded=()) -> QuantumCircuit:
    """Marks every position i where sample[i] != reference[i].

    |i>|0>|0>  --QROM-->  |i>|ref_i>|sam_i>  --CNOT-->  |i>|ref_i>|ref_i XOR sam_i>
    The XOR register is non-zero exactly at a mutation, so we phase-flip on that.
    Positions in `excluded` (already found) and padding load sam = ref, i.e. no mutation.
    """
    if len(reference) != len(sample):
        raise ValueError("reference and sample must be aligned and equal length")
    L = len(reference)
    n = index_bits(L)
    N = 2**n

    ref_table = [encode_base(reference[i]) if i < L else 0 for i in range(N)]
    sam_table = [
        encode_base(sample[i]) if i < L and i not in excluded else ref_table[i]
        for i in range(N)
    ]

    idx = list(range(n))
    r = [n, n + 1]
    s = [n + 2, n + 3]
    anc = n + 4
    qc = QuantumCircuit(n + 5, name="mutation_oracle")

    load(qc, idx, r, ref_table)
    load(qc, idx, s, sam_table)
    qc.cx(r[0], s[0])
    qc.cx(r[1], s[1])

    # Phase flip when s != 00  (= NOT(s == 00)).
    qc.x(s)
    qc.mcx(s, anc)
    qc.x(s)
    qc.x(anc)

    # Uncompute work registers.
    qc.cx(r[1], s[1])
    qc.cx(r[0], s[0])
    load(qc, idx, s, sam_table)
    load(qc, idx, r, ref_table)
    return qc


def motif_oracle(sequence: str, motif: str, excluded=()) -> QuantumCircuit:
    """Marks every start position i where sequence[i:i+k] == motif.

    QROM loads the k-mer window (2k qubits) plus a 'invalid' flag qubit that is
    set for padding positions and for positions in `excluded`.
    """
    k = len(motif)
    n_windows = len(sequence) - k + 1
    if n_windows < 1:
        raise ValueError("motif longer than sequence")
    n = index_bits(n_windows)
    N = 2**n

    flag_bit = 1 << (2 * k)
    table = []
    for i in range(N):
        if i < n_windows and i not in excluded:
            table.append(encode_kmer(sequence[i:i + k]))
        else:
            table.append(flag_bit)

    idx = list(range(n))
    data = list(range(n, n + 2 * k + 1))  # 2k window qubits + flag qubit
    anc = n + 2 * k + 1
    qc = QuantumCircuit(anc + 1, name="motif_oracle")

    load(qc, idx, data, table)

    # Match pattern: window bits == motif bits AND flag == 0.
    target = encode_kmer(motif)
    zeros = [data[b] for b in range(2 * k + 1) if not (target >> b) & 1]
    qc.x(zeros)
    qc.mcx(data, anc)
    qc.x(zeros)

    load(qc, idx, data, table)
    return qc
