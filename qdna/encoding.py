"""DNA <-> qubit encoding and sequence utilities.

Each nucleotide is stored in 2 qubits:  A=00  C=01  G=10  T=11
"""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
BASE_TO_CODE = {b: i for i, b in enumerate(BASES)}


def clean(seq: str) -> str:
    """Uppercase, drop whitespace, and reject anything that isn't A/C/G/T."""
    seq = "".join(seq.split()).upper()
    bad = set(seq) - set(BASES)
    if bad:
        raise ValueError(f"Unsupported symbols in sequence: {sorted(bad)}")
    return seq


def encode_base(base: str) -> int:
    return BASE_TO_CODE[base]


def encode_kmer(kmer: str) -> int:
    """Pack a k-mer into an integer; base j occupies bits (2j, 2j+1)."""
    value = 0
    for j, base in enumerate(kmer):
        value |= encode_base(base) << (2 * j)
    return value


def kmer_frequency(seq: str, k: int) -> np.ndarray:
    """Count vector of length 4**k (feature vector for similarity / ML)."""
    counts = np.zeros(4**k)
    for i in range(len(seq) - k + 1):
        counts[encode_kmer(seq[i:i + k])] += 1
    return counts


def parse_fasta(text: str) -> dict[str, str]:
    """Minimal FASTA parser: {record_name: sequence}."""
    records: dict[str, str] = {}
    name = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0] if line[1:].strip() else f"record{len(records) + 1}"
            records[name] = ""
        elif name is not None:
            records[name] += line
    return {k: clean(v) for k, v in records.items()}


def read_fasta(path: str) -> dict[str, str]:
    with open(path) as fh:
        return parse_fasta(fh.read())


def random_sequence(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(BASES), size=length))


def inject_snps(seq: str, positions, rng: np.random.Generator) -> str:
    """Return a copy of seq with a different base at each given position."""
    out = list(seq)
    for p in positions:
        out[p] = rng.choice([b for b in BASES if b != seq[p]])
    return "".join(out)
