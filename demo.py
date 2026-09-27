"""End-to-end demo of the quantum DNA analysis prototype.

    python demo.py                 # synthetic data
    python demo.py --seed 7 --snps 4
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from qdna import classical
from qdna.encoding import encode_base, inject_snps, random_sequence
from qdna.grover import find_all, optimal_iterations, run_distribution
from qdna.oracles import index_bits, motif_oracle, mutation_oracle
from qdna.similarity import quantum_similarity


def header(title: str) -> None:
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def histogram(counts: dict[int, int], marked: set[int], top: int = 8) -> None:
    total = sum(counts.values())
    for idx, c in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
        bar = "#" * round(40 * c / total)
        tag = "  <- marked" if idx in marked else ""
        print(f"  pos {idx:>3}  {c / total:6.1%}  {bar}{tag}")


def score(predicted, truth) -> str:
    p, t = set(predicted), set(truth)
    tp = len(p & t)
    prec = tp / len(p) if p else 1.0
    rec = tp / len(t) if t else 1.0
    return f"precision={prec:.2f}  recall={rec:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length", type=int, default=32, help="sequence length (<=64 keeps it fast)")
    ap.add_argument("--snps", type=int, default=3)
    ap.add_argument("--motif", default="GAT")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ data
    reference = random_sequence(args.length, rng)
    true_snps = sorted(rng.choice(args.length, size=args.snps, replace=False).tolist())
    sample = inject_snps(reference, true_snps, rng)

    header("1. Encoding  (A=00 C=01 G=10 T=11)")
    print(f"  reference : {reference}")
    print(f"  sample    : {sample}")
    print("  diff      : " + "".join("^" if a != b else " " for a, b in zip(reference, sample)))
    print(f"  first 4 bases as qubits: "
          + " ".join(f"{b}={encode_base(b):02b}" for b in reference[:4]))

    # ------------------------------------------------------ mutation search
    header("2. Mutation detection  (Grover search over positions)")
    n = index_bits(len(reference))
    oracle = mutation_oracle(reference, sample)
    print(f"  qubits: {oracle.num_qubits}  ({n} index + 4 data + 1 ancilla)"
          f"   search space N={2**n}")

    iters = optimal_iterations(n, len(true_snps))
    print(f"\n  Amplitude amplification with {iters} Grover iterations "
          f"(illustration, M={len(true_snps)} known):")
    histogram(run_distribution(oracle, n, iters), set(true_snps))

    print("\n  Full search, M unknown (BBHT + exclude-and-repeat):")
    t0 = time.perf_counter()
    stats = find_all(
        lambda excl: mutation_oracle(reference, sample, excl), n,
        verify=lambda i: i < len(reference) and reference[i] != sample[i], rng=rng,
    )
    dt = time.perf_counter() - t0
    classic = classical.find_mutations(reference, sample)
    print(f"  quantum  found : {stats.found}")
    print(f"  classical found: {classic}")
    print(f"  truth          : {true_snps}")
    print(f"  {score(stats.found, true_snps)}")
    print(f"  Grover iterations={stats.grover_iterations}  circuits={stats.circuits_run}"
          f"  (classical scan reads all {len(reference)} positions)  time={dt:.1f}s")

    # --------------------------------------------------------- motif search
    header(f"3. Motif discovery  (find '{args.motif}' in the sample)")
    seq = list(sample)
    k = len(args.motif)
    for p in rng.choice(len(seq) - k, size=2, replace=False):  # plant motif twice
        seq[p:p + k] = args.motif
    seq = "".join(seq)[: args.length]
    print(f"  sequence: {seq}")
    classic = classical.find_motif(seq, args.motif)
    n = index_bits(len(seq) - k + 1)
    oracle = motif_oracle(seq, args.motif)
    print(f"  qubits: {oracle.num_qubits}  ({n} index + {2 * k} window + 1 flag + 1 ancilla)")
    print(f"\n  Amplitude amplification ({optimal_iterations(n, len(classic))} iterations):")
    histogram(run_distribution(oracle, n, optimal_iterations(n, len(classic))), set(classic))

    stats = find_all(
        lambda excl: motif_oracle(seq, args.motif, excl), n,
        verify=lambda i: seq[i:i + k] == args.motif, rng=rng,
    )
    print(f"\n  quantum  found : {stats.found}")
    print(f"  classical found: {classic}")
    print(f"  {score(stats.found, classic)}   Grover iterations={stats.grover_iterations}")

    # ------------------------------------------------------------ similarity
    header("4. Sequence similarity  (swap test on 2-mer profiles, 9 qubits)")
    unrelated = random_sequence(args.length, rng)
    gc_rich = "".join(rng.choice(list("GC"), size=args.length))
    pairs = [
        ("reference vs itself", reference, reference),
        ("reference vs mutated sample", reference, sample),
        ("reference vs random sequence", reference, unrelated),
        ("reference vs GC-only sequence", reference, gc_rich),
    ]
    print(f"  {'pair':<32}{'quantum':>9}{'classical':>11}")
    for name, a, b in pairs:
        q = quantum_similarity(a, b)
        c = classical.cosine_similarity_sq(a, b)
        print(f"  {name:<32}{q:>9.3f}{c:>11.3f}")
    print()


if __name__ == "__main__":
    main()
