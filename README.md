# Quantum DNA Sequence Analysis — Prototype

Hybrid quantum-classical prototype for mutation detection and genomic pattern search (Qiskit + Aer simulator).

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python demo.py            # options: --seed 7 --snps 4 --motif GATC --length 32
.venv\Scripts\python -m streamlit run app.py   # web UI at http://localhost:8501
```

## Web UI (`app.py`)
- **Data sources**: synthetic (with injected SNPs), pasted sequences, or a FASTA upload (try `data/sample.fasta`)
- **Sequences**: colour-coded viewer with highlighted differences, GC content and the qubit encoding table
- **Mutation Detection**: Grover probability chart with an iteration slider (shows over-rotation), full BBHT search compared with classical results and ground truth, and a circuit diagram
- **Motif Discovery**: search for any 1–5 base motif, with optional planting for demos
- **Similarity**: swap test compared with classical similarity for k = 1–3, plus a custom sequence to compare against

## Modules
| File | Purpose |
|---|---|
| `qdna/encoding.py` | 2-bit base encoding (A=00 C=01 G=10 T=11), k-mer features, FASTA reader, synthetic data |
| `qdna/qrom.py` | Quantum ROM: loads sequence data into superposition over positions |
| `qdna/oracles.py` | Grover oracles: mutation (ref ≠ sample) and motif (window == motif) |
| `qdna/grover.py` | Grover driver: fixed-iteration run + BBHT search for an unknown number of hits |
| `qdna/similarity.py` | Swap test estimating k-mer profile similarity |
| `qdna/classical.py` | Classical baselines for comparison |

## Pipeline
1. **Encode** each base into 2 qubits; a QROM loads base *i* conditioned on the index register |i⟩.
2. **Mutation oracle**: load ref_i and sample_i, XOR them, and flip the phase if the result is non-zero; then uncompute.
3. **Motif oracle**: load the k-mer window at *i* and flip the phase if it equals the motif.
4. **Grover / BBHT** amplifies the marked positions; each hit is verified classically, then excluded, and the search repeats.
5. **Swap test**: amplitude-encoded 2-mer frequency vectors give |⟨a|b⟩|² = 2·P(0) − 1.

## Known limitations
- Qubit count grows as log2(L) + const, but QROM gate count grows linearly in L. Loading classical data costs O(L), which cancels Grover's √L advantage end-to-end. This is the well-known data-loading bottleneck (qRAM).
- The BBHT stopping rule spends ~9√N iterations proving that no hits remain. For small N this exceeds a classical scan.
- The simulator is noiseless. Real hardware runs need noise models and error mitigation.
