# Quantum DNA Sequence Analysis

A quantum computing system for detecting mutations and hidden patterns in DNA sequences. It uses **Grover's search algorithm** (with a quantum ROM oracle) and the **swap test**, built with Qiskit and run on the Aer simulator, with an optional hardware noise model.

## Setup
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Run the command-line demo:
```
.venv\Scripts\python demo.py            # options: --seed 7 --snps 4 --motif GATC --length 32
```

Launch the interactive app (http://localhost:8501):
```
.venv\Scripts\python -m streamlit run app.py
```

## Quantum pipeline
1. **Basis encoding**: each nucleotide maps to 2 qubits: A=|00⟩, C=|01⟩, G=|10⟩, T=|11⟩.
2. **QROM (quantum read-only memory)**: an index register in uniform superposition over all positions |i⟩ loads the base at position *i* into a data register, using multi-controlled X gates.
3. **Mutation oracle**: loads ref*ᵢ* and sample*ᵢ*, XORs them, and phase-flips |i⟩ (phase kickback onto a |−⟩ ancilla) when the XOR register is non-zero. It then uncomputes, leaving only the phase.
4. **Motif oracle**: loads the k-mer window starting at *i*, plus a validity flag, and phase-flips when it equals the target motif.
5. **Grover search**: oracle plus diffuser (inversion about the mean) amplifies the marked positions. The optimal iteration count is ⌊π/4·√(N/M)⌋.
6. **BBHT search**: when the number of solutions M is unknown, randomized Grover (Boyer–Brassard–Høyer–Tapp) finds one hit. The hit is verified classically in O(1), excluded from the oracle, and the search repeats.
7. **Swap test**: k-mer frequency vectors are amplitude-encoded, and P(ancilla=0) = ½ + ½|⟨a|b⟩|² gives the similarity between sequences.

## App sections
| Section | Contents |
|---|---|
| **Encoding** | DNA → qubit mapping and the register layout (index, data, ancilla) |
| **Mutations** | Amplitude amplification vs iteration count, simulated success probability against the sin²((2j+1)θ) theory curve, a position inspector showing a position's qubit/XOR state, full BBHT search compared with the classical baseline, circuit diagram and OpenQASM export |
| **Motifs** | Grover motif search, multi-motif scan, circuit and OpenQASM export |
| **Similarity** | Swap test vs classical cosine similarity (k = 1–3), pairwise swap-test matrix |
| **Noise Lab** | Depolarizing and readout noise model (NISQ-style), ideal vs noisy distributions, CX count and depth, noise sweep |
| **Benchmark** | Scaling of qubits, oracle gates and Grover iterations with sequence length |
| **How it works** | Pipeline and limitations |

Input can be synthetic (random sequences with injected SNPs), pasted sequences, or a FASTA upload (try `data/sample.fasta`).

## Project structure
| File | Purpose |
|---|---|
| `qdna/encoding.py` | 2-bit base encoding, k-mer feature vectors, FASTA parsing, synthetic data |
| `qdna/qrom.py` | Quantum ROM: loads classical data indexed by a superposed address |
| `qdna/oracles.py` | Grover oracles for mutation detection and motif search |
| `qdna/grover.py` | Diffuser, Grover circuit, fixed-iteration runs, BBHT search |
| `qdna/similarity.py` | Swap-test circuit and quantum similarity estimate |
| `qdna/noise.py` | NISQ noise model (depolarizing + readout) on an `rz, sx, x, cx` basis |
| `qdna/classical.py` | Classical baselines used for comparison |
| `app.py`, `ui/` | Streamlit front-end |
| `demo.py` | Command-line end-to-end demo |

## Findings and limitations
- **Qubits scale well, gates don't.** The index register needs only ⌈log₂L⌉ qubits, but the QROM needs O(L) gates to load the data. This cancels Grover's O(√N) query advantage end-to-end on classical data: the well-known data-loading (qRAM) bottleneck.
- **Unknown solution count costs extra.** BBHT spends ~9√N iterations confirming that no hits remain, so for small N it uses more queries than a classical scan.
- **Noise destroys the signal quickly.** Under realistic error rates, the Grover amplification is lost beyond ~8 bp (thousands of CX gates after transpilation). Error correction or mitigation would be required on real hardware.
- Sequences are capped at 128 bp for simulation speed.
