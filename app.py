"""Streamlit UI for the Quantum DNA Sequence Analysis prototype.

    .venv\\Scripts\\python -m streamlit run app.py
"""
from __future__ import annotations

import time

import altair as alt
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

from qdna import classical
from qdna.encoding import (
    BASES, clean, inject_snps, kmer_frequency, parse_fasta, random_sequence,
)
from qdna.grover import find_all, grover_circuit, optimal_iterations, run_distribution
from qdna.oracles import index_bits, motif_oracle, mutation_oracle
from qdna.similarity import quantum_similarity, swap_test_circuit

MAX_LEN = 128
BASE_COLORS = {"A": "#2e9d5b", "C": "#2f6fd6", "G": "#d9822b", "T": "#d14545"}

st.set_page_config(page_title="Quantum DNA Analyzer", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
      .seq {font-family: ui-monospace, Consolas, monospace; font-size: 15px;
            line-height: 1.9; word-break: break-all; letter-spacing: 1px;}
      .seq span {padding: 1px 2px; border-radius: 3px;}
      .seq .hit {outline: 2px solid #f5c518; background: rgba(245,197,24,.28); font-weight: 700;}
      .seq-label {font-size: 12px; opacity: .7; margin-bottom: -6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers
def seq_html(seq: str, highlight: set[int] = frozenset(), notes: dict[int, str] | None = None) -> str:
    notes = notes or {}
    parts = []
    for i, b in enumerate(seq):
        cls = ' class="hit"' if i in highlight else ""
        title = f"pos {i}" + (f": {notes[i]}" if i in notes else "")
        parts.append(f'<span{cls} style="color:{BASE_COLORS[b]}" title="{title}">{b}</span>')
        if (i + 1) % 10 == 0:
            parts.append(" ")
    return f'<div class="seq">{"".join(parts)}</div>'


def show_seq(label: str, seq: str, highlight=frozenset(), notes=None) -> None:
    st.markdown(f'<div class="seq-label">{label} · {len(seq)} bp</div>', unsafe_allow_html=True)
    st.markdown(seq_html(seq, set(highlight), notes), unsafe_allow_html=True)


def prob_chart(counts: dict[int, int], n_index: int, marked: set[int]) -> alt.Chart:
    total = sum(counts.values()) or 1
    df = pd.DataFrame({
        "position": range(2**n_index),
        "probability": [counts.get(i, 0) / total for i in range(2**n_index)],
        "type": ["marked" if i in marked else "other" for i in range(2**n_index)],
    })
    return (
        alt.Chart(df).mark_bar().encode(
            x=alt.X("position:O", title="Position (index register)"),
            y=alt.Y("probability:Q", title="Measurement probability", axis=alt.Axis(format="%")),
            color=alt.Color("type:N", scale=alt.Scale(domain=["marked", "other"],
                                                      range=["#f5a623", "#8a94a6"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["position", alt.Tooltip("probability:Q", format=".1%"), "type"],
        ).properties(height=280)
    )


def score(pred, truth) -> tuple[float, float]:
    p, t = set(pred), set(truth)
    tp = len(p & t)
    return (tp / len(p) if p else 1.0, tp / len(t) if t else 1.0)


def draw_circuit(qc):
    fig = qc.draw("mpl", fold=-1, scale=0.7)
    fig.patch.set_facecolor("white")
    return fig


# ------------------------------------------------ cached quantum computations
@st.cache_data(show_spinner=False)
def cached_mutation_distribution(ref, sample, iters, shots):
    return run_distribution(mutation_oracle(ref, sample), index_bits(len(ref)), iters, shots)


@st.cache_data(show_spinner=False)
def cached_mutation_search(ref, sample, seed):
    t0 = time.perf_counter()
    stats = find_all(lambda ex: mutation_oracle(ref, sample, ex), index_bits(len(ref)),
                     verify=lambda i: i < len(ref) and ref[i] != sample[i],
                     rng=np.random.default_rng(seed))
    return stats, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def cached_motif_distribution(seq, motif, iters, shots):
    n = index_bits(len(seq) - len(motif) + 1)
    return run_distribution(motif_oracle(seq, motif), n, iters, shots)


@st.cache_data(show_spinner=False)
def cached_motif_search(seq, motif, seed):
    k = len(motif)
    t0 = time.perf_counter()
    stats = find_all(lambda ex: motif_oracle(seq, motif, ex), index_bits(len(seq) - k + 1),
                     verify=lambda i: seq[i:i + k] == motif, rng=np.random.default_rng(seed))
    return stats, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def cached_similarity(a, b, k, shots):
    return quantum_similarity(a, b, k=k, shots=shots), classical.cosine_similarity_sq(a, b, k=k)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 Quantum DNA Analyzer")
st.sidebar.caption("Hybrid quantum-classical mutation & pattern detection · Qiskit Aer simulator")

source = st.sidebar.radio("Data source", ["Synthetic", "Paste sequences", "Upload FASTA"])
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)
rng = np.random.default_rng(seed)
true_snps: list[int] | None = None
reference = sample = ""

try:
    if source == "Synthetic":
        length = st.sidebar.slider("Sequence length (bp)", 8, MAX_LEN, 32, step=4)
        n_snps = st.sidebar.slider("Injected SNPs", 0, min(8, length), 3)
        reference = random_sequence(length, rng)
        true_snps = sorted(rng.choice(length, size=n_snps, replace=False).tolist())
        sample = inject_snps(reference, true_snps, rng)
    elif source == "Paste sequences":
        reference = clean(st.sidebar.text_area(
            "Reference", "ATGCCTAGAAGTGTGTGATCGCATTGCTGCCA", height=90))
        sample = clean(st.sidebar.text_area(
            "Sample (aligned, same length)", "ATTCCTAGAAGTGTGTGCTCGCATTGCTCCCA", height=90))
    else:
        up = st.sidebar.file_uploader("FASTA file (≥2 records)", type=["fa", "fasta", "fna", "txt"])
        if up is not None:
            records = parse_fasta(up.getvalue().decode("utf-8", errors="replace"))
            names = list(records)
            if len(names) < 2:
                st.sidebar.error("Need at least two records (reference + sample).")
            else:
                r = st.sidebar.selectbox("Reference record", names, 0)
                s = st.sidebar.selectbox("Sample record", names, 1)
                reference, sample = records[r], records[s]
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

shots = st.sidebar.select_slider("Shots per circuit", [256, 512, 1024, 2048, 4096, 8192], 2048)

if not reference or not sample:
    st.info("Load a reference and a sample sequence from the sidebar to begin.")
    st.stop()
if len(reference) > MAX_LEN or len(sample) > MAX_LEN:
    st.error(f"Sequences longer than {MAX_LEN} bp are too slow to simulate. "
             f"Trim them (e.g. to a gene region) first.")
    st.stop()
if len(reference) != len(sample):
    st.error(f"Reference ({len(reference)} bp) and sample ({len(sample)} bp) must be aligned "
             "and the same length for position-wise mutation detection.")
    st.stop()

classic_mut = classical.find_mutations(reference, sample)
truth = true_snps if true_snps is not None else classic_mut

# -------------------------------------------------------------------- tabs
st.title("Quantum DNA Sequence Analysis")
PAGES = ["Sequences", "Mutation Detection", "Motif Discovery", "Similarity", "How it works"]
page = st.segmented_control("Section", PAGES, default=PAGES[0], key="page",
                            label_visibility="collapsed") or PAGES[0]

# ---------------------------------------------------------------- Sequences
if page == "Sequences":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", f"{len(reference)} bp")
    c2.metric("Differences", len(classic_mut))
    gc = lambda s: (s.count("G") + s.count("C")) / len(s)
    c3.metric("GC content (ref)", f"{gc(reference):.0%}")
    c4.metric("GC content (sample)", f"{gc(sample):.0%}", f"{gc(sample) - gc(reference):+.0%}")

    notes = {i: f"{reference[i]}→{sample[i]}" for i in classic_mut}
    show_seq("Reference", reference, classic_mut, notes)
    show_seq("Sample", sample, classic_mut, notes)
    st.caption("Highlighted = positions that differ. Hover a base for details. "
               "Legend: " + " ".join(f"<b style='color:{c}'>{b}</b>" for b, c in BASE_COLORS.items()),
               unsafe_allow_html=True)

    st.subheader("Qubit encoding")
    enc = pd.DataFrame({"Base": list(BASES), "Qubits |q1 q0⟩": ["00", "01", "10", "11"],
                        "Count (ref)": [reference.count(b) for b in BASES],
                        "Count (sample)": [sample.count(b) for b in BASES]})
    st.dataframe(enc, hide_index=True, width="content")

# --------------------------------------------------------- Mutation detection
if page == "Mutation Detection":
    n = index_bits(len(reference))
    oracle = mutation_oracle(reference, sample)
    m_est = len(classic_mut)
    opt = optimal_iterations(n, m_est)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qubits", oracle.num_qubits, help=f"{n} index + 4 data + 1 ancilla")
    c2.metric("Search space N", 2**n)
    c3.metric("Optimal iterations", opt, help="π/4·√(N/M), using the classical M for illustration")
    c4.metric("Oracle size", f"{sum(oracle.count_ops().values())} gates")

    st.subheader("1 · Amplitude amplification")
    st.caption("Grover's algorithm boosts the probability of measuring mutated positions. "
               "Try going past the optimum to see over-rotation.")
    iters = st.slider("Grover iterations", 0, max(6, 2 * opt + 2), opt, key="mut_iters")
    with st.spinner("Simulating circuit…"):
        counts = cached_mutation_distribution(reference, sample, iters, shots)
    st.altair_chart(prob_chart(counts, n, set(classic_mut)), width="stretch")
    p_marked = sum(counts.get(i, 0) for i in classic_mut) / shots
    st.write(f"Probability of measuring a mutated position: **{p_marked:.1%}** "
             f"(random guess: {m_est / 2**n:.1%})")

    st.subheader("2 · Full quantum search (number of mutations unknown)")
    st.caption("BBHT randomized Grover → verify candidate classically → exclude it → repeat.")
    if st.button("Run quantum mutation search", type="primary", key="run_mut"):
        with st.spinner("Running BBHT Grover search…"):
            st.session_state["mut_result"] = (reference, sample, *cached_mutation_search(reference, sample, seed))

    res = st.session_state.get("mut_result")
    if res and res[0] == reference and res[1] == sample:
        _, _, stats, dt = res
        prec, rec = score(stats.found, truth)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Found", len(stats.found))
        c2.metric("Precision", f"{prec:.2f}")
        c3.metric("Recall", f"{rec:.2f}")
        c4.metric("Grover iterations", stats.grover_iterations,
                  help=f"Classical scan reads {len(reference)} positions")
        c5.metric("Time", f"{dt:.1f}s")
        rows = sorted(set(stats.found) | set(classic_mut) | set(truth))
        st.dataframe(pd.DataFrame({
            "Position": rows,
            "Change": [f"{reference[i]} → {sample[i]}" for i in rows],
            "Quantum": ["✅" if i in stats.found else "❌" for i in rows],
            "Classical": ["✅" if i in classic_mut else "❌" for i in rows],
            "Ground truth": ["✅" if i in truth else "—" for i in rows],
        }), hide_index=True)
        show_seq("Sample with quantum-detected mutations", sample, stats.found,
                 {i: f"{reference[i]}→{sample[i]}" for i in stats.found})

    with st.expander("Circuit"):
        st.caption("One Grover iteration = Oracle (QROM load → XOR → phase flip → uncompute) + Diffuser.")
        st.pyplot(draw_circuit(grover_circuit(oracle, n, min(max(iters, 1), 2), boxed=True)),
                  width="content")

# -------------------------------------------------------------- Motif search
if page == "Motif Discovery":
    c1, c2 = st.columns([1, 2])
    motif = c1.text_input("Motif to find", "GAT", max_chars=5).upper().strip()
    plant = c2.slider("Plant motif into the sample N times (for demo)", 0, 4, 2)
    motif_error = None
    try:
        motif = clean(motif)
        if not 1 <= len(motif) <= 5:
            motif_error = "Motif length must be 1–5 bases"
    except ValueError as e:
        motif_error = str(e)
    if motif_error:
        st.error(motif_error)
    else:

        seq = list(sample)
        k = len(motif)
        prng = np.random.default_rng(seed + 1)
        if plant and len(seq) >= k:
            starts = prng.choice(len(seq) - k + 1, size=min(plant, len(seq) - k + 1), replace=False)
            for p in starts:
                seq[p:p + k] = motif
        seq = "".join(seq)
        hits = classical.find_motif(seq, motif)
        hit_bases = {i + j for i in hits for j in range(k)}

        n = index_bits(len(seq) - k + 1)
        oracle = motif_oracle(seq, motif)
        opt = optimal_iterations(n, len(hits))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qubits", oracle.num_qubits, help=f"{n} index + {2 * k} window + 1 flag + 1 ancilla")
        c2.metric("Search space N", 2**n)
        c3.metric("Occurrences (classical)", len(hits))
        c4.metric("Optimal iterations", opt)
        show_seq(f"Sequence searched for '{motif}'", seq, hit_bases)

        iters = st.slider("Grover iterations", 0, max(6, 2 * opt + 2), opt, key="motif_iters")
        with st.spinner("Simulating circuit…"):
            counts = cached_motif_distribution(seq, motif, iters, shots)
        st.altair_chart(prob_chart(counts, n, set(hits)), width="stretch")

        if st.button("Run quantum motif search", type="primary", key="run_motif"):
            with st.spinner("Running BBHT Grover search…"):
                st.session_state["motif_result"] = (seq, motif, *cached_motif_search(seq, motif, seed))
        res = st.session_state.get("motif_result")
        if res and res[0] == seq and res[1] == motif:
            _, _, stats, dt = res
            prec, rec = score(stats.found, hits)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Found at", ", ".join(map(str, stats.found)) or "none")
            c2.metric("Precision / Recall", f"{prec:.2f} / {rec:.2f}")
            c3.metric("Grover iterations", stats.grover_iterations)
            c4.metric("Time", f"{dt:.1f}s")

# ---------------------------------------------------------------- Similarity
if page == "Similarity":
    st.caption("Each sequence → k-mer frequency vector → amplitude-encoded quantum state. "
               "The swap test estimates |⟨a|b⟩|² (squared cosine similarity).")
    c1, c2 = st.columns([1, 2])
    k = c1.radio("k-mer size", [1, 2, 3], index=1, horizontal=True,
                 help="Vector size 4^k → 2k qubits per sequence")
    custom = c2.text_input("Custom sequence to compare (optional)", "")
    c1.metric("Qubits", 1 + 4 * k)

    pairs = {
        "Reference vs itself": (reference, reference),
        "Reference vs sample": (reference, sample),
        "Reference vs random": (reference, random_sequence(len(reference), np.random.default_rng(seed + 2))),
        "Reference vs GC-only": (reference, "".join(np.random.default_rng(seed + 3).choice(list("GC"), len(reference)))),
    }
    if custom.strip():
        try:
            pairs["Reference vs custom"] = (reference, clean(custom))
        except ValueError as e:
            st.error(str(e))

    rows = []
    with st.spinner("Running swap tests…"):
        for name, (a, b) in pairs.items():
            if len(b) < k:
                continue
            q, c = cached_similarity(a, b, k, shots)
            rows.append({"Pair": name, "Quantum (swap test)": q, "Classical": c, "Error": abs(q - c)})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Quantum (swap test)": "{:.3f}", "Classical": "{:.3f}", "Error": "{:.3f}"}),
                 hide_index=True)
    long = df.melt(id_vars="Pair", value_vars=["Quantum (swap test)", "Classical"],
                   var_name="Method", value_name="Similarity")
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            y=alt.Y("Pair:N", title=None, sort=None),
            x=alt.X("Similarity:Q", scale=alt.Scale(domain=[0, 1])),
            yOffset="Method:N",
            color=alt.Color("Method:N", scale=alt.Scale(domain=["Quantum (swap test)", "Classical"],
                                            range=["#6c5ce7", "#8a94a6"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Pair", "Method", alt.Tooltip("Similarity:Q", format=".3f")],
        ).properties(height=60 * len(df)),
        width="stretch",
    )
    with st.expander("Swap test circuit"):
        u = kmer_frequency(reference, k)
        st.pyplot(draw_circuit(swap_test_circuit(u, kmer_frequency(sample, k))), width="content")

# ---------------------------------------------------------------- About
if page == "How it works":
    st.markdown(
        """
### Pipeline
1. **Encoding**: each base → 2 qubits (A=00, C=01, G=10, T=11).
2. **QROM**: loads the base at position *i* conditioned on an index register in superposition over all positions.
3. **Mutation oracle**: loads ref*ᵢ* and sample*ᵢ*, XORs them, phase-flips if non-zero, then uncomputes.
4. **Motif oracle**: loads the k-mer window at *i*, phase-flips if it equals the motif.
5. **Grover / BBHT**: amplifies marked positions; each hit is verified classically, excluded, and the search repeats.
6. **Swap test**: compares amplitude-encoded k-mer profiles → similarity.

### Limitations (honest assessment)
- **Data-loading bottleneck**: QROM needs O(N) gates, which cancels Grover's O(√N) query advantage end-to-end.
- **BBHT stopping rule** spends ~9√N iterations to confirm no hits remain, so it costs more than a classical scan at small N.
- **Noiseless simulator**: real hardware needs noise models and error mitigation.
- Sequence length is capped at 128 bp for simulation speed.
        """
    )
