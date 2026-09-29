"""Streamlit front-end for the Quantum DNA Sequence Analysis project.

    .venv\\Scripts\\python -m streamlit run app.py
"""
from __future__ import annotations

import math
import time

import altair as alt
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import qasm2, transpile

from qdna import classical
from qdna.encoding import (
    BASES, clean, encode_base, inject_snps, kmer_frequency, parse_fasta, random_sequence,
)
from qdna.grover import find_all, grover_circuit, optimal_iterations, run_distribution
from qdna.noise import BASIS, noisy_simulator
from qdna.oracles import index_bits, motif_oracle, mutation_oracle
from qdna.similarity import quantum_similarity, swap_test_circuit
from ui.components import (
    BASE_COLORS, CSS, MUTED, QUANTUM, compare_chart, draw_circuit, iteration_curve,
    prob_chart, score, show_seq,
)

MAX_LEN = 128

st.set_page_config(page_title="Quantum DNA Analyzer", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False)
def cached_noisy_distribution(ref, sample, iters, shots, p1, p2, readout):
    sim = noisy_simulator(p1, p2, readout)
    return run_distribution(mutation_oracle(ref, sample), index_bits(len(ref)), iters, shots,
                            simulator=sim)


@st.cache_data(show_spinner=False)
def cached_physical_stats(ref, sample, iters):
    qc = grover_circuit(mutation_oracle(ref, sample), index_bits(len(ref)), iters)
    tq = transpile(qc, basis_gates=BASIS, optimization_level=1)
    ops = tq.count_ops()
    return {"cx": ops.get("cx", 0), "gates": sum(v for k, v in ops.items() if k != "measure"),
            "depth": tq.depth()}


@st.cache_data(show_spinner=False)
def cached_benchmark_row(length, n_snps, seed, physical):
    rng = np.random.default_rng(seed + length)
    ref = random_sequence(length, rng)
    truth = sorted(rng.choice(length, size=min(n_snps, length), replace=False).tolist())
    sample = inject_snps(ref, truth, rng)
    oracle = mutation_oracle(ref, sample)
    n = index_bits(length)
    stats, dt = cached_mutation_search(ref, sample, seed)
    prec, rec = score(stats.found, truth)
    row = {
        "length": length, "qubits": oracle.num_qubits, "search space N": 2**n,
        "oracle gates (logical)": sum(oracle.count_ops().values()),
        "Grover iterations": stats.grover_iterations, "circuits run": stats.circuits_run,
        "classical checks": length, "√N": round(math.sqrt(2**n), 1),
        "precision": prec, "recall": rec, "seconds": round(dt, 2),
    }
    if physical:
        tq = transpile(oracle, basis_gates=BASIS, optimization_level=1)
        row["oracle CX (physical)"] = tq.count_ops().get("cx", 0)
    return row


def validate_motif(text: str) -> tuple[str | None, str | None]:
    try:
        m = clean(text)
    except ValueError as e:
        return None, str(e)
    if not 1 <= len(m) <= 5:
        return None, "Motif length must be 1–5 bases"
    return m, None


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧬 Quantum DNA Analyzer")
st.sidebar.caption("Quantum mutation & pattern detection · Qiskit Aer simulator")

source = st.sidebar.radio("Data source", ["Synthetic", "Paste sequences", "Upload FASTA"])
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)
rng = np.random.default_rng(seed)
true_snps: list[int] | None = None
reference = sample = ""
records: dict[str, str] = {}

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
    reference = sample = ""

shots = st.sidebar.select_slider("Shots per circuit", [256, 512, 1024, 2048, 4096, 8192], 2048)

# Pages that need data show this message instead of running (the nav bar stays usable).
data_problem: tuple[str, str] | None = None
if not reference or not sample:
    data_problem = ("info", "Load a reference and a sample sequence from the sidebar to begin.")
elif len(reference) > MAX_LEN or len(sample) > MAX_LEN:
    data_problem = ("error", f"Sequences longer than {MAX_LEN} bp are too slow to simulate. "
                             "Trim them (e.g. to a gene region) first.")
elif len(reference) != len(sample):
    data_problem = ("error", f"Reference ({len(reference)} bp) and sample ({len(sample)} bp) must be "
                             "aligned and the same length for position-wise mutation detection.")

st.session_state.setdefault("motif_input", "GAT")
classic_mut = truth = []
if data_problem is None:
    classic_mut = classical.find_mutations(reference, sample)
    truth = true_snps if true_snps is not None else classic_mut
    if not records:
        records = {"reference": reference, "sample": sample}


# ================================================================= Encoding
def page_encoding() -> None:
    show_seq("Reference", reference, set(classic_mut),
             {i: f"{reference[i]}→{sample[i]}" for i in classic_mut})
    show_seq("Sample", sample, set(classic_mut),
             {i: f"{reference[i]}→{sample[i]}" for i in classic_mut})
    st.caption("Cyan = positions that differ. Hover a base for details. Legend: "
               + " ".join(f"<b style='color:{c}'>{b}</b>" for b, c in BASE_COLORS.items()),
               unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Basis-state encoding**: 2 qubits per nucleotide")
        st.dataframe(pd.DataFrame({"Base": list(BASES), "Qubits |q1 q0⟩": ["00", "01", "10", "11"]}),
                     hide_index=True, width="content")
    with c2:
        n = index_bits(len(reference))
        st.markdown("**Register layout** for the mutation oracle")
        st.dataframe(pd.DataFrame([
            {"Register": "Index", "Qubits": n, "Role": f"Superposition over all {2**n} positions |i⟩"},
            {"Register": "Reference data", "Qubits": 2, "Role": "QROM loads ref[i]"},
            {"Register": "Sample data", "Qubits": 2, "Role": "QROM loads sample[i], then XOR with ref[i]"},
            {"Register": "Phase ancilla", "Qubits": 1, "Role": "|−⟩ state for phase kickback"},
            {"Register": "Total", "Qubits": n + 5, "Role": ""},
        ]), hide_index=True, width="stretch")
        st.caption("A sequence of L bases needs only ⌈log₂L⌉ index qubits, but loading it (the QROM) "
                   "costs O(L) gates. This is the data-loading bottleneck.")


# ================================================================ Mutations
def page_mutations() -> None:
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
    j_max = max(6, 2 * opt + 2)
    iters = st.slider("Grover iterations", 0, j_max, opt, key=f"mut_iters_{n}_{m_est}")
    with st.spinner("Simulating circuit…"):
        counts = cached_mutation_distribution(reference, sample, iters, shots)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(prob_chart(counts, n, set(classic_mut)), width="stretch")
        p_marked = sum(counts.get(i, 0) for i in classic_mut) / shots
        st.write(f"Probability of measuring a mutated position: **{p_marked:.1%}** "
                 f"(random guess: {m_est / 2**n:.1%})")
    with right:
        if st.toggle("Success probability vs iterations", value=len(reference) <= 64,
                     help="Simulates every iteration count and overlays the sin²((2j+1)θ) theory curve"):
            with st.spinner("Simulating each iteration count…"):
                pts = [(j, sum(cached_mutation_distribution(reference, sample, j, shots).get(i, 0)
                               for i in classic_mut) / shots) for j in range(j_max + 1)]
            st.altair_chart(iteration_curve(pts, n, m_est, iters), width="stretch")
            st.caption("Dashed = theory · dots = simulation · red = current slider value")

    with st.expander("🔎 Position inspector: what the oracle sees", expanded=False):
        pos = st.selectbox("Position", range(len(reference)),
                           format_func=lambda i: f"{i}  ({reference[i]}→{sample[i]})" if i in classic_mut
                           else f"{i}  ({reference[i]})")
        r, s = encode_base(reference[pos]), encode_base(sample[pos])
        cc = st.columns(4)
        cc[0].metric("Reference", f"{reference[pos]} = |{r:02b}⟩")
        cc[1].metric("Sample", f"{sample[pos]} = |{s:02b}⟩")
        cc[2].metric("XOR register", f"|{r ^ s:02b}⟩", "marked" if r ^ s else "not marked", delta_color="off")
        cc[3].metric(f"P(measure) @ {iters} it.", f"{counts.get(pos, 0) / shots:.1%}")
        st.caption(f"Index register |{pos:0{n}b}⟩ · the oracle phase-flips this state only if the XOR register ≠ |00⟩.")

    st.subheader("2 · Full quantum search (number of mutations unknown)")
    st.caption("BBHT randomized Grover → verify candidate classically → exclude it → repeat.")
    if st.button("Run quantum mutation search", type="primary", key="run_mut"):
        with st.spinner("Running BBHT Grover search…"):
            stats, dt = cached_mutation_search(reference, sample, seed)
        st.session_state.mut_result = (reference, sample, stats, dt)

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
        st.download_button("⬇ Export circuit (OpenQASM 2)", qasm2.dumps(grover_circuit(oracle, n, iters)),
                           file_name=f"grover_mutation_{iters}it.qasm", mime="text/plain",
                           help="Run it on IBM Quantum or any QASM-compatible tool")


# =================================================================== Motifs
def page_motifs() -> None:
    c1, c2 = st.columns([1, 2])
    c1.text_input("Motif to find", max_chars=5, key="motif_input")
    plant = c2.slider("Plant motif into the sample N times (for demo)", 0, 4, 2)
    motif, err = validate_motif(st.session_state.motif_input.upper().strip())
    if err:
        st.error(err)
        return

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

    iters = st.slider("Grover iterations", 0, max(6, 2 * opt + 2), opt, key=f"motif_iters_{n}_{len(hits)}")
    with st.spinner("Simulating circuit…"):
        counts = cached_motif_distribution(seq, motif, iters, shots)
    st.altair_chart(prob_chart(counts, n, set(hits)), width="stretch")

    if st.button("Run quantum motif search", type="primary", key="run_motif"):
        with st.spinner("Running BBHT Grover search…"):
            stats, dt = cached_motif_search(seq, motif, seed)
        st.session_state.motif_result = (seq, motif, stats, dt)
    res = st.session_state.get("motif_result")
    if res and res[0] == seq and res[1] == motif:
        _, _, stats, dt = res
        prec, rec = score(stats.found, hits)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Found at", ", ".join(map(str, stats.found)) or "none")
        c2.metric("Precision / Recall", f"{prec:.2f} / {rec:.2f}")
        c3.metric("Grover iterations", stats.grover_iterations)
        c4.metric("Time", f"{dt:.1f}s")

    with st.container(border=True):
        st.markdown("**📋 Multi-motif scan**: one Grover search per pattern in the sequence above")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        raw = c1.text_input("Motifs (comma-separated, up to 6)", "GAT, TAG, CC, ATG")
        go = c2.button("Scan with Grover", width="stretch")
        motifs, bad = [], []
        for t in [t.strip().upper() for t in raw.split(",") if t.strip()][:6]:
            m, e = validate_motif(t)
            (motifs if m else bad).append(m or t)
        if bad:
            st.warning(f"Skipped invalid motifs: {', '.join(bad)}")
        if go and motifs:
            rows, bar = [], st.progress(0.0, "Scanning…")
            for idx, m in enumerate(motifs):
                bar.progress(idx / len(motifs), f"Grover search for '{m}'…")
                h = classical.find_motif(seq, m)
                stats, dt = cached_motif_search(seq, m, seed)
                p, r = score(stats.found, h)
                rows.append({"motif": m, "quantum positions": ", ".join(map(str, stats.found)) or "—",
                             "classical positions": ", ".join(map(str, h)) or "—", "count": len(h),
                             "precision": p, "recall": r, "Grover iterations": stats.grover_iterations,
                             "seconds": round(dt, 2)})
            bar.empty()
            st.session_state.scan_result = (seq, tuple(motifs), pd.DataFrame(rows))
        res = st.session_state.get("scan_result")
        if res and res[0] == seq and res[1] == tuple(motifs):
            st.dataframe(res[2], hide_index=True, width="stretch")

    with st.expander("Circuit"):
        st.pyplot(draw_circuit(grover_circuit(oracle, n, min(max(iters, 1), 2), boxed=True)),
                  width="content")
        st.download_button("⬇ Export circuit (OpenQASM 2)", qasm2.dumps(grover_circuit(oracle, n, iters)),
                           file_name=f"grover_motif_{motif}_{iters}it.qasm", mime="text/plain")


# =============================================================== Similarity
def page_similarity() -> None:
    st.caption("Each sequence → k-mer frequency vector → amplitude-encoded quantum state. "
               "The swap test estimates |⟨a|b⟩|² (squared cosine similarity).")
    c1, c2 = st.columns([1, 2])
    k = c1.radio("k-mer size", [1, 2, 3], index=1, horizontal=True,
                 help="Vector size 4^k → 2k qubits per sequence")
    custom = c2.text_input("Custom sequence to compare (optional)", "")
    c1.metric("Qubits", 1 + 4 * k)

    pool = {
        "Reference": reference,
        "Sample": sample,
        "Random": random_sequence(len(reference), np.random.default_rng(seed + 2)),
        "GC-only": "".join(np.random.default_rng(seed + 3).choice(list("GC"), len(reference))),
    }
    if custom.strip():
        try:
            pool["Custom"] = clean(custom)
        except ValueError as e:
            st.error(str(e))
    for name, s in records.items():
        if s not in pool.values():
            pool[f"FASTA: {name}"] = s

    rows = []
    with st.spinner("Running swap tests…"):
        for name in ("Reference", "Sample", "Random", "GC-only", "Custom"):
            b = pool.get(name)
            if b is None or len(b) < k:
                continue
            q, c = cached_similarity(reference, b, k, shots)
            rows.append({"Pair": "Reference vs " + ("itself" if name == "Reference" else name.lower()),
                         "Quantum (swap test)": q, "Classical": c, "Error": abs(q - c)})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Quantum (swap test)": "{:.3f}", "Classical": "{:.3f}", "Error": "{:.3f}"}),
                 hide_index=True)
    long = df.melt(id_vars="Pair", value_vars=["Quantum (swap test)", "Classical"],
                   var_name="Method", value_name="Similarity")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y("Pair:N", title=None, sort=None),
        x=alt.X("Similarity:Q", scale=alt.Scale(domain=[0, 1])),
        yOffset="Method:N",
        color=alt.Color("Method:N", scale=alt.Scale(domain=["Quantum (swap test)", "Classical"],
                                                    range=[QUANTUM, MUTED]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Pair", "Method", alt.Tooltip("Similarity:Q", format=".3f")],
    ).properties(height=60 * len(df)), width="stretch")

    with st.container(border=True):
        st.markdown("**🗺️ Pairwise swap-test matrix**")
        names = list(pool)
        chosen = st.multiselect("Sequences", names, default=names[:min(5, len(names))], max_selections=6)
        method = st.radio("Show", ["Quantum (swap test)", "Classical", "Difference"], horizontal=True)
        chosen = [c for c in chosen if len(pool[c]) >= k]
        if len(chosen) >= 2:
            cells = []
            with st.spinner("Computing matrix…"):
                for a in chosen:
                    for b in chosen:
                        x, y = sorted((a, b))
                        q, c = cached_similarity(pool[x], pool[y], k, shots)
                        v = {"Quantum (swap test)": q, "Classical": c, "Difference": q - c}[method]
                        cells.append({"a": a, "b": b, "value": v})
            cm = pd.DataFrame(cells)
            scale = (alt.Scale(scheme="redblue", domain=[-0.1, 0.1]) if method == "Difference"
                     else alt.Scale(scheme="purples", domain=[0, 1]))
            base = alt.Chart(cm).encode(x=alt.X("a:N", title=None, sort=chosen),
                                        y=alt.Y("b:N", title=None, sort=chosen))
            st.altair_chart((base.mark_rect().encode(color=alt.Color("value:Q", scale=scale, title=None),
                                                      tooltip=["a", "b", alt.Tooltip("value:Q", format=".3f")])
                             + base.mark_text(fontSize=13).encode(
                                 text=alt.Text("value:Q", format=".2f"),
                                 color=alt.condition(alt.expr.abs(alt.datum.value) > 0.6, alt.value("white"),
                                                     alt.value("black"))))
                            .properties(height=70 * len(chosen)), width="stretch")
        else:
            st.info("Pick at least two sequences.")

    with st.expander("Swap test circuit"):
        st.pyplot(draw_circuit(swap_test_circuit(kmer_frequency(reference, k), kmer_frequency(sample, k))),
                  width="content")


# ================================================================ Noise Lab
def page_noise() -> None:
    st.caption("Real quantum hardware is noisy. This lab reruns the Grover mutation search on a short "
               "window with depolarizing gate errors and readout errors, like a NISQ device.")
    L = len(reference)
    c1, c2 = st.columns([1, 3])
    W = c1.radio("Window (bp)", [w for w in (8, 16) if w <= L] or [L], horizontal=True,
                 help="16 bp is ~40× slower under noise (thousands of CX gates)")
    first = classic_mut[0] if classic_mut else 0
    default_start = min(max(0, first - W // 2), L - W)
    start = c2.slider("Window start", 0, L - W, default_start, key=f"noise_start_{W}_{L}") if L > W else 0
    ref_w, sam_w = reference[start:start + W], sample[start:start + W]
    marked = [i for i in range(W) if ref_w[i] != sam_w[i]]
    show_seq(f"Window {start}–{start + W - 1} (sample)", sam_w, set(marked))
    if not marked:
        st.warning("No mutations inside this window: move the window so there is something to find.")
        return

    c1, c2, c3 = st.columns(3)
    p1 = c1.select_slider("1-qubit gate error", [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01], 0.0001)
    p2 = c2.select_slider("2-qubit (CX) gate error", [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05], 0.001)
    ro = c3.select_slider("Readout error", [0.0, 0.005, 0.01, 0.02, 0.05, 0.1], 0.01)
    n = index_bits(W)
    iters = optimal_iterations(n, len(marked))
    phys = cached_physical_stats(ref_w, sam_w, iters)
    cc = st.columns(4)
    cc[0].metric("Grover iterations", iters)
    cc[1].metric("CX gates (after transpile)", f"{phys['cx']:,}")
    cc[2].metric("Circuit depth", f"{phys['depth']:,}")
    cc[3].metric("Expected CX errors", f"{phys['cx'] * p2:.1f}",
                 help="CX count × CX error rate. Above ~1, the result is mostly noise.")

    key = (ref_w, sam_w, iters, shots, p1, p2, ro)
    if st.button("Run noisy simulation", type="primary"):
        with st.spinner("Simulating with noise…"):
            ideal = cached_mutation_distribution(ref_w, sam_w, iters, shots)
            noisy = cached_noisy_distribution(ref_w, sam_w, iters, shots, p1, p2, ro)
        st.session_state.noise_result = (key, ideal, noisy)
    res = st.session_state.get("noise_result")
    if res and res[0] == key:
        _, ideal, noisy = res
        pi = sum(ideal.get(i, 0) for i in marked) / shots
        pn = sum(noisy.get(i, 0) for i in marked) / shots
        rand = len(marked) / 2**n
        cc = st.columns(3)
        cc[0].metric("P(marked), ideal", f"{pi:.1%}")
        cc[1].metric("P(marked), noisy", f"{pn:.1%}", f"{pn - pi:+.1%}")
        cc[2].metric("Random guess", f"{rand:.1%}")
        st.altair_chart(compare_chart(ideal, noisy, n, set(marked)), width="stretch")
        if pn < rand * 1.5:
            st.error("The signal is lost: noisy Grover is barely better than guessing. "
                     "This is why error correction / mitigation matters.")
        elif pn < pi * 0.7:
            st.warning("The signal is degraded but still above random guessing.")
        else:
            st.success("The amplification survives this noise level.")

    with st.container(border=True):
        st.markdown("**📉 Noise sweep**: success probability as CX error grows (1q error = CX/10)")
        sweep = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
        if W > 8:
            st.caption("⚠️ A sweep on a 16 bp window takes a few minutes.")
        if st.button("Run noise sweep"):
            pts, bar = [], st.progress(0.0)
            for i, e in enumerate(sweep):
                bar.progress(i / len(sweep), f"CX error {e}…")
                d = cached_noisy_distribution(ref_w, sam_w, iters, shots, e / 10, e, ro)
                pts.append({"CX error": str(e), "P(marked)": sum(d.get(j, 0) for j in marked) / shots})
            bar.empty()
            st.session_state.sweep_result = ((ref_w, sam_w, iters, shots, ro), pd.DataFrame(pts))
        res = st.session_state.get("sweep_result")
        if res and res[0] == (ref_w, sam_w, iters, shots, ro):
            df = res[1]
            line = alt.Chart(df).mark_line(point=True, color="#e05d5d").encode(
                x=alt.X("CX error:N", sort=None), y=alt.Y("P(marked):Q", axis=alt.Axis(format="%"),
                                                          scale=alt.Scale(domain=[0, 1])),
                tooltip=["CX error", alt.Tooltip("P(marked):Q", format=".1%")])
            rule = alt.Chart(pd.DataFrame({"y": [len(marked) / 2**n]})).mark_rule(
                strokeDash=[4, 4], color=MUTED).encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=260), width="stretch")
            st.caption("Dashed line = random guessing.")


# ================================================================ Benchmark
def page_benchmark() -> None:
    st.caption("Scaling study: how qubits, gates and Grover iterations grow with sequence length, "
               "compared with a classical scan.")
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    lengths = c1.multiselect("Sequence lengths (bp)", [8, 16, 32, 64, 128], [8, 16, 32, 64])
    snps = c2.number_input("Mutations per sequence", 1, 6, 2)
    physical = c3.checkbox("Physical CX count", help="Transpile to hardware gates (slower)")
    if 128 in lengths:
        st.caption("⚠️ 128 bp can take a minute or more.")
    if st.button("▶ Run benchmark", type="primary", disabled=not lengths):
        rows, bar = [], st.progress(0.0)
        for i, L in enumerate(sorted(lengths)):
            bar.progress(i / len(lengths), f"Benchmarking {L} bp…")
            rows.append(cached_benchmark_row(L, int(snps), seed, physical))
        bar.empty()
        st.session_state.bench = pd.DataFrame(rows)

    df = st.session_state.get("bench")
    if df is None:
        return
    st.dataframe(df, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    q = df.melt(id_vars="length", value_vars=["Grover iterations", "classical checks", "√N"],
                var_name="metric", value_name="count")
    c1.markdown("**Queries: quantum vs classical**")
    c1.altair_chart(alt.Chart(q).mark_line(point=True).encode(
        x=alt.X("length:Q", title="Sequence length (bp)", scale=alt.Scale(type="log", base=2)),
        y=alt.Y("count:Q", title="Queries"),
        color=alt.Color("metric:N", scale=alt.Scale(domain=["Grover iterations", "classical checks", "√N"],
                                                    range=[QUANTUM, MUTED, "#2e9d5b"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum.metric == "√N", alt.value([4, 4]), alt.value([0])),
        tooltip=["length", "metric", "count"]).properties(height=280), width="stretch")
    gate_cols = [c for c in ("oracle gates (logical)", "oracle CX (physical)") if c in df]
    g = df.melt(id_vars="length", value_vars=gate_cols, var_name="metric", value_name="gates")
    c2.markdown("**Oracle size (data-loading cost)**")
    c2.altair_chart(alt.Chart(g).mark_line(point=True).encode(
        x=alt.X("length:Q", title="Sequence length (bp)", scale=alt.Scale(type="log", base=2)),
        y=alt.Y("gates:Q", title="Gates"),
        color=alt.Color("metric:N", legend=alt.Legend(orient="top", title=None)),
        tooltip=["length", "metric", "gates"]).properties(height=280), width="stretch")
    st.info("Grover iterations grow like √N, but the oracle's gate count grows linearly with length "
            "(every position must be loaded), so end-to-end there is no speedup on classical data.")


# ==================================================================== About
def page_about() -> None:
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
- **Noise**: as the Noise Lab shows, realistic error rates destroy the signal beyond ~8 bp without error correction.
- Sequence length is capped at 128 bp for simulation speed.
        """
    )


# --------------------------------------------------------------- navigation
# (title, function, material icon, one-line description shown under the page title)
PAGE_SPECS = {
    "": [("Encoding", page_encoding, "memory", "How DNA is mapped onto qubit registers")],
    "Algorithms": [
        ("Mutations", page_mutations, "biotech", "Grover search for positions where the sample differs"),
        ("Motifs", page_motifs, "pattern", "Grover search for every occurrence of a motif"),
        ("Similarity", page_similarity, "compare_arrows", "Swap-test similarity of k-mer profiles"),
    ],
    "Experiments": [
        ("Noise Lab", page_noise, "graphic_eq", "How realistic hardware noise affects the search"),
        ("Benchmark", page_benchmark, "speed", "Scaling of qubits, gates and queries with sequence length"),
    ],
    "About": [("How it works", page_about, "info", "Pipeline and limitations")],
}
DESCRIPTIONS = {title: desc for specs in PAGE_SPECS.values() for title, _, _, desc in specs}
nav = st.navigation(
    {section: [st.Page(fn, title=title, icon=f":material/{icon}:",
                       url_path=title.lower().replace(" ", "-"), default=(title == "Encoding"))
               for title, fn, icon, _ in specs]
     for section, specs in PAGE_SPECS.items()},
    position="top",
)
st.markdown(f'<div class="page-head"><h2>{nav.title}</h2><p>{DESCRIPTIONS[nav.title]}</p></div>',
            unsafe_allow_html=True)
if data_problem and nav.title != "How it works":
    getattr(st, data_problem[0])(data_problem[1])
else:
    nav.run()
