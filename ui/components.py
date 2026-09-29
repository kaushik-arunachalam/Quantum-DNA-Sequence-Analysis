"""Reusable Streamlit/Altair building blocks for the Quantum DNA Analyzer UI."""
from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

BASE_COLORS = {"A": "#2e9d5b", "C": "#2f6fd6", "G": "#d9822b", "T": "#d14545"}
ACCENT = "#22d3ee"  # cyan: detected mutations / motifs / marked states
MUTED = "#8a94a6"
QUANTUM = "#6c5ce7"

CSS = """
<style>
  .seq {font-family: ui-monospace, Consolas, monospace; font-size: 15px;
        line-height: 1.9; word-break: break-all; letter-spacing: 1px;}
  .seq span {padding: 1px 2px; border-radius: 3px;}
  .seq .hit {outline: 2px solid #22d3ee; background: rgba(34,211,238,.16); font-weight: 700;
             box-shadow: 0 0 8px rgba(34,211,238,.45);}
  .seq-label {font-size: 12px; opacity: .7; margin-bottom: -6px;}

  /* Top navigation bar: blend the header into the page */
  [data-testid="stHeader"] {
    background: color-mix(in srgb, var(--st-background-color, #0e1117) 82%, transparent);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(139,124,246,.18);
  }
  [data-testid="stMainBlockContainer"] {padding-top: 4.75rem;}

  /* Page title + subtitle */
  .page-head {margin-bottom: 1.1rem; padding-bottom: .8rem;
              border-bottom: 1px solid rgba(128,128,128,.18);}
  .page-head h2 {margin: 0; padding: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -.01em;}
  .page-head p {margin: .25rem 0 0; opacity: .65; font-size: .95rem;}
</style>
"""


# ------------------------------------------------------------ sequence views
def seq_html(seq: str, highlight=frozenset(), notes: dict[int, str] | None = None) -> str:
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


# -------------------------------------------------------------------- charts
def prob_chart(counts: dict[int, int], n_index: int, marked: set[int], height: int = 280) -> alt.Chart:
    total = sum(counts.values()) or 1
    df = pd.DataFrame({
        "position": range(2**n_index),
        "probability": [counts.get(i, 0) / total for i in range(2**n_index)],
        "type": ["marked" if i in marked else "other" for i in range(2**n_index)],
    })
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("position:O", title="Position (index register)"),
        y=alt.Y("probability:Q", title="Measurement probability", axis=alt.Axis(format="%")),
        color=alt.Color("type:N", scale=alt.Scale(domain=["marked", "other"], range=[ACCENT, MUTED]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["position", alt.Tooltip("probability:Q", format=".1%"), "type"],
    ).properties(height=height)


def compare_chart(ideal: dict[int, int], noisy: dict[int, int], n_index: int, marked: set[int]) -> alt.Chart:
    rows = []
    for label, counts in (("Ideal", ideal), ("Noisy", noisy)):
        total = sum(counts.values()) or 1
        for i in range(2**n_index):
            rows.append({"position": i, "probability": counts.get(i, 0) / total, "run": label,
                         "marked": "marked" if i in marked else "other"})
    return alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("position:O", title="Position"),
        xOffset="run:N",
        y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format="%")),
        color=alt.Color("run:N", scale=alt.Scale(domain=["Ideal", "Noisy"], range=[ACCENT, "#e05d5d"]),
                        legend=alt.Legend(title=None, orient="top")),
        opacity=alt.condition(alt.datum.marked == "marked", alt.value(1), alt.value(0.45)),
        tooltip=["position", "run", "marked", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=280)


def iteration_curve(sim_points: list[tuple[int, float]], n_index: int, n_marked: int,
                    selected: int) -> alt.Chart:
    """Simulated success probability vs Grover iterations, with the sin² theory curve."""
    N = 2**n_index
    j_max = max(j for j, _ in sim_points)
    theta = math.asin(math.sqrt(n_marked / N)) if n_marked else 0.0
    theory = pd.DataFrame({
        "iterations": [j / 10 for j in range(0, j_max * 10 + 1)],
    })
    theory["probability"] = [math.sin((2 * j + 1) * theta) ** 2 for j in theory["iterations"]]
    sim = pd.DataFrame(sim_points, columns=["iterations", "probability"])
    sim["selected"] = sim["iterations"] == selected

    line = alt.Chart(theory).mark_line(color=MUTED, strokeDash=[5, 4]).encode(
        x=alt.X("iterations:Q", title="Grover iterations", scale=alt.Scale(domain=[0, j_max])),
        y=alt.Y("probability:Q", title="P(measure a marked position)",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
    )
    pts = alt.Chart(sim).mark_circle(size=90).encode(
        x="iterations:Q", y="probability:Q",
        color=alt.condition(alt.datum.selected, alt.value("#e05d5d"), alt.value(ACCENT)),
        tooltip=["iterations", alt.Tooltip("probability:Q", format=".1%")],
    )
    return (line + pts + pts.mark_line(color=ACCENT, opacity=0.6)).properties(height=260)


def draw_circuit(qc):
    fig = qc.draw("mpl", fold=-1, scale=0.7)
    fig.patch.set_facecolor("white")
    return fig


# ----------------------------------------------------------------- metrics
def score(pred, truth) -> tuple[float, float]:
    p, t = set(pred), set(truth)
    tp = len(p & t)
    return (tp / len(p) if p else 1.0, tp / len(t) if t else 1.0)
