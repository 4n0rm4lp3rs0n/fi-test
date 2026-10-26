import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GA Evolution Explorer v2", layout="wide")
st.title("NASBench-101 GA · Evolution Explorer")
st.caption("Checkpoint 1: inspect population snapshots and genome distribution without changing your GA or original viewer.")

# Safety: load a copy or upload a trace; this app never writes to the source trace.
default_trace = Path("evolution_trace.json")
uploaded = st.sidebar.file_uploader("Load evolution_trace.json", type=["json"])
if uploaded is not None:
    events = json.load(uploaded)
elif default_trace.exists():
    events = json.loads(default_trace.read_text(encoding="utf-8"))
else:
    st.warning("Put evolution_trace.json beside this file, or upload it in the sidebar.")
    st.stop()

if not isinstance(events, list) or not events:
    st.error("Trace must be a non-empty JSON list of event objects.")
    st.stop()

# Population snapshots are the source of truth for generation-level charts.
population_events = [
    e for e in events
    if e.get("phase") in ("initialization", "generation_complete")
    and isinstance(e.get("population"), list)
]
pop_rows = []
for e in population_events:
    phase = e.get("phase")
    generation = int(e.get("generation", 0))
    label = "Initialization (G0)" if phase == "initialization" else "Generation {}".format(generation)
    for slot, genome in enumerate(e.get("population", []), start=1):
        ops = genome.get("operations", [])
        if isinstance(ops, str):
            try:
                ops = json.loads(ops)
            except Exception:
                ops = [ops]
        code = str(genome.get("code") or "")
        signature = genome.get("signature") or "{}|{}".format(code, ",".join(ops if isinstance(ops, list) else []))
        pop_rows.append({
            "population_label": label,
            "generation": generation,
            "phase": phase,
            "slot": slot,
            "id": genome.get("id") or "slot-{}".format(slot),
            "code": code,
            "operations": ops if isinstance(ops, list) else [ops],
            "fitness": genome.get("fitness"),
            "signature": signature,
            "metrics": genome.get("metrics", {}),
        })

pop = pd.DataFrame(pop_rows)
if pop.empty:
    st.error(
        "No population snapshots found. This view needs `initialization` and "
        "`generation_complete` events with a `population` list. Existing event-only traces "
        "cannot reconstruct populations that were never saved."
    )
    st.stop()

# If the same generation snapshot was logged more than once, keep the last row for each ID/slot.
pop = pop.drop_duplicates(subset=["population_label", "id", "slot"], keep="last")
labels = list(dict.fromkeys(pop["population_label"].tolist()))
with st.sidebar:
    st.header("Population selection")
    chosen_label = st.selectbox("Generation snapshot", labels, index=0)
    show_duplicates = st.checkbox("Show duplicate architectures", value=True)

view = pop[pop["population_label"] == chosen_label].copy()
if not show_duplicates:
    view = view.drop_duplicates(subset=["signature"], keep="first")

fitness = pd.to_numeric(view["fitness"], errors="coerce")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Individuals shown", len(view))
c2.metric("Unique architectures", view["signature"].nunique())
c3.metric("Duplicate rate", "{:.1f}%".format(100 * (1 - view["signature"].nunique() / max(len(view), 1))))
c4.metric("Mean fitness", "—" if fitness.dropna().empty else "{:.5f}".format(fitness.mean()))

tab_pop, tab_dist, tab_compare, tab_events = st.tabs([
    "Genome population",
    "Gene distribution",
    "Generation comparison",
    "Trace events"
])

with tab_pop:
    st.subheader("Every genome in this population")
    st.write("Each row is one individual. Hover over a point or select a genome below to inspect its genotype.")
    plot_view = view.copy()
    plot_view["fitness_numeric"] = pd.to_numeric(plot_view["fitness"], errors="coerce")
    if plot_view["fitness_numeric"].notna().any():
        fig = px.scatter(
            plot_view,
            x="slot",
            y="fitness_numeric",
            color="fitness_numeric",
            hover_name="id",
            hover_data={"code": True, "operations": True, "signature": True, "slot": True, "fitness_numeric": ":.6f"},
            labels={"slot": "Population slot", "fitness_numeric": "Fitness", "color": "Fitness"},
            title="Fitness distribution across individuals"
        )
        fig.update_layout(coloraxis_colorbar_title="Fitness")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Fitness values are missing in this snapshot; genotype distribution is still available below.")

    genome_options = ["{} · slot {}".format(r.id, r.slot) for r in view.itertuples()]
    selected_genome_label = st.selectbox("Inspect an individual", genome_options, key="selected_genome")
    selected_slot = int(selected_genome_label.rsplit(" ", 1)[-1])
    selected = view[view["slot"] == selected_slot].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Connection genes (`code`)**")
        code_bits = [b for b in str(selected["code"]).replace("-", "") if b in "01"]
        if code_bits:
            bit_fig = go.Figure(data=go.Heatmap(
                z=[[int(b) for b in code_bits]],
                x=["b{}".format(i + 1) for i in range(len(code_bits))],
                y=[str(selected["id"])],
                zmin=0, zmax=1,
                colorscale=[[0, "#e8edf3"], [1, "#2563eb"]],
                showscale=False,
                hovertemplate="%{x} = %{z}<extra></extra>"
            ))
            bit_fig.update_layout(height=170, margin=dict(l=10, r=10, t=20, b=20), xaxis_title="Flattened connection bit")
            st.plotly_chart(bit_fig, use_container_width=True)
            st.caption("Bit order follows the flattened `code` string after removing separators. This is a genotype view, not a drawing of the CNN graph.")
        st.markdown("**Operation genes**")
        ops = selected["operations"] or []
        if ops:
            op_df = pd.DataFrame({
                "vertex": ["input"] + ["vertex {}".format(i) for i in range(1, max(len(ops)-1, 1))] + ["output"],
                "operation": ops
            })
            st.dataframe(op_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Selected genome record**")
        st.json({
            "generation": selected["population_label"],
            "id": selected["id"],
            "fitness": selected["fitness"],
            "code": selected["code"],
            "operations": selected["operations"],
            "metrics": selected["metrics"],
            "signature": selected["signature"],
        })

with tab_dist:
    st.subheader("Which genes are becoming common?")
    bit_rows, op_rows = [], []
    for row in pop.itertuples():
        flat = "".join(ch for ch in str(row.code) if ch in "01")
        for i, bit in enumerate(flat, start=1):
            bit_rows.append({
                "generation": row.population_label,
                "generation_order": labels.index(row.population_label),
                "bit": "b{}".format(i),
                "bit_index": i,
                "value": int(bit),
            })
        ops = row.operations if isinstance(row.operations, list) else []
        for i, op in enumerate(ops[1:-1], start=1):
            op_rows.append({
                "generation": row.population_label,
                "generation_order": labels.index(row.population_label),
                "vertex": "v{}".format(i),
                "operation": str(op),
            })

    if bit_rows:
        bits = pd.DataFrame(bit_rows)
        bit_freq = bits.groupby(["generation", "generation_order", "bit", "bit_index"], as_index=False)["value"].mean()
        bit_freq = bit_freq.sort_values(["generation_order", "bit_index"])
        fig = px.line(
            bit_freq, x="bit_index", y="value", color="generation", markers=True,
            hover_data=["bit"], labels={"bit_index": "Connection-bit index", "value": "Fraction equal to 1"},
            title="Connection-bit frequency across populations"
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        heat = bit_freq.pivot(index="generation", columns="bit", values="value").reindex(labels)
        st.markdown("**Connection-bit frequency table**")
        st.dataframe(heat.round(3), use_container_width=True)
    if op_rows:
        op = pd.DataFrame(op_rows)
        op_count = op.groupby(["generation", "generation_order", "vertex", "operation"]).size().reset_index(name="count")
        op_count["proportion"] = op_count["count"] / op_count.groupby(["generation", "vertex"])["count"].transform("sum")
        fig = px.bar(
            op_count, x="vertex", y="proportion", color="operation",
            facet_col="generation", facet_col_wrap=2, barmode="stack",
            category_orders={"generation": labels},
            title="Operation mix at each internal vertex"
        )
        fig.update_yaxes(range=[0, 1], title="Fraction of individuals")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No internal operation genes found in the population snapshots.")

with tab_compare:
    st.subheader("Is the population improving or collapsing to duplicates?")
    summary = pop.copy()
    summary["fitness_numeric"] = pd.to_numeric(summary["fitness"], errors="coerce")
    summary["generation_order"] = summary["population_label"].map({label: i for i, label in enumerate(labels)})
    summary_rows = []
    for label, group in summary.groupby("population_label", sort=False):
        f = group["fitness_numeric"].dropna()
        summary_rows.append({
            "generation": label,
            "generation_order": labels.index(label),
            "population_size": len(group),
            "unique_architectures": group["signature"].nunique(),
            "duplicate_rate": 1 - group["signature"].nunique() / max(len(group), 1),
            "mean_fitness": f.mean() if not f.empty else None,
            "best_fitness": f.max() if not f.empty else None,
            "worst_fitness": f.min() if not f.empty else None,
            "fitness_std": f.std() if len(f) > 1 else 0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("generation_order")
    lcol, rcol = st.columns(2)
    with lcol:
        fig = px.line(summary_df, x="generation", y=["mean_fitness", "best_fitness", "worst_fitness"],
                      markers=True, title="Fitness by generation")
        st.plotly_chart(fig, use_container_width=True)
    with rcol:
        fig = px.line(summary_df, x="generation", y=["unique_architectures", "population_size"],
                      markers=True, title="Diversity and population size")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary_df.drop(columns=["generation_order"]).round(5), use_container_width=True, hide_index=True)

with tab_events:
    st.subheader("Original trace events (read-only)")
    event_df = pd.json_normalize(events, sep=".")
    phase_values = sorted(event_df["phase"].dropna().astype(str).unique()) if "phase" in event_df else []
    selected_phases = st.multiselect("Filter event phases", phase_values, default=phase_values, key="raw_phase_filter")
    if selected_phases and "phase" in event_df:
        event_df = event_df[event_df["phase"].isin(selected_phases)]
    st.write("{} events".format(len(event_df)))
    st.dataframe(event_df, use_container_width=True)
    st.download_button(
        "Download filtered events",
        event_df.to_json(orient="records", indent=2),
        file_name="filtered_evolution_trace.json",
        mime="application/json"
    )
