
# viewer.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import networkx as nx


def draw_architecture(matrix, operations):
    """
    matrix: square adjacency matrix
    operations: one operation label per node
    """
    graph = nx.DiGraph()
    n = len(matrix)

    for i in range(n):
        operation = (
            str(operations[i])
            if i < len(operations)
            else f"Node {i}"
        )
        graph.add_node(i, label=f"{i}: {operation}")

    for i in range(n):
        for j in range(n):
            if matrix[i][j]:
                graph.add_edge(i, j)

    positions = nx.spring_layout(graph, seed=7)

    labels = nx.get_node_attributes(graph, "label")

    fig, ax = plt.subplots(figsize=(9, 5))
    nx.draw_networkx(
        graph,
        positions,
        labels=labels,
        with_labels=True,
        node_size=2400,
        font_size=8,
        arrows=True,
        ax=ax,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="GA Evolution Viewer",
    layout="wide",
)

st.title("Genetic Algorithm Evolution Viewer")
st.caption("Explore recorded NASBench-101 evolution events.")

trace_path = st.sidebar.text_input(
    "Trace file",
    "evolution_trace.json",
)

if not Path(trace_path).is_file():
    st.warning(
        f"Cannot find {trace_path}. Run the GA and save its trace first."
    )
    st.stop()

try:
    with open(trace_path, "r", encoding="utf-8") as f:
        events = json.load(f)
except (json.JSONDecodeError, OSError) as exc:
    st.error(f"Could not read trace: {exc}")
    st.stop()

if not isinstance(events, list) or not events:
    st.warning("The trace is empty or is not a JSON list of events.")
    st.stop()

df = pd.json_normalize(events, sep=".")



# Adapt to the fields your logger actually records.
if "generation" not in df.columns:
    st.error("Events need a 'generation' field.")
    st.stop()

if "phase" not in df.columns:
    st.error("Events need a 'phase' field.")
    st.stop()

generations = sorted(
    df["generation"].dropna().unique().tolist()
)
phases = sorted(
    df["phase"].dropna().astype(str).unique().tolist()
)

selected_generations = st.sidebar.multiselect(
    "Generations",
    generations,
    default=generations,
)
selected_phases = st.sidebar.multiselect(
    "Phases",
    phases,
    default=phases,
)

filtered = df[
    df["generation"].isin(selected_generations)
    & df["phase"].astype(str).isin(selected_phases)
]

c1, c2, c3 = st.columns(3)
c1.metric("Total events", len(events))
c2.metric("Filtered events", len(filtered))
c3.metric("Generations", df["generation"].nunique())

st.subheader("Event counts by phase")

counts = (
    filtered.groupby(["generation", "phase"])
    .size()
    .reset_index(name="event_count")
)


if not counts.empty:
    chart_data = counts.pivot(
        index="generation",
        columns="phase",
        values="event_count",
    ).fillna(0)

    st.bar_chart(chart_data)

st.subheader("Evolution timeline")

for idx, event in filtered.iterrows():
    generation = event["generation"]
    phase = event["phase"]
    step = event.get("step", idx)

    with st.expander(
        f"Step {step} · Generation {generation} · {phase}"
    ):
        st.json(events[int(idx)])

st.subheader("Raw event table")
st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download filtered events",
    data=filtered.to_json(
        orient="records", indent=2
    ),
    file_name="filtered_evolution_trace.json",
    mime="application/json",
)
