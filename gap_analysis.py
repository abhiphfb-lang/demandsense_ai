"""
04 - Gap Analysis

Compares topic momentum (external demand, signal_detection.py) vs. library
coverage (internal supply, via the KG from knowledge_graph.py) to rank
under-covered-vs-market-interest topics.

Output: data/processed/gap_scores.csv, consumed by agent_orchestration.py.
"""

import os

import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def coverage_for_topic(G, topic: str) -> int:
    topic_node = f"topic:{topic}"
    if topic_node not in G:
        return 0
    doc_count = 0
    for cat_node in G.predecessors(topic_node):
        doc_count += sum(1 for _ in G.predecessors(cat_node))
    return doc_count


def normalize(series):
    rng = series.max() - series.min()
    if rng == 0:
        return series * 0
    return (series - series.min()) / rng


def compute_gap_scores(topic_momentum_df, G):
    topic_momentum_df = topic_momentum_df.copy()
    topic_momentum_df["coverage"] = topic_momentum_df["topic"].apply(lambda t: coverage_for_topic(G, t))

    gap_df = topic_momentum_df[topic_momentum_df["coverage"] >= 0].copy()
    gap_df["norm_momentum"] = normalize(gap_df["momentum"])
    gap_df["norm_coverage"] = normalize(gap_df["coverage"])
    gap_df["gap_score"] = gap_df["norm_momentum"] - gap_df["norm_coverage"]

    return gap_df.sort_values("gap_score", ascending=False)


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    topic_momentum_df = pd.read_csv(os.path.join(PROCESSED_DIR, "topic_momentum.csv"))
    G = nx.read_graphml(os.path.join(DATA_DIR, "kg", "graph.graphml"))

    gap_df = compute_gap_scores(topic_momentum_df, G)

    cols = ["topic", "momentum", "coverage", "gap_score"]
    if "source" in gap_df.columns:
        cols.insert(1, "source")
    print(gap_df[cols].head(20))

    gap_df.to_csv(os.path.join(PROCESSED_DIR, "gap_scores.csv"), index=False)
    print("Saved:", os.path.join(PROCESSED_DIR, "gap_scores.csv"))
    return gap_df


if __name__ == "__main__":
    run()
