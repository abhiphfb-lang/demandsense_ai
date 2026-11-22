"""
03 - Knowledge Graph

Builds a lightweight knowledge graph (networkx, no Neo4j dependency) linking:
topic (from GDELT/search-trend signals) <-> category (library) <-> document
(library docs), based on co-occurrence/category match.

Output: data/kg/graph.graphml, consumed by gap_analysis.py and
agent_orchestration.py.
"""

import os

import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
KG_DIR = os.path.join(DATA_DIR, "kg")


def build_graph(library_df, topic_momentum_df, top_n_topics=100):
    G = nx.DiGraph()

    for _, row in library_df.iterrows():
        doc_node = f"document:{row['id']}"
        cat_node = f"category:{row['category']}"
        G.add_node(doc_node, type="document", title=row["title"])
        G.add_node(cat_node, type="category")
        G.add_edge(doc_node, cat_node, relation="belongs_to")

    categories = library_df["category"].unique()
    top_topics = topic_momentum_df.sort_values("momentum", ascending=False).head(top_n_topics)

    for cat in categories:
        cat_node = f"category:{cat}"
        cat_words = set(cat.lower().replace("-", " ").split())
        for _, trow in top_topics.iterrows():
            topic = str(trow["topic"])
            topic_node = f"topic:{topic}"
            topic_words = set(topic.lower().replace("_", " ").split())
            if cat_words & topic_words or cat.lower() in topic.lower() or topic.lower() in cat.lower():
                G.add_node(topic_node, type="topic", momentum=float(trow["momentum"]))
                G.add_edge(cat_node, topic_node, relation="matched_by")

    return G


def topics_for_category(G, category: str):
    cat_node = f"category:{category}"
    if cat_node not in G:
        return []
    return [n.split(":", 1)[1] for n in G.successors(cat_node) if G.nodes[n]["type"] == "topic"]


def documents_for_category(G, category: str):
    cat_node = f"category:{category}"
    if cat_node not in G:
        return []
    return [n for n in G.predecessors(cat_node) if G.nodes[n]["type"] == "document"]


def categories_for_topic(G, topic: str):
    topic_node = f"topic:{topic}"
    if topic_node not in G:
        return []
    return [n.split(":", 1)[1] for n in G.predecessors(topic_node) if G.nodes[n]["type"] == "category"]


def run():
    os.makedirs(KG_DIR, exist_ok=True)

    topic_momentum_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "topic_momentum.csv"))
    library_df = pd.read_csv(os.path.join(DATA_DIR, "library", "library.csv"))

    G = build_graph(library_df, topic_momentum_df)
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    for cat in library_df["category"].unique():
        print(cat, "->", topics_for_category(G, cat))

    nx.write_graphml(G, os.path.join(KG_DIR, "graph.graphml"))
    print("Saved:", os.path.join(KG_DIR, "graph.graphml"))
    return G


if __name__ == "__main__":
    run()
