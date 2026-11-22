"""
05 - Agent Orchestration (LangGraph)

Wires signal detection, RAG, knowledge graph, and gap analysis into a single
LangGraph agent that plans which components a question needs, retrieves/
analyzes via those components, then synthesizes a cited answer.

Flow: plan -> signals -> rag -> kg -> gap -> synthesize, with each node
early-returning an empty result when the planner decided it isn't needed.
"""
import os
import re

import pandas as pd
import networkx as nx
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
try:
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\ANANDAB\Downloads\Hexion\.env")
except Exception as e:
    print(e)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

GAP_KEYWORDS = ["gap", "under-covered", "undercovered", "coverage", "covering"]
SIGNAL_KEYWORDS = ["emerging", "surge", "trend", "momentum", "recent signals", "signals"]

EXAMPLE_QUESTIONS = [
    "What topics related to AI in healthcare have surged recently?",
    "What are key issues APAC manufacturers faced this quarter?",
    "Which topics are under-covered versus market interest?",
    "Summarize recent signals around supply-chain resilience.",
]

_resources = {}


def load_resources():
    """Load and cache all data/model resources needed by the agent nodes."""
    if _resources:
        return _resources

    _resources["topic_momentum_df"] = pd.read_csv(os.path.join(DATA_DIR, "processed", "topic_momentum.csv"))
    _resources["gap_df"] = pd.read_csv(os.path.join(DATA_DIR, "processed", "gap_scores.csv"))
    _resources["G"] = nx.read_graphml(os.path.join(DATA_DIR, "kg", "graph.graphml"))
    _resources["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "vectorstore"))
    _resources["collection"] = client.get_or_create_collection("library")
    return _resources


def stub_llm(prompt: str) -> str:
    return (
        "[stub-llm synthesis - set OPENAI_ENDPOINT/OPENAI_MODEL in .env for a real LLM answer]\n"
        + prompt[-1200:]
    )


def _build_challenger_llm():
    from langchain_openai import ChatOpenAI
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    openai_endpoint = os.environ["OPENAI_ENDPOINT"]
    openai_model = os.environ["OPENAI_MODEL"]

    openai_token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )

    return ChatOpenAI(
        base_url=openai_endpoint, api_key=openai_token_provider, model=openai_model,
        use_responses_api=True, temperature=0,
    )


_llm = None


def get_llm():
    global _llm
    if _llm is None and os.environ.get("OPENAI_ENDPOINT") and os.environ.get("OPENAI_MODEL"):
        try:
            _llm = _build_challenger_llm()
        except Exception as e:
            print("Could not initialize Azure OpenAI LLM, falling back to stub:", e)
    return _llm


def call_llm(prompt: str) -> str:
    llm = get_llm()
    if llm is not None:
        return llm.invoke(prompt).content[0]["text"]
    return stub_llm(prompt)


def retrieve(question: str, k: int = 4):
    res = load_resources()
    query_emb = res["embedder"].encode([question]).tolist()
    results = res["collection"].query(query_embeddings=query_emb, n_results=k)
    hits = []
    for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        hits.append({"id": doc_id, "title": meta.get("title"), "category": meta.get("category"), "text": doc})
    return hits


def extract_keywords(question: str):
    words = re.findall(r"[a-zA-Z]{4,}", question.lower())
    stop = {"what", "which", "this", "that", "with", "have", "from", "about", "their", "recent"}
    return [w for w in words if w not in stop]


def plan_node(state: dict) -> dict:
    q = state["question"].lower()
    need_gap = any(k in q for k in GAP_KEYWORDS)
    need_signals = need_gap or any(k in q for k in SIGNAL_KEYWORDS)
    need_kg = need_gap or need_signals
    need_rag = True  # always ground the answer in the library when possible
    return {**state, "need_signals": need_signals, "need_rag": need_rag,
            "need_kg": need_kg, "need_gap": need_gap}


def signals_node(state: dict) -> dict:
    if not state.get("need_signals"):
        return {**state, "signals": []}
    res = load_resources()
    kws = extract_keywords(state["question"])
    topic_momentum_df = res["topic_momentum_df"]
    mask = topic_momentum_df["topic"].str.lower().apply(lambda t: any(k in t for k in kws))
    hits = topic_momentum_df[mask].sort_values("momentum", ascending=False).head(10)
    return {**state, "signals": hits.to_dict("records")}


def rag_node(state: dict) -> dict:
    if not state.get("need_rag"):
        return {**state, "rag_hits": []}
    return {**state, "rag_hits": retrieve(state["question"], k=4)}


def kg_node(state: dict) -> dict:
    if not state.get("need_kg"):
        return {**state, "kg_context": []}
    G = load_resources()["G"]
    kws = extract_keywords(state["question"])
    context = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "topic" and any(k in node.lower() for k in kws):
            cats = [n.split(":", 1)[1] for n in G.predecessors(node) if G.nodes[n].get("type") == "category"]
            if cats:
                context.append(f"Topic '{node.split(':', 1)[1]}' relates to library categories: {cats}")
    return {**state, "kg_context": context[:10]}


def gap_node(state: dict) -> dict:
    if not state.get("need_gap"):
        return {**state, "gap_results": []}
    gap_df = load_resources()["gap_df"]
    kws = extract_keywords(state["question"])
    if kws:
        mask = gap_df["topic"].str.lower().apply(lambda t: any(k in t for k in kws))
        subset = gap_df[mask] if mask.any() else gap_df
    else:
        subset = gap_df
    hits = subset.sort_values("gap_score", ascending=False).head(10)
    return {**state, "gap_results": hits.to_dict("records")}


def synthesize_node(state: dict) -> dict:
    parts = [f"Question: {state['question']}"]

    if state.get("signals"):
        parts.append("External signal momentum:\n" + "\n".join(
            f"- {s['topic']} (momentum={s['momentum']:.2f})" for s in state["signals"]))

    if state.get("kg_context"):
        parts.append("Knowledge graph context:\n" + "\n".join(state["kg_context"]))

    if state.get("gap_results"):
        parts.append("Coverage gaps (market interest vs. library):\n" + "\n".join(
            f"- {g['topic']}: gap_score={g['gap_score']:.2f} (momentum={g['momentum']:.2f}, coverage={g['coverage']})"
            for g in state["gap_results"]))

    sources = []
    if state.get("rag_hits"):
        sources = [{"id": h["id"], "title": h["title"]} for h in state["rag_hits"]]
        parts.append("Library context:\n" + "\n".join(
            f"[{h['id']}] {h['title']}: {h['text']}" for h in state["rag_hits"]))

    prompt = (
        "You are a research assistant for senior business leaders. Using only the context below, "
        "answer the question, cite library source ids in brackets where relevant, and explicitly call out "
        "any coverage gaps.\n\nContext:\n" + "\n\n".join(parts)
    )
    answer = call_llm(prompt)
    return {**state, "answer": answer, "sources": sources}


def build_agent():
    workflow = StateGraph(dict)
    workflow.add_node("plan", plan_node)
    workflow.add_node("signals", signals_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("kg", kg_node)
    workflow.add_node("gap", gap_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "signals")
    workflow.add_edge("signals", "rag")
    workflow.add_edge("rag", "kg")
    workflow.add_edge("kg", "gap")
    workflow.add_edge("gap", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def run():
    agent = build_agent()
    for q in EXAMPLE_QUESTIONS:
        result = agent.invoke({"question": q})
        print("-" * 80)
        print("Q:", q)
        print("-" * 80)
        print(result["answer"])
        print("Sources:", result.get("sources"))
        print()


if __name__ == "__main__":
    run()
