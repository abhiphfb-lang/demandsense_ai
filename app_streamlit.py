import os
import re
from typing import TypedDict, List, Dict, Any

import pandas as pd
import networkx as nx
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv(r"C:\Users\ANANDAB\Downloads\Hexion\.env")

DATA_DIR = "data"


@st.cache_resource
def load_resources():
    topic_momentum_df = pd.read_csv(f"{DATA_DIR}/processed/topic_momentum.csv")
    gap_df = pd.read_csv(f"{DATA_DIR}/processed/gap_scores.csv")
    G = nx.read_graphml(f"{DATA_DIR}/kg/graph.graphml")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=f"{DATA_DIR}/vectorstore")
    collection = client.get_or_create_collection("library")
    return topic_momentum_df, gap_df, G, embedder, collection


topic_momentum_df, gap_df, G, embedder, collection = load_resources()


def stub_llm(prompt: str) -> str:
    return (
        "[stub-llm synthesis \u2014 set OPENAI_ENDPOINT/OPENAI_MODEL in .env for a real LLM answer]\n"
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


@st.cache_resource
def get_llm():
    if os.environ.get("OPENAI_ENDPOINT") and os.environ.get("OPENAI_MODEL"):
        try:
            return _build_challenger_llm()
        except Exception as e:
            st.warning(f"Could not initialize Azure OpenAI LLM, falling back to stub: {e}")
    return None


_llm = get_llm()


def call_llm(prompt: str) -> str:
    if _llm is not None:
        return _llm.invoke(prompt).content[0]["text"]
    return stub_llm(prompt)


def retrieve(question: str, k: int = 4):
    query_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=k)
    hits = []
    for doc, meta, doc_id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
        hits.append({"id": doc_id, "title": meta.get("title"), "category": meta.get("category"), "text": doc})
    return hits


class AgentState(TypedDict, total=False):
    question: str
    need_signals: bool
    need_rag: bool
    need_kg: bool
    need_gap: bool
    signals: List[Dict[str, Any]]
    rag_hits: List[Dict[str, Any]]
    kg_context: List[str]
    gap_results: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]


GAP_KEYWORDS = ["gap", "under-covered", "undercovered", "coverage", "covering"]
SIGNAL_KEYWORDS = ["emerging", "surge", "trend", "momentum", "recent signals", "signals"]


def plan_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    need_gap = any(k in q for k in GAP_KEYWORDS)
    need_signals = need_gap or any(k in q for k in SIGNAL_KEYWORDS)
    need_kg = need_gap or need_signals
    need_rag = True  # always ground the answer in the library when possible
    return {**state, "need_signals": need_signals, "need_rag": need_rag,
            "need_kg": need_kg, "need_gap": need_gap}


def extract_keywords(question: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{4,}", question.lower())
    stop = {"what", "which", "this", "that", "with", "have", "from", "about", "their", "recent"}
    return [w for w in words if w not in stop]


def signals_node(state: AgentState) -> AgentState:
    if not state.get("need_signals"):
        return {**state, "signals": []}
    kws = extract_keywords(state["question"])
    mask = topic_momentum_df["topic"].str.lower().apply(lambda t: any(k in t for k in kws))
    hits = topic_momentum_df[mask].sort_values("momentum", ascending=False).head(10)
    return {**state, "signals": hits.to_dict("records")}


def rag_node(state: AgentState) -> AgentState:
    if not state.get("need_rag"):
        return {**state, "rag_hits": []}
    return {**state, "rag_hits": retrieve(state["question"], k=4)}


def kg_node(state: AgentState) -> AgentState:
    if not state.get("need_kg"):
        return {**state, "kg_context": []}
    kws = extract_keywords(state["question"])
    context = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "topic" and any(k in node.lower() for k in kws):
            cats = [n.split(":", 1)[1] for n in G.predecessors(node) if G.nodes[n].get("type") == "category"]
            if cats:
                context.append(f"Topic '{node.split(':', 1)[1]}' relates to library categories: {cats}")
    return {**state, "kg_context": context[:10]}


def gap_node(state: AgentState) -> AgentState:
    if not state.get("need_gap"):
        return {**state, "gap_results": []}
    kws = extract_keywords(state["question"])
    if kws:
        mask = gap_df["topic"].str.lower().apply(lambda t: any(k in t for k in kws))
        subset = gap_df[mask] if mask.any() else gap_df
    else:
        subset = gap_df
    hits = subset.sort_values("gap_score", ascending=False).head(10)
    return {**state, "gap_results": hits.to_dict("records")}


def synthesize_node(state: AgentState) -> AgentState:
    parts = [f"Question: {state['question']}"]

    if state.get("signals"):
        parts.append("External signal momentum (GDELT):\n" + "\n".join(
            f"- {s['topic']} (momentum={s['momentum']:.2f})" for s in state["signals"]
        ))

    if state.get("kg_context"):
        parts.append("Knowledge graph context:\n" + "\n".join(state["kg_context"]))

    if state.get("gap_results"):
        parts.append("Coverage gaps (market interest vs. library):\n" + "\n".join(
            f"- {g['topic']}: gap_score={g['gap_score']:.2f} (momentum={g['momentum']:.2f}, coverage={g['coverage']})"
            for g in state["gap_results"]
        ))

    sources = []
    if state.get("rag_hits"):
        sources = [{"id": h["id"], "title": h["title"]} for h in state["rag_hits"]]
        parts.append("Library context:\n" + "\n".join(
            f"[{h['id']}] ({h['title']}): {h['text']}" for h in state["rag_hits"]
        ))

    prompt = (
        "You are a research assistant for senior business leaders. Using only the context below, "
        "answer the question, cite library source ids in brackets where relevant, and explicitly call out "
        "any coverage gaps.\n\nContext:\n" + "\n\n".join(parts)
    )

    answer = call_llm(prompt)
    return {**state, "answer": answer, "sources": sources}


@st.cache_resource
def build_agent():
    workflow = StateGraph(AgentState)
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


agent = build_agent()

st.set_page_config(page_title="Research Gap Assistant", page_icon="\U0001F50E", layout="wide")
st.title("\U0001F50E Research Gap Assistant")
st.caption("Ask a question \u2014 the agent pulls signals, internal library (RAG), knowledge graph, and gap analysis as needed.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Example questions")
    examples = [
        "What topics related to AI in healthcare have surged recently?",
        "What are key issues APAC manufacturers faced this quarter?",
        "Which topics are under-covered versus market interest?",
        "Summarize recent signals around supply-chain resilience.",
    ]
    chosen_example = None
    for ex in examples:
        if st.button(ex, use_container_width=True):
            chosen_example = ex

question = st.chat_input("Ask a research question...") or chosen_example

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.write(f"[{s['id']}] {s['title']}")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"question": question})
        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.write(f"[{s['id']}] {s['title']}")
    st.session_state.history.append({
        "question": question, "answer": result["answer"], "sources": result.get("sources", []),
    })
