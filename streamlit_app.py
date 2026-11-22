"""
06 - Streamlit Q&A App

Wraps the LangGraph agent from agent_orchestration.py (same
plan -> signals -> rag -> kg -> gap -> synthesize flow) behind a simple chat
UI.

Run via `streamlit run streamlit_app.py`, or via main.py.
"""

import streamlit as st

import agent_orchestration as orch


@st.cache_resource
def get_agent():
    orch.load_resources()
    return orch.build_agent()


agent = get_agent()

st.set_page_config(page_title="Research Gap Assistant", page_icon="\U0001F50E", layout="wide")
st.title("\U0001F50E Research Gap Assistant")
st.caption("Ask a question - the agent pulls signals, internal library (RAG), knowledge graph, and gap analysis as needed.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Example questions")
    chosen_example = None
    for ex in orch.EXAMPLE_QUESTIONS:
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
