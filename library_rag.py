"""
02 - Internal Library + RAG

Builds a simulated internal research library, embeds it into a local Chroma
vector store, and provides a rag_answer(question) function returning an
answer with cited sources.

Output: persisted Chroma DB at data/vectorstore/, library metadata at
data/library/library.csv (consumed by knowledge_graph.py / gap_analysis.py /
agent_orchestration.py).
"""

import os
import json

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(r"C:\Users\ANANDAB\Downloads\Hexion\.env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
LIBRARY_DIR = os.path.join(DATA_DIR, "library")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")

KAGGLE_PATH = os.path.join(LIBRARY_DIR, "News_Category_Dataset_v3.json")

SYNTHETIC_LIBRARY = [
    {"id": "doc1", "title": "Generative AI adoption in enterprise software", "category": "artificial intelligence",
     "text": "Enterprises are piloting generative AI copilots for software development, customer support, and internal knowledge search. Early ROI is strongest in code generation and support deflection."},
    {"id": "doc2", "title": "AI in healthcare diagnostics: a 2024 review", "category": "artificial intelligence",
     "text": "Machine learning models for radiology triage are moving from pilots to regulated deployments, with FDA clearances accelerating for imaging diagnostics tools."},
    {"id": "doc3", "title": "Supply chain resilience after recent disruptions", "category": "supply chain",
     "text": "Manufacturers are diversifying supplier bases and nearshoring to reduce exposure to single-region shocks, trading some cost efficiency for resilience."},
    {"id": "doc4", "title": "APAC manufacturing outlook", "category": "manufacturing",
     "text": "APAC manufacturers faced rising input costs and logistics bottlenecks this quarter, prompting automation investment to offset labor shortages."},
    {"id": "doc5", "title": "Generative AI regulation landscape", "category": "artificial intelligence",
     "text": "Regulators in the EU and US are converging on risk-tiered AI governance frameworks, with disclosure requirements for high-risk generative AI use cases."},
    {"id": "doc6", "title": "Semiconductor supply constraints easing", "category": "supply chain",
     "text": "Lead times for key semiconductor components have normalized, though advanced-node capacity remains concentrated and geopolitically sensitive."},
    {"id": "doc7", "title": "Workforce reskilling for automation", "category": "manufacturing",
     "text": "Manufacturers are investing in reskilling programs as robotics and AI-driven quality inspection reduce demand for manual line roles."},
    {"id": "doc8", "title": "Client demand signals in financial services", "category": "financial services",
     "text": "Banks are prioritizing fraud-detection AI and real-time payments infrastructure as top investment areas for the coming year."},
]

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client.get_or_create_collection("library")


def load_library(sample_size=3000):
    if os.path.exists(KAGGLE_PATH):
        records = []
        with open(KAGGLE_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                row = json.loads(line)
                records.append({
                    "id": f"huffpost_{i}",
                    "title": row.get("headline", ""),
                    "category": row.get("category", "").lower(),
                    "text": f"{row.get('headline', '')}. {row.get('short_description', '')}",
                })
        print(f"Loaded {len(records)} docs from Kaggle HuffPost dataset")
        return records
    print("Kaggle dataset not found at", KAGGLE_PATH, "-> using synthetic library")
    return SYNTHETIC_LIBRARY


def index_library(library_docs, collection, embedder):
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    new_docs = [d for d in library_docs if d["id"] not in existing_ids]

    if new_docs:
        embeddings = embedder.encode([d["text"] for d in new_docs]).tolist()
        collection.add(
            ids=[d["id"] for d in new_docs],
            documents=[d["text"] for d in new_docs],
            metadatas=[{"title": d["title"], "category": d["category"]} for d in new_docs],
            embeddings=embeddings,
        )
        print(f"Indexed {len(new_docs)} new documents")
    else:
        print("Collection already up to date, count=", collection.count())


def stub_llm(prompt: str) -> str:
    """Fallback 'LLM': templated extractive synthesis, used only if Azure OpenAI env vars are missing."""
    return (
        "[stub-llm synthesis - set OPENAI_ENDPOINT/OPENAI_MODEL in .env for a real LLM answer]\n"
        + prompt.split("Context:")[-1].strip()[:1000]
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
        return llm.invoke(prompt).content
    return stub_llm(prompt)


def retrieve(question: str, k: int = 4, collection=None, embedder=None):
    collection = collection or get_collection()
    embedder = embedder or get_embedder()
    query_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=k)
    hits = []
    for doc, meta, doc_id, dist in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0], results["distances"][0]
    ):
        hits.append({"id": doc_id, "title": meta.get("title"), "category": meta.get("category"),
                     "text": doc, "distance": dist})
    return hits


def rag_answer(question: str, k: int = 4):
    hits = retrieve(question, k=k)
    context = "\n\n".join(f"[{h['id']}] ({h['title']}): {h['text']}" for h in hits)
    prompt = (
        "Answer the question using only the context below. Cite source ids in brackets.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    answer = call_llm(prompt)
    sources = [{"id": h["id"], "title": h["title"]} for h in hits]
    return {"question": question, "answer": answer, "sources": sources}


def run():
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    library_docs = load_library()
    library_df = pd.DataFrame(library_docs)
    library_df.to_csv(os.path.join(LIBRARY_DIR, "library.csv"), index=False)

    embedder = get_embedder()
    collection = get_collection()
    index_library(library_docs, collection, embedder)

    result = rag_answer("What are key issues APAC manufacturers faced this quarter?")
    print(result["answer"])
    print("\nSources:", result["sources"])
    return library_df


if __name__ == "__main__":
    run()
