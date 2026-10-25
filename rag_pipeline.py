import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from prompt import MASTER_PROMPTT


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Loading ────────────────────────────────────────────────────────────────
def process_all_pdfs(pdf_directory) -> List[Document]:
    docs = []
    for pdf in Path(pdf_directory).glob("**/*.pdf"):
        try:
            pages = PyPDFLoader(str(pdf)).load()
            for d in pages:
                d.metadata.update(source_file=pdf.name, file_type="pdf")
            docs.extend(pages)
        except Exception as e:
            print(f"Error loading {pdf.name}: {e}")
    return docs


def flatten_json_to_text(data, prefix="") -> List[str]:
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            lines += flatten_json_to_text(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines += flatten_json_to_text(item, f"{prefix}[{i}]")
    elif isinstance(data, (str, int, float, bool)):
        lines.append(f"{prefix}: {data}")
    return lines


def process_all_jsons(json_directory) -> List[Document]:
    docs = []
    for jf in Path(json_directory).glob("**/*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            for idx, rec in enumerate(data if isinstance(data, list) else [data]):
                if not isinstance(rec, dict):
                    continue
                content = "\n".join(flatten_json_to_text(rec))
                if content.strip():
                    docs.append(Document(page_content=content, metadata={
                        "source_file": jf.name, "file_type": "json", "record_index": idx}))
        except Exception as e:
            print(f"Error loading {jf.name}: {e}")
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# ── Embeddings + vector store ──────────────────────────────────────────────
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class VectorStore:
    def __init__(self, collection_name="knowledge_base", persistent_directory="./data/vector_store"):
        self.collection_name = collection_name
        os.makedirs(persistent_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_directory)
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_collection()

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of docs must match number of embeddings")
        ids, metas, texts, vecs = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            meta = dict(doc.metadata)
            meta.update(doc_index=i, content_length=len(doc.page_content),
                        content_hash=content_hash(doc.page_content))
            metas.append(meta)
            texts.append(doc.page_content)
            vecs.append(emb.tolist())
        self.collection.add(ids=ids, embeddings=vecs, metadatas=metas, documents=texts)
        print(f"Added {len(ids)} chunks. Collection size: {self.collection.count()}")


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrive(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        q = self.embedding_manager.generate_embeddings([query])[0]
        try:
            res = self.vector_store.collection.query(query_embeddings=[q.tolist()], n_results=top_k)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
        out = []
        if res["documents"] and res["documents"][0]:
            for i, (did, doc, meta, dist) in enumerate(zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])):
                score = 1 - dist  # cosine distance -> similarity
                if score >= score_threshold:
                    out.append({"id": did, "content": doc, "metadata": meta,
                                "similarity_score": score, "distance": dist, "rank": i + 1})
        return out


# ── RAG ────────────────────────────────────────────────────────────────────
def rag_advanced_v2(query, retriever, llm, *, retrieve_k=12, final_k=4, min_score=0.25,
                    max_context_chars=3500, return_context=False):
    def empty(msg):
        return {"answer": msg, "confidence": 0.0, "sources": [],
                "context": "" if return_context else None}

    retrieved = retriever.retrive(query, top_k=retrieve_k, score_threshold=0.0)
    if not retrieved:
        return empty("I don't know. No relevant information was found.")

    filtered = sorted([d for d in retrieved if d["similarity_score"] >= min_score],
                      key=lambda d: d["similarity_score"], reverse=True)
    if not filtered:
        return empty("I don't know. The available information is too weak to answer reliably.")

    seen, selected = set(), []
    for d in filtered:
        h = d["metadata"].get("content_hash")
        if h and h in seen:
            continue
        seen.add(h)
        selected.append(d)
        if len(selected) >= final_k:
            break

    chunks, total = [], 0
    for d in selected:
        c = d["content"].strip()
        if total + len(c) > max_context_chars:
            break
        chunks.append(c)
        total += len(c)
    context = "\n\n---\n\n".join(chunks)

    prompt = f"{MASTER_PROMPTT}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    answer = llm.invoke(prompt).content.strip()

    out = {
        "answer": answer,
        "confidence": round(max(d["similarity_score"] for d in selected), 3),
        "sources": [{"source": d["metadata"].get("source_file", "unknown"),
                     "score": round(d["similarity_score"], 3),
                     "preview": d["content"][:200] + "..."} for d in selected],
    }
    if return_context:
        out["context"] = context
    return out


def build_rag_v2(persist_dir="./data/vector_store", collection_name="knowledge_base",
                 model_name="openai/gpt-oss-120b"):
    load_dotenv()
    retriever = RAGRetriever(VectorStore(collection_name, persist_dir), EmbeddingManager())
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model=model_name,
                   temperature=0.1, max_tokens=1024)

    def ask(question, *, retrieve_k=12, final_k=4, min_score=0.25,
            max_context_chars=3500, return_context=False):
        return rag_advanced_v2(question, retriever, llm, retrieve_k=retrieve_k, final_k=final_k,
                               min_score=min_score, max_context_chars=max_context_chars,
                               return_context=return_context)
    return ask
