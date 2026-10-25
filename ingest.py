from rag_pipeline import (EmbeddingManager, VectorStore, process_all_jsons,
                          process_all_pdfs, split_documents)


def ingest(pdf_dir=None, json_dir=None, persist_dir="./data/vector_store",
           collection_name="knowledge_base", reset_collection=False):
    docs = []
    if pdf_dir:
        docs += process_all_pdfs(pdf_dir)
    if json_dir:
        docs += process_all_jsons(json_dir)
    if not docs:
        print("[!] Nothing to ingest.")
        return
    chunks = split_documents(docs)
    store = VectorStore(collection_name=collection_name, persistent_directory=persist_dir)
    if reset_collection:
        store.reset_collection()
    embeddings = EmbeddingManager().generate_embeddings([c.page_content for c in chunks])
    store.add_documents(chunks, embeddings)
