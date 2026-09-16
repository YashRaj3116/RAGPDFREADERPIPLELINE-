import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        store = QdrantStorage()

        # If this source was already ingested before, wipe its old vectors first
        # so re-uploading the same PDF overwrites rather than duplicates it.
        if store.source_exists(source_id):
            store.delete_source(source_id)

        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load_and_chunk-pdf", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/rag_query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5, source_ids: list[str] | None = None) -> RAGSearchResult:
         query_vec = embed_texts([question])[0]
         store = QdrantStorage()
         found = store.search(query_vec, top_k, source_ids=source_ids)
         return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source_ids = ctx.event.data.get("source_ids") or None

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k, source_ids), output_type=RAGSearchResult)

    if not found.contexts:
        return {
            "answer": "I couldn't find anything relevant  to your question in the inngested documents.",
            "sources": [],
            "num_contexts": 0
        }
    context_block = "\n\n".join(f"-{c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Use OpenAI adapter configured to target Ollama's local v1 endpoint
    adapter = ai.openai.Adapter(
        model="llama3",  # Make sure this model is pulled in Ollama
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        auth_key="ollama"  # Ollama doesn't require a strict key, but the adapter expects one
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,  # OpenAI style parameter supported by Ollama's /v1 endpoint
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    # Parse using OpenAI format since we used the OpenAI adapter
    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

@inngest_client.create_function(
    fn_id="RAG: Delete Document",
    trigger=inngest.TriggerEvent(event="rag/delete_document")
)
async def rag_delete_document(ctx: inngest.Context):
    def _delete(source_id: str) -> dict:
        store = QdrantStorage()
        existed = store.source_exists(source_id)
        if existed:
            store.delete_source(source_id)
        return {"source_id": source_id, "deleted": existed}

    source_id = ctx.event.data["source_id"]
    return await ctx.step.run("delete-source", lambda: _delete(source_id))


app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai, rag_delete_document],
)
