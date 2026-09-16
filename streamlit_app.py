import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> str:
    client = get_inngest_client()
    event_ids = await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return event_ids[0]


async def send_rag_delete_event(source_id: str) -> str:
    client = get_inngest_client()
    event_ids = await client.send(
        inngest.Event(
            name="rag/delete_document",
            data={"source_id": source_id},
        )
    )
    return event_ids[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                detail = run.get("output") or run.get("error") or "no error detail returned by Inngest"
                raise RuntimeError(f"Function run {status}: {detail}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    path = save_uploaded_pdf(uploaded)
    with st.spinner(f"Ingesting {path.name} — chunking, embedding, and storing..."):
        try:
            event_id = asyncio.run(send_rag_ingest_event(path))
            output = wait_for_run_output(event_id)
        except (RuntimeError, TimeoutError) as e:
            st.error(f"Ingestion failed for {path.name}: {e}")
        else:
            ingested = output.get("ingested")
            if ingested is not None:
                st.success(f"Ingested {path.name} — {ingested} chunks stored.")
            else:
                st.success(f"Ingested {path.name}.")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Manage ingested documents")


def refresh_sources() -> list[str]:
    # Sources are read directly from Qdrant rather than via Inngest,
    # since listing documents doesn't need to go through a workflow step.
    from vector_db import QdrantStorage
    try:
        return QdrantStorage().list_sources()
    except Exception as e:
        st.warning(f"Could not reach Qdrant to list documents: {e}")
        return []


sources = refresh_sources()
if not sources:
    st.caption("No documents ingested yet.")
else:
    for src in sources:
        col1, col2 = st.columns([4, 1])
        col1.write(src)
        if col2.button("Delete", key=f"delete_{src}"):
            with st.spinner(f"Deleting {src}..."):
                try:
                    event_id = asyncio.run(send_rag_delete_event(src))
                    output = wait_for_run_output(event_id, timeout_s=30.0)
                except (RuntimeError, TimeoutError) as e:
                    st.error(f"Delete failed for {src}: {e}")
                else:
                    if output.get("deleted"):
                        st.success(f"Deleted {src}.")
                    else:
                        st.info(f"{src} was not found in the store.")
            st.rerun()

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int, source_ids: list[str] | None = None) -> str:
    client = get_inngest_client()
    data = {
        "question": question,
        "top_k": top_k,
    }
    if source_ids:
        data["source_ids"] = source_ids
    result = await client.send(
        inngest.Event(
            name="rag/rag_query_pdf_ai",  # fixed to match main.py's trigger
            data=data,
        )
    )
    return result[0]


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    scoped_sources = st.multiselect(
        "Limit search to specific document(s) (leave empty to search all)",
        options=sources if sources else [],
    )
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before switching.")
        else:
            with st.spinner("Sending event and generating answer..."):
                event_id = asyncio.run(
                    send_rag_query_event(question.strip(), int(top_k), scoped_sources or None)
                )
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                answer_sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if answer_sources:
            st.caption("Sources")
            for s in answer_sources:
                st.write(f"- {s}")
