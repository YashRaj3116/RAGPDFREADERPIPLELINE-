from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection_name="docs", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection_name = collection_name
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def _source_filter(self, source_id: str) -> Filter:
        return Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        )

    def source_exists(self, source_id: str) -> bool:
        """Check whether any points are already stored for this source_id."""
        found, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._source_filter(source_id),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(found) > 0

    def delete_source(self, source_id: str) -> None:
        """Remove all points belonging to this source_id (e.g. before re-ingesting or on delete)."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._source_filter(source_id),
        )

    def list_sources(self) -> list[str]:
        """Return the distinct set of source_ids currently stored."""
        sources = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                src = (p.payload or {}).get("source")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
        return sorted(sources)

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k=5, source_ids=None):
        """
        Search for the top_k most relevant chunks.
        If source_ids is given (a list of source filenames), restrict the
        search to only those documents instead of the whole collection.
        """
        query_filter = None
        if source_ids:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchAny(any=list(source_ids)))]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=top_k
        ).points

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
            if source:
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
