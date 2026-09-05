from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection_name="docs", dim=768):
 
        self.client = QdrantClient(url=url, timeout=30)
      
        self.collection_name = collection_name
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) 
            for i in range(len(ids))
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k=5):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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