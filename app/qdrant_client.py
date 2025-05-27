import os
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# Connect to qdrant cloud
client = QdrantClient(
    url=os.environ.get("QDRANT_HOST"),
    api_key=os.environ.get("QDRANT_API_KEY")
)

def search_qdrant(vector, top_k=5, collection_name="content_vectors"):
    hits = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128)  # optional
    )
    return hits
