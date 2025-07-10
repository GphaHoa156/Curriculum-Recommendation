from utils import (
    encode_query_to_vector,
    search_qdrant_vector,
    rerank_candidates,
    map_passages_to_payload
)
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
client = QdrantClient(
    url=os.environ["QDRANT_HOST"],
    api_key=os.environ["QDRANT_API_KEY"]
)
custom_query = "chemistry"
query_vector = encode_query_to_vector(custom_query)

search_result = search_qdrant_vector(client=client, 
                                    collection_name="content_vectors", 
                                    vector=query_vector, 
                                    language="en")

passages = [hit.payload["text"] for hit in search_result]
reranked_texts = rerank_candidates(custom_query, passages)
payload_map = {hit.payload["text"]: hit.payload for hit in search_result if "text" in hit.payload}

for i, passage in enumerate(reranked_texts[:10]):
    payload = payload_map.get(passage, {})
    print(f"\n-- Kết quả #{i+1} --")
    print("Title:", payload.get("title"))

