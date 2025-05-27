from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    text: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: int
    content: str
    score: float

class QueryResponse(BaseModel):
    results: List[SearchResult]
