import os
import time
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient
from utils import (
    encode_query_to_vector,
    search_qdrant_vector,
    rerank_candidates,
    map_passages_to_payload
)

# Initialize FastAPI application
app = FastAPI()

# Connect to Qdrant using environment variables for host and API key
client = QdrantClient(
    url=os.environ["QDRANT_HOST"],
    api_key=os.environ["QDRANT_API_KEY"]
)

# Mount static files and HTML templates
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
templates = Jinja2Templates(directory="src/app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render the home page.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        TemplateResponse: Rendered index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, custom_query: str = Form(...), language: str = Form(None)):
    """
    Handle search requests: encode query, perform vector search in Qdrant, rerank results, and return top matches.

    Args:
        request (Request): The incoming HTTP request.
        custom_query (str): The user-entered search query.
        language (str, optional): Optional language filter.

    Returns:
        TemplateResponse: Search results rendered in index.html or an error message.
    """
    start = time.time()

    # Handle empty input
    if not custom_query.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please fill your query."
        })

    # Encode the query into a vector
    try:
        query_vector = encode_query_to_vector(custom_query)
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error occured when calling API /encode: {e}"
        })
    print("Encode time: ", time.time() - start)

    # Search in Qdrant vector database
    try:
        search_result = search_qdrant_vector(
            client=client, 
            collection_name="content_vectors", 
            vector=query_vector, 
            language=language
        )
    except Exception as e:
        print("Encounter error when reranking!", e)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error occured with Qdrant: {e}"
        })
    print("Completed search with input: ", custom_query)
    print("search time: ", time.time() - start)

    # Extract text from payloads
    passages = [hit.payload["text"] for hit in search_result]

    # Rerank the retrieved passages
    try:
        reranked_texts = rerank_candidates(custom_query, passages)
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error occured with API /rerank: {e}"
        })
    print("Rerank time: ", time.time() - start)

    # Map reranked texts to original payloads
    payload_map = map_passages_to_payload(search_result)
    top_results = [{"payload": payload_map[txt], "score": i} for i, txt in enumerate(reranked_texts[:10])]

    print("Total search time:", time.time() - start)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": top_results,
        "topic": None
    })


@app.get("/filters/languages")
def get_languages() -> dict:
    """
    Retrieve all unique languages from the 'topic_vectors' collection.

    Returns:
        dict: A dictionary containing a sorted list of unique languages.
    """
    language_set = set()
    offset = None

    # Scroll through all points to collect languages
    while True:
        result, offset = client.scroll(
            collection_name="topic_vectors",  # update with actual collection name if needed
            with_payload=True,
            limit=100,
            offset=offset
        )
        for point in result:
            lang = point.payload.get("language")
            if lang:
                language_set.add(lang)
        if offset is None:
            break

    # Sort language list alphabetically (case-insensitive)
    sorted_languages = sorted(language_set, key=lambda x: x.lower())
    return {"languages": sorted_languages}
