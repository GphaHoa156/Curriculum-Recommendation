from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.embedding import embed_text
from app.qdrant_client import search_qdrant

app = FastAPI()

# Kết nối template và static folder
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search_web", response_class=HTMLResponse)
def search_web(request: Request, text: str = Form(...)):
    vector = embed_text(text)
    hits = search_qdrant(vector, top_k=5)
    results = [{"content": h.payload.get("content", ""), "score": h.score, "id": h.id} for h in hits]
    return templates.TemplateResponse("index.html", {"request": request, "results": results})
