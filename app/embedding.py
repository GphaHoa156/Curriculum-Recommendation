from sentence_transformers import SentenceTransformer

# Load model 
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def embed_text(text: str):
    """
    Nhận input là 1 đoạn văn, trả về embedding vector dạng list[float]
    """
    return model.encode([text])[0].tolist()
