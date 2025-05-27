import pandas as pd
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# === CẤU HÌNH ===
QDRANT_HOST = "https://fd60c49a-d5e7-43d6-8ddc-c1d2532774b9.us-east4-0.gcp.cloud.qdrant.io:6333" 
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.v65qZseHQZxaXYK6UyETSMI2suWA3iiZR6FAkFyvZCQ" 

# === TẢI DỮ LIỆU ===
def load_data_qdrant(COLLECTION_NAME, CSV_PATH, VECTORS_PATH):

    print("Reading data...")
    df = pd.read_csv(CSV_PATH)
    if COLLECTION_NAME == "topic_vectors":
        df = df[df['has_content'] == True].reset_index(drop=True)
    vectors = np.load(VECTORS_PATH)
    print("Sucess load csv and vectors")
    assert len(df) == len(vectors), "Number of lines in CSV and vectors does not match"

    # === KẾT NỐI QDRANT ===
    print("Connecting to Qdrant server...")
    client = QdrantClient(
        url=os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    print("Sucess connecting to Qdrant server")

    # === TẠO COLLECTION (nếu chưa có) ===
    print("Creating collection...")
    vector_size = vectors.shape[1]
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": vector_size, "distance": "Cosine"}
    )
    print("Sucess create collection")

    # === UPLOAD DỮ LIỆU THEO BATCH ===
    print("Data uploading...")
    batch_size = 512

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        points = [
            PointStruct(
                id=int(row.name),               
                vector=vectors[idx].tolist(),   
                payload=row.iloc[1:].to_dict()  
            )
            for idx, row in batch_df.iterrows()
        ]
        client.upload_points(collection_name=COLLECTION_NAME, points=points)

    print("✅ Upload hoàn tất" + COLLECTION_NAME)

load_data_qdrant(COLLECTION_NAME = "topic_vectors", CSV_PATH = "Data/topics.csv", VECTORS_PATH = "Model/topic_embeddings.npy")
load_data_qdrant(COLLECTION_NAME = "content_vectors", CSV_PATH = "Data/content.csv", VECTORS_PATH = "Model/content_embeddings.npy")