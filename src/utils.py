import os
import re
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, MatchAny
from tqdm.contrib.concurrent import thread_map
from common.constants import MAX_CHAR, DATA_DIR

load_dotenv()
COLAB_API_BASE = os.environ["COLAB_API"]

def encode_query_to_vector(text: str) -> list[float]:
    """
    Send the input text to the remote /encode API and return the embedding vector.

    Args:
        text (str): The query or sentence to be encoded.

    Returns:
        list[float]: A list representing the embedding vector of the input text.
    """
    res = requests.post(f"{COLAB_API_BASE}/encode", json={"text": text})
    res.raise_for_status()
    return res.json()["embedding"]


def search_qdrant_vector(client,
                         collection_name: str,
                         vector: list[float],
                         limit: int = 50,
                         language: str = None):
    """
    Perform a vector search in Qdrant, optionally filtered by language.

    Args:
        client: Qdrant client instance.
        collection_name (str): Name of the collection to search.
        vector (list[float]): The query vector.
        limit (int, optional): Maximum number of results. Defaults to 50.
        language (str, optional): Language filter.

    Returns:
        list: A list of search results from Qdrant with payloads.
    """
    print("🔍 Starting vector search...")

    # No language filter
    if language is None:
        print("🌐 No language filter applied")
        return client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True
        )

    # Scroll to get content IDs matching the language
    scroll_result, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language)
                )
            ]
        ),
        with_payload=True
    )

    # Extract content IDs from payloads
    content_ids = [
        pt.payload.get("id") for pt in scroll_result
        if pt.payload and "id" in pt.payload
    ]

    # Ensure index on ID field
    client.create_payload_index(
        collection_name="content_vectors",
        field_name="id",
        field_schema="keyword"
    )

    # Create filter to search only in matching content IDs
    filter_content_id = Filter(
        should=[
            FieldCondition(
                key="id",
                match=MatchValue(value=cid)
            ) for cid in content_ids
        ],
        must=[],
        must_not=[]
    )

    # Perform filtered search
    return client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit,
        with_payload=True,
        query_filter=filter_content_id
    )


def rerank_candidates(query: str, passages: list[str]) -> list[str]:
    """
    Call the external /rerank API to rerank the retrieved passages based on query relevance.

    Args:
        query (str): The original search query.
        passages (list[str]): List of candidate passages to rerank.

    Returns:
        list[str]: The reranked list of passages sorted by relevance.
    """
    res = requests.post(f"{COLAB_API_BASE}/rerank", json={
        "query": query,
        "passages": passages
    })
    res.raise_for_status()
    return res.json()["sorted_passages"]


def map_passages_to_payload(search_results) -> dict:
    """
    Create a mapping from passage text to the corresponding payload object.

    Args:
        search_results (list): List of search result objects from Qdrant.

    Returns:
        dict: Dictionary mapping passage text to payload data.
    """
    return {hit.payload["text"]: hit.payload for hit in search_results}


def load_data_qdrant(COLLECTION_NAME, CSV_PATH, VECTORS_PATH):
    """
    Load data from CSV and vector file, and upload to Qdrant in batches.

    Args:
        COLLECTION_NAME (str): Name of the Qdrant collection to create or replace.
        CSV_PATH (str): Path to the CSV file with metadata.
        VECTORS_PATH (str): Path to the NumPy .npz file containing vectors and IDs.

    Raises:
        AssertionError: If the number of vectors does not match the number of rows in the CSV.
    """
    print("📄 Reading data...")
    data = np.load(VECTORS_PATH)
    df = pd.read_csv(CSV_PATH).fillna("")

    if COLLECTION_NAME == "topic_vectors":
        df = df[df.has_content]
        vectors = data["topic_emb"]
        ids = data["topic_ids"]
    elif COLLECTION_NAME == "content_vectors":
        vectors = data["content_emb"]
        ids = data["content_ids"]

    print("✅ CSV and vectors loaded successfully.")
    assert len(df) == len(vectors), "❌ CSV and vector count mismatch."

    print("🔗 Connecting to Qdrant server...")
    load_dotenv()
    client = QdrantClient(
        url=os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    print("✅ Connected to Qdrant.")

    print("📦 Creating collection...")
    vector_size = vectors.shape[1]
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": vector_size, "distance": "Cosine"}
    )
    print("✅ Collection created.")

    print("📤 Uploading vectors in batches...")
    batch_size = 512
    payload_map = {str(row["id"]): row.to_dict() for _, row in df.iterrows()}

    for i in tqdm(range(0, len(ids), batch_size), desc="Uploading..."):
        points = [
            PointStruct(
                id=i + j,
                vector=vectors[i + j].tolist(),
                payload=payload_map[str(ids[i + j])]
            )
            for j in range(min(batch_size, len(ids) - i))
        ]
        client.upload_points(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Upload complete: {COLLECTION_NAME}")


def clean_and_limit_text(text, max_chars=MAX_CHAR):
    """
    Clean and truncate text to a maximum number of characters.

    Args:
        text (Any): Input text to clean.
        max_chars (int): Maximum allowed character length.

    Returns:
        str: Cleaned and truncated string.
    """
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned[:max_chars]


def translate_text(text, source_lang, target_lang="en"):
    """
    Translate text using Google Translate API.

    Args:
        text (str): Text to be translated.
        source_lang (str): Source language code.
        target_lang (str): Target language code (default is 'en').

    Returns:
        str: Translated text or original if translation fails.
    """
    if source_lang == target_lang or not text:
        return text
    url = os.environ["GOOGLE_TRANSLATE_URL"]

    def send_request(src_lang):
        params = {
            'q': text,
            'source': src_lang,
            'target': target_lang,
            'format': 'text',
            'key': os.environ["API_KEY"]
        }
        response = requests.post(url, data=params, timeout=10)
        response.raise_for_status()
        return response.json()['data']['translations'][0]['translatedText']

    try:
        return send_request(source_lang)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            try:
                return send_request('auto')
            except:
                translate_text.skipped += 1
                return ""
        else:
            translate_text.skipped += 1
            return ""
    except:
        translate_text.skipped += 1
        return ""


def translate_dataframe(df: pd.DataFrame,
                        file_name: str,
                        save_file: bool = False,
                        lang_col: str = 'language',
                        target_lang: str = 'en',
                        num_workers: int = 4):
    """
    Translate all text columns in a DataFrame to a target language.

    Args:
        df (pd.DataFrame): The input DataFrame to translate.
        file_name (str): Name of the CSV file to save (if save_file=True).
        save_file (bool): Whether to save the translated DataFrame.
        lang_col (str): Column name indicating language of each row.
        target_lang (str): Target language code for translation.
        num_workers (int): Number of parallel workers for translation.

    Returns:
        pd.DataFrame: Translated DataFrame.
    """
    df_translated = df.copy()
    text_columns = [col for col in df.columns if col != lang_col]
    records = df_translated.to_dict(orient='records')

    for col in text_columns:
        print(f"🔁 Translating column '{col}' using {num_workers} workers...")

        def translate_one(row):
            text = clean_and_limit_text(row[col])
            lang = row[lang_col]
            return translate_text(text, lang, target_lang)

        results = thread_map(translate_one, records, max_workers=num_workers, desc=f"Translating '{col}'")
        df_translated[f"{col}_translated"] = results

    if save_file:
        save_dir = DATA_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)

        if os.path.exists(save_path):
            replace = input("⚠️ File exists. Replace it? (Y/N): ").strip().lower()
            if replace == "y":
                df_translated.to_csv(save_path, index=False, encoding='utf-8-sig')
                print(f"💾 File replaced: {save_path}")
        else:
            df_translated.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"💾 File saved: {save_path}")

    return df_translated


