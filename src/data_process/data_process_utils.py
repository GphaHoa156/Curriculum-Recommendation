import pandas as pd 
from common.constants import DATA_DIR
from utils import translate_dataframe, translate_text
from utils import load_data_qdrant

# Load raw topic and content CSV files
topics_df = pd.read_csv(DATA_DIR + "topics.csv")
content_df = pd.read_csv(DATA_DIR + "content.csv")

# Filter non-English entries that need translation
non_en_topic_df = topics_df[(topics_df.language != "en")]
non_en_content_df = content_df[(content_df.language != "en")]

# Select only text-related columns to translate
non_en_topic_df = non_en_topic_df[["language", "title", "description"]]
non_en_content_df = non_en_content_df[["language", "title", "description", "text"]]

# Translate non-English topics
translate_text.skipped = 0
topic_translated = translate_dataframe(
    non_en_topic_df, 
    file_name="topics_translated.csv",
    lang_col='language', 
    target_lang='en', 
    num_workers=10
)
print(f"\n✅ Skipped {translate_text.skipped} topic rows due to translation errors (including fallback failures).")

# Translate non-English content
translate_text.skipped = 0
content_translated = translate_dataframe(
    non_en_content_df, 
    file_name="content_translated.csv",
    lang_col='language', 
    target_lang='en', 
    num_workers=10
)
print(f"\n✅ Skipped {translate_text.skipped} content rows due to translation errors (including fallback failures).")

# Upload topic data to Qdrant
load_data_qdrant(
    COLLECTION_NAME="topic_vectors", 
    CSV_PATH=DATA_DIR + "topics.csv", 
    VECTORS_PATH=DATA_DIR + "all_embeddings.npz"
)

# Upload content data to Qdrant
load_data_qdrant(
    COLLECTION_NAME="content_vectors", 
    CSV_PATH=DATA_DIR + "content.csv", 
    VECTORS_PATH=DATA_DIR + "all_embeddings.npz"
)
