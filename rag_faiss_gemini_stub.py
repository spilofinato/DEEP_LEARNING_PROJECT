import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List

# --- CONFIG ---
JSON_PATH = "DATA/progress_dataset_dedup.json"
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "rag_metadata.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# # --- LOAD AND FLATTEN JSON WITH CONTEXT ---
# with open(JSON_PATH, "r", encoding="utf-8") as f:
#     dataset = json.load(f)
#
# flat_messages = []
# for progress in dataset:
#     progress_title = progress.get("title", "")
#     progress_description = progress.get("description", "")
#     for msg in progress["messages"]:
#         enriched_content = (
#             f"[Progress Title: {progress_title}]\n\n"
#             f"[Progress Description: {progress_description.strip()}]\n\n"
#             f"{msg['content'].strip()}"
#         )
#         flat_messages.append({
#             "progress_id": progress["progress_id"],
#             "progress_title": progress_title,
#             "message_id": msg["message_id"],
#             "timestamp": msg["timestamp"],
#             "author": msg["author"],
#             "original_content": msg["content"],
#             "enriched_content": enriched_content
#         })
#
# df = pd.DataFrame(flat_messages)
# df = df.dropna(subset=["enriched_content"])
# df = df[df["enriched_content"].str.len() > 10].reset_index(drop=True)
#
# # --- EMBEDDINGS ---
# print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# print("Encoding enriched messages...")
# embeddings = model.encode(df["enriched_content"].tolist(), show_progress_bar=True, convert_to_numpy=True)
#
# # --- FAISS INDEXING ---
# print("Building FAISS index...")
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)
#
# # Save FAISS index
# faiss.write_index(index, INDEX_PATH)
# print(f"FAISS index saved to {INDEX_PATH}")
#
# # Save metadata
# df.to_json(METADATA_PATH, orient="records", force_ascii=False, indent=2)
# print(f"Metadata saved to {METADATA_PATH}")

# --- RETRIEVAL FUNCTION ---
def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[dict]:
    q_emb = model.encode([query])[0].astype("float32").reshape(1, -1)
    D, I = index.search(q_emb, top_k)
    return df.iloc[I[0]][["timestamp", "author", "progress_title", "original_content"]].to_dict(orient="records")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    q = "Problemi di login in piattaforma"
    results = retrieve_relevant_chunks(q, top_k=5)
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['timestamp']} — {r['author']} ({r['progress_title']}):\n{r['original_content'][:300]}{'...' if len(r['original_content']) > 300 else ''}\n")
