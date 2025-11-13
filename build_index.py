import os, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
INDEX_PATH = Path("index.faiss")
META_PATH = Path("metadata.json")
MODEL = "all-MiniLM-L6-v2"

def read_files():
    docs = []
    for f in DATA_DIR.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        docs.append({"id": f.name, "text": text})
    return docs

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def build_embeddings(docs):
    model = SentenceTransformer(MODEL)
    texts, meta = [], []
    for d in docs:
        for i, chunk in enumerate(chunk_text(d["text"])):
            texts.append(chunk)
            meta.append({"id": f"{d['id']}__{i}", "source": d["id"], "text": chunk})
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb, meta

def main():
    docs = read_files()
    emb, meta = build_embeddings(docs)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_PATH))
    json.dump(meta, open(META_PATH, "w"), indent=2)
    print("✅ Index + metadata saved!")

if __name__ == "__main__":
    main()
