import os
import json
import requests
import numpy as np
import faiss

from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────

load_dotenv()

DATA_DIR = Path("data")
INDEX_PATH = Path("index.faiss")
META_PATH = Path("metadata.json")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_EMBED_MODEL = os.getenv(
    "NVIDIA_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"


# ─────────────────────────────────────────────
# Read Markdown files
# ─────────────────────────────────────────────

def read_files():
    docs = []

    for f in DATA_DIR.glob("*.md"):
        text = f.read_text(encoding="utf-8")

        if text.strip():
            docs.append({
                "id": f.name,
                "text": text
            })

    return docs


# ─────────────────────────────────────────────
# Split text into chunks
# ─────────────────────────────────────────────

def chunk_text(text, max_words=200):
    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────
# NVIDIA Embeddings
# ─────────────────────────────────────────────

def get_embeddings(texts, input_type="passage"):
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. "
            "Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NVIDIA_EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "END"
    }

    print(
        f"Requesting embeddings from NVIDIA "
        f"for {len(texts)} chunks..."
    )

    response = requests.post(
        EMBED_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"NVIDIA embedding API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    data = result.get("data")

    if not data:
        raise RuntimeError(
            f"NVIDIA API returned no embeddings: {result}"
        )

    # Make sure embeddings are in original input order
    data.sort(key=lambda x: x["index"])

    embeddings = np.array(
        [item["embedding"] for item in data],
        dtype="float32"
    )

    return embeddings


# ─────────────────────────────────────────────
# Build embeddings + metadata
# ─────────────────────────────────────────────

def build_embeddings(docs):
    texts = []
    metadata = []

    for document in docs:

        chunks = chunk_text(
            document["text"],
            max_words=200
        )

        for i, chunk in enumerate(chunks):

            texts.append(chunk)

            metadata.append({
                "id": f"{document['id']}__{i}",
                "source": document["id"],
                "text": chunk
            })

    if not texts:
        raise RuntimeError(
            "No text chunks were generated."
        )

    print(f"Total chunks: {len(texts)}")

    # Documents/passages use input_type=passage
    embeddings = get_embeddings(
        texts,
        input_type="passage"
    )

    # Normalize for cosine similarity / inner product
    faiss.normalize_L2(embeddings)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings, metadata


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():

    print("========================================")
    print("Building NVIDIA Embedding FAISS Index")
    print("========================================")

    print("Embedding model:", NVIDIA_EMBED_MODEL)
    print("API key present:", bool(NVIDIA_API_KEY))

    docs = read_files()

    if not docs:
        raise RuntimeError(
            f"No .md files found in {DATA_DIR}"
        )

    print(f"Found {len(docs)} Markdown files.")

    embeddings, metadata = build_embeddings(docs)

    # ─────────────────────────────────────────
    # Create FAISS index
    # ─────────────────────────────────────────

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # ─────────────────────────────────────────
    # Save index
    # ─────────────────────────────────────────

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # ─────────────────────────────────────────
    # Save metadata
    # ─────────────────────────────────────────

    with open(
        META_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("========================================")
    print("✅ FAISS index created successfully!")
    print("========================================")
    print("Files:", len(docs))
    print("Chunks:", len(metadata))
    print("Embedding dimension:", dimension)
    print("Index:", INDEX_PATH)
    print("Metadata:", META_PATH)


if __name__ == "__main__":
    main()