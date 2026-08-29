import os
import json
import time
import requests
import numpy as np
import faiss

from requests.exceptions import RequestException
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
)

NVIDIA_EMBED_MODEL = os.getenv(
    "NVIDIA_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


print("+++ NVIDIA LLM MODEL:", NVIDIA_MODEL)
print("+++ NVIDIA EMBEDDING MODEL:", NVIDIA_EMBED_MODEL)
print("+++ NVIDIA API KEY present:", bool(NVIDIA_API_KEY))


# ─────────────────────────────────────────────
# Load FAISS index and metadata
# ─────────────────────────────────────────────

try:

    index = faiss.read_index("index.faiss")

    with open(
        "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        meta = json.load(f)

    print("+++ FAISS index loaded")
    print("+++ FAISS vectors:", index.ntotal)
    print("+++ FAISS dimension:", index.d)

except Exception as e:

    print("❌ Failed to load FAISS index:", e)

    raise


# ─────────────────────────────────────────────
# FastAPI setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Portfolio Chatbot",
    description="RAG chatbot powered by NVIDIA APIs and FAISS",
    version="1.0.0"
)


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):

    question: str

    top_k: int = 5

    temperature: float = 0.4


class QueryResponse(BaseModel):

    answer: str

    sources: List[str]


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/")
def health():

    return {
        "status": "ok",
        "service": "Portfolio RAG API"
    }


# ─────────────────────────────────────────────
# NVIDIA Embedding
# ─────────────────────────────────────────────

def get_query_embedding(query: str):

    if not NVIDIA_API_KEY:

        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    headers = {

        "Authorization":
            f"Bearer {NVIDIA_API_KEY}",

        "Content-Type":
            "application/json"
    }

    payload = {

        "model":
            NVIDIA_EMBED_MODEL,

        "input":
            [query],

        # Query embedding
        "input_type":
            "query",

        "encoding_format":
            "float",

        "truncate":
            "END"
    }

    try:

        response = requests.post(

            EMBED_URL,

            headers=headers,

            json=payload,

            timeout=60
        )

    except RequestException as e:

        raise RuntimeError(
            f"NVIDIA embedding request failed: {e}"
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
            f"No embedding returned by NVIDIA: "
            f"{result}"
        )

    embedding = np.array(
        [data[0]["embedding"]],
        dtype="float32"
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding)

    return embedding


# ─────────────────────────────────────────────
# FAISS Search
# ─────────────────────────────────────────────

def search(query: str, k: int = 5):

    # Protect the API from unreasonable values
    k = max(1, min(k, 10))

    query_embedding = get_query_embedding(
        query
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        result = meta[idx].copy()

        # Include similarity score
        result["score"] = float(distance)

        results.append(result)

    return results


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are PortfolioAssistant, a friendly and factual assistant.

Rules:

1. Use ONLY the given CONTEXT from Suprith M's portfolio.

2. If the answer isn't found in CONTEXT, say:
"I’m not sure about that — please check the portfolio or feel free to get in touch with Suprith M."

3. Do not make up information.

4. When the answer is present in CONTEXT, respond naturally and directly.

5. Do not say things like:
"Based on the provided context..."
"According to the context..."
"From the context..."

6. Keep answers concise and useful.

7. You may mention relevant technologies, projects,
experience, skills, education, or other portfolio information
only when it is present in CONTEXT.
"""


# ─────────────────────────────────────────────
# Build RAG prompt
# ─────────────────────────────────────────────

def build_prompt(chunks, question):

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"[{chunk['id']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    return f"""
{SYSTEM_PROMPT}

CONTEXT:

{context}

QUESTION:

{question}

Answer only using the CONTEXT.
"""


# ─────────────────────────────────────────────
# NVIDIA LLM Generation
# ─────────────────────────────────────────────

def nvidia_generate(
    prompt,
    temp=0.4,
    max_tokens=1024,
    retries=2,
    backoff=1.5
):

    if not NVIDIA_API_KEY:

        return (
            "NVIDIA_API_KEY is not configured."
        )

    headers = {

        "Authorization":
            f"Bearer {NVIDIA_API_KEY}",

        "Content-Type":
            "application/json",

        "Accept":
            "application/json"
    }

    payload = {

        "model":
            NVIDIA_MODEL,

        "messages": [

            {
                "role": "system",

                "content":
                    "You are PortfolioAssistant, "
                    "a friendly and factual assistant."
            },

            {
                "role": "user",

                "content":
                    prompt
            }
        ],

        "temperature":
            temp,

        "max_tokens":
            max_tokens,

        "stream":
            False
    }

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            response = requests.post(

                CHAT_URL,

                headers=headers,

                json=payload,

                timeout=120
            )

        except RequestException as e:

            print(
                f"[NVIDIA] Request exception "
                f"(attempt {attempt}): {e}"
            )

            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue

            return f"Request error: {e}"

        print(
            f"[NVIDIA] status="
            f"{response.status_code} "
            f"attempt={attempt}"
        )

        # ─────────────────────────────────────
        # Success
        # ─────────────────────────────────────

        if response.status_code == 200:

            try:

                data = response.json()

                answer = (
                    data["choices"][0]
                    ["message"]["content"]
                    .strip()
                )

                return answer

            except Exception as e:

                print(
                    f"[NVIDIA] Parse error: {e}"
                )

                return response.text

        # ─────────────────────────────────────
        # Authentication
        # ─────────────────────────────────────

        elif response.status_code in (
            401,
            403
        ):

            return (
                f"Authentication error "
                f"{response.status_code}: "
                "check NVIDIA_API_KEY"
            )

        # ─────────────────────────────────────
        # Model no longer available
        # ─────────────────────────────────────

        elif response.status_code == 410:

            return (
                f"NVIDIA model "
                f"'{NVIDIA_MODEL}' "
                "is no longer available."
            )

        # ─────────────────────────────────────
        # Rate limit
        # ─────────────────────────────────────

        elif response.status_code == 429:

            print(
                "[NVIDIA] Rate limited."
            )

            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue

            return (
                "NVIDIA API rate limit reached. "
                "Please try again later."
            )

        # ─────────────────────────────────────
        # Temporary unavailable
        # ─────────────────────────────────────

        elif response.status_code == 503:

            print(
                "[NVIDIA] Model temporarily "
                "unavailable."
            )

            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue

        # ─────────────────────────────────────
        # Other errors
        # ─────────────────────────────────────

        else:

            print(
                f"[NVIDIA] Unexpected "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            )

            return (
                f"NVIDIA API error "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            )

    return (
        "NVIDIA inference failed after retries."
    )


# ─────────────────────────────────────────────
# Main API route
# ─────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse
)
async def query(req: QueryRequest):

    question = req.question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    # Limit top_k
    top_k = max(
        1,
        min(req.top_k, 10)
    )

    # ─────────────────────────────────────────
    # 1. Retrieve relevant chunks
    # ─────────────────────────────────────────

    try:

        top_chunks = search(
            question,
            top_k
        )

    except Exception as e:

        print(
            f"[SEARCH ERROR] {e}"
        )

        return {
            "answer":
                "I’m having trouble searching "
                "the portfolio right now.",
            "sources": []
        }

    # ─────────────────────────────────────────
    # 2. Build prompt
    # ─────────────────────────────────────────

    prompt = build_prompt(
        top_chunks,
        question
    )

    # ─────────────────────────────────────────
    # 3. Generate answer
    # ─────────────────────────────────────────

    answer = nvidia_generate(
        prompt,
        temp=req.temperature
    )

    # ─────────────────────────────────────────
    # 4. Return response
    # ─────────────────────────────────────────

    return {

        "answer":
            answer,

        "sources":
            [
                chunk["id"]
                for chunk in top_chunks
            ]
    }