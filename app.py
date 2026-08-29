# ============================================================
# Portfolio RAG API
#
# FastAPI + FAISS + NVIDIA Embeddings + NVIDIA LLM
# ============================================================

import os
import json
import time
import requests
import numpy as np
import faiss

from requests.exceptions import RequestException

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from typing import List

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# LLM used for generating the final answer
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
)

# Embedding model used for both indexing and searching
NVIDIA_EMBED_MODEL = os.getenv(
    "NVIDIA_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

# NVIDIA API endpoints
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ============================================================
# STARTUP LOGGING
# ============================================================

print("============================================")
print("Portfolio RAG API")
print("============================================")

print(
    "+++ NVIDIA LLM MODEL:",
    NVIDIA_MODEL
)

print(
    "+++ NVIDIA EMBEDDING MODEL:",
    NVIDIA_EMBED_MODEL
)

# Never print the actual API key
print(
    "+++ NVIDIA API KEY PRESENT:",
    bool(NVIDIA_API_KEY)
)


# ============================================================
# LOAD FAISS INDEX + METADATA
# ============================================================

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

    print("❌ Failed to load FAISS index:")
    print(e)

    raise


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Portfolio Chatbot API",
    description=(
        "Portfolio RAG chatbot powered by "
        "NVIDIA Embeddings, FAISS and NVIDIA LLM"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================
#
# This allows your Flutter Web application to call this API.
#
# For development:
#     allow_origins=["*"]
#
# Once your portfolio is deployed, replace "*" with your
# actual website domain for better security.
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about Suprith's portfolio"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )

    temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class QueryResponse(BaseModel):

    answer: str

    sources: List[str]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():
    """
    Simple endpoint to check whether the API is running.
    """

    return {
        "status": "ok",
        "service": "Portfolio RAG API",
        "embedding_model": NVIDIA_EMBED_MODEL,
        "llm_model": NVIDIA_MODEL,
        "faiss_vectors": index.ntotal
    }


# ============================================================
# NVIDIA EMBEDDING
# ============================================================

def get_query_embedding(query: str):
    """
    Convert the user's question into an embedding.

    IMPORTANT:
    We use input_type="query".

    build_index.py uses input_type="passage"
    for portfolio documents.
    """

    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured."
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": NVIDIA_EMBED_MODEL,

        # NVIDIA expects input as a list
        "input": [query],

        # User question = query
        "input_type": "query",

        "encoding_format": "float",

        "truncate": "END"
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

    try:

        result = response.json()

    except Exception as e:

        raise RuntimeError(
            f"Invalid JSON from NVIDIA embedding API: {e}"
        )

    data = result.get("data")

    if not data:

        raise RuntimeError(
            f"No embedding returned by NVIDIA: {result}"
        )

    # Convert embedding to float32 for FAISS
    embedding = np.array(
        [data[0]["embedding"]],
        dtype="float32"
    )

    # Normalize because our FAISS index uses
    # normalized vectors + Inner Product.
    faiss.normalize_L2(embedding)

    return embedding


# ============================================================
# FAISS SEARCH
# ============================================================

def search(query: str, k: int = 5):
    """
    Search the portfolio vector database.

    Returns the most relevant portfolio chunks.
    """

    # Safety limit
    k = max(1, min(k, 10))

    # Convert user question to vector
    query_embedding = get_query_embedding(query)

    # Search FAISS
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

        # Similarity score
        result["score"] = float(distance)

        results.append(result)

    return results


# ============================================================
# SYSTEM PROMPT
# ============================================================
#
# This is intentionally concise.
#
# The goal is NOT to make the model repeat the retrieved
# chunks. It should understand the question and synthesize
# an answer from the relevant information.
# ============================================================

SYSTEM_PROMPT = """
You are PortfolioAssistant, a friendly and factual assistant
for Suprith M's professional portfolio.

Use ONLY the information provided in CONTEXT.

Your job is to answer the user's QUESTION, not to repeat or
dump the retrieved CONTEXT.

Understand the question first, identify the relevant information,
and synthesize it into a clear, complete, natural response.

When multiple relevant pieces of information are available,
combine them into a coherent answer.

If the question asks for an explanation, explain using the
available context rather than simply listing names.

If the question asks for projects, mention the relevant projects
and briefly explain them when the context provides enough detail.

Do not invent technologies, responsibilities, achievements,
dates, companies, project details, or other information.

Do not mention the retrieval process or say:
"Based on the context..."
"According to the context..."
"The retrieved information..."

If the answer cannot be determined from the CONTEXT, respond exactly:

"I’m not sure about that — please check the portfolio or feel free to get in touch with Suprith M."
"""


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_prompt(chunks, question):
    """
    Creates the final prompt sent to the LLM.
    """

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"[{chunk['id']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
CONTEXT:
{context}

QUESTION:
{question}

Answer the QUESTION using the CONTEXT.
Synthesize the information into a useful response.
Do not simply copy the CONTEXT.
"""


# ============================================================
# NVIDIA LLM GENERATION
# ============================================================

def nvidia_generate(
    prompt,
    temp=0.4,
    max_tokens=1024,
    retries=2,
    backoff=1.5
):
    """
    Send the RAG prompt to NVIDIA's chat completion API.
    """

    if not NVIDIA_API_KEY:

        return (
            "NVIDIA_API_KEY is not configured."
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {

        "model": NVIDIA_MODEL,

        "messages": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        "temperature": temp,

        "max_tokens": max_tokens,

        "stream": False
    }


    # ========================================================
    # RETRY LOOP
    # ========================================================

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


        # ====================================================
        # SUCCESS
        # ====================================================

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


        # ====================================================
        # AUTHENTICATION
        # ====================================================

        elif response.status_code in (
            401,
            403
        ):

            return (
                f"Authentication error "
                f"{response.status_code}: "
                "check NVIDIA_API_KEY"
            )


        # ====================================================
        # MODEL NO LONGER AVAILABLE
        # ====================================================

        elif response.status_code == 410:

            return (
                f"NVIDIA model "
                f"'{NVIDIA_MODEL}' "
                "is no longer available."
            )


        # ====================================================
        # RATE LIMIT
        # ====================================================

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


        # ====================================================
        # TEMPORARILY UNAVAILABLE
        # ====================================================

        elif response.status_code == 503:

            print(
                "[NVIDIA] Model temporarily unavailable."
            )

            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue


        # ====================================================
        # OTHER ERROR
        # ====================================================

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


# ============================================================
# MAIN QUERY ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse
)
async def query(req: QueryRequest):
    """
    Main RAG endpoint.

    Flutter sends:

    POST /query

    {
        "question": "What projects has Suprith worked on?",
        "top_k": 5,
        "temperature": 0.4
    }
    """

    # --------------------------------------------------------
    # Clean the question
    # --------------------------------------------------------

    question = req.question.strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # Limit retrieved chunks
    # --------------------------------------------------------

    top_k = max(
        1,
        min(req.top_k, 10)
    )


    # ========================================================
    # STEP 1: RETRIEVE RELEVANT INFORMATION
    # ========================================================

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


    # ========================================================
    # STEP 2: BUILD RAG PROMPT
    # ========================================================

    prompt = build_prompt(
        top_chunks,
        question
    )


    # ========================================================
    # STEP 3: GENERATE ANSWER
    # ========================================================

    answer = nvidia_generate(
        prompt,
        temp=req.temperature
    )


    # ========================================================
    # STEP 4: RETURN ANSWER TO FLUTTER
    # ========================================================

    return {
        "answer": answer,

        "sources": [
            chunk["id"]
            for chunk in top_chunks
        ]
    }