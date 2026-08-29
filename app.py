# ============================================================
# app.py
#
# Portfolio RAG API
#
# FastAPI
# NVIDIA Embeddings
# FAISS
# NVIDIA LLM
#
# Designed for:
#     Flutter Web
#     Flutter Mobile
#     Cloud Run
# ============================================================


# ============================================================
# IMPORTS
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
# ENVIRONMENT
# ============================================================

load_dotenv()


NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)


NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
)


NVIDIA_EMBED_MODEL = os.getenv(
    "NVIDIA_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b"
)


# ============================================================
# NVIDIA ENDPOINTS
# ============================================================

EMBED_URL = (
    "https://integrate.api.nvidia.com/v1/embeddings"
)


CHAT_URL = (
    "https://integrate.api.nvidia.com/v1/chat/completions"
)


# ============================================================
# STARTUP LOGGING
# ============================================================

print("============================================")
print("Portfolio RAG API")
print("============================================")

print(
    "+++ LLM:",
    NVIDIA_MODEL
)

print(
    "+++ EMBEDDING:",
    NVIDIA_EMBED_MODEL
)

print(
    "+++ API KEY PRESENT:",
    bool(NVIDIA_API_KEY)
)


# ============================================================
# LOAD FAISS
# ============================================================

try:

    index = faiss.read_index(
        "index.faiss"
    )


    with open(
        "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:

        meta = json.load(f)


    print(
        "+++ FAISS vectors:",
        index.ntotal
    )

    print(
        "+++ FAISS dimension:",
        index.d
    )


except Exception as e:

    print(
        "❌ Could not load FAISS index:"
    )

    print(e)

    raise


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title="Suprith M Portfolio API",

    description=(
        "Portfolio RAG API using "
        "NVIDIA Embeddings + FAISS + NVIDIA LLM"
    ),

    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================
#
# This is required for Flutter Web.
#
# During development:
#
#     allow_origins=["*"]
#
# Once your website is deployed, you can replace "*"
# with your real domain.
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    question: str = Field(

        ...,

        min_length=1,

        max_length=2000

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

    return {

        "status": "ok",

        "service":
            "Suprith M Portfolio RAG",

        "embedding_model":
            NVIDIA_EMBED_MODEL,

        "llm_model":
            NVIDIA_MODEL,

        "vectors":
            index.ntotal

    }


# ============================================================
# NVIDIA QUERY EMBEDDING
# ============================================================

def get_query_embedding(
    query: str
):
    """
    Convert the user's question into an embedding.

    IMPORTANT:

    Index:
        input_type="passage"

    Query:
        input_type="query"
    """

    if not NVIDIA_API_KEY:

        raise RuntimeError(
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
            NVIDIA_EMBED_MODEL,

        "input":
            [query],

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
            f"Embedding request failed: {e}"
        )


    if response.status_code != 200:

        raise RuntimeError(

            f"NVIDIA embedding error "
            f"{response.status_code}: "
            f"{response.text}"

        )


    result = response.json()


    data = result.get(
        "data"
    )


    if not data:

        raise RuntimeError(
            f"No embedding returned: {result}"
        )


    embedding = np.array(

        [
            data[0]["embedding"]
        ],

        dtype="float32"

    )


    # Normalize query vector.
    #
    # This matches the normalization used
    # while building the FAISS index.

    faiss.normalize_L2(
        embedding
    )


    return embedding


# ============================================================
# SEARCH FAISS
# ============================================================

def search(
    query: str,
    k: int = 5
):
    """
    Search portfolio information using FAISS.
    """

    # Safety limit
    k = max(
        1,
        min(k, 10)
    )


    # Get query vector
    query_embedding = (
        get_query_embedding(query)
    )


    # Search FAISS
    distances, indices = (
        index.search(
            query_embedding,
            k
        )
    )


    results = []


    for distance, idx in zip(

        distances[0],

        indices[0]

    ):

        if idx < 0:
            continue


        result = meta[idx].copy()


        result["score"] = float(
            distance
        )


        results.append(
            result
        )


    return results


# ============================================================
# SYSTEM PROMPT
# ============================================================
#
# This prompt is intentionally concise.
#
# We don't want to waste tokens.
#
# The important improvement comes from structured retrieval
# metadata in build_index.py.
# ============================================================

SYSTEM_PROMPT = """
You are PortfolioAssistant for Suprith M's professional portfolio.

Use ONLY the information provided in CONTEXT.

Answer the user's question directly and completely.
Do not simply copy or dump retrieved content.

IMPORTANT:

1. Always distinguish CURRENT employment from PREVIOUS employment
   using the employment dates and section names.

2. Morgan Stanley is Suprith's current company.
   The portfolio currently provides his role, domain, team and
   technology stack there, but does NOT provide specific Morgan Stanley
   project descriptions.

3. Mbb Labs Pvt Ltd (Product Division, Maybank) is his previous company.
   Projects under "Work Projects — Mbb Labs Pvt Ltd" belong to Mbb Labs,
   NOT Morgan Stanley.

4. Never attribute a project to a company unless the CONTEXT explicitly
   associates that project with the company.

5. Projects under "Hobby & Open Source Projects" are personal/open-source
   projects and must not be presented as employment projects.

6. If the user asks about current Morgan Stanley projects and the context
   contains no Morgan Stanley project details, explicitly say that the
   portfolio does not currently provide specific project details for
   Morgan Stanley. You may provide the available Morgan Stanley role,
   domain, team and technology information instead.

7. If the user asks about Mbb Labs projects, provide only the projects
   associated with Mbb Labs.

8. If a question mentions both current and previous companies, clearly
   distinguish them instead of combining their projects.

9. Synthesize information into a useful answer. Do not merely return
   project names or raw retrieved text.

10. Never invent project names, responsibilities, technologies, dates,
    achievements or company associations.

11. Do not mention the retrieval process or say:
    "Based on the context..."
    "According to the context..."

If the requested information cannot be determined from the CONTEXT,
respond:

"I’m not sure about that — please check the portfolio or feel free to get in touch with Suprith M."
"""


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    chunks,
    question
):
    """
    Build the final prompt.

    We include the metadata so the model knows
    which company/section/project each chunk belongs to.
    """

    context_parts = []


    for chunk in chunks:

        context_parts.append(

            f"""
[SOURCE]
ID: {chunk.get("id", "")}

SECTION:
{chunk.get("section", "")}

COMPANY:
{chunk.get("company", "")}

PROJECT:
{chunk.get("project", "")}

CONTENT:
{chunk.get("text", "")}
""".strip()

        )


    context = "\n\n".join(
        context_parts
    )


    return f"""
CONTEXT:

{context}

USER QUESTION:

{question}

Answer the USER QUESTION using only the CONTEXT.

Synthesize the relevant information.
Do not simply repeat the context.
"""


# ============================================================
# NVIDIA LLM
# ============================================================

def nvidia_generate(
    prompt,
    temperature=0.4,
    max_tokens=1024,
    retries=2,
    backoff=1.5
):
    """
    Send prompt to NVIDIA's LLM endpoint.
    """

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

                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT

            },

            {

                "role":
                    "user",

                "content":
                    prompt

            }

        ],

        "temperature":
            temperature,

        "max_tokens":
            max_tokens,

        "stream":
            False

    }


    # ========================================================
    # RETRY
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
                f"[NVIDIA] Request error "
                f"attempt={attempt}: {e}"
            )


            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue


            return (
                f"Request error: {e}"
            )


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
        # AUTH ERROR
        # ====================================================

        if response.status_code in (

            401,

            403

        ):

            return (

                "Authentication error. "
                "Please check NVIDIA_API_KEY."

            )


        # ====================================================
        # MODEL DEPRECATED
        # ====================================================

        if response.status_code == 410:

            return (

                f"The NVIDIA model "
                f"'{NVIDIA_MODEL}' "
                "is no longer available."

            )


        # ====================================================
        # RATE LIMIT
        # ====================================================

        if response.status_code == 429:

            print(
                "[NVIDIA] Rate limit reached."
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
        # TEMPORARY ERROR
        # ====================================================

        if response.status_code == 503:

            print(
                "[NVIDIA] Service temporarily unavailable."
            )


            if attempt < retries:

                time.sleep(
                    backoff * attempt
                )

                continue


        # ====================================================
        # OTHER ERROR
        # ====================================================

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
# QUERY ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse
)
async def query(
    req: QueryRequest
):
    """
    Main RAG endpoint.

    Example request:

    {
        "question":
            "Which is his current company and what projects
             did he work on at his previous company?",

        "top_k": 6,

        "temperature": 0.4
    }
    """

    # --------------------------------------------------------
    # Clean question
    # --------------------------------------------------------

    question = req.question.strip()


    if not question:

        return {

            "answer":
                "Please enter a question.",

            "sources":
                []

        }


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    try:

        top_chunks = search(

            question,

            req.top_k

        )


    except Exception as e:

        print(
            f"[SEARCH ERROR] {e}"
        )


        return {

            "answer":
                "I’m having trouble searching "
                "the portfolio right now.",

            "sources":
                []

        }


    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(

        top_chunks,

        question

    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = nvidia_generate(

        prompt,

        temperature=req.temperature

    )


    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "answer":
            answer,

        "sources":
            [
                chunk["id"]
                for chunk in top_chunks
            ]

    }