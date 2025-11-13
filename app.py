import os, json, time, requests
from requests.exceptions import RequestException
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

print("+++ NVIDIA MODEL:", NVIDIA_MODEL)
print("+++ NVIDIA API KEY present:", bool(NVIDIA_API_KEY))

# ─────────────────────────────────────────────
# Load FAISS index and metadata
# ─────────────────────────────────────────────
index = faiss.read_index("index.faiss")
meta = json.load(open("metadata.json"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# FastAPI app setup
# ─────────────────────────────────────────────
app = FastAPI(title="Portfolio Chatbot (NVIDIA API)")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.4

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# ─────────────────────────────────────────────
# Search function using FAISS
# ─────────────────────────────────────────────
def search(query, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    res = [meta[i] for i in I[0]]
    return res

# ─────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are PortfolioAssistant, a friendly and factual assistant.
Rules:
1. Use ONLY the given CONTEXT from Suprith M's portfolio.
2. If the answer isn't found in CONTEXT, say: "I’m not sure about that — please check the portfolio or feel free to get in touch with Suprith M."
3. You may be concise yet creative in style but don't make up details.
4. When the CONTEXT is matched, DO NOT include phrases like "Based on the provided CONTEXT" or similar — respond naturally and directly as if you know the information.
"""

def build_prompt(chunks, question):
    ctx = "\n\n".join([f"[{c['id']}]\n{c['text']}" for c in chunks])
    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer only using CONTEXT."

# ─────────────────────────────────────────────
# NVIDIA API call
# ─────────────────────────────────────────────
def nvidia_generate(prompt, temp=0.4, max_tokens=350, retries=2, backoff=1.5):
    if not NVIDIA_API_KEY:
        return "(NVIDIA_API_KEY not set) Please set NVIDIA_API_KEY in .env"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": "You are PortfolioAssistant, a friendly and factual assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": max_tokens
    }

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except RequestException as e:
            print(f"[nvidia_generate] request exception (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            return f"Request error: {e}"

        print(f"[nvidia_generate] status={resp.status_code} attempt={attempt}")

        if resp.status_code == 200:
            try:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Parse error: {e}")
                return resp.text

        elif resp.status_code in (401, 403):
            return f"Authentication error {resp.status_code}: check NVIDIA_API_KEY"
        elif resp.status_code == 503:
            print(f"[nvidia_generate] model loading/unavailable. attempt {attempt}")
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
        else:
            print(f"[nvidia_generate] unexpected {resp.status_code}: {resp.text[:200]}")

    return "NVIDIA inference failed after retries."

# ─────────────────────────────────────────────
# Main API route
# ─────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    top_chunks = search(req.question, req.top_k)
    prompt = build_prompt(top_chunks, req.question)
    ans = nvidia_generate(prompt, req.temperature)
    return {"answer": ans, "sources": [c["id"] for c in top_chunks]}
