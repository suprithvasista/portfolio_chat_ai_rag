# ============================================================
# build_index.py
#
# Builds a FAISS vector index from your portfolio Markdown file.
#
# Embedding model:
#     NVIDIA nvidia/nemotron-3-embed-1b
#
# Important:
#     Documents use input_type="passage"
#     User questions use input_type="query"
#
# This file should be run whenever you update your portfolio.
# ============================================================

import os
import json
import requests
import numpy as np
import faiss

from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DATA_DIR = Path("data")

INDEX_PATH = Path("index.faiss")

META_PATH = Path("metadata.json")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_EMBED_MODEL = os.getenv(
    "NVIDIA_EMBED_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

EMBED_URL = (
    "https://integrate.api.nvidia.com/v1/embeddings"
)


# ============================================================
# READ MARKDOWN FILES
# ============================================================

def read_files():
    """
    Read all .md files from the data directory.
    """

    documents = []

    for file_path in DATA_DIR.glob("*.md"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            continue

        documents.append({
            "id": file_path.name,
            "text": text
        })

    return documents


# ============================================================
# PARSE PORTFOLIO INTO MEANINGFUL SECTIONS
# ============================================================

def parse_portfolio(text, source_file):
    """
    Convert the Markdown portfolio into meaningful chunks.

    Instead of blindly splitting every 200 words, we preserve
    the structure of the portfolio.

    Example:

        ## Work Projects — Mbb Labs Pvt Ltd

        - Project: Cloudera Data Platform (CDP) Migration
            - Led migration...
            - Technologies...

    becomes one searchable chunk containing:

        Company
        Section
        Project
        Description
        Technologies

    This helps the LLM understand relationships between:
        company -> projects
        section -> projects
        current -> previous employment
    """

    lines = text.splitlines()

    chunks = []

    current_section = ""

    current_company = ""

    current_project = None

    current_project_lines = []


    # --------------------------------------------------------
    # Save the current project
    # --------------------------------------------------------

    def save_project():

        nonlocal current_project
        nonlocal current_project_lines

        if not current_project:
            return

        project_text = "\n".join(
            current_project_lines
        ).strip()

        if not project_text:
            return

        # Build a structured searchable document.
        structured_text = f"""
Source: {source_file}

Section: {current_section}

Company: {current_company}

Project: {current_project}

Details:
{project_text}
""".strip()

        chunks.append({
            "id": (
                f"{source_file}__"
                f"{current_project.replace(' ', '_')}"
            ),

            "source": source_file,

            "section": current_section,

            "company": current_company,

            "project": current_project,

            "text": structured_text
        })

        current_project = None
        current_project_lines = []


    # ========================================================
    # PROCESS LINES
    # ========================================================

    for line in lines:

        stripped = line.strip()

        # ----------------------------------------------------
        # Empty line
        # ----------------------------------------------------

        if not stripped:

            if current_project:
                current_project_lines.append("")

            continue


        # ----------------------------------------------------
        # H1
        #
        # Example:
        # # Suprith M — Portfolio
        # ----------------------------------------------------

        if stripped.startswith("# ") and not stripped.startswith("## "):

            if current_project:
                save_project()

            continue


        # ----------------------------------------------------
        # H2
        #
        # Example:
        #
        # ## Current Professional Experience — Morgan Stanley
        #
        # ## Professional Experience — Mbb Labs Pvt Ltd
        #
        # ## Work Projects — Mbb Labs Pvt Ltd
        #
        # ## Hobby & Open Source Projects
        # ----------------------------------------------------

        if stripped.startswith("## "):

            if current_project:
                save_project()

            current_section = stripped[3:].strip()

            # ------------------------------------------------
            # Determine company from section title
            # ------------------------------------------------

            section_lower = current_section.lower()

            if "morgan stanley" in section_lower:

                current_company = "Morgan Stanley"

            elif "mbb labs" in section_lower:

                current_company = (
                    "Mbb Labs Pvt Ltd "
                    "(Product Division, Maybank)"
                )

            elif (
                "hobby" in section_lower
                or "open source" in section_lower
            ):

                current_company = (
                    "Hobby & Open Source"
                )

            else:

                current_company = ""

            continue


        # ----------------------------------------------------
        # PROJECT LINE
        #
        # Example:
        #
        # - Project: Cloudera Data Platform (CDP) Migration
        # ----------------------------------------------------

        if stripped.lower().startswith(
            "- project:"
        ):

            if current_project:
                save_project()

            current_project = (
                stripped[len("- Project:"):].strip()
            )

            current_project_lines = []

            continue


        # ----------------------------------------------------
        # Everything else belongs to current project
        # ----------------------------------------------------

        if current_project:

            current_project_lines.append(
                stripped
            )


    # ========================================================
    # SAVE FINAL PROJECT
    # ========================================================

    if current_project:

        save_project()


    return chunks


# ============================================================
# CREATE FALLBACK CHUNKS
# ============================================================

def create_general_chunks(text, source_file):
    """
    Some portfolio information is not a project.

    Examples:

        About
        Current Professional Experience
        Professional Experience
        Achievements

    We still need this information in FAISS.

    This function creates structured chunks for those sections.
    """

    lines = text.splitlines()

    chunks = []

    current_section = ""

    current_lines = []

    current_company = ""


    def save_section():

        if not current_section:
            return

        content = "\n".join(
            current_lines
        ).strip()

        if not content:
            return

        chunks.append({

            "id": (
                f"{source_file}__"
                f"section__"
                f"{current_section.replace(' ', '_')}"
            ),

            "source": source_file,

            "section": current_section,

            "company": current_company,

            "project": "",

            "text": (
                f"Source: {source_file}\n\n"
                f"Section: {current_section}\n\n"
                f"Company: {current_company}\n\n"
                f"Details:\n{content}"
            )
        })


    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue


        # H2 section
        if stripped.startswith("## "):

            save_section()

            current_section = (
                stripped[3:].strip()
            )

            current_lines = []


            section_lower = (
                current_section.lower()
            )


            if "morgan stanley" in section_lower:

                current_company = "Morgan Stanley"

            elif "mbb labs" in section_lower:

                current_company = (
                    "Mbb Labs Pvt Ltd "
                    "(Product Division, Maybank)"
                )

            elif (
                "hobby" in section_lower
                or "open source" in section_lower
            ):

                current_company = (
                    "Hobby & Open Source"
                )

            else:

                current_company = ""

            continue


        # Don't duplicate project content here.
        if stripped.lower().startswith(
            "- project:"
        ):
            continue


        current_lines.append(
            stripped
        )


    save_section()

    return chunks


# ============================================================
# NVIDIA EMBEDDING FUNCTION
# ============================================================

def get_embeddings(
    texts,
    input_type="passage"
):
    """
    Generate embeddings using NVIDIA.

    For portfolio documents:
        input_type = "passage"

    For user questions:
        input_type = "query"
    """

    if not NVIDIA_API_KEY:

        raise RuntimeError(
            "NVIDIA_API_KEY is not set."
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
            texts,

        "input_type":
            input_type,

        "encoding_format":
            "float",

        "truncate":
            "END"
    }


    print(
        f"Requesting NVIDIA embeddings "
        f"for {len(texts)} texts..."
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
            f"No embeddings returned: {result}"
        )


    # NVIDIA returns an index for each embedding.
    # Sort to guarantee original order.
    data.sort(
        key=lambda x: x["index"]
    )


    embeddings = np.array(

        [
            item["embedding"]
            for item in data
        ],

        dtype="float32"
    )


    return embeddings


# ============================================================
# BUILD INDEX
# ============================================================

def build_index():

    print()
    print("============================================")
    print("Building Portfolio FAISS Index")
    print("============================================")

    print(
        "Embedding model:",
        NVIDIA_EMBED_MODEL
    )


    # --------------------------------------------------------
    # Read documents
    # --------------------------------------------------------

    documents = read_files()


    if not documents:

        raise RuntimeError(
            "No .md files found inside data/"
        )


    print(
        f"Found {len(documents)} Markdown file(s)."
    )


    all_chunks = []


    # ========================================================
    # PARSE DOCUMENTS
    # ========================================================

    for document in documents:

        source_file = document["id"]

        text = document["text"]


        # Project chunks
        project_chunks = parse_portfolio(
            text,
            source_file
        )


        # General/non-project chunks
        general_chunks = create_general_chunks(
            text,
            source_file
        )


        all_chunks.extend(
            project_chunks
        )

        all_chunks.extend(
            general_chunks
        )


    if not all_chunks:

        raise RuntimeError(
            "No chunks were generated."
        )


    print(
        f"Total chunks created: "
        f"{len(all_chunks)}"
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_chunks = []

    seen = set()


    for chunk in all_chunks:

        key = (
            chunk["section"],
            chunk["project"],
            chunk["text"]
        )


        if key in seen:
            continue


        seen.add(key)

        unique_chunks.append(
            chunk
        )


    all_chunks = unique_chunks


    print(
        f"Unique chunks: "
        f"{len(all_chunks)}"
    )


    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    texts = [

        chunk["text"]

        for chunk in all_chunks

    ]


    # ========================================================
    # GENERATE EMBEDDINGS
    # ========================================================

    embeddings = get_embeddings(

        texts,

        input_type="passage"

    )


    # ========================================================
    # NORMALIZE EMBEDDINGS
    #
    # We use IndexFlatIP.
    #
    # Normalized vectors + Inner Product
    # = cosine similarity.
    # ========================================================

    faiss.normalize_L2(
        embeddings
    )


    dimension = embeddings.shape[1]


    print(
        "Embedding dimension:",
        dimension
    )


    # ========================================================
    # CREATE FAISS INDEX
    # ========================================================

    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )


    # ========================================================
    # SAVE INDEX
    # ========================================================

    faiss.write_index(

        index,

        str(INDEX_PATH)

    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    with open(

        META_PATH,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            all_chunks,

            f,

            indent=2,

            ensure_ascii=False

        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("============================================")
    print("✅ INDEX CREATED SUCCESSFULLY")
    print("============================================")

    print(
        "Documents:",
        len(documents)
    )

    print(
        "Chunks:",
        len(all_chunks)
    )

    print(
        "Vectors:",
        index.ntotal
    )

    print(
        "Dimension:",
        index.d
    )

    print(
        "Index:",
        INDEX_PATH
    )

    print(
        "Metadata:",
        META_PATH
    )

    print("============================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    build_index()