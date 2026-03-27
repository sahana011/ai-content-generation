"""
vector_store.py
---------------
Handles loading the wedding card JSON dataset, creating embeddings,
and storing/querying a FAISS vector database via LangChain.
Includes caching so embeddings are only computed once.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Cache directory for persisted FAISS index
CACHE_DIR = Path(".cache")
FAISS_INDEX_PATH = CACHE_DIR / "faiss_index"
DATASET_HASH_PATH = CACHE_DIR / "dataset_hash.pkl"


def _compute_dataset_hash(data: list) -> str:
    """Compute a hash of the dataset to detect changes."""
    content = json.dumps(data, sort_keys=True).encode()
    return hashlib.md5(content).hexdigest()


def _get_embeddings_model() -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace sentence-transformers embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_dataset(json_path: str) -> list:
    """Load the wedding card JSON dataset from file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def dataset_to_documents(data: list) -> List[Document]:
    """
    Convert JSON dataset entries into LangChain Document objects.
    Each document's page_content is the Description field.
    Metadata stores SKU, dimensions, weight, and image URL.
    """
    documents = []
    for item in data:
        # Handle both list-of-dicts and dict-of-dicts formats
        if isinstance(item, dict):
            record = item
        else:
            continue

        description = record.get("Description", "").strip()
        if not description:
            continue

        doc = Document(
            page_content=description,
            metadata={
                "sku": record.get("SKU", "N/A"),
                "height": record.get("Height", "N/A"),
                "width": record.get("Width", "N/A"),
                "weight": record.get("Weight", "N/A"),
                "image_url": record.get("Image URL", record.get("image_url", "")),
            },
        )
        documents.append(doc)

    return documents


def build_vector_store(json_path: str, force_rebuild: bool = False) -> FAISS:
    """
    Build or load a cached FAISS vector store from the dataset.

    Args:
        json_path: Path to the JSON dataset file
        force_rebuild: If True, ignore cache and rebuild from scratch

    Returns:
        FAISS vector store instance
    """
    CACHE_DIR.mkdir(exist_ok=True)

    data = load_dataset(json_path)
    current_hash = _compute_dataset_hash(data)
    embeddings = _get_embeddings_model()

    # Check if we can use cached index
    if not force_rebuild and FAISS_INDEX_PATH.exists() and DATASET_HASH_PATH.exists():
        with open(DATASET_HASH_PATH, "rb") as f:
            cached_hash = pickle.load(f)

        if cached_hash == current_hash:
            # Load from cache
            vectorstore = FAISS.load_local(
                str(FAISS_INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return vectorstore

    # Build fresh index
    documents = dataset_to_documents(data)
    if not documents:
        raise ValueError("No valid documents found in the dataset. Check JSON structure.")

    vectorstore = FAISS.from_documents(documents, embeddings)

    # Persist to cache
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    with open(DATASET_HASH_PATH, "wb") as f:
        pickle.dump(current_hash, f)

    return vectorstore


def retrieve_similar_cards(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Retrieve the top-k most similar wedding card descriptions from the vector store.

    Args:
        vectorstore: Initialized FAISS vector store
        query: Natural language query string (from image features)
        k: Number of results to retrieve

    Returns:
        List of (Document, similarity_score) tuples, sorted by relevance
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results
