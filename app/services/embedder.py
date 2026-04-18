"""
Embeds code chunks with Sentence Transformers and manages FAISS indexes.
One FAISS index per repo, persisted to disk.
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.config import settings


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.embedding_model,
            cache_folder=settings.model_dir,
        )
    return _model


def embed_nodes(nodes: list) -> np.ndarray:
    """
    Build embedding matrix for a list of CodeNode-like objects.
    Each node is embedded as:  node_type + name + code_snippet
    """
    model = get_model()
    texts = [
        f"[{n['node_type']}] {n['name']}\n{n['code_snippet']}"
        for n in nodes
    ]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,   # cosine sim via inner product
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, repo_id: str) -> str:
    """
    Builds a flat inner-product FAISS index (cosine similarity on normalized vecs).
    Returns the path where the index is saved.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(settings.faiss_dir, f"{repo_id}.faiss")
    faiss.write_index(index, index_path)
    return index_path


def load_faiss_index(repo_id: str) -> faiss.Index:
    index_path = os.path.join(settings.faiss_dir, f"{repo_id}.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index for repo {repo_id}")
    return faiss.read_index(index_path)


def query_index(
    query: str,
    repo_id: str,
    nodes: list,
    top_k: int = 8,
) -> list[dict]:
    """
    Embeds the NL query and retrieves top-k most similar code nodes.
    Returns list of node dicts augmented with 'similarity' score.
    """
    model  = get_model()
    index  = load_faiss_index(repo_id)

    q_vec = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(q_vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < len(nodes):
            node = dict(nodes[idx])
            node["similarity"] = float(score)
            node["rank"] = rank + 1
            results.append(node)
    return results
