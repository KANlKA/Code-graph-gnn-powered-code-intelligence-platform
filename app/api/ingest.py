"""
POST /ingest
Clones a GitHub repo, builds graph, embeds all nodes, runs GNN, stores in MongoDB.
"""
import os
import shutil
import hashlib
import tempfile
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from git import Repo
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.services.graph_builder import GraphBuilder
from app.services.embedder import embed_nodes, build_faiss_index
from app.services.gnn_inference import score_nodes

router = APIRouter()


class IngestRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class IngestResponse(BaseModel):
    repo_id: str
    message: str
    node_count: int
    edge_count: int
    total_lines: int


async def _run_ingest(repo_url: str, branch: str, db) -> dict:
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Clone
        print(f"[Ingest] Cloning {repo_url}")
        git_repo = Repo.clone_from(repo_url, tmp_dir, branch=branch, depth=1)
        commit_sha = git_repo.head.commit.hexsha

        # Derive a stable repo_id
        repo_id = hashlib.md5(f"{repo_url}#{commit_sha}".encode()).hexdigest()

        # Check if already ingested
        existing = await db.repos.find_one({"repo_id": repo_id})
        if existing:
            return {"repo_id": repo_id, "cached": True, **existing}

        # 2. Build graph
        print(f"[Ingest] Building code graph")
        builder = GraphBuilder()
        nodes, edges, total_lines = builder.build_from_directory(tmp_dir)
        print(f"[Ingest] {len(nodes)} nodes, {len(edges)} edges, {total_lines} lines")

        if len(nodes) == 0:
            raise ValueError("No Python nodes found in repository.")

        # 3. Embed all nodes
        print(f"[Ingest] Embedding {len(nodes)} nodes")
        node_dicts = [
            {
                "node_id": n.node_id,
                "name": n.name,
                "node_type": n.node_type,
                "file": n.file,
                "line_start": n.line_start,
                "line_end": n.line_end,
                "code_snippet": n.code_snippet,
                "qualified_name": n.qualified_name,
            }
            for n in nodes
        ]
        embeddings = embed_nodes(node_dicts)

        # 4. Build FAISS index
        print(f"[Ingest] Building FAISS index")
        faiss_path = build_faiss_index(embeddings, repo_id)

        # 5. GNN scoring
        print(f"[Ingest] Running GNN inference")
        edge_dicts = [{"src": e.src, "dst": e.dst, "edge_type": e.edge_type} for e in edges]
        scored_nodes = score_nodes(node_dicts, edge_dicts, embeddings)

        suspect_count = sum(1 for n in scored_nodes if n["is_suspect"])
        print(f"[Ingest] {suspect_count} suspect nodes identified")

        # 6. Store in MongoDB
        doc = {
            "repo_id": repo_id,
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "nodes": scored_nodes,
            "edges": edge_dicts,
            "total_lines": total_lines,
            "faiss_index_path": faiss_path,
            "node_count": len(scored_nodes),
            "edge_count": len(edge_dicts),
        }
        await db.repos.insert_one(doc)
        print(f"[Ingest] Stored in MongoDB with repo_id={repo_id}")
        return doc

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repo(req: IngestRequest, background_tasks: BackgroundTasks):
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]

    try:
        result = await _run_ingest(req.repo_url, req.branch, db)
        return IngestResponse(
            repo_id=result["repo_id"],
            message="Repository ingested successfully" if not result.get("cached") else "Already ingested (cached)",
            node_count=result["node_count"],
            edge_count=result["edge_count"],
            total_lines=result["total_lines"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        client.close()