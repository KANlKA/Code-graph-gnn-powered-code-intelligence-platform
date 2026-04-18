"""
GET /graph/{repo_id}
Returns the full graph structure for frontend visualization.
"""
from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

router = APIRouter()


@router.get("/graph/{repo_id}")
async def get_graph(repo_id: str):
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]

    try:
        doc = await db.repos.find_one(
            {"repo_id": repo_id},
            {"_id": 0, "nodes": 1, "edges": 1, "repo_url": 1, "total_lines": 1},
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Repo not found")

        # Strip embeddings, keep only what the frontend needs
        lightweight_nodes = [
            {
                "node_id": n["node_id"],
                "name": n["name"],
                "node_type": n["node_type"],
                "file": n["file"],
                "line_start": n["line_start"],
                "is_suspect": n.get("is_suspect", False),
                "gnn_score": n.get("gnn_score", 0.0),
            }
            for n in doc["nodes"]
        ]

        return {
            "repo_id": repo_id,
            "repo_url": doc["repo_url"],
            "total_lines": doc["total_lines"],
            "nodes": lightweight_nodes,
            "edges": doc["edges"],
        }
    finally:
        client.close()


@router.get("/repos")
async def list_repos():
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]
    try:
        repos = await db.repos.find(
            {}, {"_id": 0, "repo_id": 1, "repo_url": 1, "node_count": 1, "total_lines": 1}
        ).to_list(50)
        return repos
    finally:
        client.close()