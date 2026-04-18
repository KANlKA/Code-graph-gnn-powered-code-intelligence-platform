"""
POST /query
Takes NL query + repo_id, runs RAG retrieval, combines with GNN suspects,
streams Claude debug trace.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.services.embedder import query_index
from app.services.ollama_service import astream_debug_trace

router = APIRouter()


class QueryRequest(BaseModel):
    repo_id: str
    query: str
    top_k: int = 8


@router.post("/query")
async def query_repo(req: QueryRequest):
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db]

    try:
        repo_doc = await db.repos.find_one({"repo_id": req.repo_id})
        if not repo_doc:
            raise HTTPException(status_code=404, detail="Repo not found. Run /ingest first.")

        nodes = repo_doc["nodes"]

        # 1. RAG: retrieve top-k relevant nodes via FAISS
        rag_results = query_index(
            query=req.query,
            repo_id=req.repo_id,
            nodes=nodes,
            top_k=req.top_k,
        )

        # 2. GNN suspects: pull pre-scored suspect nodes, sorted by score
        suspect_nodes = sorted(
            [n for n in nodes if n.get("is_suspect")],
            key=lambda n: n.get("gnn_score", 0),
            reverse=True,
        )

        # 3. Stream Claude trace
        async def generate():
            # Send metadata header first
            import json
            meta = {
                "type": "meta",
                "suspect_count": len(suspect_nodes),
                "rag_count": len(rag_results),
                "top_suspects": [
                    {"name": n["name"], "gnn_score": n["gnn_score"], "file": n["file"]}
                    for n in suspect_nodes[:3]
                ],
            }
            yield f"data: {json.dumps(meta)}\n\n"

            # Stream Claude response
            async for chunk in astream_debug_trace(req.query, suspect_nodes, rag_results):
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        client.close()