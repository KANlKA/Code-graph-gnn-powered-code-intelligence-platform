from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class NodeModel(BaseModel):
    node_id: int
    name: str
    node_type: str          # "function" | "class" | "import"
    file: str
    line_start: int
    line_end: int
    code_snippet: str
    embedding_index: int    # index into FAISS
    is_suspect: bool = False
    gnn_score: float = 0.0


class EdgeModel(BaseModel):
    src: int
    dst: int
    edge_type: str          # "calls" | "inherits" | "imports"


class RepoGraph(BaseModel):
    repo_url: str
    commit_sha: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    nodes: list[NodeModel] = []
    edges: list[EdgeModel] = []
    total_lines: int = 0
    faiss_index_path: Optional[str] = None
    gnn_scored: bool = False