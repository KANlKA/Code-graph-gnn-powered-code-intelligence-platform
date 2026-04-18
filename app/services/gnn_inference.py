"""
Loads the trained GNN checkpoint and scores all nodes in a repo graph.
Returns each node annotated with is_suspect + gnn_score.
"""
import torch
import numpy as np
from torch_geometric.data import Data
from app.models.gnn import BugLocalizationGNN
from app.config import settings

_gnn_model: BugLocalizationGNN | None = None


def _load_gnn(in_dim: int) -> BugLocalizationGNN:
    global _gnn_model
    if _gnn_model is None:
        model = BugLocalizationGNN(in_dim=in_dim)
        checkpoint_path = settings.gnn_checkpoint
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"[GNN] Loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print("[GNN] No checkpoint found — using random weights (run training first)")
        model.eval()
        _gnn_model = model
    return _gnn_model


def build_node_features(nodes: list, embeddings: np.ndarray) -> torch.Tensor:
    """
    Node features = sentence embedding + 3 structural features:
      - node_type one-hot [function, class, import]
      - normalized degree (filled later)
      - code length (normalized)
    """
    type_map = {"function": [1, 0, 0], "class": [0, 1, 0], "import": [0, 0, 1]}
    structural = []
    for n in nodes:
        t = type_map.get(n.get("node_type", "function"), [1, 0, 0])
        code_len = min(len(n.get("code_snippet", "")) / 1200.0, 1.0)
        structural.append(t + [code_len])

    structural_tensor = torch.tensor(structural, dtype=torch.float32)   # [N, 4]
    embed_tensor      = torch.tensor(embeddings, dtype=torch.float32)   # [N, D]
    return torch.cat([embed_tensor, structural_tensor], dim=1)           # [N, D+4]


def score_nodes(
    nodes: list,
    edges: list,
    embeddings: np.ndarray,
) -> list[dict]:
    """
    Run GNN inference. Returns nodes dicts with 'gnn_score' and 'is_suspect'.
    """
    if len(nodes) == 0:
        return []

    x = build_node_features(nodes, embeddings)

    # Build edge_index tensor
    if edges:
        src = [e.get("src", e["src"]) for e in edges]
        dst = [e.get("dst", e["dst"]) for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    model = _load_gnn(in_dim=x.shape[1])

    with torch.no_grad():
        logits = model(x, edge_index)                         # [N, 2]
        probs  = torch.softmax(logits, dim=1)[:, 1].numpy()  # P(buggy)

    threshold = settings.gnn_suspect_threshold
    scored_nodes = []
    for node, score in zip(nodes, probs):
        d = dict(node)
        d["gnn_score"]  = round(float(score), 4)
        d["is_suspect"] = bool(score >= threshold)
        scored_nodes.append(d)

    return scored_nodes