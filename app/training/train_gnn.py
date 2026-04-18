"""
Trains the BugLocalizationGNN on labeled repo graphs.

Pipeline:
  1. For each repo in REPO_LIST, build graph + embeddings
  2. Apply mined labels to mark buggy nodes
  3. Train GNN with weighted cross-entropy (class imbalance)
  4. Evaluate on held-out repos → report precision

Usage:
    python -m app.training.train_gnn \
        --repos_file repos.txt \
        --labels_file labels.json \
        --epochs 50 \
        --output ./models_cache/gnn_checkpoint.pt
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from app.models.gnn import BugLocalizationGNN
from app.services.graph_builder import GraphBuilder
from app.services.embedder import embed_nodes
from app.services.gnn_inference import build_node_features


def load_repo_graph(repo_path: str, labels: dict) -> Data | None:
    """
    Build PyG Data object for one repo.
    labels: dict mapping function_name → 1 (buggy)
    """
    builder = GraphBuilder()
    nodes, edges, _ = builder.build_from_directory(repo_path)
    if len(nodes) < 5:
        return None

    node_dicts = [
        {
            "node_id": n.node_id, "name": n.name, "node_type": n.node_type,
            "file": n.file, "line_start": n.line_start, "line_end": n.line_end,
            "code_snippet": n.code_snippet,
        }
        for n in nodes
    ]

    embeddings = embed_nodes(node_dicts)
    x = build_node_features(node_dicts, embeddings)

    # Build labels tensor (default = 0, apply mined positives)
    y = torch.zeros(len(nodes), dtype=torch.long)
    for i, n in enumerate(nodes):
        if n.name in labels:
            y[i] = 1

    if edges:
        src = [e.src for e in edges]
        dst = [e.dst for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def train(args):
    repos = Path(args.repos_file).read_text().strip().splitlines()
    raw_labels = json.loads(Path(args.labels_file).read_text())

    # Build per-function label lookup
    label_lookup = {item["function_name"] for item in raw_labels if item["label"] == 1}

    # Split repos 80/20
    split = int(0.8 * len(repos))
    train_repos, test_repos = repos[:split], repos[split:]

    print(f"[Train] Building graphs for {len(train_repos)} train repos...")
    train_data, test_data = [], []

    for path in tqdm(train_repos):
        data = load_repo_graph(path.strip(), label_lookup)
        if data:
            train_data.append(data)

    for path in tqdm(test_repos):
        data = load_repo_graph(path.strip(), label_lookup)
        if data:
            test_data.append(data)

    if not train_data:
        print("[Train] No training data — check repo paths and labels.")
        return

    in_dim = train_data[0].x.shape[1]
    model  = BugLocalizationGNN(in_dim=in_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Class-weighted loss for heavy imbalance
    pos_weight = torch.tensor([1.0, 15.0])
    criterion  = torch.nn.CrossEntropyLoss(weight=pos_weight)

    print(f"[Train] Training for {args.epochs} epochs on {len(train_data)} graphs")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for data in train_data:
            optimizer.zero_grad()
            out  = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} — loss: {total_loss/len(train_data):.4f}")

    # Evaluation on held-out repos
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for data in test_data:
            logits = model(data.x, data.edge_index)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = (probs >= 0.65).long()
            all_preds.extend(preds.tolist())
            all_true.extend(data.y.tolist())

    p = precision_score(all_true, all_preds, zero_division=0)
    r = recall_score(all_true, all_preds, zero_division=0)
    f = f1_score(all_true, all_preds, zero_division=0)
    print(f"\n[Eval] Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f}")

    # Save checkpoint
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[Train] Checkpoint saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos_file",  required=True, help="Text file with one repo path per line")
    parser.add_argument("--labels_file", required=True, help="JSON from mine_labels.py")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--output",      default="./models_cache/gnn_checkpoint.pt")
    args = parser.parse_args()
    train(args)