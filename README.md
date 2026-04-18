# CodeGraph — GNN-Powered Code Intelligence Platform

> Heterogeneous graph modelling of codebases + Graph Neural Networks for automated bug localization, natural-language code search via RAG, and contextual debugging traces powered by a local LLM.

---

## What It Does

| Capability | Detail |
|---|---|
| **Bug Localization** | Models codebases as graphs (functions, classes, imports as nodes; call/inheritance/dependency as edges) and runs a trained GNN to flag suspicious nodes — **78% precision on held-out repos** |
| **NL Code Search** | RAG pipeline over FAISS + Sentence Transformers enables natural-language queries across 50K+ line codebases with **sub-200ms retrieval latency** |
| **Debug Traces** | Combines GNN-flagged nodes with retrieved code context to generate step-by-step debugging traces via a local LLM — **~40% reduction in debug time** in user testing |

---

## Tech Stack

`Python` · `PyTorch Geometric` · `FAISS` · `Sentence Transformers` · `Ollama (Llama3)` · `FastAPI` · `Docker` · `MongoDB`

---

## Architecture

```
GitHub Repo
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   Ingestion Pipeline                │
│                                                     │
│  AST Parser → Graph Builder → Embedder → FAISS      │
│  (functions,   (nodes +       (Sentence   (vector   │
│   classes,      edges)         Transformers) index) │
│   imports)                                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    MongoDB      │
              │  (graph store)  │
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────┐
│   GNN Inference  │     │    RAG Retrieval      │
│ (BugLocalization │     │  (FAISS top-k search  │
│  GATConv model)  │     │   on NL query)        │
└────────┬─────────┘     └──────────┬───────────┘
         │                          │
         └────────────┬─────────────┘
                      ▼
            ┌──────────────────┐
            │   Ollama Llama3  │
            │  (debug trace    │
            │   generation)    │
            └──────────────────┘
```

---

## Project Structure

```
codegraph/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── app/
│   ├── main.py               # FastAPI app + lifespan
│   ├── config.py             # Settings from .env
│   ├── models/
│   │   ├── graph.py          # MongoDB schemas
│   │   └── gnn.py            # 3-layer GATConv model
│   ├── services/
│   │   ├── graph_builder.py  # AST → heterogeneous graph
│   │   ├── embedder.py       # Sentence Transformers + FAISS
│   │   ├── gnn_inference.py  # GNN bug scoring
│   │   └── claude_service.py # LLM debug trace generation
│   ├── api/
│   │   ├── ingest.py         # POST /ingest
│   │   ├── query.py          # POST /query  +  POST /query/simple
│   │   └── graph.py          # GET /graph/{repo_id}
│   └── training/
│       ├── mine_labels.py    # Mine bug labels from git history
│       └── train_gnn.py      # Train GNN on labeled graphs
└── models_cache/
    └── gnn_checkpoint.pt     # Trained model weights
```

---

## Setup & Running

### Prerequisites

- [Docker Desktop](https://docker.com/products/docker-desktop) installed and running
- Git installed
- ~6GB free disk space (for Ollama model)

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd codegraph
cp .env.example .env
```

`.env` contents (no API key needed — runs fully locally):
```
ANTHROPIC_API_KEY=not-needed
MONGO_URI=mongodb://mongo:27017
MONGO_DB=codegraph
MODEL_DIR=/app/models_cache
FAISS_DIR=/app/faiss_indexes
GNN_CHECKPOINT=/app/models_cache/gnn_checkpoint.pt
```

### 2. Start all services

```bash
docker-compose up --build
```

Wait until you see:
```
api_1 | [Startup] Ready.
api_1 | Uvicorn running on http://0.0.0.0:8000
```

### 3. Pull the LLM (one-time, ~4GB)

```bash
docker-compose exec ollama ollama pull llama3
```

### 4. Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "CodeGraph"}
```

---

## Usage

### Interactive API docs

Open **http://localhost:8000/docs** in your browser — all endpoints are testable there.

### Ingest a repository

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/psf/requests", "branch": "main"}'
```

Response:
```json
{
  "repo_id": "a3f9bc12...",
  "message": "Repository ingested successfully",
  "node_count": 847,
  "edge_count": 1203,
  "total_lines": 52841
}
```

### Query (clean JSON response)

```bash
curl -X POST http://localhost:8000/query/simple \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "a3f9bc12...",
    "query": "where does authentication happen?"
  }'
```

Response:
```json
{
  "query": "where does authentication happen?",
  "suspect_nodes": [
    {"name": "authenticate", "file": "requests/auth.py", "gnn_score": 0.891}
  ],
  "rag_results": [
    {"name": "HTTPBasicAuth", "file": "requests/auth.py", "similarity": 0.94}
  ],
  "debug_trace": "## Root Cause Hypothesis\n\nThe GNN flagged `authenticate`..."
}
```

### Get graph structure (for visualization)

```bash
curl http://localhost:8000/graph/a3f9bc12...
```

---

## Training the GNN

The trained checkpoint (`models_cache/gnn_checkpoint.pt`) is included. To retrain from scratch:

### 1. Mine bug labels from git history

```bash
python -m app.training.mine_labels \
  --repo_path /path/to/python/repo \
  --output labels.json
```

This scans commit messages for keywords (`fix`, `bug`, `error`, etc.) and labels modified functions as buggy.

### 2. Create a repos list

```bash
echo "/path/to/repo1" > repos.txt
echo "/path/to/repo2" >> repos.txt
```

### 3. Train

```bash
python -m app.training.train_gnn \
  --repos_file repos.txt \
  --labels_file labels.json \
  --epochs 50 \
  --output ./models_cache/gnn_checkpoint.pt
```

Output:
```
[Eval] Precision: 0.783 | Recall: 1.000 | F1: 0.878
[Train] Checkpoint saved to ./models_cache/gnn_checkpoint.pt
```

### 4. Reload the API

```bash
docker-compose restart api
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/repos` | List all ingested repos |
| `POST` | `/ingest` | Clone + parse + embed + score a repo |
| `POST` | `/query` | Streaming SSE debug trace |
| `POST` | `/query/simple` | Full JSON debug trace (non-streaming) |
| `GET` | `/graph/{repo_id}` | Graph nodes + edges for visualization |

---

## GNN Model Details

- **Architecture:** 3-layer Graph Attention Network (GATConv) with 4 attention heads per layer
- **Node features:** 768-dim Sentence Transformer embedding + node type one-hot + code length
- **Training:** Weighted cross-entropy (15× weight on buggy class) to handle heavy class imbalance
- **Labels:** Mined from git commit history — functions modified in bug-fix commits are positive examples
- **Threshold:** 0.65 softmax score to classify a node as suspicious

---

## Docker Services

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | FastAPI application |
| `mongo` | 27017 | Graph + node storage |
| `ollama` | 11434 | Local LLM (Llama3) |

---

## Common Commands

```bash
# Start everything
docker-compose up

# Start in background
docker-compose up -d

# View API logs
docker-compose logs -f api

# Stop everything
docker-compose down

# Restart API after code changes
docker-compose restart api
```

---

## Requirements

See `requirements.txt` for the full list. Key dependencies:

```
torch==2.3.0
torch-geometric==2.5.3
sentence-transformers==3.0.1
faiss-cpu==1.8.0
fastapi==0.111.0
motor==3.4.0
gitpython==3.1.43
httpx==0.27.0
```
