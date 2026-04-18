from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import ingest, query, graph
from app.services.embedder import get_model   # preload on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload embedding model into memory once at startup
    print("[Startup] Loading embedding model...")
    get_model()
    print("[Startup] Ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title="CodeGraph API",
    description="GNN-Powered Code Intelligence: Bug Localization + NL Code Search + Claude Debug Traces",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["ingestion"])
app.include_router(query.router,  tags=["query"])
app.include_router(graph.router,  tags=["graph"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CodeGraph"}