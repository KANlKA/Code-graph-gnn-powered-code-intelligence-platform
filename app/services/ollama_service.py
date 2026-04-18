import httpx
import json

OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_MODEL = "llama3"


def _format_suspect_nodes(suspects: list[dict]) -> str:
    if not suspects:
        return "No nodes flagged as suspicious by the GNN."
    lines = []
    for n in suspects[:5]:
        lines.append(
            f"  - [{n['node_type'].upper()}] {n['name']} "
            f"(file: {n['file']}, lines {n['line_start']}-{n['line_end']}, "
            f"gnn_score: {n['gnn_score']:.3f})\n"
            f"    Code:\n    {n['code_snippet'][:400]}"
        )
    return "\n".join(lines)


def _format_rag_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant code retrieved."
    lines = []
    for c in chunks[:6]:
        lines.append(
            f"  [{c['rank']}] {c['name']} - {c['file']} "
            f"(similarity: {c['similarity']:.3f})\n"
            f"    Code:\n    {c['code_snippet'][:500]}"
        )
    return "\n".join(lines)


def build_debug_prompt(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
) -> str:
    return f"""You are CodeGraph, an expert debugging assistant that analyzes Python codebases.

## GNN Analysis - Suspicious Nodes
The Graph Neural Network flagged these nodes as potentially buggy:

{_format_suspect_nodes(suspect_nodes)}

## Retrieved Code (most relevant to the query)
{_format_rag_chunks(rag_chunks)}

## User Query
{user_query}

## Your Task
Give a step-by-step debugging trace:
1. Root Cause Hypothesis - what is most likely wrong?
2. Suspicious Code Walkthrough - why is each flagged node suspicious?
3. Call Chain Analysis - how do the nodes relate to each other?
4. Likely Bug Location - exact file, function, line range
5. Recommended Fix - concrete code changes

Be specific. Reference actual function names and line numbers."""


def stream_debug_trace(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
):
    """Sync generator — yields text chunks from Ollama."""
    prompt = build_debug_prompt(user_query, suspect_nodes, rag_chunks)

    with httpx.stream(
        "POST",
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        timeout=120,
    ) as response:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break


async def astream_debug_trace(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
):
    """Async generator — for FastAPI StreamingResponse."""
    prompt = build_debug_prompt(user_query, suspect_nodes, rag_chunks)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break