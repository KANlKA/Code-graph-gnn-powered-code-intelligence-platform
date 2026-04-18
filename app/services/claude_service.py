"""
Combines GNN-flagged suspicious nodes + RAG-retrieved context
and streams a step-by-step debugging trace from Claude.
"""
import anthropic
from app.config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _format_suspect_nodes(suspects: list[dict]) -> str:
    if not suspects:
        return "No nodes were flagged as suspicious by the GNN."
    lines = []
    for n in suspects[:5]:   # cap at top-5 to stay within context
        lines.append(
            f"  • [{n['node_type'].upper()}] {n['name']} "
            f"(file: {n['file']}, lines {n['line_start']}–{n['line_end']}, "
            f"gnn_score: {n['gnn_score']:.3f})\n"
            f"    ```python\n    {n['code_snippet'][:400]}\n    ```"
        )
    return "\n".join(lines)


def _format_rag_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant code retrieved."
    lines = []
    for c in chunks[:6]:
        lines.append(
            f"  [{c['rank']}] {c['name']} — {c['file']} "
            f"(similarity: {c['similarity']:.3f})\n"
            f"    ```python\n    {c['code_snippet'][:500]}\n    ```"
        )
    return "\n".join(lines)


def build_debug_prompt(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
) -> str:
    return f"""You are CodeGraph, an expert debugging assistant that analyzes Python codebases using Graph Neural Network analysis and semantic code search.

## GNN Analysis — Suspicious Nodes
The Graph Neural Network has flagged the following nodes as potentially containing bugs (scored by learned graph patterns):

{_format_suspect_nodes(suspect_nodes)}

## RAG Retrieval — Semantically Relevant Code
The following code chunks were retrieved as most relevant to the user's query:

{_format_rag_chunks(rag_chunks)}

## User Query
{user_query}

## Your Task
Produce a precise, step-by-step debugging trace. Structure your response as:

1. **Root Cause Hypothesis** — What is most likely wrong, based on GNN scores and code patterns?
2. **Suspicious Code Walkthrough** — Walk through each flagged node, explaining *why* the GNN likely flagged it.
3. **Call Chain Analysis** — How do the suspicious nodes relate to each other in the call graph?
4. **Likely Bug Location** — The single most probable file, function, and line range.
5. **Recommended Fix** — Concrete code-level changes to investigate or apply.

Be specific. Reference actual function names, line numbers, and code patterns. Do not be generic."""


def stream_debug_trace(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
):
    """
    Generator that yields text chunks from Claude's streaming response.
    Usage: for chunk in stream_debug_trace(...): print(chunk, end="", flush=True)
    """
    prompt = build_debug_prompt(user_query, suspect_nodes, rag_chunks)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


async def astream_debug_trace(
    user_query: str,
    suspect_nodes: list[dict],
    rag_chunks: list[dict],
):
    """
    Async generator for FastAPI StreamingResponse.
    """
    prompt = build_debug_prompt(user_query, suspect_nodes, rag_chunks)

    async with anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key
    ).messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text