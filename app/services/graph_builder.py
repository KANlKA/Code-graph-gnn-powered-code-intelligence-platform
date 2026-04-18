"""
Parses a Python repository into a heterogeneous graph.
Nodes: functions, classes, imports
Edges: calls, inherits, imports
"""
import ast
import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import networkx as nx


@dataclass
class CodeNode:
    node_id: int
    name: str
    node_type: str           # "function" | "class" | "import"
    file: str
    line_start: int
    line_end: int
    code_snippet: str
    qualified_name: str      # module.ClassName.method_name


@dataclass
class CodeEdge:
    src: int
    dst: int
    edge_type: str           # "calls" | "inherits" | "imports"


class GraphBuilder:
    def __init__(self):
        self.nodes: list[CodeNode] = []
        self.edges: list[CodeEdge] = []
        self._node_index: dict[str, int] = {}   # qualified_name → node_id
        self._counter = 0

    def _next_id(self) -> int:
        i = self._counter
        self._counter += 1
        return i

    def _get_or_create_node(
        self,
        qualified_name: str,
        name: str,
        node_type: str,
        file: str,
        line_start: int,
        line_end: int,
        code_snippet: str,
    ) -> int:
        if qualified_name in self._node_index:
            return self._node_index[qualified_name]
        nid = self._next_id()
        self.nodes.append(CodeNode(
            node_id=nid,
            name=name,
            node_type=node_type,
            file=file,
            line_start=line_start,
            line_end=line_end,
            code_snippet=code_snippet,
            qualified_name=qualified_name,
        ))
        self._node_index[qualified_name] = nid
        return nid

    def _extract_snippet(self, source_lines: list[str], start: int, end: int) -> str:
        snippet = "".join(source_lines[start - 1 : end])
        return snippet[:1200]   # cap at 1200 chars

    def _parse_file(self, filepath: str, module_prefix: str, source: str):
        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            return

        source_lines = source.splitlines(keepends=True)
        # symbol table: local name → qualified_name  (for resolving calls)
        local_symbols: dict[str, str] = {}

        # ── First pass: register all definitions ─────────────────────────
        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):
                qname = f"{module_prefix}.{node.name}"
                end   = getattr(node, "end_lineno", node.lineno)
                self._get_or_create_node(
                    qname, node.name, "class", filepath,
                    node.lineno, end,
                    self._extract_snippet(source_lines, node.lineno, end),
                )
                local_symbols[node.name] = qname

                # methods inside the class
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item is not node:
                        mname  = f"{qname}.{item.name}"
                        mend   = getattr(item, "end_lineno", item.lineno)
                        self._get_or_create_node(
                            mname, item.name, "function", filepath,
                            item.lineno, mend,
                            self._extract_snippet(source_lines, item.lineno, mend),
                        )
                        local_symbols[item.name] = mname

            elif isinstance(node, ast.FunctionDef) and not isinstance(
                getattr(node, "parent", None), ast.ClassDef
            ):
                qname = f"{module_prefix}.{node.name}"
                end   = getattr(node, "end_lineno", node.lineno)
                self._get_or_create_node(
                    qname, node.name, "function", filepath,
                    node.lineno, end,
                    self._extract_snippet(source_lines, node.lineno, end),
                )
                local_symbols[node.name] = qname

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    iname = alias.asname or alias.name
                    qname = f"import.{alias.name}"
                    if qname not in self._node_index:
                        nid = self._next_id()
                        self.nodes.append(CodeNode(
                            node_id=nid, name=iname, node_type="import",
                            file=filepath, line_start=node.lineno,
                            line_end=node.lineno,
                            code_snippet=ast.unparse(node),
                            qualified_name=qname,
                        ))
                        self._node_index[qname] = nid
                    local_symbols[iname] = qname

        # ── Second pass: register edges ───────────────────────────────────
        for node in ast.walk(tree):

            # CALL edges
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller_qname = local_symbols.get(node.name)
                if not caller_qname:
                    continue
                caller_id = self._node_index.get(caller_qname)
                if caller_id is None:
                    continue
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = None
                        if isinstance(child.func, ast.Name):
                            callee_name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            callee_name = child.func.attr
                        if callee_name and callee_name in local_symbols:
                            callee_id = self._node_index.get(local_symbols[callee_name])
                            if callee_id is not None and callee_id != caller_id:
                                self.edges.append(CodeEdge(caller_id, callee_id, "calls"))

            # INHERITS edges
            elif isinstance(node, ast.ClassDef):
                child_qname = f"{module_prefix}.{node.name}"
                child_id    = self._node_index.get(child_qname)
                if child_id is None:
                    continue
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name and base_name in local_symbols:
                        parent_id = self._node_index.get(local_symbols[base_name])
                        if parent_id is not None:
                            self.edges.append(CodeEdge(child_id, parent_id, "inherits"))

            # IMPORTS edges  (module-level function → import node)
            elif isinstance(node, ast.FunctionDef):
                fn_qname = local_symbols.get(node.name)
                if not fn_qname:
                    continue
                fn_id = self._node_index.get(fn_qname)
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id in local_symbols:
                        dep_qname = local_symbols[child.id]
                        dep_id    = self._node_index.get(dep_qname)
                        if dep_id is not None:
                            n = self.nodes[dep_id]
                            if n.node_type == "import" and fn_id is not None:
                                self.edges.append(CodeEdge(fn_id, dep_id, "imports"))

    def build_from_directory(self, repo_path: str) -> tuple[list[CodeNode], list[CodeEdge], int]:
        total_lines = 0
        for root, _, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    source = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                total_lines += source.count("\n")
                # derive module prefix from relative path
                rel = os.path.relpath(fpath, repo_path)
                module_prefix = rel.replace(os.sep, ".").removesuffix(".py")
                self._parse_file(fpath, module_prefix, source)

        # deduplicate edges
        seen = set()
        unique_edges = []
        for e in self.edges:
            key = (e.src, e.dst, e.edge_type)
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)
        self.edges = unique_edges

        return self.nodes, self.edges, total_lines