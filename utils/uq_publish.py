"""Opt-in understand-quickly registry publish for PocketFlow-Tutorial-Codebase-Knowledge.

Emits a small `generic@1` knowledge-graph projection of the generated tutorial
(nodes = chapters, edges = chapter relationships) and, if a token is set,
fires a `repository_dispatch` at the registry.

Stdlib-only — no new dependencies.

Spec: https://github.com/looptech-ai/understand-quickly/blob/main/docs/spec/code-graph-protocol.md
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess  # nosec B404 — fixed argv, no shell
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOOL_NAME = "pocketflow-tutorial-codebase-knowledge"
TOOL_VERSION = "0.1.0"
REGISTRY_REPO = "looptech-ai/understand-quickly"
TOKEN_ENV = "UNDERSTAND_QUICKLY_TOKEN"
DISPATCH_EVENT_TYPE = "uq-publish"


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(  # nosec B603
            ["git", *args], cwd=str(cwd), capture_output=True, text=True,
            check=False, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _git_head(repo_dir: Path) -> str | None:
    sha = _git(["rev-parse", "HEAD"], repo_dir)
    return sha if sha and len(sha) == 40 else None


def _detect_repo_slug(repo_dir: Path, repo_url: str | None = None) -> str | None:
    """Best-effort `owner/repo` slug — honours `repo_url` first (PocketFlow
    typically tutorialises a remote repo, not the cwd)."""
    candidates: list[str] = []
    if repo_url:
        candidates.append(repo_url)
    origin = _git(["remote", "get-url", "origin"], repo_dir)
    if origin:
        candidates.append(origin)
    for url in candidates:
        for prefix in ("https://github.com/", "git@github.com:"):
            if url.startswith(prefix):
                slug = url[len(prefix):].removesuffix(".git")
                if slug and "/" in slug:
                    return slug
    return None


def build_generic_graph(
    *,
    project_name: str,
    abstractions: list[dict],
    chapter_order: list[int],
    relationships: dict,
    repo_url: str | None,
    source_dir: Path | None,
) -> dict:
    """Project the tutorial onto a `generic@1` node/edge graph.

    Each abstraction becomes a node (kind=abstraction). The edges capture the
    `relationships.details` produced by AnalyzeRelationships and chapter ordering.
    """
    nodes: list[dict] = []
    for i, abstr in enumerate(abstractions):
        nodes.append({
            "id": f"A{i}",
            "label": abstr.get("name", f"abstraction {i}"),
            "kind": "abstraction",
            "description": abstr.get("description", ""),
            "files": list(abstr.get("files", [])),
            "chapter_index": chapter_order.index(i) if i in chapter_order else None,
        })
    edges: list[dict] = []
    for rel in (relationships or {}).get("details", []):
        edges.append({
            "source": f"A{rel['from']}",
            "target": f"A{rel['to']}",
            "label": rel.get("label", ""),
            "kind": "relationship",
        })
    # Chapter-order edges (A_i -> A_{i+1}) for prerequisite-style traversal.
    for prev, curr in zip(chapter_order, chapter_order[1:]):
        edges.append({
            "source": f"A{prev}",
            "target": f"A{curr}",
            "kind": "next_chapter",
        })

    commit = _git_head(source_dir) if source_dir else None
    metadata: dict[str, Any] = {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "project_name": project_name,
        "summary": (relationships or {}).get("summary", ""),
    }
    if commit:
        metadata["commit"] = commit
    if repo_url:
        metadata["repo_url"] = repo_url
    return {
        "schema": "generic@1",
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
    }


def write_graph(graph: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return output_path


def dispatch(repo_slug: str, *, token: str, schema: str, graph_path: str,
             commit: str | None = None, timeout: float = 10.0) -> int:
    payload = {
        "event_type": DISPATCH_EVENT_TYPE,
        "client_payload": {
            "repo": repo_slug, "schema": schema, "graph_path": graph_path,
            "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
            **({"commit": commit} if commit else {}),
        },
    }
    req = urllib.request.Request(  # nosec B310 — fixed https URL
        f"https://api.github.com/repos/{REGISTRY_REPO}/dispatches",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": f"{TOOL_NAME}/{TOOL_VERSION}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return resp.status


def publish(
    graph: dict,
    output_path: Path,
    *,
    repo_url: str | None = None,
    source_dir: Path | None = None,
    token_env: str = TOKEN_ENV,
    log: Any = None,
) -> dict[str, Any]:
    """Write the graph and (if token set) dispatch. Never raises on network errors."""
    log = log or sys.stderr
    write_graph(graph, output_path)
    metadata = graph.get("metadata", {})

    token = os.environ.get(token_env, "").strip()
    if not token:
        print(
            f"[uq-publish] wrote {output_path}; ${token_env} unset — "
            f"skipping registry dispatch (see "
            f"https://github.com/looptech-ai/uq-publish-action for CI use).",
            file=log,
        )
        return {"dispatched": False, "metadata": metadata}

    repo_slug = _detect_repo_slug(source_dir or Path.cwd(), repo_url)
    if not repo_slug:
        print("[uq-publish] could not detect github repo slug — skipping dispatch.",
              file=log)
        return {"dispatched": False, "metadata": metadata}

    try:
        status = dispatch(
            repo_slug, token=token, schema=graph.get("schema", "generic@1"),
            graph_path=str(output_path), commit=metadata.get("commit"),
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"[uq-publish] {repo_slug} not in registry — register once with: "
                  "npx @understand-quickly/cli add", file=log)
            return {"dispatched": False, "metadata": metadata, "registered": False}
        print(f"[uq-publish] dispatch failed ({exc.code}); local file written.",
              file=log)
        return {"dispatched": False, "metadata": metadata, "error": str(exc)}
    except (urllib.error.URLError, OSError) as exc:
        print(f"[uq-publish] dispatch failed ({exc}); local file written.", file=log)
        return {"dispatched": False, "metadata": metadata, "error": str(exc)}

    print(f"[uq-publish] dispatched to {REGISTRY_REPO} (HTTP {status}) for "
          f"{repo_slug}.", file=log)
    return {"dispatched": True, "metadata": metadata, "status": status}
