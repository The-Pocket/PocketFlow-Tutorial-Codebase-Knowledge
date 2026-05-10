"""Tests for utils/uq_publish.py — opt-in understand-quickly publish."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add project root to path so `from utils.uq_publish import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.uq_publish import build_generic_graph, publish, TOKEN_ENV  # noqa: E402


# Mirrors the actual upstream shape: IdentifyAbstractions emits `files` as a
# list of integer indices into shared["files"] (the (path, content) tuples).
SAMPLE_ABSTRACTIONS = [
    {"name": "Flow", "description": "Pipeline orchestrator", "files": [0]},
    {"name": "Node", "description": "Unit of work", "files": [1]},
]
SAMPLE_FILES_DATA = [
    ("flow.py", "class Flow:\n    pass\n"),
    ("nodes.py", "class Node:\n    pass\n"),
]
SAMPLE_RELATIONSHIPS = {
    "summary": "PocketFlow runs nodes in a flow.",
    "details": [{"from": 0, "to": 1, "label": "contains"}],
}


class BuildGenericGraphTests(unittest.TestCase):
    def test_emits_generic_at_1_with_metadata(self) -> None:
        graph = build_generic_graph(
            project_name="demo",
            abstractions=SAMPLE_ABSTRACTIONS,
            chapter_order=[0, 1],
            relationships=SAMPLE_RELATIONSHIPS,
            repo_url="https://github.com/example/demo",
            source_dir=None,
            files_data=SAMPLE_FILES_DATA,
        )
        self.assertEqual(graph["schema"], "generic@1")
        md = graph["metadata"]
        self.assertEqual(md["tool"], "pocketflow-tutorial-codebase-knowledge")
        self.assertEqual(md["project_name"], "demo")
        self.assertTrue(md["generated_at"].endswith("Z"))
        # Two abstractions -> two nodes; one relationship + one chapter-order edge.
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 2)
        kinds = sorted(e["kind"] for e in graph["edges"])
        self.assertEqual(kinds, ["next_chapter", "relationship"])
        # File indices resolved to repo-relative paths via files_data.
        self.assertEqual(graph["nodes"][0]["files"], ["flow.py"])
        self.assertEqual(graph["nodes"][1]["files"], ["nodes.py"])
        # chapter_index precomputed from chapter_order.
        self.assertEqual(graph["nodes"][0]["chapter_index"], 0)
        self.assertEqual(graph["nodes"][1]["chapter_index"], 1)

    def test_no_files_data_exposes_indices_under_renamed_field(self) -> None:
        graph = build_generic_graph(
            project_name="demo",
            abstractions=SAMPLE_ABSTRACTIONS,
            chapter_order=[0, 1],
            relationships=SAMPLE_RELATIONSHIPS,
            repo_url=None,
            source_dir=None,
        )
        # Without files_data, expose integer indices as `file_indices` (not
        # `files`) so downstream consumers know they're not paths.
        self.assertNotIn("files", graph["nodes"][0])
        self.assertEqual(graph["nodes"][0]["file_indices"], [0])


class PublishTests(unittest.TestCase):
    def test_no_token_writes_file_and_skips_dispatch(self) -> None:
        graph = build_generic_graph(
            project_name="demo",
            abstractions=SAMPLE_ABSTRACTIONS,
            chapter_order=[0, 1],
            relationships=SAMPLE_RELATIONSHIPS,
            repo_url=None,
            source_dir=None,
        )
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "tutorial.json"
                result = publish(graph, out, source_dir=Path(tmp))
                self.assertFalse(result["dispatched"])
                self.assertTrue(out.exists())
                data = json.loads(out.read_text())
                self.assertEqual(data["schema"], "generic@1")
                self.assertEqual(data["metadata"]["tool"],
                                 "pocketflow-tutorial-codebase-knowledge")


if __name__ == "__main__":
    unittest.main()
