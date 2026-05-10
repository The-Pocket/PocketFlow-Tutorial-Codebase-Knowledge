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


SAMPLE_ABSTRACTIONS = [
    {"name": "Flow", "description": "Pipeline orchestrator", "files": ["flow.py"]},
    {"name": "Node", "description": "Unit of work", "files": ["nodes.py"]},
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
