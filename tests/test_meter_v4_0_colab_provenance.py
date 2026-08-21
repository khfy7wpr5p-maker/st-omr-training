from __future__ import annotations

import json
from pathlib import Path
import unittest


class MeterV40ColabProvenanceTests(unittest.TestCase):
    def test_notebook_uses_domain_separated_sha256_binding_for_git_commit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook_path = root / "notebooks" / "st_omr_meter_v4_0_numerator_representation_audit_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        self.assertIn('git_commit_sha = subprocess.run', source)
        self.assertIn('len(git_commit_sha) != 40', source)
        self.assertIn('sha256(("git-commit-sha1:" + git_commit_sha).encode("ascii")).hexdigest()', source)
        self.assertIn('repository_sha=repository_sha', source)
        self.assertIn('repository-provenance.json', source)


if __name__ == "__main__":
    unittest.main()
