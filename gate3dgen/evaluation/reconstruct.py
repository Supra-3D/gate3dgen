"""Generate Pixal3D GLBs from matched before/after purification images."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .io import iter_jsonl, write_jsonl
from .official_manifest import build_condition_rows


def condition_rows(manifest: Path, root: Path) -> list[dict]:
    rows = list(iter_jsonl(manifest))
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique case IDs")
    for row in rows:
        if not isinstance(row.get("restored"), str) or not row["restored"]:
            raise ValueError(f"{row['case_id']}: restored image is required")
        row["restored_s01"] = row["restored"]
    result = build_condition_rows(rows, steps=(1,), path_root=root, validate=True)
    for row in result:
        if not row.get("degraded") or not Path(row["degraded"]).is_file():
            raise FileNotFoundError(f"{row['case_id']}: degraded input is required")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("assets/examples/eval/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/new_generation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output = args.output.resolve()
    if output == (root / "assets") or (root / "assets") in output.parents:
        parser.error("Keep new reconstructions outside assets")
    rows = condition_rows(args.manifest, root)
    repo = os.environ.get("PIXAL3D_REPO")
    python = os.environ.get("PIXAL3D_PYTHON", sys.executable)
    model = os.environ.get("PIXAL3D_MODEL", str(root / "checkpoints/upstream/Pixal3D"))
    if not repo or not (Path(repo) / "inference.py").is_file():
        parser.error("Set PIXAL3D_REPO to the installed official Pixal3D repository")
    command = [python, "-m", "gate3dgen.evaluation.generate_pixal3d",
               "--manifest", str(output / "conditions.jsonl"), "--output-dir", str(output),
               "--repo", repo, "--model", model,
               "--variant", "degraded", "--variant", "restored_s01", "--no-renders"]
    if args.dry_run:
        print(json.dumps({"cases": len(rows), "variants": ["degraded", "restored_s01"],
                          "command": command}, indent=2))
        return 0
    if not Path(model).is_dir():
        parser.error("Download Pixal3D with python setup/download_models.py pixal3d or set PIXAL3D_MODEL")
    output.mkdir(parents=True, exist_ok=True)
    conditions = output / "conditions.jsonl"
    if conditions.exists():
        parser.error("Use a new output directory; conditions.jsonl already exists")
    write_jsonl(conditions, rows)
    subprocess.run(command, check=True, cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
