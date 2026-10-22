"""Download the complete Gate3DGen inference release from Hugging Face."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "hllee0615/gate3dgen"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main", help="Use a commit hash for reproducibility")
    parser.add_argument("--output-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without network or writes")
    args = parser.parse_args(argv)
    destination = args.output_root.expanduser().resolve()
    if destination == ROOT or (ROOT in destination.parents and not destination.is_relative_to(ROOT / "checkpoints")):
        parser.error("In-repository downloads must stay under checkpoints/; external directories are allowed")
    plan = {"repo_id": args.repo_id, "revision": args.revision, "local_dir": str(destination),
            "authentication": "anonymous", "validation": "weights-only readiness, not full hashing or inference"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from huggingface_hub import snapshot_download
    # Explicitly anonymous: the public core release must not require access approval.
    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=destination, token=False)
    validate = runpy.run_path(str(ROOT / "setup/check.py"))["validate_checkpoints"]
    ready = validate(destination / "router/checkpoint-6000", destination / "purifier/checkpoint-15000",
                     destination / "runtime")
    print(json.dumps({**plan, "weights": ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
