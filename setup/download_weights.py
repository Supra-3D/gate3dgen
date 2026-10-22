"""Download Gate3DGen replacement tensors and reconstruct complete inference weights."""
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
    parser.add_argument("--release-dir", type=Path, help="Use an existing release root containing finetuned/, without downloading it")
    parser.add_argument("--firered", type=Path, help="Use an existing matching FireRed base, without downloading it")
    parser.add_argument("--verify-only", action="store_true", help="Verify all reconstructed tensors without writing full checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without network or writes")
    args = parser.parse_args(argv)
    destination = args.output_root.expanduser().resolve()
    if destination == ROOT or (ROOT in destination.parents and not destination.is_relative_to(ROOT / "checkpoints")):
        parser.error("In-repository downloads must stay under checkpoints/; external directories are allowed")
    restore = runpy.run_path(str(ROOT / "setup/restore_original_weights.py"))
    release = args.release_dir.expanduser().resolve() if args.release_dir else destination / "release"
    firered = args.firered.expanduser().resolve() if args.firered else destination / "upstream/FireRed-Image-Edit-1.1"
    plan = {"repo_id": args.repo_id, "revision": args.revision, "local_dir": str(destination),
            "release_dir": str(release), "firered": str(firered),
            "base_revision": restore["REVISION"], "authentication": "anonymous",
            "validation": "all reconstructed tensor hashes; no GPU inference", "verify_only": args.verify_only}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from huggingface_hub import snapshot_download
    # Explicitly anonymous: the public core release must not require access approval.
    if args.release_dir is None:
        snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=release,
                          allow_patterns=["finetuned/**"], token=False)
    # Fail before downloading the large base if the replacement release is incomplete.
    for kind in ("restoration", "gateb"):
        manifest = json.loads((release / "finetuned" / kind / "manifest.json").read_text())
        for shard in manifest["shards"]:
            path = release / "finetuned" / kind / shard["path"]
            if not path.is_file() or path.stat().st_size != shard["bytes"]:
                raise ValueError("Missing/truncated replacement shard: " + str(path))
    if args.firered is None:
        snapshot_download(repo_id=restore["BASE_REPO"], revision=restore["REVISION"], local_dir=firered, token=False)
    ready = restore["reconstruct"](release, firered, destination, verify_only=args.verify_only)
    print(json.dumps({**plan, "weights": ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
