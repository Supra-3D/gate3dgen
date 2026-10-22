"""Download official upstream checkpoints; use --output-root for external test storage."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "firered": ("FireRedTeam/FireRed-Image-Edit-1.1", "FireRed-Image-Edit-1.1"),
    "pixal3d": ("TencentARC/Pixal3D", "Pixal3D"),
}
FIRERED_REVISION = "3bc3f2a12722fd9883eb6357500de191d56baaf5"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*MODELS, "all"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "checkpoints/upstream")
    args = parser.parse_args(argv)
    destination = args.output_root.expanduser().resolve()
    upstream = ROOT / "checkpoints/upstream"
    if destination == ROOT or (ROOT in destination.parents and not destination.is_relative_to(upstream)):
        parser.error("In-repository downloads must stay under checkpoints/upstream; use an external path for tests")
    from huggingface_hub import snapshot_download
    for key in MODELS if args.model == "all" else (args.model,):
        repo, folder = MODELS[key]
        options = {"revision": FIRERED_REVISION, "token": False} if key == "firered" else {}
        snapshot_download(repo_id=repo, local_dir=destination / folder, **options)
        print(f"{key}: {destination / folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
