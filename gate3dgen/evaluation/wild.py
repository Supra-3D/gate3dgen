"""Table 4 FD-DINOv2 on photographs versus four independent rendered-view samples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .io import iter_jsonl
from .dino_features import LazyDinoV2Provider, frechet_distance


def populations(rows, generators=("pixal3d", "trellis2"), views=4):
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Require nonempty unique case IDs")
    photos = [Path(r["original"]) for r in rows]
    methods = set(rows[0]["renders"][generators[0]])
    if not methods:
        raise ValueError("No declared methods")
    groups = {}
    for gen in generators:
        for method in sorted(methods):
            paths = []
            for row in rows:
                if set(row["renders"][gen]) != methods:
                    raise ValueError("Every generator/case must have the same methods")
                values = row["renders"][gen][method]
                if len(values) != views or len(set(values)) != views:
                    raise ValueError("Declare exactly four distinct shaded views per asset and condition")
                paths.extend(Path(x) for x in values)
            groups[(gen, method)] = paths
    for path in photos + [p for group in groups.values() for p in group]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return photos, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dino-hub-dir", required=True)
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-cases", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(args.manifest))
    if len(rows) != args.expected_cases:
        raise ValueError("Population size differs from --expected-cases")
    photos, groups = populations(rows)
    provider = LazyDinoV2Provider(device=args.device, batch_size=args.batch_size, backend="hub",
        hub_dir=args.dino_hub_dir, hub_model="dinov2_vitl14_reg", weights=args.dino_weights, image_size=224)
    real = provider.features(photos)
    results = [{"generator": gen, "method": method, "fd_dinov2": frechet_distance(provider.features(paths), real),
                "photographs": len(photos), "render_samples": len(paths)} for (gen, method), paths in groups.items()]
    digest = hashlib.sha256()
    with Path(args.dino_weights).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    payload = {"results": results, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "model": "dinov2_vitl14_reg", "weights_sha256": digest.hexdigest(), "resize": [224, 224],
               "normalization": "ImageNet; L2-normalized CLS", "view_pooling": False,
               "reference_is_true_gt": False, "paper_population": len(rows) == 96}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
