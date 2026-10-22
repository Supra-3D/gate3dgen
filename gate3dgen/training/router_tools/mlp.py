"""Independent pooled-VAE MLP baseline; never trains the purifier or changes its checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gate3dgen.evaluation.io import write_jsonl
from gate3dgen.evaluation.router import load_samples, summarize
from gate3dgen.training.latent_mlp import LatentRouterMLP


def load_cache(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    features, labels, rows = value["features"], value["labels"], value["samples"]
    if features.ndim != 2 or len(features) != len(labels) or len(features) != len(rows) or not len(rows):
        raise ValueError("Invalid pooled feature cache")
    if not torch.isfinite(features).all() or set(labels.tolist()) - {0, 1}:
        raise ValueError("Require finite features and binary labels")
    if any(not r.get("asset_id") or not r.get("image_sha256") for r in rows):
        raise ValueError("Cache requires asset identities and image hashes")
    return value


def assert_disjoint(a, b):
    for field in ("asset_id", "image_sha256"):
        if {r[field] for r in a["samples"]} & {r[field] for r in b["samples"]}:
            raise ValueError(f"Train/development {field} overlap")
    if a["vae_identity"] != b["vae_identity"] or a["features"].shape[1] != b["features"].shape[1]:
        raise ValueError("Feature caches use different VAE contracts")


def cache_features(args):
    from diffusers import AutoencoderKLQwenImage
    rows = load_samples(args.manifest, args.data_root, args.texverse_root, args.texverse_full_root)
    if any(not r.get("asset_id") for r in rows):
        raise ValueError("Every labeled image must declare asset_id")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderKLQwenImage.from_pretrained(args.base_model, subfolder="vae", local_files_only=True,
                                              torch_dtype=dtype).to(device).eval().requires_grad_(False)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    features = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image"]) as image:
                if image.size != (1024, 1024):
                    raise ValueError("Supply declared 1024x1024 inputs; implicit resizing is forbidden")
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
            pixels = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].to(device, dtype)
            z = (vae.encode(pixels).latent_dist.mode().float() - mean) / std
            b, c, t, h, w = z.shape
            if t != 1 or h % 2 or w % 2:
                raise ValueError("VAE latent does not meet singleton-temporal 2x2 patch contract")
            packed = z.view(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c*4)
            features.append(packed.mean(1)[0].cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.stack(features), "labels": torch.tensor([r["label"] for r in rows]),
                "samples": rows, "vae_identity": args.vae_identity,
                "encoding": "posterior mode; normalized latents; 2x2 pack; token mean",
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}, args.output)


def train_head(args):
    train, development = load_cache(args.train_cache), load_cache(args.development_cache)
    assert_disjoint(train, development)
    if set(train["labels"].tolist()) != {0, 1}:
        raise ValueError("Training requires both classes")
    if min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise ValueError("Positive steps, batch size and learning rate are required")
    torch.manual_seed(args.seed)
    model = LatentRouterMLP(train["features"].shape[1], 128).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.steps):
        indices = torch.randint(len(train["samples"]), (args.batch_size,), generator=generator)
        logits = model(train["features"][indices].to(args.device)[:, None])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train["labels"][indices].to(args.device).float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "input_dim": model.input_dim, "hidden_dim": 128, "vae_identity": train["vae_identity"],
                "train_asset_ids": sorted({r["asset_id"] for r in train["samples"]}),
                "train_image_hashes": sorted({r["image_sha256"] for r in train["samples"]}),
                "recipe": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                           "weight_decay": args.weight_decay, "seed": args.seed,
                           "selection": "fixed last step; threshold selected separately on development",
                           "original_paper_checkpoint_binding": False}}, args.output)


def evaluate_head(args):
    cache = load_cache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if cache["vae_identity"] != checkpoint["vae_identity"]:
        raise ValueError("VAE identity differs from training")
    if {r["asset_id"] for r in cache["samples"]} & set(checkpoint["train_asset_ids"]) or \
       {r["image_sha256"] for r in cache["samples"]} & set(checkpoint["train_image_hashes"]):
        raise ValueError("Evaluation overlaps training")
    model = LatentRouterMLP(checkpoint["input_dim"], checkpoint["hidden_dim"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.inference_mode():
        scores = model(cache["features"].to(args.device)[:, None]).sigmoid().cpu().tolist()
    rows = [{**r, "purify_probability": p} for r, p in zip(cache["samples"], scores)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summarize(rows, args.threshold), indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--base-model", type=Path, required=True)
    cache.add_argument("--vae-identity", required=True, help="Immutable revision or content hash, same for every split")
    cache.add_argument("--data-root", type=Path, default=Path.cwd())
    cache.add_argument("--texverse-root", type=Path, default=Path("data/TexVerse"))
    cache.add_argument("--texverse-full-root", type=Path, default=Path("data/TexVerse-full"))
    cache.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--train-cache", type=Path, required=True)
    train.add_argument("--development-cache", type=Path, required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--learning-rate", type=float, required=True)
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=1234)
    train.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    for command in (cache, train, evaluate):
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    target = getattr(args, "output", None)
    if target and target.exists():
        raise FileExistsError(target)
    if args.command == "cache":
        cache_features(args)
    elif args.command == "train":
        train_head(args)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Use a new evaluation directory")
        evaluate_head(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
