"""Run original native editing baselines with shared prompts and recorded sampler settings."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from PIL import Image
from .io import iter_jsonl, write_jsonl
from gate3dgen.training.prompts import restoration_instructions, DEFAULT_PROMPT_LIBRARY

BACKENDS = {"firered_image_edit_1_1": 40, "qwen_image_edit_2511": 40, "mage_flow_edit": 30,
            "step1x_edit": 50, "omnigen2": 50, "flux_kontext": 30, "gpt_image_2_5_flare": None}
SAMPLERS = {
    "firered_image_edit_1_1": {"true_cfg_scale": 4., "guidance_scale": 1., "negative_prompt": " "},
    "qwen_image_edit_2511": {"true_cfg_scale": 4., "guidance_scale": 1., "negative_prompt": " "},
    "mage_flow_edit": {"cfg_scale": 5., "vlm_long_edge": 384},
    "step1x_edit": {"true_cfg_scale": 6., "thinking": False, "reflection": False},
    "omnigen2": {"text_guidance_scale": 5., "image_guidance_scale": 2., "cfg_range": [0., 1.]},
    "flux_kontext": {"guidance_scale": 2.5},
    "gpt_image_2_5_flare": {"mode": "import original provider outputs; sampler not exposed"},
}


def stable_seed(base_seed, case_id):
    return (int(base_seed) + int.from_bytes(hashlib.sha256(str(case_id).encode()).digest()[:4], "big")) % (2**31)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(args.manifest))
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Require nonempty unique cases")
    for row in rows:
        if Path(row["case_id"]).name != row["case_id"] or row["case_id"] in {".", ".."}:
            raise ValueError("Unsafe case ID")
        with Image.open(row["degraded"]) as image:
            if image.size != (1024, 1024):
                raise ValueError("Native benchmark editors require predeclared 1024x1024 inputs")
    prompt = restoration_instructions({"prompt_library": {"path": str(DEFAULT_PROMPT_LIBRARY), "id": "gate3dgen_purify_input_v1"}})["instruction"]
    metadata = {"backend": args.backend, "steps": BACKENDS[args.backend], "seed_policy": "base_seed + case SHA256 first four bytes mod 2**31",
                "seed": args.seed, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_resolution": [1024, 1024], "protocol": "force-purify", "prompt_rewrite": False,
                "sampler": SAMPLERS[args.backend],
                "model": str(args.model) if args.model else None, "code_root": str(args.code_root) if args.code_root else None}
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty baseline output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "gpt_image_2_5_flare":
        for row in rows:
            if not Path(row.get("baseline_restored", "")).is_file():
                raise ValueError("Closed-source baseline requires original externally generated baseline_restored files")
        runtime = None
        metadata["sampler_parameters"] = "not exposed by closed-source provider; retain original request/response records"
    elif args.backend in {"firered_image_edit_1_1", "qwen_image_edit_2511"}:
        if not args.model:
            parser.error("--model is required")
        import torch
        from diffusers import QwenImageEditPlusPipeline
        runtime = QwenImageEditPlusPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True).to(args.device)
        metadata.update(true_cfg_scale=4., guidance_scale=1., negative_prompt=" ")
    else:
        if not args.model:
            parser.error("--model is required")
        module = importlib.import_module(f".baseline_backends.{args.backend}", package=__package__)
        kwargs = {"device": args.device}
        if args.code_root and args.backend in {"omnigen2", "flux_kontext"}:
            kwargs["code_root"] = args.code_root
        if args.code_root and args.backend == "mage_flow_edit":
            import sys
            sys.path.insert(0, str(args.code_root))
        runtime = module.load_backend(args.model, **kwargs)
        metadata["native"] = runtime.metadata()
    output = []
    for row in rows:
        path = args.output_dir / row["case_id"] / "restored.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = stable_seed(args.seed, row["case_id"])
        if runtime is None:
            with Image.open(row["baseline_restored"]) as image:
                if image.size != (1024, 1024):
                    raise ValueError("Imported baseline dimensions differ from benchmark contract")
                image.convert("RGB").save(path)
        else:
            with Image.open(row["degraded"]) as image:
                image = image.convert("RGB")
                if args.backend in {"firered_image_edit_1_1", "qwen_image_edit_2511"}:
                    result = runtime(image=[image], prompt=prompt, negative_prompt=" ", num_inference_steps=40,
                        true_cfg_scale=4., guidance_scale=1., generator=torch.Generator(device=args.device).manual_seed(seed)).images[0]
                else:
                    result = module.edit_image(runtime, image, prompt, seed=seed)
            if result.size != (1024, 1024):
                raise ValueError("Native backend output dimensions differ from benchmark contract")
            result.save(path)
        output.append({**row, "restored": str(path.resolve()), "baseline": args.backend, "seed": seed, "protocol": "force-purify"})
        write_jsonl(args.output_dir / "manifest.jsonl", output)
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
