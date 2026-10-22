from __future__ import annotations

import argparse
import gc
import json
import math
import os
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
DEFAULT_MODEL_CACHE = Path(os.environ.get("GATE3DGEN_MODEL_CACHE", _PROJECT_ROOT / "checkpoints/cache"))
os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE / "torch"))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np
from PIL import Image

from .io import VARIANTS, available_variants, iter_jsonl, variant_path, write_jsonl
from .provenance import write_run_metadata
from .rembg_adapter import DEFAULT_RMBG2_MODEL, patch_birefnet_class
from .runtime_profile import RuntimeProfiler, aggregate_runtime, generation_record, mark_cached, stage_record


def _valid_glb(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if not path.is_file() or size < 20:
            return False
        with path.open("rb") as handle:
            magic, version, declared = struct.unpack("<4sII", handle.read(12))
        return magic == b"glTF" and version == 2 and declared == size
    except (OSError, struct.error):
        return False


def _atomic_export_glb(glb, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{time.time_ns()}.partial.glb"
    )
    try:
        glb.export(str(partial), extension_webp=True)
        if not _valid_glb(partial):
            raise RuntimeError(f"export produced an incomplete GLB v2: {partial}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _official_variants(case: dict) -> tuple[str, ...]:
    """Select official condition ids while retaining legacy VARIANTS support."""
    if case.get("schema_version") == "official_aligned_v1":
        declared = available_variants(case)
        if case.get("dataset_kind") == "wild_no_gt":
            return tuple(name for name in declared if name == "original" or name.startswith("restored_s"))
        return tuple(name for name in declared if name in {"degraded", "clean_gt"} or name.startswith("restored_s"))
    return available_variants(case, VARIANTS)


def _resolution_value(value: str | int | None) -> list[int] | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(str(value).replace("_cascade", ""))
    except ValueError:
        return None
    # Image-to-3D pipelines expose a scalar cascade size, but the runtime
    # schema deliberately represents 3D resolution as [x, y, z].
    return [parsed, parsed, parsed]


def _as_resolution(value) -> list[int] | None:
    if value in (None, ""):
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (tuple, list)):
            if len(value) == 1:
                value = value[0]
            elif len(value) >= 3:
                return [int(value[0]), int(value[1]), int(value[2])]
            elif len(value) == 2:
                # A 2D value can occur in older pipeline releases. Preserve
                # both axes and make the depth axis explicit for the new
                # volumetric runtime contract.
                return [int(value[0]), int(value[1]), int(value[1])]
        parsed = int(value)
        return [parsed, parsed, parsed]
    except (TypeError, ValueError):
        return None


def _sync_cuda() -> None:
    """Synchronize public stage timing when the optional CUDA runtime exists."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # Stage timing must remain usable for CPU tests and environments where
        # torch is intentionally not importable.
        pass


@contextmanager
def _timed_stage(stages: list[dict], name: str):
    _sync_cuda()
    started = time.perf_counter()
    try:
        yield
    except Exception as error:
        _sync_cuda()
        stages.append(stage_record(
            name=name,
            status="error",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{type(error).__name__}: {error}",
        ))
        raise
    else:
        _sync_cuda()
        stages.append(stage_record(
            name=name,
            status="ok",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        ))


def _append_runtime(output_dir: Path, record: dict) -> None:
    """Atomically rewrite Pixal3D runtime records and aggregate summary."""
    directory = output_dir / "runtime" / "pixal3d"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "per_case.jsonl"
    previous = list(iter_jsonl(path)) if path.is_file() else []
    key = _runtime_key(record)
    previous = [row for row in previous if _runtime_key(row) != key]
    previous.append(record)
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, previous)
    temporary.replace(path)
    summary = aggregate_runtime(previous)
    summary_path = directory / "summary.json"
    summary_tmp = summary_path.with_name(summary_path.name + ".tmp")
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_tmp.replace(summary_path)


def _runtime_key(record: dict) -> tuple[str, ...]:
    """Identify one sweep point without collapsing resolution variants."""

    return tuple(
        str(record.get(name, ""))
        for name in (
            "case_id",
            "variant",
            "requested_resolution",
            "pipeline_type",
            "seed",
            "render_resolution",
        )
    )


def save_renders(mesh, repo: Path, destination: Path, views: int, resolution: int) -> None:
    import cv2
    import torch
    from pixal3d.renderers import EnvMap
    from pixal3d.utils import render_utils

    hdri = cv2.imread(str(repo / "assets/hdri/studio.exr"), cv2.IMREAD_UNCHANGED)
    envmap = EnvMap(torch.tensor(cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB), dtype=torch.float32, device="cuda"))
    rendered = render_utils.render_snapshot(mesh, resolution=resolution, nviews=views, r=10, fov=8, envmap=envmap)
    for channel in ("shaded", "base_color", "normal"):
        for index, array in enumerate(rendered[channel]):
            path = destination / channel / f"view_{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Pixal3D generation for legacy or official-aligned image conditions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Local Pixal3D repository")
    parser.add_argument("--model", default="TencentARC/Pixal3D")
    parser.add_argument(
        "--dino-model", default=None,
        help="Override every Pixal3D image-condition DINOv3 checkpoint (repo id or local path)",
    )
    parser.add_argument(
        "--moge-model", default=None,
        help="Override the MoGe-2 camera checkpoint (repo id or local path)",
    )
    parser.add_argument(
        "--rembg-model",
        default=str(DEFAULT_RMBG2_MODEL),
        help=(
            "Local offline RMBG-2.0 checkpoint used for every RGB input "
            f"(default: {DEFAULT_RMBG2_MODEL}); override only with another explicit local checkpoint"
        ),
    )
    parser.add_argument("--resolution", type=int, default=1536, choices=[1024, 1536])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0, help="Radians; >0 overrides automatic MoGe-2 camera estimation")
    parser.add_argument(
        "--camera-policy",
        choices=("gt_shared", "input_shared", "per_image"),
        default="gt_shared",
        help="Estimate once from GT, once from degraded/original input, or independently per image",
    )
    parser.add_argument(
        "--segmentation-policy",
        choices=("independent", "shared_reference"),
        default="independent",
        help=(
            "Segment/crop every condition independently (legacy behavior), or reuse one image-space "
            "reference alpha and therefore one crop for every condition in a case. The shared policy "
            "prefers the manifest object_mask and otherwise segments clean_gt/original once."
        ),
    )
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument(
        "--variant", action="append",
        help="Condition id (repeatable); official manifests accept clean_gt/restored_s01...s50, legacy defaults to VARIANTS",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render-views", type=int, default=8)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--no-renders", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed condition in runtime JSONL and continue with the remaining conditions",
    )
    parser.add_argument(
        "--skip-known-failures",
        action="store_true",
        help="On resume, keep prior structured error rows unavailable instead of retrying them",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import inference as official
    import o_voxel
    import torch

    if args.dino_model:
        for config in official.IMAGE_COND_CONFIGS.values():
            config["model_name"] = args.dino_model

    if args.rembg_model:
        from pixal3d.pipelines.rembg import BiRefNet
        # The shared adapter forces the explicit local RMBG-2.0 checkpoint and
        # normalizes its remote-code output without any network fallback.
        patch_birefnet_class(BiRefNet, args.rembg_model, device="cuda")

    requested_variants = tuple(args.variant) if args.variant else None
    runtime_path = args.output_dir / "runtime" / "pixal3d" / "per_case.jsonl"
    known_failures = {
        (str(row.get("case_id", "")), str(row.get("variant", "")))
        for row in (iter_jsonl(runtime_path) if args.skip_known_failures and runtime_path.is_file() else ())
        if str(row.get("status", "ok")).lower() in {"error", "failed", "failure", "oom", "unavailable"}
        # Missing inputs are transient when an upstream restoration shard is
        # repaired.  They must be retried after the manifest path becomes
        # valid instead of being cached forever by --skip-known-failures.
        and not str(row.get("error") or row.get("reason") or "").startswith("missing_input:")
    }
    write_run_metadata(
        args.output_dir / "run_metadata" / "pixal3d.json",
        component="generate_pixal3d", arguments=vars(args), manifest=args.manifest, repo=args.repo,
        extra={"checkpoint": args.model, "variants": requested_variants, "official_max_quality": args.resolution == 1536,
               "input_contract": "ordinary RGB/RGBA accepted; official rembg+crop+resize preprocessing"},
    )
    pipeline = official.init_pipeline(args.model, low_vram=args.low_vram)
    profiler = RuntimeProfiler("cuda")
    moge = None if args.fov > 0 else official.load_moge_model(
        device="cuda",
        model_name=args.moge_model or official.MOGE_MODEL_NAME,
    )
    sampler_ss = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}
    sampler_shape = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    sampler_tex = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}

    def camera_from_processed(processed: Image.Image, temp: Path) -> dict:
        processed.save(temp)
        try:
            return official.get_camera_params_wild_moge(
                str(temp), moge, image_resolution=512
            )
        finally:
            temp.unlink(missing_ok=True)

    def fixed_camera() -> dict:
        grid = torch.tensor([-1.0, 0.0, 0.0])
        distance = official.distance_from_fov(
            args.fov, grid, torch.tensor([0, 511]), 1.0, 512
        )["distance_from_x"]
        return {"camera_angle_x": args.fov, "distance": distance, "mesh_scale": 1.0}

    def meaningful_alpha(image: Image.Image) -> Image.Image | None:
        if image.mode != "RGBA":
            return None
        alpha = image.getchannel("A")
        extrema = alpha.getextrema()
        return alpha if extrema and extrema[0] < 255 else None

    def segment_alpha(image: Image.Image) -> Image.Image:
        existing = meaningful_alpha(image)
        if existing is not None:
            return existing
        # Use the exact remover owned by the initialized official pipeline;
        # this keeps shared-mask preprocessing tied to the checkpoint recorded
        # in run metadata instead of silently loading another model.
        if pipeline.low_vram:
            pipeline.rembg_model.to(getattr(pipeline, "device", getattr(pipeline, "_device", "cuda")))
        segmented = pipeline.rembg_model(image.convert("RGB"))
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
        return segmented.convert("RGBA").getchannel("A")

    def apply_alpha(image: Image.Image, alpha: Image.Image) -> Image.Image:
        resized = alpha if alpha.size == image.size else alpha.resize(image.size, Image.Resampling.BILINEAR)
        rgba = image.convert("RGBA")
        rgba.putalpha(resized)
        return rgba

    for case in iter_jsonl(args.manifest):
        variants = _official_variants(case) if requested_variants is None else available_variants(case, requested_variants)
        case_root = args.output_dir / "pixal3d" / case["case_id"]
        case_root.mkdir(parents=True, exist_ok=True)
        camera_path = case_root / "camera_params.json"
        previous_camera = (
            json.loads(camera_path.read_text(encoding="utf-8"))
            if camera_path.is_file()
            else {}
        )
        shared_camera = None
        shared_alpha = None
        source_key = None
        try:
            if args.camera_policy in {"gt_shared", "input_shared"} or args.segmentation_policy == "shared_reference":
                if args.camera_policy == "input_shared":
                    source_key = "degraded"
                else:
                    source_key = "original" if case.get("dataset_kind") == "wild_no_gt" else "clean_gt"
                shared_path = Path(variant_path(case, source_key) or "")
                if args.camera_policy == "gt_shared" and not shared_path.is_file():
                    source_key = "gt"
                    shared_path = Path(variant_path(case, source_key) or "")
                if args.camera_policy == "input_shared" and not shared_path.is_file():
                    source_key = "original"
                    shared_path = Path(variant_path(case, source_key) or "")
                if not shared_path.is_file():
                    raise FileNotFoundError(f"{case['case_id']}:{source_key}: {shared_path}")
                with Image.open(shared_path) as image:
                    reference_image = image.copy()
                if args.segmentation_policy == "shared_reference":
                    declared_mask = case.get("object_mask") or case.get("object_mask_path")
                    if declared_mask and Path(str(declared_mask)).is_file():
                        with Image.open(Path(str(declared_mask))) as mask_image:
                            shared_alpha = mask_image.convert("L").resize(reference_image.size, Image.Resampling.NEAREST)
                        segmentation_source = "manifest_object_mask"
                    else:
                        shared_alpha = segment_alpha(reference_image)
                        segmentation_source = f"segmented_{source_key}"
                    segmentation_root = args.output_dir / "segmentation" / "pixal3d" / case["case_id"]
                    segmentation_root.mkdir(parents=True, exist_ok=True)
                    shared_alpha.save(segmentation_root / "shared_reference_alpha.png")
                    (segmentation_root / "policy.json").write_text(
                        json.dumps(
                            {
                                "policy": args.segmentation_policy,
                                "source_key": source_key,
                                "source_path": str(shared_path),
                                "segmentation_source": segmentation_source,
                                "rembg_model": args.rembg_model,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    reference_image = apply_alpha(reference_image, shared_alpha)
                gt_processed = pipeline.preprocess_image(reference_image)
            if args.fov > 0:
                shared_camera = fixed_camera()
            elif args.camera_policy in {"gt_shared", "input_shared"}:
                shared_camera = camera_from_processed(
                    gt_processed, case_root / ".gt_shared.camera.png"
                )
        except Exception as error:
            # Camera estimation is shared by every requested condition for a
            # case and therefore happens before the per-condition try block.
            # Treat it like any other recoverable generation failure: persist
            # one unavailable record per requested condition and let the shard
            # continue instead of terminating the whole multi-GPU stage.
            error_text = f"shared_camera:{type(error).__name__}: {error}"
            for variant in variants:
                if (str(case["case_id"]), variant) in known_failures:
                    continue
                _append_runtime(args.output_dir, generation_record(
                    case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                    status="error", error=error_text,
                    requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                    stage_granularity="public_wrapper", camera_policy=args.camera_policy,
                ))
            if not args.continue_on_error:
                raise
            print(
                f"[skip case error] {case['case_id']}: {error_text}",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        camera_records = dict(previous_camera.get("variants", {}))
        for variant in variants:
            if (str(case["case_id"]), variant) in known_failures:
                print(f"[skip known failure] {case['case_id']}:{variant}")
                continue
            source = variant_path(case, variant)
            if not source or not Path(source).is_file():
                if args.allow_missing:
                    print(f"[skip missing] {case['case_id']}:{variant}: {source}")
                    _append_runtime(args.output_dir, generation_record(
                        case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                        status="error", error=f"missing_input:{source}",
                        requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                        stage_granularity="public_wrapper",
                    ))
                    continue
                raise FileNotFoundError(f"{case['case_id']}:{variant}: {source}")
            destination = args.output_dir / "pixal3d" / case["case_id"] / f"{variant}.glb"
            render_root = args.output_dir / "renders" / "pixal3d" / case["case_id"] / variant
            expected_renders = [
                render_root / channel / f"view_{view:02d}.png"
                for channel in ("shaded", "base_color", "normal")
                for view in range(args.render_views)
            ]
            complete = _valid_glb(destination) and (
                args.no_renders or all(path.is_file() for path in expected_renders)
            )
            if complete and not args.overwrite:
                print(f"[skip] {destination}")
                _append_runtime(args.output_dir, mark_cached(generation_record(
                    case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                    status="ok", requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                    glb_path=destination, stage_granularity="public_wrapper",
                ), output_path=destination))
                continue
            stages: list[dict] = []
            measured: dict = {}
            source_resolution = None
            realized_resolution = None
            try:
                with Image.open(source) as source_image:
                    source_resolution = [source_image.height, source_image.width]
                with profiler.measure(
                    case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                    input_resolution=source_resolution,
                    requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                    resolution_dimensions=3,
                    metadata={"stage_granularity": "public_wrapper"},
                ) as measured:
                    with _timed_stage(stages, "preprocess"):
                        with Image.open(source) as image:
                            raw_image = image.copy()
                        if args.segmentation_policy == "shared_reference":
                            if shared_alpha is None:
                                raise RuntimeError("shared_reference segmentation alpha was not initialized")
                            raw_image = apply_alpha(raw_image, shared_alpha)
                        processed = pipeline.preprocess_image(raw_image)
                        conditioning_path = args.output_dir / "conditioning" / "pixal3d" / case["case_id"] / f"{variant}.png"
                        conditioning_path.parent.mkdir(parents=True, exist_ok=True)
                        processed.save(conditioning_path)
                    with _timed_stage(stages, "camera"):
                        if shared_camera is not None:
                            camera = dict(shared_camera)
                            stages[-1]["status"] = "shared_reused"
                        else:
                            camera = camera_from_processed(
                                processed, destination.parent / f".{variant}.camera.png"
                            )
                    camera_records[variant] = camera
                    with _timed_stage(stages, "generation"):
                        meshes, (_, _, resolution) = pipeline.run(
                            processed,
                            camera_params=camera,
                            seed=args.seed,
                            sparse_structure_sampler_params=sampler_ss,
                            shape_slat_sampler_params=sampler_shape,
                            tex_slat_sampler_params=sampler_tex,
                            preprocess_image=False,
                            return_latent=True,
                            pipeline_type=f"{args.resolution}_cascade",
                            max_num_tokens=49152,
                        )
                        realized_resolution = _as_resolution(resolution)
                    mesh = meshes[0]
                    if not args.no_renders:
                        with _timed_stage(stages, "render"):
                            save_renders(
                                mesh, args.repo,
                                args.output_dir / "renders" / "pixal3d" / case["case_id"] / variant,
                                args.render_views, args.render_resolution,
                            )
                    else:
                        stages.append(stage_record(name="render", status="not_requested", elapsed_ms=0.0))
                    with _timed_stage(stages, "export"):
                        glb = o_voxel.postprocess.to_glb(
                            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                            grid_size=resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=1_000_000, texture_size=4096,
                            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                        rotation = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
                        glb.apply_transform(rotation)
                        _atomic_export_glb(glb, destination)
                measured["output_resolution"] = source_resolution
                record = generation_record(
                    case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                    status="ok", total_elapsed_ms=measured.get("elapsed_ms"),
                    requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                    realized_resolution=realized_resolution,
                    peak_allocated_bytes=measured.get("peak_allocated_bytes"),
                    peak_reserved_bytes=measured.get("peak_reserved_bytes"),
                    glb_path=destination,
                    mesh_stats={"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))},
                    stages=stages,
                    stage_granularity="public_wrapper",
                    render_resolution=[args.render_resolution, args.render_resolution],
                    camera_policy=args.camera_policy,
                    segmentation_policy=args.segmentation_policy,
                )
                _append_runtime(args.output_dir, record)
                print(f"[done] {destination} seconds={float(measured.get('elapsed_ms', 0.0)) / 1000.0:.1f}")
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                _append_runtime(args.output_dir, generation_record(
                    case_id=str(case["case_id"]), baseline="pixal3d", variant=variant,
                    status="error", total_elapsed_ms=measured.get("elapsed_ms"),
                    requested_resolution=_resolution_value(f"{args.resolution}_cascade"),
                    realized_resolution=realized_resolution,
                    peak_allocated_bytes=measured.get("peak_allocated_bytes"),
                    peak_reserved_bytes=measured.get("peak_reserved_bytes"),
                    glb_path=destination, stages=stages,
                    stage_granularity="public_wrapper", error=error_text,
                    camera_policy=args.camera_policy,
                    segmentation_policy=args.segmentation_policy,
                ))
                if not args.continue_on_error:
                    raise
                # A bad restored image or a model/export failure must not
                # discard the other conditions in the shard.  Keep the
                # structured runtime failure as coverage evidence and release
                # per-condition objects before attempting the next input.
                print(
                    f"[skip error] {case['case_id']}:{variant}: {error_text}",
                    file=sys.stderr,
                    flush=True,
                )
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
        camera_payload = {
            "policy": "fixed_fov" if args.fov > 0 else args.camera_policy,
            "source_variant": None if args.fov > 0 or args.camera_policy == "per_image" else source_key,
            "variants": camera_records,
        }
        camera_path.write_text(
            json.dumps(camera_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        render_camera_path = args.output_dir / "renders" / "pixal3d" / str(case["case_id"]) / "camera_params.json"
        render_camera_path.parent.mkdir(parents=True, exist_ok=True)
        render_camera_path.write_text(
            json.dumps(camera_payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
