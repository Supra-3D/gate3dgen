from __future__ import annotations

import argparse
import gc
import json
import os
import signal
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

from PIL import Image

from .io import VARIANTS, available_variants, iter_jsonl, variant_path, write_jsonl
from .provenance import write_run_metadata
from .rembg_adapter import DEFAULT_RMBG2_MODEL, patch_birefnet_class
from .runtime_profile import RuntimeProfiler, aggregate_runtime, generation_record, mark_cached, stage_record


class CaseTimeoutError(TimeoutError):
    """Raised when one pathological TRELLIS.2 case exceeds its wall-time budget."""


@contextmanager
def _case_timeout(seconds: int, case_id: str, variant: str):
    """Bound one case without turning a native xatlas stall into a shard stall."""

    seconds = int(seconds)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame):
        raise CaseTimeoutError(
            f"case_timeout:{seconds}s:{case_id}:{variant}"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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


def _official_variants(case: dict) -> tuple[str, ...]:
    """Select conditions from both the long-form and legacy manifests."""
    if case.get("schema_version") == "official_aligned_v1":
        declared = available_variants(case)
        if case.get("dataset_kind") == "wild_no_gt":
            return tuple(name for name in declared if name == "original" or name.startswith("restored_s"))
        return tuple(name for name in declared if name in {"degraded", "clean_gt"} or name.startswith("restored_s"))
    return available_variants(case, VARIANTS)


def _resolution_value(value: str | int | None) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return [value, value, value]
    text = str(value).replace("_cascade", "")
    try:
        parsed = int(text)
    except ValueError:
        return None
    return [parsed, parsed, parsed]


def _realized_resolution(mesh) -> list[int] | None:
    for name in ("resolution", "grid_size", "grid_resolution"):
        value = getattr(mesh, name, None)
        if value is not None:
            try:
                parsed = int(value)
                return [parsed, parsed, parsed]
            except (TypeError, ValueError):
                pass
    return None


def _sync_cuda() -> None:
    """Synchronize public stage timing when the optional CUDA runtime exists."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # Keep the wrapper importable and testable without CUDA/TRELLIS.
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
    """Atomically rewrite one baseline runtime log and its aggregate."""
    directory = output_dir / "runtime" / "trellis2"
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
    import json
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


def inject_black_background_alpha(image: Image.Image, threshold: int = 2) -> Image.Image:
    """Turn a pure-black render background into an auditable silhouette alpha.

    Filling the largest external contour is intentional: thresholding pixels alone
    would erase genuinely black material inside the object.  This mode is only
    enabled explicitly for intrinsic renders whose background is known to be black.
    """
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    foreground = (rgb.max(axis=2) > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("black-background alpha extraction found no foreground")
    contour = max(contours, key=cv2.contourArea)
    alpha = np.zeros(foreground.shape, dtype=np.uint8)
    cv2.drawContours(alpha, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    if int((alpha > 0).sum()) < 64:
        raise ValueError("black-background alpha extraction produced a degenerate silhouette")
    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, mode="RGBA")


def save_renders(mesh, repo: Path, destination: Path, views: int, resolution: int) -> None:
    import cv2
    import numpy as np
    import torch
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils

    hdri = cv2.imread(str(repo / "assets/hdri/studio.exr"), cv2.IMREAD_UNCHANGED)
    envmap = EnvMap(torch.tensor(cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB), dtype=torch.float32, device="cuda"))
    rendered = render_utils.render_snapshot(mesh, resolution=resolution, nviews=views, r=10, fov=8, envmap=envmap)
    for channel in ("shaded", "base_color", "normal"):
        for index, array in enumerate(rendered[channel]):
            path = destination / channel / f"view_{index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)


def _to_glb_kwargs(mesh, pipeline, decimation_target: int, texture_size: int, *, remesh: bool) -> dict:
    return {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "voxel_size": mesh.voxel_size,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "remesh": remesh,
        "remesh_band": 1,
        "remesh_project": 0,
        "verbose": True,
    }


def _is_cumesh_cuda_config_error(error: BaseException) -> bool:
    text = str(error)
    return "CuMesh" in text or "invalid configuration argument" in text


def export_mesh(mesh, pipeline, destination: Path, decimation_target: int, texture_size: int) -> None:
    import o_voxel

    try:
        glb = o_voxel.postprocess.to_glb(
            **_to_glb_kwargs(mesh, pipeline, decimation_target, texture_size, remesh=True)
        )
    except RuntimeError as error:
        if not _is_cumesh_cuda_config_error(error):
            raise
        print(
            f"[export fallback] remesh=False after CuMesh export failure: {error}",
            flush=True,
        )
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        glb = o_voxel.postprocess.to_glb(
            **_to_glb_kwargs(mesh, pipeline, decimation_target, texture_size, remesh=False)
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TRELLIS.2 generation for legacy or official-aligned image conditions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Local TRELLIS.2 repository")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--sparse-model",
        default=None,
        help=(
            "Override the sparse-structure decoder checkpoint stem, e.g. "
            "/local/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        ),
    )
    parser.add_argument(
        "--rembg-model",
        default=str(DEFAULT_RMBG2_MODEL),
        help=(
            "Local offline RMBG-2.0 checkpoint used for every RGB input "
            f"(default: {DEFAULT_RMBG2_MODEL}); override only with another explicit local checkpoint"
        ),
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override the gated official DINOv3 checkpoint (diagnostic fallback recorded in provenance)",
    )
    parser.add_argument("--pipeline-type", default="1536_cascade", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant", action="append",
        help="Condition id (repeatable); official manifests accept clean_gt/restored_s01...s50, legacy defaults to VARIANTS",
    )
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--render-views", type=int, default=8)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument(
        "--black-background-alpha",
        action="store_true",
        help="Use the largest non-black external contour as alpha (for known black-background intrinsic renders)",
    )
    parser.add_argument("--black-background-threshold", type=int, default=2)
    parser.add_argument("--no-renders", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--case-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Maximum wall time for generation+simplify+export of one condition; "
            "0 disables the timeout"
        ),
    )
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
    if args.case_timeout_seconds < 0:
        raise ValueError("--case-timeout-seconds must be non-negative")

    import sys
    sys.path.insert(0, str(args.repo.resolve()))
    if args.dino_model:
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

        original_dino_init = DinoV3FeatureExtractor.__init__

        def dino_override(self, model_name: str = args.dino_model, image_size: int = 512):
            return original_dino_init(self, model_name=args.dino_model, image_size=image_size)

        DinoV3FeatureExtractor.__init__ = dino_override

    if args.rembg_model:
        from trellis2.pipelines.rembg import BiRefNet
        # Always patch the repository wrapper so the explicit local RMBG-2.0
        # checkpoint (rather than the pipeline.json fallback) is used.
        patch_birefnet_class(BiRefNet, args.rembg_model, device="cuda")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if args.sparse_model:
        import trellis2.models as trellis_models

        original_model_from_pretrained = trellis_models.from_pretrained
        sparse_suffix = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"

        def local_model_from_pretrained(path: str, **kwargs):
            # Pipeline.from_pretrained first tries <local-root>/<declared-ref>
            # and then the declared remote ref.  Intercept both forms so the
            # copied sparse decoder is always used without an HF lookup.
            if str(path).replace("\\", "/").endswith(sparse_suffix):
                return original_model_from_pretrained(args.sparse_model, **kwargs)
            return original_model_from_pretrained(path, **kwargs)

        trellis_models.from_pretrained = local_model_from_pretrained

    requested_variants = tuple(args.variant) if args.variant else None
    runtime_path = args.output_dir / "runtime" / "trellis2" / "per_case.jsonl"
    known_failures = {
        (str(row.get("case_id", "")), str(row.get("variant", "")))
        for row in (iter_jsonl(runtime_path) if args.skip_known_failures and runtime_path.is_file() else ())
        if str(row.get("status", "ok")).lower() in {"error", "failed", "failure", "oom", "unavailable"}
        # An upstream restoration retry can make a previously missing image
        # available.  Do not permanently cache that transient failure.
        and not str(row.get("error") or row.get("reason") or "").startswith("missing_input:")
    }
    write_run_metadata(
        args.output_dir / "run_metadata" / "trellis2.json",
        component="generate_trellis2", arguments=vars(args), manifest=args.manifest, repo=args.repo,
        extra={"checkpoint": args.model, "variants": requested_variants, "official_max_quality": args.pipeline_type == "1536_cascade",
               "input_contract": "ordinary RGB/RGBA accepted; official rembg+crop+resize preprocessing"},
    )
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    profiler = RuntimeProfiler("cuda")
    for case in iter_jsonl(args.manifest):
        variants = _official_variants(case) if requested_variants is None else available_variants(case, requested_variants)
        for variant in variants:
            if (str(case["case_id"]), variant) in known_failures:
                print(f"[skip known failure] {case['case_id']}:{variant}")
                continue
            source = variant_path(case, variant)
            if not source or not Path(source).is_file():
                if args.allow_missing:
                    print(f"[skip missing] {case['case_id']}:{variant}: {source}")
                    _append_runtime(args.output_dir, generation_record(
                        case_id=str(case["case_id"]), baseline="trellis2", variant=variant,
                        status="error", error=f"missing_input:{source}",
                        requested_resolution=_resolution_value(args.pipeline_type),
                        stage_granularity="public_wrapper",
                    ))
                    continue
                raise FileNotFoundError(f"{case['case_id']}:{variant}: {source}")
            destination = args.output_dir / "trellis2" / case["case_id"] / f"{variant}.glb"
            render_root = args.output_dir / "renders" / "trellis2" / case["case_id"] / variant
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
                    case_id=str(case["case_id"]), baseline="trellis2", variant=variant,
                    status="ok", requested_resolution=_resolution_value(args.pipeline_type),
                    glb_path=destination, stage_granularity="public_wrapper",
                ), output_path=destination))
                continue
            stages: list[dict] = []
            measured: dict = {}
            source_resolution = None
            try:
                with Image.open(source) as source_image:
                    source_resolution = [source_image.height, source_image.width]
                with profiler.measure(
                    case_id=str(case["case_id"]), baseline="trellis2", variant=variant,
                    input_resolution=source_resolution,
                    requested_resolution=_resolution_value(args.pipeline_type),
                    resolution_dimensions=3,
                    metadata={"stage_granularity": "public_wrapper"},
                ) as measured:
                    with _timed_stage(stages, "preprocess"):
                        with Image.open(source) as image:
                            condition = image.convert("RGB")
                            if args.black_background_alpha:
                                condition = inject_black_background_alpha(condition, args.black_background_threshold)
                            # Raw RGB/RGBA are accepted; this is the official
                            # pipeline's own rembg/crop/resize preprocessing.
                            condition = pipeline.preprocess_image(condition)
                            condition_path = (
                                args.output_dir / "conditioning" / "trellis2" /
                                case["case_id"] / f"{variant}.png"
                            )
                            condition_path.parent.mkdir(parents=True, exist_ok=True)
                            condition.save(condition_path)
                    # TRELLIS.2 has a fixed camera in this public wrapper; no
                    # hidden camera estimation is performed.
                    stages.append(stage_record(name="camera", status="not_applicable", elapsed_ms=0.0, reason="fixed_public_renderer_camera"))
                    with _case_timeout(
                        args.case_timeout_seconds, str(case["case_id"]), variant
                    ):
                        with _timed_stage(stages, "generation"):
                            mesh = pipeline.run(
                                condition, seed=args.seed, pipeline_type=args.pipeline_type,
                                preprocess_image=False,
                            )[0]
                        # Rendering the raw 1536-cascade surface can exceed
                        # nvdiffrast's practical triangle limit before export.
                        with _timed_stage(stages, "simplify"):
                            mesh.simplify(args.decimation_target)
                        with _timed_stage(stages, "export"):
                            export_mesh(mesh, pipeline, destination, args.decimation_target, args.texture_size)
                    if not args.no_renders:
                        with _timed_stage(stages, "render"):
                            save_renders(
                                mesh, args.repo,
                                args.output_dir / "renders" / "trellis2" / case["case_id"] / variant,
                                args.render_views, args.render_resolution,
                            )
                    else:
                        stages.append(stage_record(name="render", status="not_requested", elapsed_ms=0.0))
                measured["output_resolution"] = source_resolution
                record = generation_record(
                    case_id=str(case["case_id"]), baseline="trellis2", variant=variant,
                    status="ok", total_elapsed_ms=measured.get("elapsed_ms"),
                    requested_resolution=_resolution_value(args.pipeline_type),
                    realized_resolution=_realized_resolution(mesh),
                    peak_allocated_bytes=measured.get("peak_allocated_bytes"),
                    peak_reserved_bytes=measured.get("peak_reserved_bytes"),
                    glb_path=destination,
                    mesh_stats={"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))},
                    stages=stages,
                    stage_granularity="public_wrapper",
                    render_resolution=[args.render_resolution, args.render_resolution],
                )
                _append_runtime(args.output_dir, record)
                print(f"[done] {destination} seconds={float(measured.get('elapsed_ms', 0.0)) / 1000.0:.1f}")
            except Exception as error:
                # RuntimeProfiler already synchronized CUDA and filled peak
                # memory in ``measured`` before re-raising.
                error_text = f"{type(error).__name__}: {error}"
                _append_runtime(args.output_dir, generation_record(
                    case_id=str(case["case_id"]), baseline="trellis2", variant=variant,
                    status="error", total_elapsed_ms=measured.get("elapsed_ms"),
                    requested_resolution=_resolution_value(args.pipeline_type),
                    peak_allocated_bytes=measured.get("peak_allocated_bytes"),
                    peak_reserved_bytes=measured.get("peak_reserved_bytes"),
                    glb_path=destination, stages=stages,
                    stage_granularity="public_wrapper", error=error_text,
                ))
                if not args.continue_on_error:
                    raise
                print(
                    f"[skip error] {case['case_id']}:{variant}: {error_text}",
                    file=sys.stderr,
                    flush=True,
                )
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
        # The public TRELLIS.2 snapshot renderer uses a deterministic camera;
        # persist its contract beside renders so paired PBR evaluators can
        # distinguish it from independently rendered assets.
        camera_path = args.output_dir / "renders" / "trellis2" / str(case["case_id"]) / "camera_params.json"
        camera_path.parent.mkdir(parents=True, exist_ok=True)
        camera_path.write_text(json.dumps({
            "policy": "fixed_public_renderer_camera",
            "camera": {"radius": 10.0, "fov": 8.0, "views": args.render_views},
            "render_resolution": [args.render_resolution, args.render_resolution],
            "pipeline_type": args.pipeline_type,
        }, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
