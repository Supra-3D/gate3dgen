"""Render metric-grade, shared-camera PBR channels from geometry metric rows.

Run inside Blender.  Prediction rows carry the normalized prediction-to-GT
similarity estimated by ``metrics3d_official``.  Both sides are normalized to
the same unit-cube frame and rendered with the same deterministic camera grid.
These PNGs are quantitative inputs; they are independent of the qualitative
360-degree MP4 renderer.

``turntable`` keeps the historical four-view unit-cube camera.  ``benchmark``
reads the exact camera matrix from ``benchmark_plan.json`` and uses the DTC
benchmark object frame for both GT and prediction.  The latter is the
production contract shared by Pixal3D and TRELLIS.2; it never consults a
Pixal3D ``camera_params.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

# Blender has its own Python ABI. Do not put the host conda site-packages here.
runtime_path = os.environ.get("GATE3DGEN_BLENDER_PYTHONPATH")
if runtime_path:
    sys.path.insert(0, runtime_path)

import bpy
from mathutils import Matrix, Vector

# Ubuntu's Blender 3.0 glTF importer still references ``np.bool``.  Local
# rendering supplies NumPy from a compatible Python-3.10 conda environment;
# restore this removed alias without affecting newer Blender importers.
try:
    import numpy as _numpy
    if "bool" not in _numpy.__dict__:
        _numpy.bool = bool
except ImportError:
    pass


CHANNELS = ("normal", "mask", "base_color", "metallic", "roughness", "alpha", "shaded")


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prediction", "gt", "wild"), required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--light-scale", type=float,
        default=float(os.environ.get("GATE3DGEN_PBR_LIGHT_SCALE", "1.0")),
        help="scale the shaded-preview area-light energies; 1.0 keeps the original brightness",
    )
    parser.add_argument(
        "--channel", action="append", dest="channels", choices=CHANNELS,
        help="render only selected channels; repeatable, defaults to all seven",
    )
    parser.add_argument(
        "--camera-mode", choices=("turntable", "benchmark"),
        default=os.environ.get("GATE3DGEN_PBR_CAMERA_MODE", "turntable"),
        help="turntable keeps the legacy camera; benchmark uses benchmark_plan.json",
    )
    parser.add_argument("--camera-plan", type=Path, help="benchmark_plan.json for --camera-mode benchmark")
    parser.add_argument("--camera-fov", type=float, default=40.0, help="benchmark camera horizontal FOV in degrees")
    parser.add_argument(
        "--frame-mode", choices=("fit", "plan"),
        default=os.environ.get("GATE3DGEN_PBR_FRAME_MODE", "fit"),
        help=(
            "benchmark framing: 'fit' reframes each plan view to the object's "
            "bounding sphere so the whole asset is visible (default); 'plan' "
            "uses the raw baked plan matrices (legacy tight input-view crop)"
        ),
    )
    parser.add_argument(
        "--frame-margin", type=float,
        default=float(os.environ.get("GATE3DGEN_PBR_FRAME_MARGIN", "1.1")),
        help="extra distance factor beyond the exact bounding-sphere fit (fit mode)",
    )
    parser.add_argument(
        "--shaded-light-mode", choices=("three_point", "uniform"),
        default=os.environ.get("GATE3DGEN_PBR_SHADED_LIGHT_MODE", "three_point"),
        help=(
            "shaded-preview lighting: 'three_point' keeps the directional "
            "key/fill/rim rig (default); 'uniform' uses an even environment "
            "(world) light with no strong specular hotspots"
        ),
    )
    parser.add_argument(
        "--uniform-light-strength", type=float,
        default=float(os.environ.get("GATE3DGEN_PBR_UNIFORM_LIGHT_STRENGTH", "1.0")),
        help="world background strength for --shaded-light-mode uniform",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compatibility-python", default=os.environ.get("GATE3DGEN_PYTHON"),
                        help="host Python with Pillow; never Blender's different-ABI Python")
    parser.add_argument("--max-rows", type=int, default=0, help="optional smoke-test row limit")
    parser.add_argument(
        "--case-id", action="append", dest="case_ids", default=None,
        help="render only the specified case id(s); repeatable for an isolated smoke test",
    )
    parser.add_argument(
        "--compatibility-cache-root",
        type=Path,
        help=(
            "optional NFS cache for Blender-3-compatible GLB copies; prediction/wild "
            "assets using EXT_texture_webp are transcoded to embedded PNG without "
            "modifying the canonical source GLB"
        ),
    )
    args = parser.parse_args(values)
    if args.compatibility_cache_root is None:
        args.compatibility_cache_root = args.render_root / ".compatibility_cache"
    return args


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


WORKSPACE_MOUNTS = (
)


def resolve_workspace_file(value: str | Path) -> Path:
    """Resolve an artifact recorded under either distributed or local mount."""
    path = Path(value)
    if path.is_file():
        return path
    for recorded_root in WORKSPACE_MOUNTS:
        try:
            relative = path.relative_to(recorded_root)
        except ValueError:
            continue
        for mounted_root in WORKSPACE_MOUNTS:
            candidate = mounted_root / relative
            if candidate.is_file():
                return candidate
    return path


def blender_compatible_source(source: Path, cache_root: Path | None,
                              compatibility_python: str | None = None) -> Path:
    """Return a Blender-3-readable source, caching WebP conversions on NFS.

    Pixal3D exports embedded textures through ``EXT_texture_webp``.  Ubuntu's
    Blender 3.0 glTF importer rejects that extension before it reaches the
    image decoder.  Keep Blender identical for GT and prediction renders by
    transcoding only the incompatible texture payloads to lossless PNG in a
    content-identity cache.  The canonical generation artifact is read-only.
    """
    if cache_root is None:
        return source
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from glb_compat import blender_compatible, convert

    if blender_compatible(source):
        return source
    source_stat = source.stat()
    identity = "\0".join((
        str(source.resolve()),
        str(source_stat.st_size),
        str(source_stat.st_mtime_ns),
    ))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    destination = cache_root / digest[:2] / f"{digest}.glb"
    if not blender_compatible(destination):
        if compatibility_python:
            try:
                completed = subprocess.run([compatibility_python, "-I", str(tools_dir / "glb_compat.py"),
                    "--input", str(source), "--output", str(destination)], check=True,
                    capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"host WebP conversion failed for {source}: {error.stderr}") from error
            result = json.loads(completed.stdout)
        else:
            try:
                result = convert(source, destination)
            except ImportError as error:
                raise RuntimeError("WebP conversion needs --compatibility-python /path/to/conda/bin/python; "
                                   "use python -m gate3dgen.evaluation.render_pbr") from error
        if int(result.get("converted_images", 0)) <= 0:
            raise RuntimeError(
                f"incompatible GLB conversion produced no PNG textures: {source}"
            )
        if not blender_compatible(destination):
            raise RuntimeError(f"GLB compatibility validation failed: {destination}")
        print(
            f"[compat converted] {source} -> {destination} "
            f"images={result['converted_images']}",
            flush=True,
        )
    else:
        print(f"[compat cached] {source} -> {destination}", flush=True)
    return destination


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def mesh_objects() -> list[bpy.types.Object]:
    result = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not result:
        raise RuntimeError("imported GLB contains no mesh objects")
    return result


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    extent = max(high[index] - low[index] for index in range(3))
    if not math.isfinite(extent) or extent <= 1e-8:
        raise RuntimeError("GLB has a degenerate bounding box")
    return (low + high) * 0.5, extent


def _find_case_plan(plan: Any, case_id: str) -> dict[str, Any]:
    """Find one case record in the nested benchmark plan."""

    if isinstance(plan, dict):
        if str(plan.get("case_id", "")) == str(case_id):
            return plan
        for value in plan.values():
            found = _find_case_plan(value, case_id)
            if found:
                return found
    elif isinstance(plan, list):
        for value in plan:
            found = _find_case_plan(value, case_id)
            if found:
                return found
    return {}


def normalize_benchmark(objects: list[bpy.types.Object]) -> None:
    """Apply the DTC benchmark object frame used by input-view rendering."""

    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("no mesh bounds")
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    extent = high - low
    longest = float(max(extent))
    if not math.isfinite(longest) or longest <= 1e-8:
        raise RuntimeError("GLB has a degenerate bounding box")
    center = (low + high) * 0.5
    transform = Matrix.Scale(2.0 / longest, 4) @ Matrix.Translation(
        Vector((-center.x, -center.y, -low.z))
    )
    imported = set(objects)
    roots = [
        obj for obj in bpy.context.scene.objects
        if obj in imported and obj.parent not in imported
    ]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world


def normalize_and_align(objects: list[bpy.types.Object], alignment: list[list[float]]) -> None:
    center, extent = world_bounds(objects)
    normalize = (
        Matrix.Translation(Vector((0.5, 0.5, 0.5)))
        @ Matrix.Scale(1.0 / extent, 4)
        @ Matrix.Translation(-center)
    )
    align = Matrix(alignment)
    # Imported object hierarchies may contain empty parents.  Premultiplying
    # every root object moves the complete hierarchy exactly once.
    imported = set(objects)
    def descendants(obj: bpy.types.Object):
        # ``children_recursive`` was added after the Ubuntu Blender 3.0 build
        # used by the local evaluator.  Walk the hierarchy explicitly so the
        # same renderer supports both that build and newer Blender releases.
        for child in obj.children:
            yield child
            yield from descendants(child)

    roots = [obj for obj in bpy.context.scene.objects if obj in imported or any(child in imported for child in descendants(obj))]
    roots = [obj for obj in roots if obj.parent not in roots]
    for obj in roots:
        obj.matrix_world = align @ normalize @ obj.matrix_world


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def object_bounding_sphere(objects: list[bpy.types.Object]) -> tuple[Vector, float]:
    """Return the world-space bounding-box center and enclosing-sphere radius.

    Used to frame the benchmark cameras.  The center is the axis-aligned
    bounding-box center (stable across conditions of the same asset) and the
    radius is the farthest bound-box corner from that center, guaranteeing the
    whole object projects inside the camera frustum once the fit distance is
    respected.
    """
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not points:
        raise RuntimeError("no mesh bounds for camera framing")
    low = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    high = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    center = (low + high) * 0.5
    radius = max((point - center).length for point in points)
    if not math.isfinite(radius) or radius <= 1e-8:
        raise RuntimeError("degenerate bounding sphere for camera framing")
    return center, radius


def fit_benchmark_camera_matrix(
    plan_matrix: list[list[float]],
    center: Vector,
    radius: float,
    camera_fov_deg: float,
    *,
    margin: float = 1.1,
    min_distance: float = 1e-3,
) -> Matrix:
    """Reframe a benchmark camera so the whole object is visible.

    The plan matrix fixes the *viewing direction* (its +Z column points from the
    look-at target back toward the camera, matching Blender's
    ``to_track_quat('-Z', 'Y')`` convention).  We keep that direction but move
    the camera to a distance that makes the object's bounding sphere fit inside
    the (square) field of view with a small margin, and re-aim it at the object
    center.  Because every condition of a case is rendered from the same
    per-case direction and the object is normalized identically, GT and
    prediction stay framed consistently.
    """
    matrix = Matrix(plan_matrix)
    view_axis = matrix.col[2].to_3d()
    if view_axis.length <= 1e-8:
        raise ValueError("benchmark camera matrix has a degenerate view axis")
    view_axis = view_axis.normalized()
    half_fov = math.radians(float(camera_fov_deg)) * 0.5
    sin_half = math.sin(half_fov)
    if sin_half <= 1e-6:
        raise ValueError("benchmark camera FOV is too small to frame the object")
    distance = max(radius / sin_half * float(margin), float(min_distance))
    location = center + view_axis * distance
    rotation = (center - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(location) @ rotation


def setup_scene(
    resolution: int,
    samples: int = 16,
    *,
    light_scale: float = 1.0,
    camera_mode: str = "turntable",
    camera_matrix: list[list[float]] | None = None,
    camera_fov_deg: float = 40.0,
    light_mode: str = "three_point",
    uniform_light_strength: float = 1.0,
) -> bpy.types.Object:
    scene = bpy.context.scene
    # EEVEE requires a headless EGL/OpenGL context.  The distributed CUDA image
    # used by the distributed evaluator has no libEGL.so.1, so EEVEE aborts
    # the whole Blender process before Python can handle an exception.
    # Cycles CUDA/OptiX is genuinely headless and respects CUDA_VISIBLE_DEVICES
    # (one evaluator worker per GPU), matching the proven turntable path.
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    requested_device = os.environ.get("GATE3DGEN_PBR_RENDER_DEVICE", "OPTIX").upper()
    actual_device = "CPU"
    if requested_device in {"GPU", "CUDA", "OPTIX"}:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        backend = "OPTIX" if requested_device in {"GPU", "OPTIX"} else "CUDA"
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            enabled = 0
            for device in preferences.devices:
                device.use = device.type != "CPU"
                enabled += int(bool(device.use))
            if enabled == 0:
                raise RuntimeError(f"Cycles {backend} found no visible GPU")
            scene.cycles.device = "GPU"
            actual_device = backend
        except Exception as error:
            print(
                f"[pbr] {backend} setup failed; falling back to CPU: {error}",
                flush=True,
            )
            for device in getattr(preferences, "devices", []):
                device.use = device.type == "CPU"
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"
    print(
        f"[pbr] renderer=CYCLES requested_device={requested_device} actual_device={actual_device} samples={samples}",
        flush=True,
    )
    # ``uniform`` lighting replaces the directional three-point rig with an
    # even environment (world) light: every surface point is lit equally from
    # all directions, so the shaded preview has no strong specular hotspots and
    # no direction-dependent shadowing.  ``three_point`` keeps the historical
    # key/fill/rim rig against a black world.
    if light_mode == "uniform":
        # Even environment light that still renders a BLACK background: the
        # white world lights the object from every direction (no hotspots),
        # but camera rays see a black background so every channel keeps the
        # historical black canvas.  A Mix Shader keyed on ``Is Camera Ray``
        # selects black for the camera and the lighting world otherwise.
        scene.world.use_nodes = True
        tree = scene.world.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputWorld")
        mix = tree.nodes.new("ShaderNodeMixShader")
        light_path = tree.nodes.new("ShaderNodeLightPath")
        lit = tree.nodes.new("ShaderNodeBackground")
        dark = tree.nodes.new("ShaderNodeBackground")
        lit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        lit.inputs["Strength"].default_value = float(uniform_light_strength)
        dark.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        dark.inputs["Strength"].default_value = 0.0
        # fac = Is Camera Ray -> 1 for camera rays (dark), 0 otherwise (lit)
        tree.links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        tree.links.new(lit.outputs["Background"], mix.inputs[1])
        tree.links.new(dark.outputs["Background"], mix.inputs[2])
        tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    else:
        scene.world.use_nodes = False
        scene.world.color = (0.0, 0.0, 0.0)
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    camera_data = bpy.data.cameras.new("MetricCamera")
    camera_data.lens = 55.0
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    if camera_mode == "benchmark":
        if camera_matrix is None:
            raise ValueError("benchmark camera mode requires a camera matrix")
        camera_data.angle = math.radians(float(camera_fov_deg))
    camera = bpy.data.objects.new("MetricCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    if camera_mode == "benchmark":
        camera.matrix_world = Matrix(camera_matrix)
    scene.camera = camera
    if light_mode != "uniform":
        target = Vector((0.5, 0.5, 0.5))
        for index, (direction, energy, size) in enumerate((
            ((1.2, -1.0, 1.5), 1100.0, 4.0),
            ((-1.0, -0.4, 0.9), 700.0, 3.0),
            ((0.2, 1.2, 0.5), 450.0, 2.5),
        )):
            data = bpy.data.lights.new(f"MetricArea{index}", "AREA")
            data.energy, data.shape, data.size = energy * light_scale, "DISK", size
            light = bpy.data.objects.new(f"MetricArea{index}", data)
            light.location = target + Vector(direction) * 2.0
            bpy.context.collection.objects.link(light)
            point_at(light, target)
    return camera


def shared_reference_cameras(source, camera_matrix, camera_views, camera_fov_deg, frame_margin):
    """Fit the reference once; predictions must use these exact matrices."""
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(source))
    objects = mesh_objects()
    normalize_benchmark(objects)
    bpy.context.view_layer.update()
    center, radius = object_bounding_sphere(objects)
    views = camera_views or [{"view_id": "view_00", "view_index": 0,
                              "expected_camera_matrix_world": camera_matrix}]
    fitted = []
    for view in views:
        result = dict(view)
        result["plan_camera_matrix_world"] = view["expected_camera_matrix_world"]
        result["expected_camera_matrix_world"] = [list(map(float, row)) for row in
            fit_benchmark_camera_matrix(view["expected_camera_matrix_world"], center, radius,
                                        camera_fov_deg, margin=frame_margin)]
        fitted.append(result)
    clear_scene()
    return fitted[0]["expected_camera_matrix_world"], fitted


def output_node(nodes):
    return next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)


def principled_node(nodes):
    return next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)


def copy_input(source, destination, links) -> None:
    if source is None:
        return
    if source.is_linked:
        links.new(source.links[0].from_socket, destination)
    else:
        value = source.default_value
        if not hasattr(value, "__len__") and destination.type == "RGBA":
            destination.default_value = (float(value), float(value), float(value), 1.0)
        else:
            destination.default_value = value[:] if hasattr(value, "__len__") else value


def emission_material(original: bpy.types.Material | None, channel: str) -> bpy.types.Material:
    material = original.copy() if original is not None else bpy.data.materials.new(f"metric_{channel}")
    material.name = f"{material.name}_{channel}"
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    output = output_node(nodes) or nodes.new("ShaderNodeOutputMaterial")
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    emission = nodes.new("ShaderNodeEmission")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    emission.inputs["Strength"].default_value = 1.0

    if channel == "mask":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        return material
    if channel == "normal":
        geometry = nodes.new("ShaderNodeNewGeometry")
        multiply = nodes.new("ShaderNodeVectorMath")
        multiply.operation = "SCALE"
        multiply.inputs[3].default_value = 0.5
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        links.new(geometry.outputs["Normal"], multiply.inputs[0])
        links.new(multiply.outputs["Vector"], add.inputs[0])
        links.new(add.outputs["Vector"], emission.inputs["Color"])
        return material

    principled = principled_node(nodes)
    names = {
        "base_color": ("Base Color",),
        "metallic": ("Metallic",),
        "roughness": ("Roughness",),
        "alpha": ("Alpha",),
    }
    socket = None
    if principled is not None:
        for name in names[channel]:
            socket = principled.inputs.get(name)
            if socket is not None:
                break
    if socket is None:
        default = 1.0 if channel in {"roughness", "alpha"} else 0.0
        if channel == "base_color":
            emission.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        else:
            emission.inputs["Color"].default_value = (default, default, default, 1.0)
    else:
        copy_input(socket, emission.inputs["Color"], links)
    return material


def ensure_pbr_material_slots(objects: list[bpy.types.Object]) -> int:
    """Give material-less imported meshes a deterministic neutral PBR slot.

    Some valid GLBs contain helper/primitive meshes (for example an
    ``Icosphere``) without a material assignment.  They still contribute to
    geometry, mask and normal renders, so aborting the entire benchmark is
    inappropriate.  A shared neutral Principled material also makes the
    shaded/base-color/metallic/roughness/alpha channels well-defined.
    """
    fallback = None
    repaired = 0

    def material() -> bpy.types.Material:
        nonlocal fallback
        if fallback is None:
            fallback = bpy.data.materials.new("TexVerseNeutralPBRFallback")
            fallback.use_nodes = True
            principled = principled_node(fallback.node_tree.nodes)
            if principled is not None:
                base = principled.inputs.get("Base Color")
                metallic = principled.inputs.get("Metallic")
                roughness = principled.inputs.get("Roughness")
                alpha = principled.inputs.get("Alpha")
                if base is not None:
                    base.default_value = (0.8, 0.8, 0.8, 1.0)
                if metallic is not None:
                    metallic.default_value = 0.0
                if roughness is not None:
                    roughness.default_value = 1.0
                if alpha is not None:
                    alpha.default_value = 1.0
        return fallback

    for obj in objects:
        if not obj.material_slots:
            obj.data.materials.append(material())
            repaired += 1
            continue
        for slot in obj.material_slots:
            if slot.material is None:
                slot.material = material()
                repaired += 1
    if repaired:
        print(f"[pbr] assigned neutral fallback material to {repaired} empty mesh slots", flush=True)
    return repaired


def channel_materials(objects: list[bpy.types.Object]) -> dict[str, list[bpy.types.Material]]:
    originals: list[bpy.types.Material | None] = []
    for obj in objects:
        if not obj.material_slots:
            raise RuntimeError(f"mesh {obj.name!r} has no PBR material slot")
        for slot in obj.material_slots:
            if slot.material is None:
                raise RuntimeError(f"mesh {obj.name!r} has an empty PBR material slot")
            originals.append(slot.material)
    unique: list[bpy.types.Material | None] = []
    for material in originals:
        if material not in unique:
            unique.append(material)
    return {
        channel: [emission_material(material, channel) for material in unique]
        for channel in CHANNELS if channel != "shaded"
    }


def assign_channel(objects: list[bpy.types.Object], channel: str, replacements: dict[str, list[bpy.types.Material]], originals: dict[int, list]) -> None:
    if channel == "shaded":
        for obj in objects:
            for slot, material in zip(obj.material_slots, originals[id(obj)]):
                slot.material = material
        return
    unique_originals: list[bpy.types.Material | None] = []
    for values in originals.values():
        for material in values or [None]:
            if material not in unique_originals:
                unique_originals.append(material)
    mapping = {id(material): replacement for material, replacement in zip(unique_originals, replacements[channel])}
    for obj in objects:
        if not obj.material_slots:
            obj.data.materials.append(replacements[channel][0])
            continue
        for slot, original in zip(obj.material_slots, originals[id(obj)]):
            slot.material = mapping[id(original)]


def _camera_fingerprint(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def camera_contract(
    views: int,
    resolution: int,
    samples: int = 16,
    *,
    camera_mode: str = "turntable",
    camera_matrix: list[list[float]] | None = None,
    camera_views: list[dict[str, Any]] | None = None,
    camera_fov_deg: float = 40.0,
    light_scale: float = 1.0,
) -> dict:
    if camera_mode == "benchmark":
        if camera_views:
            if len(camera_views) != int(views):
                raise ValueError(
                    f"benchmark camera contract has {len(camera_views)} matrices, expected {views}"
                )
            matrices = [view["expected_camera_matrix_world"] for view in camera_views]
            protocol = f"texverse_benchmark_camera_v2_multiview{int(views)}"
            camera_source = "benchmark_camera_plan_v2.evaluation_views"
        else:
            if camera_matrix is None:
                raise ValueError("benchmark camera contract requires a matrix")
            if int(views) != 1:
                raise ValueError(
                    "benchmark camera v1 has one matrix; views>1 would duplicate the anchor"
                )
            matrices = [camera_matrix]
            protocol = "texverse_benchmark_camera_v1"
            camera_source = "benchmark_plan.json.source_anchor.expected_camera_matrix_world"
        contract = {
            "protocol": protocol,
            "coordinate_frame": "dtc_benchmark_object",
            "camera_mode": "benchmark",
            "camera_matrix_world": matrices[0],
            "camera_views": camera_views or [{
                "view_id": "view_00",
                "view_index": 0,
                "role": "source_anchor",
                "expected_camera_matrix_world": matrices[0],
            }],
            "camera_fov_deg": float(camera_fov_deg),
            "camera_source": camera_source,
            "views": int(views),
            "resolution": int(resolution),
            "render_samples": int(samples),
            "channels": list(CHANNELS),
            "renderer": f"Blender_{bpy.app.version_string}_Cycles_PBR_v3",
            "render_device": os.environ.get("GATE3DGEN_PBR_RENDER_DEVICE", "OPTIX").upper(),
            "shaded_light_energy_scale": float(light_scale),
            "view_transform": "Standard",
        }
        contract["fingerprint"] = _camera_fingerprint(contract)
        return contract
    contract = {
        "protocol": "dual3d_pbr_shared_camera_v1",
        "coordinate_frame": "gt_unit_cube",
        "camera_mode": "turntable",
        "views": views,
        "resolution": resolution,
        "render_samples": samples,
        "lens_mm": 55.0,
        "radius": 2.2,
        "elevation_deg": 20.0,
        "azimuth_deg": [360.0 * index / views for index in range(views)],
        "channels": list(CHANNELS),
        "renderer": f"Blender_{bpy.app.version_string}_Cycles_PBR_v3",
        "render_device": os.environ.get("GATE3DGEN_PBR_RENDER_DEVICE", "OPTIX").upper(),
        "shaded_light_energy_scale": float(light_scale),
        "view_transform": "Standard",
    }
    contract["fingerprint"] = _camera_fingerprint(contract)
    return contract


def write_camera(path: Path, contract: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(contract)
    # ``camera_contract`` already computes the fingerprint for the benchmark
    # mode.  Legacy callers may still pass a contract without one, in which
    # case compute it once here.  Never hash a dictionary that already
    # contains its own fingerprint (hash-of-hash made old audits needlessly
    # difficult to reproduce).
    if not payload.get("fingerprint"):
        payload["fingerprint"] = _camera_fingerprint(payload)
    # Conditions of the same case share this contract and are rendered by
    # different GPUs concurrently. Publish atomically so a metric process can
    # never observe a truncated JSON document.
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def render_asset(
    source: Path,
    destination: Path,
    alignment: list[list[float]],
    views: int,
    resolution: int,
    samples: int,
    overwrite: bool,
    *,
    light_scale: float = 1.0,
    channels: tuple[str, ...] = CHANNELS,
    camera_mode: str = "turntable",
    camera_matrix: list[list[float]] | None = None,
    camera_views: list[dict[str, Any]] | None = None,
    camera_fov_deg: float = 40.0,
    frame_mode: str = "fit",
    frame_margin: float = 1.1,
    light_mode: str = "three_point",
    uniform_light_strength: float = 1.0,
    contract: dict | None = None,
) -> None:
    expected = [destination / channel / f"view_{view:02d}.png" for channel in channels for view in range(views)]
    if contract is not None:
        asset_contract = {key: value for key, value in contract.items() if key != "fingerprint"}
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        asset_contract["render_source_sha256"] = digest.hexdigest()
        metadata_path = destination / "camera.json"
        if not overwrite and any(path.is_file() for path in expected):
            try:
                previous = json.loads(metadata_path.read_text())
                previous.pop("fingerprint", None)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Cached renders have no valid per-asset camera contract: {destination}; "
                                   "use --overwrite or a new render root") from error
            if previous != asset_contract:
                raise RuntimeError(f"Cached render camera/source changed: {destination}; "
                                   "use --overwrite or a new render root")
        # Partial resumes retain their intended camera/source contract. Missing
        # channel/view files still fail metric coverage; this is not a success marker.
        write_camera(metadata_path, asset_contract)
    if not overwrite and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(f"[skip] {destination}", flush=True)
        return
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(source))
    objects = mesh_objects()
    if camera_mode == "benchmark":
        normalize_benchmark(objects)
    else:
        normalize_and_align(objects, alignment)
    # Benchmark cameras baked in the plan frame the object as a tight input-view
    # crop, which clips the object in most orthogonal views.  Reframe them to
    # the object's bounding sphere so the whole asset is visible while keeping
    # each plan view's exact direction.
    fit_camera_matrices: list[list[list[float]]] | None = None
    if camera_mode == "benchmark" and frame_mode == "fit":
        fit_center, fit_radius = object_bounding_sphere(objects)
        plan_matrices = (
            [view["expected_camera_matrix_world"] for view in camera_views]
            if camera_views
            else [camera_matrix]
        )
        fit_camera_matrices = [
            [list(map(float, row)) for row in fit_benchmark_camera_matrix(
                plan_matrix, fit_center, fit_radius, camera_fov_deg, margin=frame_margin,
            )]
            for plan_matrix in plan_matrices
        ]
        print(
            f"[fit] center={tuple(round(v,3) for v in fit_center)} radius={fit_radius:.3f} "
            f"fov={camera_fov_deg} margin={frame_margin} "
            f"cam0_loc={tuple(round(fit_camera_matrices[0][r][3],3) for r in range(3))}",
            flush=True,
        )
    camera = setup_scene(
        resolution,
        samples,
        light_scale=light_scale,
        camera_mode=camera_mode,
        camera_matrix=camera_matrix,
        camera_fov_deg=camera_fov_deg,
        light_mode=light_mode,
        uniform_light_strength=uniform_light_strength,
    )
    ensure_pbr_material_slots(objects)
    originals = {id(obj): [slot.material for slot in obj.material_slots] for obj in objects}
    replacements = channel_materials(objects)
    target = Vector((0.5, 0.5, 0.5))
    radius, elevation = 2.2, math.radians(20.0)
    for channel in channels:
        assign_channel(objects, channel, replacements, originals)
        for view in range(views):
            if camera_mode == "benchmark" and fit_camera_matrices is not None:
                camera.matrix_world = Matrix(fit_camera_matrices[view])
                bpy.context.view_layer.update()
            elif camera_mode == "benchmark" and camera_views:
                camera.matrix_world = Matrix(
                    camera_views[view]["expected_camera_matrix_world"]
                )
            elif camera_mode != "benchmark":
                azimuth = 2.0 * math.pi * view / views
                camera.location = target + Vector((
                    radius * math.cos(elevation) * math.cos(azimuth),
                    radius * math.cos(elevation) * math.sin(azimuth),
                    radius * math.sin(elevation),
                ))
                point_at(camera, target)
            output = destination / channel / f"view_{view:02d}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            # A multiview extension may seed view_00 from the completed
            # benchmark_camera_v1 run.  Skip each valid file independently so
            # that views 01..11 are rendered without recomputing the anchor.
            if not overwrite:
                try:
                    if output.is_file() and output.stat().st_size > 0:
                        continue
                except OSError:
                    pass
            bpy.context.scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
    print(f"[done] {destination}", flush=True)


def main() -> None:
    args = arguments()
    if args.views <= 0 or args.resolution <= 0 or args.samples <= 0 or args.light_scale <= 0:
        raise ValueError("views, resolution, samples and light-scale must be positive")
    rows = read_rows(args.metrics)
    if args.case_ids:
        wanted = {str(value) for value in args.case_ids}
        rows = [row for row in rows if str(row.get("case_id")) in wanted]
        if not rows:
            raise ValueError(f"none of --case-id values found in {args.metrics}: {sorted(wanted)}")
    selected_channels = tuple(dict.fromkeys(args.channels or CHANNELS))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    camera_plan = None
    if args.camera_mode == "benchmark":
        if args.camera_plan is None or not args.camera_plan.is_file():
            raise FileNotFoundError(
                f"benchmark camera mode requires --camera-plan: {args.camera_plan}"
            )
        camera_plan = json.loads(args.camera_plan.read_text(encoding="utf-8"))
    completed: set[tuple[str, ...]] = set()
    identity = [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
    for row in rows:
        case_id = str(row["case_id"])
        camera_matrix = None
        camera_views = None
        if args.camera_mode == "benchmark":
            case = _find_case_plan(camera_plan, case_id)
            anchor = case.get("source_anchor", {}) if isinstance(case, dict) else {}
            camera_matrix = anchor.get("expected_camera_matrix_world")
            if not camera_matrix:
                raise KeyError(
                    f"benchmark_plan.json has no source_anchor.expected_camera_matrix_world for {case_id}"
                )
            camera_matrix = [
                [float(value) for value in values] for values in camera_matrix
            ]
            raw_views = case.get("evaluation_views") if isinstance(case, dict) else None
            if raw_views is not None:
                if not isinstance(raw_views, list) or len(raw_views) != args.views:
                    raise ValueError(
                        f"camera plan has {len(raw_views) if isinstance(raw_views, list) else 'invalid'} "
                        f"evaluation views for {case_id}; expected {args.views}"
                    )
                camera_views = []
                for index, raw_view in enumerate(raw_views):
                    if not isinstance(raw_view, dict) or int(raw_view.get("view_index", -1)) != index:
                        raise ValueError(f"invalid evaluation view {index} for {case_id}")
                    matrix = raw_view.get("expected_camera_matrix_world")
                    if not isinstance(matrix, list) or len(matrix) != 4:
                        raise ValueError(f"evaluation view {index} has no 4x4 matrix for {case_id}")
                    normalized_view = dict(raw_view)
                    normalized_view["expected_camera_matrix_world"] = [
                        [float(value) for value in values] for values in matrix
                    ]
                    camera_views.append(normalized_view)
        render_frame_mode = args.frame_mode
        unavailable = args.mode != "gt" and str(row.get("status", "ok")).lower() in {
            "unavailable", "error", "failed", "failure", "oom"
        }
        if args.camera_mode == "benchmark" and args.frame_mode == "fit" and not unavailable:
            reference_source = resolve_workspace_file(row["reference_path"])
            if not reference_source.is_file():
                raise FileNotFoundError(reference_source)
            reference_source = blender_compatible_source(
                reference_source, args.compatibility_cache_root, args.compatibility_python)
            camera_matrix, camera_views = shared_reference_cameras(
                reference_source, camera_matrix, camera_views, args.camera_fov, args.frame_margin)
            # The fitted matrices are now final, not refitted to each prediction.
            render_frame_mode = "plan"
        contract = camera_contract(
            args.views,
            args.resolution,
            args.samples,
            camera_mode=args.camera_mode,
            camera_matrix=camera_matrix,
            camera_views=camera_views,
            camera_fov_deg=args.camera_fov,
            light_scale=args.light_scale,
        )
        contract["frame_policy"] = ("shared_reference_fit_v1" if args.frame_mode == "fit"
                                    and args.camera_mode == "benchmark" else args.frame_mode)
        contract["shaded_light_mode"] = args.shaded_light_mode
        contract["uniform_light_strength"] = args.uniform_light_strength
        contract["fingerprint"] = _camera_fingerprint({key: value for key, value in contract.items()
                                                       if key != "fingerprint"})
        # Geometry evaluation emits explicit unavailable rows for conditions
        # whose GLB generation failed.  Their GT can still be rendered once,
        # but prediction/wild rendering must skip the absent asset; downstream
        # PBR metrics will mark its missing renders unavailable.
        if args.mode != "gt" and str(row.get("status", "ok")).lower() in {
            "unavailable", "error", "failed", "failure", "oom"
        }:
            print(
                f"[skip unavailable] {case_id}:{row.get('variant') or row.get('condition')}: "
                f"{row.get('generation_error') or row.get('reason') or row.get('error')}",
                flush=True,
            )
            continue
        if args.mode == "gt":
            key = (case_id,)
            if key in completed:
                continue
            source = resolve_workspace_file(row["reference_path"])
            destination = args.render_root / "glb_gt" / case_id
            alignment = identity
        elif args.mode == "prediction":
            model, variant = str(row.get("baseline") or row.get("model")), str(row.get("variant") or row.get("condition"))
            key = (model, case_id, variant)
            if key in completed:
                continue
            source = resolve_workspace_file(row["prediction_path"])
            destination = args.render_root / model / case_id / variant
            alignment = row.get("alignment", {}).get("matrix_normalized_prediction_to_reference")
            if args.camera_mode == "benchmark":
                alignment = identity
            if alignment is None:
                raise RuntimeError(f"{model}:{case_id}:{variant}: geometry row is missing the shared PBR alignment matrix")
        else:
            # Wild has no real GLB GT.  Geometry-change rows align each
            # restored prediction to the model's original-input prediction.
            # Render that original once as the per-model reference, then use
            # the exact restored->original alignment for every restored step.
            model = str(row.get("baseline") or row.get("model"))
            variant = str(row.get("variant") or row.get("condition"))
            reference_key = (model, case_id, "original")
            if reference_key not in completed:
                reference_source = resolve_workspace_file(row["reference_path"])
                if not reference_source.is_file():
                    raise FileNotFoundError(reference_source)
                reference_source = blender_compatible_source(
                    reference_source, args.compatibility_cache_root, args.compatibility_python
                )
                render_asset(
                    reference_source,
                    args.render_root / model / case_id / "original",
                    identity,
                    args.views,
                    args.resolution,
                    args.samples,
                    args.overwrite,
                    light_scale=args.light_scale,
                    channels=selected_channels,
                    camera_mode=args.camera_mode,
                    camera_matrix=camera_matrix,
                    camera_views=camera_views,
                    camera_fov_deg=args.camera_fov,
                    frame_mode=render_frame_mode,
                    frame_margin=args.frame_margin,
                    light_mode=args.shaded_light_mode,
                    uniform_light_strength=args.uniform_light_strength,
                    contract=contract,
                )
                completed.add(reference_key)
            key = (model, case_id, variant)
            source = resolve_workspace_file(row["prediction_path"])
            destination = args.render_root / model / case_id / variant
            alignment = row.get("alignment", {}).get("matrix_normalized_prediction_to_reference")
            # Benchmark-camera texture comparisons deliberately use the DTC
            # object frame for both assets.  The geometry ICP transform is
            # retained in the geometry row for audit, but must not alter the
            # benchmark camera pose.
            if args.camera_mode == "benchmark":
                alignment = identity
            if alignment is None:
                raise RuntimeError(
                    f"{model}:{case_id}:{variant}: Wild geometry row is missing the restored-to-original alignment matrix"
                )
        if key not in completed:
            if not source.is_file():
                if args.mode != "gt":
                    print(
                        f"[skip unavailable missing GLB] {case_id}:"
                        f"{row.get('variant') or row.get('condition')}: {source}",
                        flush=True,
                    )
                    continue
                raise FileNotFoundError(source)
            source = blender_compatible_source(
                source, args.compatibility_cache_root, args.compatibility_python
            )
            render_asset(
                source,
                destination,
                alignment,
                args.views,
                args.resolution,
                args.samples,
                args.overwrite,
                light_scale=args.light_scale,
                channels=selected_channels,
                camera_mode=args.camera_mode,
                camera_matrix=camera_matrix,
                camera_views=camera_views,
                camera_fov_deg=args.camera_fov,
                frame_mode=render_frame_mode,
                frame_margin=args.frame_margin,
                light_mode=args.shaded_light_mode,
                uniform_light_strength=args.uniform_light_strength,
                contract=contract,
            )
            completed.add(key)
    print(json.dumps({"mode": args.mode, "assets": len(completed), "channels": list(selected_channels), "views": args.views}), flush=True)


if __name__ == "__main__":
    main()
