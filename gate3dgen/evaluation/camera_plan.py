#!/usr/bin/env python3
"""Build a deterministic four-view (default) or twelve-view PBR camera plan.

The input view remains ``view_00``.  Three orthogonal cameras form the other
fixed basis views. In twelve-view mode, eight more cameras are greedily selected from a 44-camera
bank by incremental GT-mesh surface coverage.  Selection uses only the oracle
mesh, never a generated asset, so every model and condition receives exactly
the same camera matrices.

Hunyuan3D 2.0 describes this policy as four fixed orthogonal views followed by
geometry-aware selection up to 8--12 views.  We approximate visible texel
coverage deterministically with area-weighted front-facing triangle coverage;
the resulting exact matrices are frozen in the output contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from .official_manifest import _read_rows


WORKSPACE_ROOTS = ()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_case(value: Any, case_id: str) -> dict[str, Any]:
    if isinstance(value, dict):
        if str(value.get("case_id") or "") == case_id:
            return value
        for child in value.values():
            found = find_case(child, case_id)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_case(child, case_id)
            if found:
                return found
    return {}


def resolve_path(value: Any, workspace: Path) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = workspace / path
    if path.is_file():
        return path.resolve()
    for root in WORKSPACE_ROOTS:
        try:
            candidate = workspace / path.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


def camera_matrix(
    azimuth_deg: float,
    elevation_deg: float,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 2.2,
) -> list[list[float]]:
    """Match Blender ``to_track_quat('-Z', 'Y')`` in the benchmark frame."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    z = np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ), dtype=np.float64)
    x = np.asarray((-z[1], z[0], 0.0), dtype=np.float64)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    location = np.asarray(center, dtype=np.float64) + float(radius) * z
    return [
        [float(x[row]), float(y[row]), float(z[row]), float(location[row])]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.asarray((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ), dtype=np.float64)


def _sample_mesh_surface(mesh: trimesh.Trimesh, maximum_faces: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) > maximum_faces:
        # Evenly spaced indices are deterministic and retain all regions of
        # exporters that group faces by primitive/material better than a head
        # slice would.  View selection needs a coverage proxy, not exact area.
        indices = np.linspace(0, len(faces) - 1, maximum_faces, dtype=np.int64)
        faces = faces[indices]
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(cross).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)
    return cross[valid] / lengths[valid, None], lengths[valid] * 0.5


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    loaded = trimesh.load(path, force="scene", process=False)
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    if isinstance(loaded, trimesh.Scene):
        normal_parts, area_parts = [], []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            mesh = loaded.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            normals, areas = _sample_mesh_surface(mesh, 10_000)
            normal_parts.append(normals); area_parts.append(areas)
        if not normal_parts:
            raise ValueError("scene has no triangle mesh")
        normals, areas = np.concatenate(normal_parts), np.concatenate(area_parts)
        if len(areas) > 20_000:
            indices = np.linspace(0, len(areas) - 1, 20_000, dtype=np.int64)
            normals, areas = normals[indices], areas[indices]
    elif isinstance(loaded, trimesh.Trimesh):
        normals, areas = _sample_mesh_surface(loaded)
    else:
        raise TypeError(f"unsupported mesh type: {type(loaded).__name__}")
    if not len(areas) or float(areas.sum()) <= 0:
        raise ValueError("mesh has no valid triangle area")
    areas /= areas.sum()
    extent = bounds[1] - bounds[0]
    longest = float(np.max(extent))
    if not np.isfinite(extent).all() or not math.isfinite(longest) or longest <= 1e-8:
        raise ValueError("mesh has invalid bounds")
    # ``normalize_benchmark`` centers x/y at zero, places the mesh base at
    # z=0 and scales the longest side to 2.  Its geometric center is therefore
    # (0, 0, extent_z / longest).
    benchmark_center = (0.0, 0.0, float(extent[2] / longest))
    return normals, areas, benchmark_center


def candidate_bank(anchor_azimuth: float) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for elevation in (-20.0, 0.0, 20.0, 40.0, 60.0):
        for offset in range(0, 360, 45):
            values.append(((anchor_azimuth + offset) % 360.0, elevation))
    for offset in (0, 90, 180, 270):
        values.append(((anchor_azimuth + offset) % 360.0, -40.0))
    assert len(values) == 44
    return values


def angular_key(azimuth: float, elevation: float) -> tuple[float, float]:
    return (round(azimuth % 360.0, 6), round(elevation, 6))


def select_views(
    normals: np.ndarray,
    areas: np.ndarray,
    anchor_azimuth: float,
    anchor_elevation: float,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 2.2,
) -> list[dict[str, Any]]:
    fixed = [
        ((anchor_azimuth + offset) % 360.0, anchor_elevation)
        for offset in (0.0, 90.0, 180.0, 270.0)
    ]
    candidates = candidate_bank(anchor_azimuth)
    selected = list(fixed)
    selected_keys = {angular_key(*value) for value in selected}
    # A triangle is considered covered when it is sufficiently front-facing;
    # projected cosine weights favor informative, non-grazing viewpoints.
    threshold = 0.10
    best_visibility = np.zeros(len(areas), dtype=np.float64)
    for azimuth, elevation in selected:
        cosine = normals @ direction(azimuth, elevation)
        best_visibility = np.maximum(best_visibility, np.where(cosine > threshold, cosine, 0.0))
    candidate_directions = np.stack([direction(*value) for value in candidates])
    candidate_cosines = candidate_directions @ normals.T
    candidate_visibility = np.where(
        candidate_cosines > threshold, candidate_cosines, 0.0
    )
    while len(selected) < 12:
        best_candidate: tuple[float, float] | None = None
        best_gain = -1.0
        best_index = -1
        for candidate_index, (azimuth, elevation) in enumerate(candidates):
            if angular_key(azimuth, elevation) in selected_keys:
                continue
            visibility = candidate_visibility[candidate_index]
            gain = float(np.sum(areas * np.maximum(visibility - best_visibility, 0.0)))
            key = angular_key(azimuth, elevation)
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15
                and best_candidate is not None
                and key < angular_key(*best_candidate)
            ):
                best_candidate, best_gain, best_index = (azimuth, elevation), gain, candidate_index
        if best_candidate is None:
            raise RuntimeError("camera candidate bank was exhausted")
        selected.append(best_candidate)
        selected_keys.add(angular_key(*best_candidate))
        best_visibility = np.maximum(
            best_visibility, candidate_visibility[best_index]
        )
    rows = []
    for index, (azimuth, elevation) in enumerate(selected):
        rows.append({
            "view_id": f"view_{index:02d}",
            "view_index": index,
            "role": "source_anchor" if index == 0 else ("orthogonal_basis" if index < 4 else "geometry_coverage_greedy"),
            "camera_azimuth_deg": float(azimuth),
            "camera_elevation_deg": float(elevation),
            "camera_radius": float(radius),
            "normalized_center": [float(value) for value in center],
            "expected_camera_matrix_world": camera_matrix(azimuth, elevation, center, radius),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-plan-v1", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--views", type=int, choices=(4, 12), default=4)
    args = parser.parse_args()
    manifest = _read_rows(args.manifest)
    if not manifest or (args.expected_cases is not None and len(manifest) != args.expected_cases):
        raise SystemExit(f"manifest count mismatch: {len(manifest) if isinstance(manifest, list) else type(manifest).__name__}")
    v1 = json.loads(args.camera_plan_v1.read_text(encoding="utf-8"))
    cases = []
    surface_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = {}
    for index, row in enumerate(manifest, 1):
        case_id = str(row.get("case_id") or row.get("pair_id") or "")
        source = find_case(v1, case_id)
        anchor = source.get("source_anchor") if isinstance(source, dict) else None
        if not case_id or not isinstance(anchor, dict):
            raise SystemExit(f"camera anchor missing for case {case_id!r}")
        mesh = resolve_path(row.get("gt_mesh") or row.get("glb_path"), args.workspace.resolve())
        if not mesh.is_file():
            raise SystemExit(f"GT mesh missing for {case_id}: {mesh}")
        cache_key = str(mesh)
        if cache_key not in surface_cache:
            surface_cache[cache_key] = load_surface(mesh)
        normals, areas, benchmark_center = surface_cache[cache_key]
        raw_matrix = np.asarray(anchor.get("expected_camera_matrix_world"), dtype=np.float64)
        if raw_matrix.shape != (4, 4) or not np.isfinite(raw_matrix).all():
            raise SystemExit(f"invalid camera matrix for {case_id}")
        if "camera_azimuth_deg" in anchor and "camera_elevation_deg" in anchor:
            anchor_azimuth = float(anchor["camera_azimuth_deg"])
            anchor_elevation = float(anchor["camera_elevation_deg"])
            raw_center = anchor.get("normalized_center", (0.5, 0.5, 0.5))
            camera_center = tuple(float(value) for value in raw_center)
            camera_radius = float(anchor.get("camera_radius", 2.2))
        else:
            # Legacy362 stores the exact camera matrix but not always its
            # spherical metadata.  Recover that metadata from the camera
            # position and the GT mesh center in the benchmark-normalized
            # frame, instead of guessing a common elevation.
            camera_center = benchmark_center
            offset = raw_matrix[:3, 3] - np.asarray(camera_center)
            camera_radius = float(np.linalg.norm(offset))
            if not math.isfinite(camera_radius) or camera_radius <= 1e-8:
                raise SystemExit(f"invalid recovered camera radius for {case_id}")
            anchor_azimuth = math.degrees(math.atan2(float(offset[1]), float(offset[0]))) % 360.0
            anchor_elevation = math.degrees(math.asin(float(offset[2]) / camera_radius))
        views = select_views(
            normals,
            areas,
            anchor_azimuth,
            anchor_elevation,
            camera_center,
            camera_radius,
        )
        if views[0]["expected_camera_matrix_world"] != anchor.get("expected_camera_matrix_world"):
            # The source matrix is the compatibility contract.  Preserve it
            # byte-for-byte even if a legacy writer rounded it differently.
            views[0]["expected_camera_matrix_world"] = anchor["expected_camera_matrix_world"]
        cases.append({
            "case_id": case_id,
            "asset_id": row.get("asset_id"),
            "dataset": row.get("dataset"),
            "family": row.get("family") or row.get("task_id"),
            "gt_mesh": str(mesh),
            "source_anchor": anchor,
            "evaluation_views": views[:args.views],
            "view_count": args.views,
            "selection_policy": "fixed_orthogonal_basis" if args.views == 4 else "hunyuan3d2_geometry_aware_4_fixed_plus_8_greedy_v1",
        })
        if index % 25 == 0 or index == len(manifest):
            print(f"[camera-plan] {index}/{len(manifest)}", flush=True)
    payload = {
        "schema_version": f"texverse-benchmark-camera-v2-multiview{args.views}",
        "protocol": f"benchmark_camera_v2_multiview{args.views}",
        "views_per_case": args.views,
        "source_camera_plan": str(args.camera_plan_v1.resolve()),
        "source_camera_plan_sha256": hashlib.sha256(args.camera_plan_v1.read_bytes()).hexdigest(),
        "selection_policy": {
            "reference": "Hunyuan3D-2.0 Section 4.1 View Selection Strategy",
            "fixed_views": 4,
            "greedy_views": args.views - 4,
            "candidate_views": 44,
            "coverage_proxy": "area_weighted_front_facing_GT_mesh_triangles",
            "generated_mesh_used_for_selection": False,
        },
        "cases": cases,
    }
    payload["contract_sha256"] = sha256_json(payload)
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "cases": len(cases), "views_per_case": args.views, "output": str(args.output.resolve()), "contract_sha256": payload["contract_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
