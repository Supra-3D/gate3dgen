"""All-surface point-to-mesh MD and visible-shell point-cloud CD/F1.

Default: 100 first-hit depth views, one million visible points per mesh.
Explicit proxies are retained only for historical diagnostics. Camera details
are pinned locally; upstream numerical reproduction is not claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .outer_shell import (SHELL_PROTOCOL, DEFAULT_VIEWS, DEFAULT_RESOLUTION,
                          camera_frame, extract_outer_shell, shell_distances)
from .io import finite_json, iter_jsonl, write_jsonl
from .official_manifest import RESTORATION_STEPS


PROTOCOL_ID = "geometry_visible_shell_v3"
DISTANCE_KERNEL_ID = "ericson_closest_point_v1"
EXACT_MD_KERNEL_ID = "trimesh_rtree_exact_closest_triangle_v1"
HIGH_QUALITY_SAMPLES = 1_000_000
SCALED_DEMO_SAMPLES = 10_000
DEFAULT_MD_F_THRESHOLD = 1e-8
DEFAULT_CD_F_THRESHOLD = 1e-6
HISTORICAL_RAW_SURFACE_MAPPING = {
    "raw_surface_proxy_distance": "outer_shell_chamfer_distance",
    "raw_surface_proxy_distance_l2": "outer_shell_chamfer_distance_l2",
    "raw_surface_proxy_distance_l1": "outer_shell_chamfer_distance_l1",
    "raw_surface_proxy_fscore": "outer_shell_fscore",
    "raw_surface_proxy_precision": "outer_shell_precision",
    "raw_surface_proxy_recall": "outer_shell_recall",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a GLB/mesh scene and apply scene transforms."""

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"no triangle mesh in {path}")
    return loaded


def normalize_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy fitted to a unit cube with no anisotropic scaling.

    The longest axis has extent one.  Shorter axes retain their proportions
    and are centered in the cube.  This operation removes arbitrary export
    units and translation while deliberately preserving aspect ratio.
    """

    result = mesh.copy()
    low = np.asarray(result.vertices).min(axis=0).astype(np.float64)
    high = np.asarray(result.vertices).max(axis=0).astype(np.float64)
    extent = high - low
    scale = float(np.max(extent))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("cannot normalize a zero-extent mesh")
    center = (low + high) * 0.5
    result.vertices = (np.asarray(result.vertices, dtype=np.float64) - center) / scale + 0.5
    return result


def _proper_axis_rotations() -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.diag(signs) @ base
            if np.linalg.det(rotation) > 0.5:
                result.append(rotation)
    return result


PROPER_AXIS_ROTATIONS = _proper_axis_rotations()


def _subsample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices]


def _similarity_initial(source: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_scale = np.sqrt(np.mean(np.sum((source - source_center) ** 2, axis=1)))
    target_scale = np.sqrt(np.mean(np.sum((target - target_center) ** 2, axis=1)))
    uniform = target_scale / max(float(source_scale), 1e-12)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = uniform * rotation
    matrix[:3, 3] = target_center - matrix[:3, :3] @ source_center
    return matrix


def _validate_similarity(matrix: np.ndarray, tolerance: float = 1e-5) -> dict[str, float]:
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise ValueError("similarity transform contains a reflection")
    scale = float(np.cbrt(determinant))
    if scale <= 0 or np.max(np.abs(singular - scale)) > tolerance * max(scale, 1.0):
        raise ValueError("similarity transform is anisotropic")
    return {
        "determinant": determinant,
        "uniform_scale": scale,
        "singular_values": [float(value) for value in singular],
        "anisotropy_ratio": float(np.max(singular) / max(np.min(singular), 1e-12)),
    }


def validate_similarity_transform(matrix: np.ndarray, tolerance: float = 1e-5) -> dict[str, float]:
    """Public validation helper used by callers before applying a transform."""

    return _validate_similarity(matrix, tolerance=tolerance)


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_alignment_points: int = 5_000,
    max_iterations: int = 60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate proper rigid + uniform-scale alignment using multi-start ICP."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape (N, 3)")
    if len(source) < 3 or len(target) < 3:
        raise ValueError("at least three points are required for alignment")
    source_fit = _subsample(source, max_alignment_points)
    target_fit = _subsample(target, max_alignment_points)
    best: tuple[np.ndarray, float] | None = None
    errors: list[str] = []
    for rotation in PROPER_AXIS_ROTATIONS:
        initial = _similarity_initial(source_fit, target_fit, rotation)
        try:
            matrix, _, cost = trimesh.registration.icp(
                source_fit,
                target_fit,
                initial=initial,
                threshold=1e-7,
                max_iterations=max_iterations,
                scale=False,
                reflection=False,
            )
            info = _validate_similarity(matrix)
            if best is None or float(cost) < best[1]:
                best = (np.asarray(matrix, dtype=np.float64), float(cost))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            errors.append(str(error))
    if best is None:
        reason = errors[-1] if errors else "no valid ICP initialization"
        raise RuntimeError(f"similarity alignment failed: {reason}")
    matrix, cost = best
    info = _validate_similarity(matrix)
    info.update({
        "method": "24_proper_axis_multistart_icp",
        "icp_cost": cost,
        "reflection_allowed": False,
        "anisotropic_scale_allowed": False,
    })
    return matrix, info


def apply_transform(mesh: trimesh.Trimesh, matrix: np.ndarray) -> trimesh.Trimesh:
    _validate_similarity(matrix)
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample surface points, their face normals, and face ids deterministically."""

    if count <= 0:
        raise ValueError("sample count must be positive")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_ids = trimesh.sample.sample_surface(mesh, int(count))
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return np.asarray(points, dtype=np.float64), normals, np.asarray(face_ids, dtype=np.int64)


def _closest_point_candidates(
    points: np.ndarray,
    triangles: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact point-to-triangle distance over a candidate triangle set.

    This is the same closest-point region test used by common triangle distance
    implementations.  ``candidates`` has shape (points, K); using a centroid
    index keeps memory bounded for very large GLBs while the small-mesh path
    below remains exhaustive.
    """

    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)[candidates]
    a, b, c = tri[:, :, 0, :], tri[:, :, 1, :], tri[:, :, 2, :]
    ab, ac, bc = b - a, c - a, c - b
    ap = points[:, None, :] - a
    bp = points[:, None, :] - b
    cp = points[:, None, :] - c
    d1 = np.sum(ab * ap, axis=2)
    d2 = np.sum(ac * ap, axis=2)
    d3 = np.sum(ab * bp, axis=2)
    d4 = np.sum(ac * bp, axis=2)
    d5 = np.sum(ab * cp, axis=2)
    d6 = np.sum(ac * cp, axis=2)

    # Ericson's closest-point region tests.  In particular, the face-interior
    # barycentric denominator is ``va + vb + vc``; using a determinant from a
    # different projection formula here silently produced non-zero distances
    # even for points sampled directly from the same triangle.
    q = a.copy()
    assigned = np.zeros(d1.shape, dtype=bool)

    # Vertex A
    mask = (d1 <= 0) & (d2 <= 0)
    q[mask] = a[mask]
    assigned |= mask
    # Vertex B
    mask_b = (d3 >= 0) & (d4 <= d3)
    q[mask_b] = b[mask_b]
    assigned |= mask_b
    # Edge AB
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t_ab = d1 / np.maximum(d1 - d3, 1e-20)
    q[mask_ab] = a[mask_ab] + ab[mask_ab] * t_ab[mask_ab, None]
    assigned |= mask_ab
    # Vertex C
    mask_c = (d6 >= 0) & (d5 <= d6)
    q[mask_c] = c[mask_c]
    assigned |= mask_c
    # Edge AC
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t_ac = d2 / np.maximum(d2 - d6, 1e-20)
    q[mask_ac] = a[mask_ac] + ac[mask_ac] * t_ac[mask_ac, None]
    assigned |= mask_ac
    # Edge BC
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-20)
    q[mask_bc] = b[mask_bc] + bc[mask_bc] * t_bc[mask_bc, None]
    assigned |= mask_bc
    # Points not covered by an edge/vertex Voronoi region project into the
    # face interior.  Degenerate residuals safely keep vertex A.
    face_denom = va + vb + vc
    mask_face = (~assigned) & (np.abs(face_denom) > 1e-20)
    v_face = np.zeros_like(face_denom)
    w_face = np.zeros_like(face_denom)
    v_face[mask_face] = vb[mask_face] / face_denom[mask_face]
    w_face[mask_face] = vc[mask_face] / face_denom[mask_face]
    q[mask_face] = (
        a[mask_face]
        + ab[mask_face] * v_face[mask_face, None]
        + ac[mask_face] * w_face[mask_face, None]
    )

    distances = np.linalg.norm(points[:, None, :] - q, axis=2)
    local = np.argmin(distances, axis=1)
    rows = np.arange(len(points))
    return q[rows, local], distances[rows, local], candidates[rows, local]


def closest_point_to_triangles(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    chunk_size: int = 4_096,
    exhaustive_chunk_size: int = 128,
    exhaustive_triangle_limit: int = 2_000,
    candidate_count: int = 64,
    kdtree_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return closest points/distances/triangle IDs and approximation status."""

    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if len(triangles) == 0:
        raise ValueError("mesh has no triangles")
    # trimesh's naive implementation is exact, and is substantially faster
    # than a Python implementation for small demo meshes.  Chunking avoids its
    # otherwise enormous point×triangle temporary array.
    if len(triangles) <= exhaustive_triangle_limit:
        closest_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        for start in range(0, len(points), max(1, exhaustive_chunk_size)):
            closest, distance, ids = trimesh.proximity.closest_point_naive(
                mesh, points[start : start + exhaustive_chunk_size]
            )
            closest_parts.append(np.asarray(closest, dtype=np.float64))
            distance_parts.append(np.asarray(distance, dtype=np.float64))
            id_parts.append(np.asarray(ids, dtype=np.int64))
        return (
            np.concatenate(closest_parts),
            np.concatenate(distance_parts),
            np.concatenate(id_parts),
            False,
        )

    centroids = triangles.mean(axis=1)
    tree = cKDTree(centroids)
    k = min(max(1, candidate_count), len(triangles))
    closest_parts, distance_parts, id_parts = [], [], []
    for start in range(0, len(points), max(1, chunk_size)):
        block = points[start : start + chunk_size]
        # Geometry shards already run concurrently (up to eight processes per
        # 72-core node).  ``workers=-1`` here made every process consume the
        # whole machine for every tiny query block, causing severe
        # oversubscription and silent multi-hour long tails.  Keep this
        # explicitly bounded; callers may opt into more workers when running
        # a single shard in isolation.
        _, candidate = tree.query(block, k=k, workers=max(1, int(kdtree_workers)))
        if k == 1:
            candidate = candidate[:, None]
        closest, distance, ids = _closest_point_candidates(block, triangles, candidate)
        closest_parts.append(closest)
        distance_parts.append(distance)
        id_parts.append(ids)
    return (
        np.concatenate(closest_parts),
        np.concatenate(distance_parts),
        np.concatenate(id_parts),
        True,
    )


def _fscore(distances_a: np.ndarray, distances_b: np.ndarray, threshold: float,
            *, strict: bool = True) -> tuple[float, float, float]:
    """F-score using the squared-distance thresholds reported by TRELLIS.2.

    The paper writes the acceptance test as ``d(p, S)^2 < tau`` and reports
    ``tau=1e-8`` for Mesh Distance and ``tau=1e-6`` for outer-shell CD.
    ``closest_point_to_triangles`` returns Euclidean distances, so square them
    here instead of accidentally treating tau as an Euclidean threshold.
    """

    compare = np.less if strict else np.less_equal
    precision = float(np.mean(compare(np.asarray(distances_a) ** 2, threshold)))
    recall = float(np.mean(compare(np.asarray(distances_b) ** 2, threshold)))
    score = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return score, precision, recall


def _normal_metrics(
    source_normals: np.ndarray,
    target_normals: np.ndarray,
    source_to_target_ids: np.ndarray,
    target_to_source_ids: np.ndarray,
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> dict[str, float]:
    # ``source_to_target_ids`` index target triangles and vice versa.  Keeping
    # this direction explicit avoids accidentally comparing a prediction's
    # normals with its own (rather than the reference's) matched face.
    source_face_normals = np.asarray(target_mesh.face_normals[source_to_target_ids], dtype=np.float64)
    target_face_normals = np.asarray(source_mesh.face_normals[target_to_source_ids], dtype=np.float64)
    source_normals = source_normals / np.maximum(np.linalg.norm(source_normals, axis=1, keepdims=True), 1e-12)
    target_normals = target_normals / np.maximum(np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12)
    cos_a = np.clip(np.abs(np.sum(source_normals * source_face_normals, axis=1)), 0.0, 1.0)
    cos_b = np.clip(np.abs(np.sum(target_normals * target_face_normals, axis=1)), 0.0, 1.0)
    angles = np.degrees(np.arccos(np.concatenate((cos_a, cos_b))))
    return {
        "normal_consistency": float(np.mean(np.concatenate((cos_a, cos_b)))),
        "normal_mean_angular_error_deg": float(np.mean(angles)),
        "normal_median_angular_error_deg": float(np.median(angles)),
        "normal_accuracy_11.25deg": float(np.mean(angles <= 11.25)),
        "normal_accuracy_22.5deg": float(np.mean(angles <= 22.5)),
        "normal_accuracy_30deg": float(np.mean(angles <= 30.0)),
    }


def _surface_proxy(mesh: trimesh.Trimesh, method: str) -> trimesh.Trimesh:
    if method == "raw_surface":
        return mesh
    if method == "convex_hull":
        return mesh.convex_hull
    raise ValueError(f"unsupported surface proxy method: {method}")


def evaluate_pair(
    prediction: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    *,
    samples: int = HIGH_QUALITY_SAMPLES,
    seed: int = 1234,
    surface_proxy_method: str | None = None,
    shell_views: int = DEFAULT_VIEWS,
    shell_resolution: int = DEFAULT_RESOLUTION,
    md_threshold: float = DEFAULT_MD_F_THRESHOLD,
    cd_threshold: float = DEFAULT_CD_F_THRESHOLD,
    chunk_size: int = 4_096,
    exhaustive_chunk_size: int = 128,
    exhaustive_triangle_limit: int = 2_000,
    candidate_count: int = 64,
    kdtree_workers: int = 1,
) -> dict[str, Any]:
    """Compute MD on all surfaces, CD/F1 on visible depth point clouds."""

    if samples <= 0:
        raise ValueError("samples must be positive")

    prediction = normalize_unit_cube(prediction)
    reference = normalize_unit_cube(reference)
    # Avoid an unnecessary stochastic ICP solve for the common identity case.
    # Besides being faster, this keeps the official tiny F-score thresholds
    # meaningful when a mesh is compared with itself (independent surface
    # samples still land exactly on the same triangles).
    same_topology = (
        prediction.faces.shape == reference.faces.shape
        and np.array_equal(prediction.faces, reference.faces)
        and prediction.vertices.shape == reference.vertices.shape
        and np.allclose(prediction.vertices, reference.vertices, rtol=1e-7, atol=1e-8)
    )
    if same_topology:
        matrix = np.eye(4, dtype=np.float64)
        alignment = {
            "determinant": 1.0,
            "uniform_scale": 1.0,
            "singular_values": [1.0, 1.0, 1.0],
            "anisotropy_ratio": 1.0,
            "method": "identity_same_normalized_mesh",
            "icp_cost": 0.0,
            "reflection_allowed": False,
            "anisotropic_scale_allowed": False,
        }
    else:
        pred_fit, _, _ = sample_surface(prediction, min(samples, 20_000), seed)
        ref_fit, _, _ = sample_surface(reference, min(samples, 20_000), seed + 1)
        matrix, alignment = estimate_similarity_transform(pred_fit, ref_fit)
    # Persist the normalized-prediction -> normalized-reference similarity so
    # the metric-grade PBR renderer can apply exactly the same alignment before
    # rendering both assets with one shared camera.  This is deliberately not
    # a per-axis fit and validate_similarity_transform remains the gate.
    alignment["matrix_normalized_prediction_to_reference"] = matrix.tolist()
    prediction = apply_transform(prediction, matrix)

    pred_points, pred_normals, _ = sample_surface(prediction, samples, seed)
    ref_points, ref_normals, _ = sample_surface(reference, samples, seed + 1)
    def md_closest(mesh, points, **options):
        if surface_proxy_method is not None:
            return closest_point_to_triangles(mesh, points, **options)
        # Full R-tree search, not a limited nearest-centroid triangle candidate set.
        distances, faces = [], []
        # This is an indexed complete search, not the historical exhaustive
        # all-triangles broadcast kernel. Use its own distance batch size:
        # trimesh rebuilds a vertex KD-tree per call, so 128-point exhaustive
        # batches impose thousands of unnecessary rebuilds for 1M samples.
        for start in range(0, len(points), chunk_size):
            _, distance, face = trimesh.proximity.closest_point(
                mesh, points[start:start + chunk_size])
            distances.append(distance)
            faces.append(face)
        return None, np.concatenate(distances), np.concatenate(faces), False

    _, pred_to_ref, pred_ref_ids, approx_a = md_closest(
        reference, pred_points, chunk_size=chunk_size,
        exhaustive_chunk_size=exhaustive_chunk_size,
        exhaustive_triangle_limit=exhaustive_triangle_limit,
        candidate_count=candidate_count, kdtree_workers=kdtree_workers,
    )
    _, ref_to_pred, ref_pred_ids, approx_b = md_closest(
        prediction, ref_points, chunk_size=chunk_size,
        exhaustive_chunk_size=exhaustive_chunk_size,
        exhaustive_triangle_limit=exhaustive_triangle_limit,
        candidate_count=candidate_count, kdtree_workers=kdtree_workers,
    )
    distances = np.concatenate((pred_to_ref, ref_to_pred))
    md_score, md_precision, md_recall = _fscore(
        pred_to_ref, ref_to_pred, md_threshold, strict=surface_proxy_method is None)
    metrics: dict[str, float] = {
        # TRELLIS.2 defines MD with squared point-to-surface distance.  Keep
        # the unsquared mean as an explicitly named project diagnostic.
        "mesh_distance": float(np.mean(distances**2)),
        "mesh_distance_l2": float(np.mean(distances**2)),
        "mesh_distance_l1": float(np.mean(distances)),
        "mesh_distance_fscore": md_score,
        "mesh_distance_precision": md_precision,
        "mesh_distance_recall": md_recall,
    }
    metrics.update(_normal_metrics(
        pred_normals, ref_normals, pred_ref_ids, ref_pred_ids, prediction, reference
    ))

    shell_metadata = None
    cd_approx_a = cd_approx_b = False
    if surface_proxy_method is None:
        center, radius = camera_frame([prediction, reference])
        pred_outer_points, pred_shell = extract_outer_shell(
            prediction, center=center, radius=radius, samples=samples,
            seed=seed + 2, views=shell_views, resolution=shell_resolution)
        ref_outer_points, ref_shell = extract_outer_shell(
            reference, center=center, radius=radius, samples=samples,
            seed=seed + 3, views=shell_views, resolution=shell_resolution)
        pred_cd, ref_cd = shell_distances(
            pred_outer_points, ref_outer_points, workers=kdtree_workers)
        score, precision, recall = _fscore(pred_cd, ref_cd, cd_threshold)
        distances = np.concatenate((pred_cd, ref_cd))
        metrics.update({
            "outer_shell_chamfer_distance": float(np.mean(distances**2)),
            "outer_shell_chamfer_distance_l2": float(np.mean(distances**2)),
            "outer_shell_chamfer_distance_l1": float(np.mean(distances)),
            "outer_shell_fscore": score,
            "outer_shell_precision": precision,
            "outer_shell_recall": recall,
        })
        shell_metadata = {"prediction": pred_shell, "reference": ref_shell}
    else:
        pred_outer = _surface_proxy(prediction, surface_proxy_method)
        ref_outer = _surface_proxy(reference, surface_proxy_method)
        pred_outer_points, _, _ = sample_surface(pred_outer, samples, seed + 2)
        ref_outer_points, _, _ = sample_surface(ref_outer, samples, seed + 3)
        _, pred_cd, _, cd_approx_a = closest_point_to_triangles(
            ref_outer, pred_outer_points, chunk_size=chunk_size,
            exhaustive_chunk_size=exhaustive_chunk_size,
            exhaustive_triangle_limit=exhaustive_triangle_limit,
            candidate_count=candidate_count, kdtree_workers=kdtree_workers,
        )
        _, ref_cd, _, cd_approx_b = closest_point_to_triangles(
            pred_outer, ref_outer_points, chunk_size=chunk_size,
            exhaustive_chunk_size=exhaustive_chunk_size,
            exhaustive_triangle_limit=exhaustive_triangle_limit,
            candidate_count=candidate_count, kdtree_workers=kdtree_workers,
        )
        cd_score, cd_precision, cd_recall = _fscore(pred_cd, ref_cd, cd_threshold, strict=False)
        outer_distances = np.concatenate((pred_cd, ref_cd))
        metrics.update({
            # Proxy statistic: squared sampled-point-to-triangle distance, NOT paper outer-surface CD.
            f"{surface_proxy_method}_proxy_distance": float(np.mean(outer_distances**2)),
            f"{surface_proxy_method}_proxy_distance_l2": float(np.mean(outer_distances**2)),
            f"{surface_proxy_method}_proxy_distance_l1": float(np.mean(outer_distances)),
            f"{surface_proxy_method}_proxy_fscore": cd_score,
            f"{surface_proxy_method}_proxy_precision": cd_precision,
            f"{surface_proxy_method}_proxy_recall": cd_recall,
        })
    return {
        "metrics": metrics,
        "alignment": alignment,
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "compliance": "visible_shell" if shell_metadata else "surface_proxy",
            "exact_reproduction": False,
            "unit_cube": True,
            "normalization": "longest_bbox_axis_to_one_preserve_aspect_ratio",
            "alignment": "proper_rotation_translation_uniform_scale",
            "reflection_allowed": False,
            "anisotropic_scale_allowed": False,
            "samples": int(samples),
            "md_fscore_threshold": md_threshold,
            "outer_shell_fscore_threshold" if shell_metadata else "surface_proxy_fscore_threshold": cd_threshold,
            "fscore_threshold_domain": "squared_euclidean_distance",
            "fscore_comparison": "strict_less_than" if shell_metadata else "less_equal_historical",
            "surface_proxy_method": surface_proxy_method,
            "point_to_triangle": True,
            "outer_shell_distance_kernel": "exact_ckdtree_nearest_point_squared" if shell_metadata else None,
            "outer_shell_extraction": shell_metadata,
            "distance_kernel_id": EXACT_MD_KERNEL_ID if shell_metadata else DISTANCE_KERNEL_ID,
            "point_to_triangle_approximation": bool(approx_a or approx_b or cd_approx_a or cd_approx_b),
            "distance_chunk_size": int(chunk_size),
            "exhaustive_chunk_size": int(exhaustive_chunk_size),
            "exhaustive_triangle_limit": int(exhaustive_triangle_limit),
            "candidate_count": int(candidate_count),
            "kdtree_workers": int(kdtree_workers),
            "outer_shell_render_protocol": shell_metadata is not None,
            "surface_proxy": surface_proxy_method,
            "surface_proxy_definition": (
                None if surface_proxy_method is None else
                "unchanged mesh surface, including any internal surfaces"
                if surface_proxy_method == "raw_surface" else "convex hull surface; not an exact outer shell"
            ),
            "paper_outer_surface_metrics_available": shell_metadata is not None,
            "metric_schema": PROTOCOL_ID,
            "historical_field_mapping": HISTORICAL_RAW_SURFACE_MAPPING if surface_proxy_method == "raw_surface" else {},
        },
    }


def official_pair_metrics(
    prediction: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    *,
    samples: int = SCALED_DEMO_SAMPLES,
    seed: int = 1234,
    surface_proxy_method: str | None = None,
    shell_views: int = DEFAULT_VIEWS,
    shell_resolution: int = DEFAULT_RESOLUTION,
    **kwargs: Any,
) -> dict[str, float]:
    """Convenience API returning only scalar metrics."""

    return evaluate_pair(
        prediction,
        reference,
        samples=samples,
        seed=seed,
        surface_proxy_method=surface_proxy_method,
        shell_views=shell_views, shell_resolution=shell_resolution,
        **kwargs,
    )["metrics"]


def identity_metrics(surface_proxy_method: str = "raw_surface") -> dict[str, float]:
    if surface_proxy_method not in {"raw_surface", "convex_hull"}:
        raise ValueError(f"unsupported surface proxy method: {surface_proxy_method}")
    return {
        "mesh_distance": 0.0,
        "mesh_distance_l2": 0.0,
        "mesh_distance_l1": 0.0,
        "mesh_distance_fscore": 1.0,
        "mesh_distance_precision": 1.0,
        "mesh_distance_recall": 1.0,
        "normal_consistency": 1.0,
        "normal_mean_angular_error_deg": 0.0,
        "normal_median_angular_error_deg": 0.0,
        "normal_accuracy_11.25deg": 1.0,
        "normal_accuracy_22.5deg": 1.0,
        "normal_accuracy_30deg": 1.0,
        f"{surface_proxy_method}_proxy_distance": 0.0,
        f"{surface_proxy_method}_proxy_distance_l2": 0.0,
        f"{surface_proxy_method}_proxy_distance_l1": 0.0,
        f"{surface_proxy_method}_proxy_fscore": 1.0,
        f"{surface_proxy_method}_proxy_precision": 1.0,
        f"{surface_proxy_method}_proxy_recall": 1.0,
    }


METRIC_DIRECTIONS: dict[str, str] = {
    **{name: ("lower" if "distance" in name else "higher")
       for name in HISTORICAL_RAW_SURFACE_MAPPING.values()},
    "mesh_distance": "lower",
    "mesh_distance_l2": "lower",
    "mesh_distance_l1": "lower",
    "mesh_distance_fscore": "higher",
    "mesh_distance_precision": "higher",
    "mesh_distance_recall": "higher",
    "normal_consistency": "higher",
    "normal_mean_angular_error_deg": "lower",
    "normal_median_angular_error_deg": "lower",
    "normal_accuracy_11.25deg": "higher",
    "normal_accuracy_22.5deg": "higher",
    "normal_accuracy_30deg": "higher",
    **{
        f"{method}_proxy_{name}": direction
        for method in ("raw_surface", "convex_hull")
        for name, direction in {
            "distance": "lower", "distance_l2": "lower", "distance_l1": "lower",
            "fscore": "higher", "precision": "higher", "recall": "higher",
        }.items()
    },
}


def _intrinsic(mesh: trimesh.Trimesh) -> dict[str, Any]:
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(labels.max() + 1) if len(labels) else 0,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def _manifest_rows(path: Path) -> Iterable[dict[str, Any]]:
    for row in iter_jsonl(path):
        yield row


def _variant_path(generation_dir: Path, baseline: str, case_id: str, variant: str) -> Path:
    return generation_dir / baseline / case_id / f"{variant}.glb"


def _generation_failure_reasons(
    generation_dir: Path, baseline: str
) -> dict[tuple[str, str], str]:
    """Load explicit generation failures published beside canonical GLBs."""

    reasons: dict[tuple[str, str], str] = {}
    for artifact_manifest in sorted(generation_dir.glob("*_artifacts.jsonl")):
        for row in iter_jsonl(artifact_manifest):
            if str(row.get("model") or row.get("baseline") or "") != baseline:
                continue
            if str(row.get("status", "ok")).lower() not in {
                "error", "failed", "failure", "oom", "unavailable", "not_available"
            }:
                continue
            key = (str(row.get("case_id") or ""), str(row.get("condition") or row.get("variant") or ""))
            if all(key):
                reasons[key] = str(row.get("error") or row.get("reason") or "generation_failed")
    return reasons


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def valid_value(row, name):
        value = row.get("metrics", {}).get(name)
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(value)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = ":".join(str(row.get(name, "")) for name in ("baseline", "reference", "variant"))
        groups[key].append(row)
    result = {}
    for key, values in sorted(groups.items()):
        names = sorted({name for row in values for name in row.get("metrics", {})})
        result[key] = {
            "count": len(values),
            "ok": sum(row.get("status", "ok") == "ok" for row in values),
            "metric_coverage": {
                name: {
                    "valid_count": sum(bool(valid_value(row, name)) for row in values),
                    "expected_count": len(values),
                    "coverage": sum(bool(valid_value(row, name)) for row in values) / len(values),
                } for name in names
            },
            "metrics": {
                name: float(np.mean([row["metrics"][name] for row in values
                                    if valid_value(row, name)])) if any(valid_value(row, name) for row in values) else None
                for name in names
            },
        }
    return {"schema_version": PROTOCOL_ID, "paper_outer_surface_metrics_available": bool(rows) and all(
        row.get("protocol", {}).get("paper_outer_surface_metrics_available", False) for row in rows), "groups": result}


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("case_id") or ""),
        str(row.get("baseline") or ""),
        str(row.get("variant") or ""),
    )


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Publish a parseable checkpoint without exposing a partial JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="official_aligned_v1 condition JSONL")
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=("trellis2", "pixal3d", "fixture"),
    )
    parser.add_argument("--variant", action="append", help="GLB variant; defaults to all declared restored steps")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--scaled-demo", action="store_true", help=f"Use {SCALED_DEMO_SAMPLES} samples and mark compliance")
    parser.add_argument("--shell-views", type=int, default=DEFAULT_VIEWS)
    parser.add_argument("--shell-resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--surface-proxy-method", choices=("raw_surface", "convex_hull"), default=None, help="historical diagnostic instead of true visible shell")
    parser.add_argument("--paper-protocol", action="store_true", help="require full 100-view 1M-point visible-shell geometry")
    parser.add_argument("--historical-report-protocol", action="store_true", help="freeze verified historical raw-surface geometry settings without misleading shell aliases")
    parser.add_argument("--distance-chunk-size", type=int, default=4_096)
    parser.add_argument("--exhaustive-chunk-size", type=int, default=128)
    parser.add_argument("--exhaustive-triangle-limit", type=int, default=2_000)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--kdtree-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="checkpoint after every variant and skip completed identities")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    samples = int(args.samples if args.samples is not None else (SCALED_DEMO_SAMPLES if args.scaled_demo else HIGH_QUALITY_SAMPLES))
    if samples <= 0 or args.shell_views <= 0 or args.shell_resolution <= 0:
        parser.error("samples, views and resolution must be positive")
    if min(args.distance_chunk_size, args.exhaustive_chunk_size, args.candidate_count) <= 0:
        parser.error("distance chunks and candidate count must be positive")
    if args.paper_protocol and (
        args.scaled_demo or samples != HIGH_QUALITY_SAMPLES or args.shell_views != 100
        or args.shell_resolution != DEFAULT_RESOLUTION or args.surface_proxy_method is not None
        or args.historical_report_protocol or args.allow_missing
    ):
        parser.error("paper protocol requires 1M samples, 100 views, resolution 512, genuine shell and no --allow-missing")
    if args.historical_report_protocol and args.surface_proxy_method is None:
        args.surface_proxy_method = "raw_surface"
    if args.historical_report_protocol and (
        args.scaled_demo or samples != 1_000_000 or args.surface_proxy_method != "raw_surface"
        or args.seed != 1234 or args.distance_chunk_size != 4096
        or args.exhaustive_chunk_size != 128 or args.exhaustive_triangle_limit != 2000
        or args.candidate_count != 64 or args.kdtree_workers != 1
    ):
        parser.error("historical-report protocol requires frozen 1M-sample raw_surface, seed=1234 and default kernel settings")
    try:
        backend_versions = {
            name: version(name) for name in (
                ("trimesh", "numpy", "scipy") if args.surface_proxy_method
                else ("trimesh", "numpy", "scipy", "embreex", "rtree")
            )
        }
    except PackageNotFoundError as error:
        parser.error(f"missing geometry backend {error}; install with: pip install -e '.[eval]'")
    metric_config = {
        "metric_schema": PROTOCOL_ID,
        "distance_kernel_id": DISTANCE_KERNEL_ID if args.surface_proxy_method else EXACT_MD_KERNEL_ID,
        "backend_versions": backend_versions,
        "samples": samples,
        "seed": int(args.seed),
        "surface_proxy_method": args.surface_proxy_method,
        "distance_chunk_size": int(args.distance_chunk_size),
        "exhaustive_chunk_size": int(args.exhaustive_chunk_size),
        "exhaustive_triangle_limit": int(args.exhaustive_triangle_limit),
        "candidate_count": int(args.candidate_count),
        "historical_report_protocol": bool(args.historical_report_protocol),
        "shell_protocol": SHELL_PROTOCOL,
        "shell_views": args.shell_views,
        "shell_resolution": args.shell_resolution,
        "kdtree_workers": args.kdtree_workers,
        "paper_protocol": args.paper_protocol,

    }
    metric_config_hash = hashlib.sha256(
        json.dumps(metric_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    baselines = args.baseline or ["trellis2", "pixal3d"]
    if len(set(baselines)) != len(baselines):
        parser.error("duplicate baselines are not allowed")
    if args.paper_protocol and (args.output_dir / "summary.json").exists():
        parser.error("paper protocol requires an output directory without an existing summary")
    expected_identities = set()

    failure_reasons = {
        baseline: _generation_failure_reasons(args.generation_dir, baseline)
        for baseline in baselines
    }
    output_path = args.output_dir / "per_case.jsonl"
    rows: list[dict[str, Any]] = []
    if args.resume and output_path.is_file():
        existing_rows = list(iter_jsonl(output_path))
        # Never reuse scores computed by an older distance kernel or a
        # different sampling/candidate protocol.  Keep them beside the shard
        # for audit instead of silently mixing incomparable values.
        # A matching unavailable row is a completed, auditable result too.
        # In particular, a Step-1 empty-token/missing-input generation failure
        # must stay cached on resume instead of being retried forever.
        rows = [
            row for row in existing_rows
            if row.get("protocol", {}).get("metric_config_hash") == metric_config_hash
            and not (
                str(row.get("status", "ok")).lower() == "unavailable"
                and row.get("reason") == "generation_failed_or_missing"
                and (
                    str(row.get("generation_error") or "").startswith("missing_input:")
                    or str(row.get("generation_error") or "").startswith("missing_prediction_glb:")
                )
            )
        ]
        if len(rows) != len(existing_rows) and existing_rows:
            backup = output_path.with_name(
                f"{output_path.stem}.incompatible.{metric_config_hash[:12]}{output_path.suffix}"
            )
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, backup)
    completed = {_row_identity(row) for row in rows}
    for case in _manifest_rows(args.manifest):
        reference_path = Path(str(case.get("gt_mesh") or ""))
        if case.get("dataset_kind") == "wild_no_gt" or not reference_path.is_file():
            if args.paper_protocol:
                raise ValueError(f"paper geometry requires mesh GT for {case.get('case_id')}")
            continue
        reference = load_mesh(reference_path)
        variants = list(args.variant or [])
        if not variants:
            variants = [f"restored_s{step:02d}" for step in case.get("restoration_steps", RESTORATION_STEPS)]
            variants = ["degraded", "clean_gt", *variants]
        for baseline in baselines:
            for variant in variants:
                identity = (str(case["case_id"]), baseline, variant)
                if identity in expected_identities:
                    raise ValueError(f"duplicate case/baseline/variant identity: {identity}")
                expected_identities.add(identity)
                if identity in completed:
                    print(json.dumps({"event": "metric_cached", "case_id": identity[0], "baseline": baseline, "variant": variant}), flush=True)
                    continue
                path = _variant_path(args.generation_dir, baseline, str(case["case_id"]), variant)
                if not path.is_file():
                    if args.allow_missing:
                        reason = failure_reasons[baseline].get(
                            (str(case["case_id"]), variant),
                            f"missing_prediction_glb:{path}",
                        )
                        rows.append({
                            "schema_version": PROTOCOL_ID,
                            "case_id": str(case["case_id"]),
                            "asset_id": case.get("asset_id"),
                            "task_id": case.get("task_id"),
                            "baseline": baseline,
                            "variant": variant,
                            "reference": "glb_gt",
                            "reference_is_true_gt": True,
                            "prediction_path": str(path.absolute()),
                            "reference_path": str(reference_path.absolute()),
                            "status": "unavailable",
                            "reason": "generation_failed_or_missing",
                            "generation_error": reason,
                            "metrics": {},
                            "alignment": {},
                            "protocol": {
                                "id": PROTOCOL_ID,
                                "compliance": "unavailable",
                                "samples": samples,
                                "distance_kernel_id": metric_config["distance_kernel_id"],
                                "metric_config_hash": metric_config_hash,
                            },
                            "intrinsic": {},
                        })
                        completed.add(identity)
                        if args.resume:
                            _atomic_write_jsonl(output_path, rows)
                        continue
                    raise FileNotFoundError(path)
                try:
                    prediction = load_mesh(path)
                    evaluated = evaluate_pair(
                        prediction,
                        reference,
                        samples=samples,
                        seed=args.seed,
                        surface_proxy_method=args.surface_proxy_method,
                        shell_views=args.shell_views, shell_resolution=args.shell_resolution,
                        chunk_size=args.distance_chunk_size,
                        exhaustive_chunk_size=args.exhaustive_chunk_size,
                        exhaustive_triangle_limit=args.exhaustive_triangle_limit,
                        candidate_count=args.candidate_count,
                        kdtree_workers=args.kdtree_workers,
                    )
                except Exception as error:
                    if not args.allow_missing:
                        raise
                    rows.append({
                        "schema_version": PROTOCOL_ID,
                        "case_id": str(case["case_id"]),
                        "asset_id": case.get("asset_id"),
                        "task_id": case.get("task_id"),
                        "baseline": baseline,
                        "variant": variant,
                        "reference": "glb_gt",
                        "reference_is_true_gt": True,
                        "prediction_path": str(path.absolute()),
                        "reference_path": str(reference_path.absolute()),
                        "status": "unavailable",
                        "reason": "metric_evaluation_failed",
                        "metric_error": f"{type(error).__name__}: {error}",
                        "metrics": {},
                        "alignment": {},
                        "protocol": {
                            "id": PROTOCOL_ID,
                            "compliance": "unavailable",
                            "samples": samples,
                            "distance_kernel_id": metric_config["distance_kernel_id"],
                            "metric_config_hash": metric_config_hash,
                        },
                        "intrinsic": {},
                    })
                    completed.add(identity)
                    if args.resume:
                        _atomic_write_jsonl(output_path, rows)
                    print(json.dumps({
                        "event": "metric_unavailable", "case_id": identity[0],
                        "baseline": baseline, "variant": variant,
                        "error": f"{type(error).__name__}: {error}",
                    }), flush=True)
                    continue
                if args.scaled_demo:
                    evaluated["protocol"]["compliance"] = "scaled_demo_" + evaluated["protocol"]["compliance"]
                evaluated["protocol"]["metric_config_hash"] = metric_config_hash
                evaluated["protocol"]["historical_report_protocol"] = bool(args.historical_report_protocol)
                rows.append({
                    "schema_version": PROTOCOL_ID,
                    "case_id": str(case["case_id"]),
                    "asset_id": case.get("asset_id"),
                    "task_id": case.get("task_id"),
                    "baseline": baseline,
                    "variant": variant,
                    "reference": "glb_gt",
                    "reference_is_true_gt": True,
                    "prediction_path": str(path.absolute()),
                    "reference_path": str(reference_path.absolute()),
                    "metrics": finite_json(evaluated["metrics"]),
                    "fixture_only": baseline == "fixture" or bool(case.get("fixture_only", False)),
                    "input_provenance": case.get("provenance"),
                    "alignment": finite_json(evaluated["alignment"]),
                    "protocol": finite_json(evaluated["protocol"]),
                    "intrinsic": finite_json(_intrinsic(prediction)),
                })
                completed.add(identity)
                if args.resume:
                    _atomic_write_jsonl(output_path, rows)
                print(json.dumps({
                    "event": "metric_done", "case_id": identity[0],
                    "baseline": baseline, "variant": variant,
                    "rows": len(rows), "samples": samples,
                }), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_path, rows)
    if args.paper_protocol:
        identities = [_row_identity(row) for row in rows]
        if set(identities) != expected_identities or len(set(identities)) != len(identities):
            raise ValueError("paper geometry populations differ from planned case/baseline/variant set")
        if not rows or any(row.get("status", "ok") != "ok" or not all(
            np.isfinite(row.get("metrics", {}).get(name, np.nan))
            for name in ("mesh_distance", "mesh_distance_fscore",
                         "outer_shell_chamfer_distance", "outer_shell_fscore")
        ) for row in rows):
            raise ValueError("incomplete paper geometry metrics; no formal summary published")
    summary = _summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "schema_version": PROTOCOL_ID,
        "component": "metrics3d_official",
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "rows": len(rows),
        "compliance": ("scaled_demo_" if args.scaled_demo else "") + ("surface_proxy" if args.surface_proxy_method else "visible_shell"),
        "exact_reproduction": False,
        "distance_kernel_id": metric_config["distance_kernel_id"],
        "metric_config": metric_config,
        "metric_config_hash": metric_config_hash,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir.absolute()), "compliance": metadata["compliance"]}, indent=2))


if __name__ == "__main__":
    main()
