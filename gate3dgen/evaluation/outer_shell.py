"""Opaque, double-sided first-hit depth sampling of view-visible surfaces.

Camera details not specified by TRELLIS.2 D.1 are explicitly pinned here,
not claimed to reproduce the authors' unpublished rasterizer numerically.
No surface-area resampling, convex hull, or visibility-of-face approximation.
"""
from __future__ import annotations

from importlib.metadata import version

import numpy as np
from scipy.spatial import cKDTree

SHELL_PROTOCOL = "fibonacci_orthographic_first_hit_v1"
DEFAULT_VIEWS = 100
DEFAULT_RESOLUTION = 512


def camera_frame(meshes):
    """One shared frame in aligned coordinates; never normalize shells separately."""
    bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=np.float64)
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    center = (low + high) / 2
    radius = float(np.linalg.norm(high - low) / 2)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("invalid shell camera bounds")
    return center, radius


def extract_outer_shell(mesh, *, center, radius, samples, seed,
                        views=DEFAULT_VIEWS, resolution=DEFAULT_RESOLUTION):
    """Render depth on pixel-center rays and unproject nearest positive hits.

Embree accelerates exact triangle intersections (subject to floating precision).
Random-priority streaming gives a uniform sample WITHOUT replacement from all
valid depth pixels, keeping memory O(samples + resolution**2), not O(views*pixels).
An insufficient dense cloud fails instead of duplicating points silently.
"""
    if samples <= 0 or views <= 0 or resolution <= 0:
        raise ValueError("shell samples, views and resolution must be positive")
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        intersector = RayMeshIntersector(mesh, scale_to_box=True)
    except ImportError as error:
        raise RuntimeError("outer-shell evaluation requires embreex: pip install 'gate3dgen[eval]'") from error
    rng = np.random.default_rng(seed)
    # Bounding sphere fits every projection with 5% margin. Origins are outside
    # the whole pair; all triangles are opaque and hit from either face side.
    half_extent = radius * 1.05
    pixel = ((np.arange(resolution) + 0.5) / resolution * 2 - 1) * half_extent
    px, py = np.meshgrid(pixel, pixel, indexing="xy")
    kept = np.empty((0, 3), dtype=np.float64)
    priorities = np.empty(0)
    hit_counts = []
    for index in range(views):
        z = 1 - 2 * (index + 0.5) / views
        phi = index * np.pi * (3 - np.sqrt(5))
        outward = np.array([np.sqrt(1 - z*z) * np.cos(phi),
                            np.sqrt(1 - z*z) * np.sin(phi), z])
        helper = np.array([0., 0., 1.]) if abs(z) < 0.99 else np.array([0., 1., 0.])
        right = np.cross(outward, helper)
        right /= np.linalg.norm(right)
        up = np.cross(right, outward)
        origins = (center + outward * (3 * radius)
                   + px.reshape(-1, 1) * right + py.reshape(-1, 1) * up)
        directions = np.broadcast_to(-outward, origins.shape).copy()
        # First-hit intersections are the unprojected orthographic depth map.
        locations, _, _ = intersector.intersects_location(
            origins, directions, multiple_hits=False)
        if not np.isfinite(locations).all():
            raise ValueError("nonfinite shell depth intersections")
        hit_counts.append(len(locations))
        kept = np.concatenate((kept, locations))
        priorities = np.concatenate((priorities, rng.random(len(locations))))
        if len(kept) > samples:
            selection = np.argpartition(priorities, samples - 1)[:samples]
            kept, priorities = kept[selection], priorities[selection]
    if sum(hit_counts) < samples:
        raise ValueError(f"only {sum(hit_counts)} valid shell depth pixels for {samples} requested points; "
                         "increase --shell-resolution (no replacement sampling is allowed)")
    return kept, {
        "protocol_id": SHELL_PROTOCOL,
        "backend": "trimesh_embreex_first_hit",
        "embreex_version": version("embreex"),
        "trimesh_version": version("trimesh"),
        "projection": "orthographic", "camera_distribution": "fibonacci_sphere",
        "views": views, "resolution": resolution,
        "center": np.asarray(center).tolist(), "bounding_radius": radius,
        "camera_radius": 3 * radius, "orthographic_half_extent": half_extent,
        "pixel_sampling": "pixel_centers", "double_sided": True,
        "materials": "opaque_geometry_ignores_alpha_and_transmission",
        "dense_point_count": sum(hit_counts), "per_view_hit_counts": hit_counts,
        "sample_count": len(kept), "seed": seed,
        "sampling": "uniform_random_priority_without_replacement_from_valid_depth_pixels",
    }


def shell_distances(pred_points, ref_points, *, workers=1):
    """Bidirectional Euclidean nearest POINT distances, never triangle distance."""
    return (cKDTree(ref_points).query(pred_points, workers=workers)[0],
            cKDTree(pred_points).query(ref_points, workers=workers)[0])
