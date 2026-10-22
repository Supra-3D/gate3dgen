from __future__ import annotations

import collections
import json
import os
import tempfile
import unittest
from pathlib import Path

from gate3dgen.data.pair_generation.common import atomic_write_json, extract_relative_source, iter_jsonl
from gate3dgen.data.pair_generation.planner import build_view_assignments


class AssignmentTests(unittest.TestCase):
    def test_full_assignment_invariants(self):
        first = build_view_assignments(3967, 12, 42, "v7")
        second = build_view_assignments(3967, 12, 42, "v7")
        self.assertEqual(first, second)
        self.assertTrue(all(collections.Counter(row) == {0: 3, 1: 3, 2: 3, 3: 3} for row in first))
        for task_index in range(12):
            counts = collections.Counter(row[task_index] for row in first)
            self.assertEqual(sorted(counts.values()), [991, 992, 992, 992])

    def test_assignment_changes_with_seed(self):
        self.assertNotEqual(
            build_view_assignments(128, 12, 42, "v7"),
            build_view_assignments(128, 12, 43, "v7"),
        )


class CommonTests(unittest.TestCase):
    def test_relative_source_extraction(self):
        path = Path("/old/mount/TexVerse/renders_crossview/abc/albedo/000.png")
        self.assertEqual(extract_relative_source(path), "renders_crossview/abc/albedo/000.png")

    def test_atomic_json_is_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"status": "success", "count": 12})
            self.assertEqual(json.loads(path.read_text()), {"status": "success", "count": 12})

    def test_generated_outputs_remap_to_new_output_root(self):
        from PIL import Image
        from gate3dgen.data.pair_generation.reporting import _metadata_output_path, _success_outputs_present

        with tempfile.TemporaryDirectory() as directory:
            new_root = Path(directory) / "new_mount/output"
            pair_dir = new_root / "pairs/task/ab/pair"
            pair_dir.mkdir(parents=True)
            for name in ("gt.png", "degraded.png"):
                Image.new("RGB", (8, 8), "white").save(pair_dir / name)
            metadata = {
                "status": "success",
                "basic_audit": {"status": "pass"},
                "gt_output": "/old/mount/output/pairs/task/ab/pair/gt.png",
                "degraded_output": "/old/mount/output/pairs/task/ab/pair/degraded.png",
            }
            self.assertEqual(_metadata_output_path(metadata, "gt_output", new_root), pair_dir / "gt.png")
            self.assertTrue(_success_outputs_present(metadata, new_root))


class BasicAuditTests(unittest.TestCase):
    def test_procedural_replay_is_pixel_exact(self):
        from PIL import Image, ImageDraw
        from gate3dgen.data.pair_generation.common import DEFAULT_LIBRARY
        from gate3dgen.data.pair_generation.generation import _atomic_save_image, basic_audit, load_tasks
        from gate3dgen.data.pair_generation.degradation_backend import apply_postprocess

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Image.new("RGB", (64, 64), "black")
            ImageDraw.Draw(source).rectangle((16, 14, 48, 52), fill=(190, 80, 35))
            source_path, gt_path, degraded_path = (root / name for name in ("source.png", "gt.png", "degraded.png"))
            source.save(source_path)
            source.save(gt_path)
            task, _ = load_tasks(DEFAULT_LIBRARY, 1234, "synthetic")
            crop = task["crop_heavy"]
            degraded, _, _ = apply_postprocess(source, crop, 1234, source=source)
            _atomic_save_image(degraded, degraded_path)
            result = basic_audit(
                {"pair_id": "synthetic", "task_id": "crop_heavy"}, crop,
                source_path, gt_path, degraded_path, 1234,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["procedural_replay"], "pixel_exact")

    def test_corrupt_completed_pair_is_regenerated(self):
        from types import SimpleNamespace
        from PIL import Image, ImageDraw
        from gate3dgen.data.pair_generation.common import DEFAULT_LIBRARY
        from gate3dgen.data.pair_generation.generation import pair_output_dir, process_pair

        class FakePipe:
            def __call__(self, *, image, **_kwargs):
                return SimpleNamespace(images=[image[0].copy()])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "renders_crossview/asset/albedo/000.png"
            source.parent.mkdir(parents=True)
            image = Image.new("RGB", (64, 64), "black")
            ImageDraw.Draw(image).rectangle((16, 14, 48, 52), fill=(190, 80, 35))
            image.save(source)
            row = {
                "pair_id": "pair-recovery", "pair_seed": 123, "sha256": "asset",
                "task_id": "crop_heavy", "degradation_type": "crop",
                "source_relative_path": "renders_crossview/asset/albedo/000.png",
                "source_original_path": str(source),
            }
            output = root / "output"
            pair_dir = pair_output_dir(output, row)
            pair_dir.mkdir(parents=True)
            (pair_dir / "gt.png").write_bytes(b"corrupt")
            (pair_dir / "degraded.png").write_bytes(b"corrupt")
            atomic_write_json(pair_dir / "metadata.json", {
                "pair_id": row["pair_id"], "status": "success",
                "accepted_attempt_seed": 1,
                "attempts": [{"status": "failed"}] * 3,
            })
            outcome, _ = process_pair(
                row, FakePipe(), DEFAULT_LIBRARY, root, output,
                model_path=root / "model", steps=1, cfg_scale=1.0,
                max_attempts=3, gpu_label="0", source_hash_cache={},
                code_sha256="test-code",
            )
            self.assertEqual(outcome, "success")
            metadata = json.loads((pair_dir / "metadata.json").read_text())
            self.assertIsNotNone(metadata["recovery_from_invalid_success"])
            self.assertEqual(len(metadata["attempts"]), 1)


class RealPlanTests(unittest.TestCase):
    def test_generated_real_plan_when_present(self):
        pair_root = Path(os.environ.get(
            "GATE3DGEN_TEST_PAIR_ROOT", Path(__file__).resolve().parents[1] / "outputs/pairs"
        ))
        root = pair_root / "plans"
        master = root / "master.jsonl"
        pilot = root / "pilot.jsonl"
        if not master.is_file() or not pilot.is_file():
            self.skipTest("real plan has not been generated")
        rows = list(iter_jsonl(master))
        pilot_rows = list(iter_jsonl(pilot))
        summary = json.loads((root / "plan_summary.json").read_text())
        self.assertEqual(len(rows), summary["pairs"])
        self.assertEqual(len(pilot_rows), summary["pilot_pairs"])
        self.assertEqual(len({row["pair_id"] for row in rows}), len(rows))
        self.assertEqual(len({row["sha256"] for row in rows}), summary["assets"])
        self.assertEqual({row["split"] for row in pilot_rows}, {"train"})
        self.assertTrue({row["pair_id"] for row in pilot_rows}.issubset(
            {row["pair_id"] for row in rows}))
        assets = collections.defaultdict(list)
        for row in rows:
            assets[row["sha256"]].append(row)
        for asset_rows in assets.values():
            self.assertEqual(len(asset_rows), 12)
            self.assertEqual(len({row["task_id"] for row in asset_rows}), 12)
            self.assertEqual(len({row["split"] for row in asset_rows}), 1)
            self.assertEqual(collections.Counter(row["view_slot"] for row in asset_rows),
                             {0: 3, 1: 3, 2: 3, 3: 3})


if __name__ == "__main__":
    unittest.main()
