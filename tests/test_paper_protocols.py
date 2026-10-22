import json
from pathlib import Path
import numpy as np
import pytest
from gate3dgen.evaluation.router import summarize
from gate3dgen.evaluation.router_metrics import calibrate_threshold
from gate3dgen.evaluation.metrics2d import foreground_metrics
from gate3dgen.evaluation.dino_features import frechet_distance
from gate3dgen.evaluation.preferences import assignments, human_summary, vlm_summary
from gate3dgen.training.ablations import ablation_config
from gate3dgen.data.statistics import manifest_statistics
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_paper_h_is_not_standard_f1_and_calibration_uses_original_rule():
    rows = [{"label": y, "purify_probability": p} for y, p in [(1,.9),(1,.1),(0,.1),(0,.1)]]
    result = summarize(rows, .5)
    assert result["h_recall_accuracy"] == pytest.approx(.6)
    assert result["f1"] == pytest.approx(2/3)
    chosen, sweep = calibrate_threshold(rows)
    assert chosen["threshold"] == .5
    assert [r["threshold"] for r in sweep] == [0,.5,1]


def test_foreground_lpips_ignores_background_but_iou_tracks_prediction_mask():
    class Learned:
        lpips_model, dino_model = object(), None
        def call_arrays(self, a, b):
            return {"lpips_alex": float(np.mean((a-b)**2))}
    gt = np.zeros((16,16,3), np.float32)
    pred = np.ones_like(gt)
    mask = np.zeros((16,16), np.uint8)
    mask[4:12,4:12] = 255
    pred[4:12,4:12] = 0
    other = np.roll(mask, 4, axis=0)
    result = foreground_metrics(pred, gt, mask, other, Learned())
    assert result["object_lpips_alex"] == 0
    assert result["foreground_mask_iou"] == pytest.approx(1/3)
    with pytest.raises(ValueError, match="empty foreground"):
        foreground_metrics(pred,gt,np.zeros_like(mask),other,Learned())


def test_distribution_distance_not_mean_paired_cosine():
    x = np.array([[0,0],[1,0],[0,1],[1,1]],float)
    assert frechet_distance(x,x) == pytest.approx(0,abs=1e-5)
    assert frechet_distance(x+2,x) == pytest.approx(8,abs=1e-5)
    with pytest.raises(ValueError,match="two samples"):
        frechet_distance(x[:1],x)


def test_human_blinding_and_invalid_ratings_exclusion():
    rows = [{"case_id":"case","methods":{"ours":"ours.png","baseline":"other.png"}}]
    prepared = assignments(rows,42)
    assert prepared == assignments(rows,42)
    slot = next(k for k,v in prepared[0]["slot_map"].items() if v=="ours")
    rating = {"participant_id":"person", "assignment_id":prepared[0]["assignment_id"], "valid":True,
              "identity":slot, "purification":slot, "quality":"tie"}
    result=human_summary(prepared,[rating])
    assert result["baselines"]["baseline"]["identity"]["ours_preference_percent"] == 100
    assert result["baselines"]["baseline"]["quality"]["ours_preference_percent"] == 0
    with pytest.raises(ValueError,match="Duplicate"):
        human_summary(prepared,[rating,rating])


def test_vlm_equal_domain_weighting_and_complete_four_view_population():
    rows=[{"case_id":"case","method":"ours","domain":"2d","identity":10,"purification":10}]
    rows += [{"case_id":"case","method":"ours","domain":"3d","generator":g,"view":v,
              "identity":2,"purification":4,"quality":6} for g in ("pixal3d","trellis2") for v in range(4)]
    result=vlm_summary(rows)["ours"]
    assert (result["identity"],result["purification"],result["quality"]) == (6,7,6)
    with pytest.raises(ValueError,match="four views"):
        vlm_summary(rows[:-1])


def test_paper_ablation_removes_both_region_losses_without_overloading_legacy_labels():
    base=yaml.safe_load((ROOT/"configs/train/purifier.yaml").read_text())
    a,b=(ablation_config(base,name) for name in ("purifier-A","purifier-B"))
    assert not a["condition_model"]["enabled"] and b["condition_model"]["enabled"]
    for config in (a,b):
        assert config["loss"]["pixel_l1_weight"] == 1
        assert all(config["loss"][name]==0 for name in ("pixel_subject_l1_weight","pixel_affected_l1_weight","pixel_subject_gradient_weight"))
    assert base["loss"]["restoration_v3"] is True


def test_statistics_fingerprint_does_not_expose_ids_and_rejects_leakage(tmp_path):
    source=tmp_path/"manifest.jsonl"
    rows=[{"pair_id":"one","sha256":"private-asset","split":"train","task_id":"blur"}]
    source.write_text("\n".join(json.dumps(r) for r in rows))
    result=manifest_statistics(source)
    assert result["splits"]["train"]["assets"]==1
    assert "private-asset" not in json.dumps(result)
    rows.append({**rows[0],"pair_id":"two","split":"test"})
    source.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match="overlap"):
        manifest_statistics(source)


def test_standalone_mlp_trains_and_evaluates_without_purifier(tmp_path):
    import torch
    from gate3dgen.training.router_tools.mlp import main, assert_disjoint, load_cache
    def cache(name):
        path = tmp_path / (name + ".pt")
        torch.save({"features": torch.arange(16).reshape(4,4).float() / 16,
                    "labels": torch.tensor([0,1,0,1]), "vae_identity": "fixture-vae-revision",
                    "samples": [{"sample_id": f"{name}-{i}", "asset_id": f"{name}-{i}",
                                 "image_sha256": f"{name}-hash-{i}", "label": i%2} for i in range(4)]}, path)
        return path
    train, dev = cache("train"), cache("dev")
    checkpoint = tmp_path / "mlp.pt"
    main(["train", "--train-cache", str(train), "--development-cache", str(dev),
          "--steps", "2", "--batch-size", "2", "--learning-rate", "0.001", "--device", "cpu", "--output", str(checkpoint)])
    main(["eval", "--cache", str(dev), "--checkpoint", str(checkpoint), "--threshold", "0.5",
          "--device", "cpu", "--output-dir", str(tmp_path / "eval")])
    assert json.loads((tmp_path / "eval/summary.json").read_text())["samples"] == 4
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(load_cache(train), load_cache(train))




def test_separate_population_audit_reports_intersections_without_private_ids(tmp_path):
    from gate3dgen.data.statistics import split_statistics
    inputs = {}
    for role in ("train", "development", "final"):
        path = tmp_path / (role + ".jsonl")
        path.write_text(json.dumps({"sample_id": role, "asset_id": "private-" + role,
                                    "image_sha256": "private-hash-" + role, "split": "source_label"}) + "\n")
        inputs[role] = path
    result = split_statistics(inputs)
    assert result["cross_split_asset_isolation_verified"] is True
    assert len(result["cross_split_checks"]) == 3
    assert all(r["shared_assets"] == r["shared_declared_image_hashes"] == 0 for r in result["cross_split_checks"])
    assert "private-" not in json.dumps(result) and str(tmp_path) not in json.dumps(result)
    assert set(result["splits"]) == set(inputs)
    inputs["final"].write_text(json.dumps({"sample_id":"different", "asset_id":"private-train"}) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        split_statistics(inputs)


def test_public_helpers_and_worker_imports_do_not_eagerly_load_legacy_heads():
    import ast
    from gate3dgen.training.checkpoint_state import normalize_state_dict_root
    for name in ("training/inference.py", "training/router_tools/mlp.py",
                 "training/vendor/train/src/forward_step.py", "training/vendor/train/src/model_provider.py"):
        tree = ast.parse((ROOT / "gate3dgen" / name).read_text())
        imports = [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert not any("experimental" in module for module in imports)
    normalized = normalize_state_dict_root({"module.proj_out.weight": 1, "module.albedo_aux_head.net.0.weight": 2})
    assert normalized == {"proj_out.weight": 1, "albedo_aux_head.net.0.weight": 2}
    # Strict loading, not this generic helper, decides whether unknown tensors are allowed.
    assert normalize_state_dict_root({"module.weight": 1, "unrelated.weight": 2}) == {"module.weight": 1, "unrelated.weight": 2}
