"""Generate explicit paper ablation configs without overloading historical A/B/C recipe names."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import yaml


def ablation_config(config, experiment):
    result = deepcopy(config)
    result["paper_ablation"] = experiment
    if experiment in {"purifier-A", "purifier-B"}:
        result["condition_model"]["enabled"] = experiment == "purifier-B"
        # Table 5 removes BOTH region terms and the edge term; L1 remains in all variants.
        result["loss"].update(restoration_v3=False, pixel_l1_weight=1., pixel_full_l1_weight=0.,
                              pixel_subject_l1_weight=0., pixel_affected_l1_weight=0., pixel_subject_gradient_weight=0.)
        result["data"]["subject_mask_required"] = False
        result["variant"] = "PAPER_" + experiment.replace("-", "_").upper()
        result["paths"] = {"cache_root": f"outputs/cache/{experiment}", "output_dir": f"outputs/training/{experiment}"}
    elif experiment == "router-language-only":
        result["variant"] = "QWEN_VL_LM_GATE_V3_A_8GPU"
        result["model"]["vision_lora"]["enabled"] = False
        result["paths"]["output_dir"] = "outputs/training/router-language-only"
    else:
        raise ValueError("Unknown paper ablation")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("purifier-A", "purifier-B", "router-language-only"))
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    base = args.base_config or Path("configs/train/router.yaml" if args.experiment.startswith("router") else "configs/train/purifier.yaml")
    config = ablation_config(yaml.safe_load(base.read_text()), args.experiment)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
