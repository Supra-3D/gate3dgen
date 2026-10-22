#!/usr/bin/env bash
# Score the retained Pixal3D example unless external inputs are supplied.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
manifest="${EVAL_MANIFEST:-assets/examples/eval/cases.jsonl}"
generation="${EVAL_GENERATION_DIR:-assets/examples/eval/generation}"
cameras="${EVAL_CAMERA_PLAN:-assets/examples/eval/camera_plan.json}"
output="${EVAL_OUTPUT_ROOT:-outputs/eval}"
python -m gate3dgen.evaluation.metrics2d --manifest "$manifest" --output-dir "$output/2d" --lpips
python -m gate3dgen.evaluation.metrics3d_official \
  --manifest "$manifest" --generation-dir "$generation" --baseline pixal3d \
  --variant degraded --variant restored_s01 --paper-protocol --output-dir "$output/geometry"
for mode in gt prediction; do
  python -m gate3dgen.evaluation.render_pbr --metrics "$output/geometry/per_case.jsonl" \
    --camera-plan "$cameras" --render-root "$output/renders" --mode "$mode"
done
python -m gate3dgen.evaluation.pbr_metrics --manifest "$manifest" --baseline pixal3d \
  --variant degraded --variant restored_s01 --views 4 --lpips --paper-protocol \
  --render-root "$output/renders" --output-dir "$output/pbr"
