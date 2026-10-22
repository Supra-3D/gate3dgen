#!/usr/bin/env bash
# Thin calls into the maintained generator/auditor; no synthetic audit decisions.
set -euo pipefail
if [[ ! -f pyproject.toml ]]; then
  echo 'Run from the Gate3DGen repository root.' >&2
  exit 2
fi
runtime="${EXAMPLE_OUTPUT_ROOT:-outputs/examples}"
pair_root="${EXAMPLE_PAIR_ROOT:-$runtime/pairs}"
pair_manifest="${EXAMPLE_PAIR_MANIFEST:-$pair_root/manifests/all_pass.jsonl}"
gpus="${EXAMPLE_GPUS:-0}"
case "${1:-help}" in
  plan)
    python -m gate3dgen.data.pair_generation.pipeline plan \
      --input assets/examples/data/assets.jsonl --dataset-root assets/examples/data \
      --output-root "$pair_root"
    ;;
  generate)
    python -m gate3dgen.data.pair_generation.pipeline check \
      --input assets/examples/data/assets.jsonl --dataset-root assets/examples/data --gpus "$gpus"
    python -m gate3dgen.data.pair_generation.pipeline run \
      --scope full --gpus "$gpus" --steps 40 --cfg-scale 4.0 \
      --dataset-root assets/examples/data --output-root "$pair_root"
    ;;
  audit)
    python -m gate3dgen.data.quality_audit.cli all \
      --scope full --manifest "$pair_manifest" --pair-root "$pair_root" \
      --dataset-root assets/examples/data --output-root "$runtime/audit" \
      --skip-vision --vlm-all-pairs --vlm-backend vllm \
      --vlm-base-url "${EXAMPLE_VLM_BASE_URL:-http://127.0.0.1:8100/v1}" \
      --vlm-served-model "${EXAMPLE_VLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct}" --gpus "$gpus"
    ;;
  export)
    python -m gate3dgen.data.prepare dataset --manifest "$pair_manifest" \
      --audit "$runtime/audit/manifests/pair_quality_audit.jsonl" --output "$runtime/manifests"
    ;;
  *)
    echo 'Usage: bash scripts/run_example_data.sh {plan|generate|audit|export}'
    [[ "${1:-help}" == help ]] || exit 2
    ;;
esac
