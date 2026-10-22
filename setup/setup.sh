#!/usr/bin/env bash
# Set up the environment and download the complete official upstream model repositories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${GATE3DGEN_ENV_NAME:-gate3dgen}"
# Do not satisfy imports accidentally from user-site or another Python runtime.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
# Keep optional conda plugins from requiring unrelated default-channel terms
# before the YAML's conda-forge/nodefaults selection is applied. No terms are
# accepted and no user conda configuration is changed.
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda/Anaconda, then rerun this script." >&2
  exit 2
fi

cd "$ROOT"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file setup/environment.yml --prune
else
  conda env create --name "$ENV_NAME" --file setup/environment.yml
fi

if [[ -n "${GATE3DGEN_BLENDER:-}" ]]; then
  BLENDER_BIN="$GATE3DGEN_BLENDER"
  conda run --name "$ENV_NAME" python -m gate3dgen.evaluation.blender_runtime \
    --blender "$BLENDER_BIN" --install
  conda run --name "$ENV_NAME" python setup/check.py --environment-only --blender "$BLENDER_BIN"
else
  conda run --name "$ENV_NAME" python setup/blender.py
  conda run --name "$ENV_NAME" python setup/check.py --environment-only
fi

MODEL_ROOT="${GATE3DGEN_UPSTREAM_ROOT:-$ROOT/checkpoints/upstream}"
if [[ "${GATE3DGEN_SKIP_UPSTREAM_DOWNLOAD:-0}" != "1" ]]; then
  conda run --name "$ENV_NAME" python setup/download_models.py all --output-root "$MODEL_ROOT"
fi

echo "Setup complete. Activate it with: conda activate $ENV_NAME"
echo "Official models: $MODEL_ROOT"
echo "Gate3DGen inference weights: python setup/download_weights.py"
