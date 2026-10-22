# Setup

This Git repository excludes checkpoints. Download the assembled Gate3DGen release
with `python setup/download_weights.py`; see [weights and external models](../docs/weights.md).
Use `GATE3DGEN_SKIP_UPSTREAM_DOWNLOAD=1 bash setup/setup.sh` to install the core
environment without also downloading the full FireRed training base and Pixal3D.
The default 3D demo still needs the separately installed Pixal3D backend and its
auxiliary models, including authorized RMBG-2.0 weights.

Run from the repository root:

```bash
bash setup/setup.sh
conda activate gate3dgen
python setup/check.py
```

The Gate3DGen inference package belongs under `checkpoints/`; see
[installation](../docs/installation.md) for its layout and checks.

`setup.sh` installs and checks the environment and Blender, then downloads the complete official
FireRed-Image-Edit-1.1 and Pixal3D model repositories. To run just the download step again:

```bash
python setup/download_models.py all
```

Downloads go to `checkpoints/upstream/` inside this checkout; these weights are ignored by Git
and must be excluded from source archives. The downloader fetches the complete contents of both official
model repositories, without file filters. Pixal3D auxiliary models hosted in other repositories
are separate dependencies. For FireRed workflows, set:

```bash
export GATE3DGEN_MODEL_PATH="$PWD/checkpoints/upstream/FireRed-Image-Edit-1.1"
```

Pixal3D also needs upstream code, compiled dependencies and auxiliary models.
Follow the [official installation](https://github.com/TencentARC/Pixal3D#installation)
in a separate environment, then set `PIXAL3D_REPO`, `PIXAL3D_PYTHON` and
`GATE3DGEN_RMBG_MODEL`. A checkpoint download alone does not install those dependencies.

For maintainer testing, keep downloads outside the checkout:

```bash
GATE3DGEN_UPSTREAM_ROOT=/path/to/external/models bash setup/setup.sh
# Or run only the downloader:
python setup/download_models.py all --output-root /path/to/external/models
export GATE3DGEN_MODEL_PATH=/path/to/external/models/FireRed-Image-Edit-1.1
export PIXAL3D_MODEL=/path/to/external/models/Pixal3D
export GATE3DGEN_MODEL_CACHE=/path/to/external/cache
export HF_HOME="$GATE3DGEN_MODEL_CACHE/huggingface"
export TORCH_HOME="$GATE3DGEN_MODEL_CACHE/torch"
```

Demo and reconstruction default to `checkpoints/upstream/Pixal3D` unless `PIXAL3D_MODEL` is set.

`python setup/check.py --environment-only` checks dependencies without model files;
`--weights-only` checks package readiness without inference or weight hashing.

If automatic Blender download is unavailable, export `GATE3DGEN_BLENDER=/path/to/official/blender`
before running the installer, checker and evaluator. A system Blender without valid OCIO
resources is rejected; it must not be used for texture metrics.
