# Model weights and external dependencies

The GitHub repository contains code, configuration, documentation and example
assets. Checkpoint tensors, optimizer states and model caches are excluded.

## Core Gate3DGen release

**Download:** [hllee0615/gate3dgen](https://huggingface.co/hllee0615/gate3dgen)

The maintainer has made this repository **Public**, with **no access gate**.
No access application or Hugging Face token is required for the core model.
At the 2026-10-22 access check, only the initial `.gitattributes` was committed;
the large-weight upload was still pending. Public visibility is verified, but
complete checkpoint availability remains pending until that upload finishes.
See [the machine-readable access check](metadata/weight_access.json).

After the weight upload has completed, run from the source checkout:

```bash
conda activate gate3dgen
python setup/download_weights.py
python setup/check.py --weights-only
```

The downloader explicitly uses anonymous access (`token=False`), downloads the
assembled release and validates checkpoint markers, indexed shards and runtime
files. It does not reconstruct from deltas or download the raw FireRed base.
Checkpoints preserve the reconstructed tensor bytes and original dtypes; the
restoration replacement tensors are FP32 and GateB replacement tensors are BF16.

```text
checkpoints/
├── router/checkpoint-6000/
│   ├── checkpoint_complete.json
│   └── exported_model/{config.json,model.safetensors.index.json,model-part-*.safetensors,...}
├── purifier/checkpoint-15000/
│   └── {checkpoint_complete.json,config.json,pytorch_model_fsdp.bin}
└── runtime/{processor,vae,scheduler,runtime_manifest.json,LICENSE,NOTICE.md}
```

For reproducible downloads, pass `--revision <published-commit-sha>` after a full
release commit exists. For external storage:

```bash
python setup/download_weights.py --output-root /absolute/path/to/weights
export GATE3DGEN_ROUTER_CHECKPOINT=/absolute/path/to/weights/router/checkpoint-6000
export GATE3DGEN_PURIFIER_CHECKPOINT=/absolute/path/to/weights/purifier/checkpoint-15000
export GATE3DGEN_RUNTIME_PATH=/absolute/path/to/weights/runtime
python setup/check.py --weights-only
```

The HF release contains `PACKAGE_METADATA.json` and `MANIFEST.sha256`. Run
`shasum -a 256 -c MANIFEST.sha256` from the downloaded weight directory for full
file hashing; the readiness checker does not hash weights or run GPU inference.

## External models

All Hugging Face model links below were checked against the official anonymous
API on 2026-10-22. A one-byte HTTP Range probe checked one representative weight
per model: public entries returned HTTP 206; gated entries returned HTTP 401.
This checks access, not full-download integrity or runtime compatibility.

| Component | Weight repository | Where it is used | Access |
| --- | --- | --- | --- |
| FireRed base | [FireRedTeam/FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) | Training initialization, synthesis and historical delta reconstruction; not needed again for assembled core inference | Public, no gate |
| Pixal3D | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | Default downstream 3D generator | Public, no gate |
| MoGe-2 | [Ruicheng/moge-2-vitl](https://huggingface.co/Ruicheng/moge-2-vitl) | Current Pixal3D camera-estimation default | Public, no gate |
| MoGe-2 normal variant | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | Alternative upstream variant; not the inspected Pixal3D default | Public, no gate |
| DINOv3, Pixal3D default mirror | [camenduru/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | Current official Pixal3D `IMAGE_COND_CONFIGS` default | Public third-party mirror, no gate; upstream DINOv3 terms still apply |
| DINOv3, original publisher | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | Optional explicit DINOv3 override | Manual access approval and login required |
| RMBG-2.0 | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | Local background-removal model required by the project's RGB-to-3D wrappers and training mask extraction | Auto gate: log in and accept the model's access terms; separate license applies |
| TRELLIS.2 | [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | Optional TRELLIS.2 backend and its declared component weights | Public, no gate |
| DINOv2 | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | Optional Transformers feature-metric provider | Public, no gate; paper FD uses the separately specified register backbone |

The current Pixal3D dependencies were traced through
[`inference.py`](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/inference.py)
and its
[`DinoV3ProjFeatureExtractor`](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py).
Upstream revisions may change these defaults; tested access revisions are recorded
in the JSON access report. Public mirrors are not a replacement for upstream licenses.

### Optional data, audit and comparison workflows

These links cover additional model families referenced by the source code. They
are not needed for the core router/purifier. The entries below were checked with
anonymous HF **metadata only**; their model tensors were not downloaded or range-probed.

| Component | Weight link | Use and access result |
| --- | --- | --- |
| Qwen image editor | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Optional pair-synthesis editor; public, no gate |
| Earlier FireRed | [FireRedTeam/FireRed-Image-Edit-1.0](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0) | Legacy synthesis option; public, no gate |
| Qwen audit VLM | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | Local/served dataset-review VLM; public, no gate |
| DINOv2 audit backbone | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) | Audit uses a local native `dinov2_vitb14` checkpoint; public HF model reference, no gate |
| DINOv2 with registers | [facebook/dinov2-with-registers-large](https://huggingface.co/facebook/dinov2-with-registers-large) | Paper FD backbone family; the current paper runner expects native Torch Hub weights, not a Transformers checkpoint interchange |
| SAM2.1 audit segmenter | [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small) | Audit default is native `sam2.1_hiera_small.pt` plus matching SAM2 code/config; public HF model reference, no gate |
| TRELLIS sparse decoder | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | Sparse-decoder dependency referenced by the optional TRELLIS wrapper; public, no gate |
| Step1X baseline | [stepfun-ai/Step1X-Edit-v1p2](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2) | Optional comparison; public, no gate |
| OmniGen2 baseline | [OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2) | Optional comparison; public, no gate |
| FLUX Kontext baseline | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Optional comparison; automatic access gate and separate model terms |
| Mage baseline | [microsoft/Mage-Flow-Edit](https://huggingface.co/microsoft/Mage-Flow-Edit) | Exact repository ID declared by the adapter; anonymous API returned HTTP 401. Availability/access could not be verified; not a confirmed public dependency |

HF links identify the model families; download formats must match each wrapper's
native loader. For example, the audit's Torch Hub DINO state dict and SAM2 native
checkpoint must match its specified code/config. The source provides no verified
automatic conversion from every listed HF format. OpenAI image baselines import
provider outputs and do not distribute downloadable HF model weights. Optional
LPIPS/torchvision metric weights follow their upstream package download mechanisms.

### Pixal3D installation and auxiliary downloads

Follow the [official Pixal3D installation](https://github.com/TencentARC/Pixal3D#installation)
in a compatible separate environment, then configure `PIXAL3D_REPO` and
`PIXAL3D_PYTHON`. Download the main model with:

```bash
python setup/download_models.py pixal3d
export PIXAL3D_MODEL="$PWD/checkpoints/upstream/Pixal3D"
```

Its MoGe and DINOv3 components can be fetched by upstream loaders. The project
requires a local RMBG-2.0 snapshot. After accepting its HF access terms and logging
in with `hf auth login`, download it explicitly:

```bash
hf download briaai/RMBG-2.0 --local-dir checkpoints/RMBG-2.0
export GATE3DGEN_RMBG_MODEL="$PWD/checkpoints/RMBG-2.0"
```

Pixal3D's high-resolution image features also load
[NAF](https://github.com/valeoai/NAF) via `torch.hub`. Its upstream weight is hosted
on a [GitHub release](https://github.com/valeoai/NAF/releases/download/model/naf_release.pth),
not Hugging Face; this is an additional non-HF download. Installing only the
Pixal3D HF checkpoint does not install NAF or compiled/code dependencies.

## Historical reconstruction tools

`setup/restore_original_weights.py` and `setup/delta_weights.py` remain available
for earlier replacement-tensor packages. Their required `checkpoints/finetuned/`
payload is not included in this source-only repository, and reconstruction is not
part of the normal download path above.
