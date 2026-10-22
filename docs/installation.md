# Installation

For the source-only GitHub release, use the [core weight download guide](weights.md).
`GATE3DGEN_SKIP_UPSTREAM_DOWNLOAD=1 bash setup/setup.sh` skips the large training-base
and downstream-model downloads; then run `python setup/download_weights.py`.

Run from the repository root:

```bash
bash setup/setup.sh
conda activate gate3dgen
```

The installer uses [setup/environment.yml](../setup/environment.yml) and
[version constraints](../setup/constraints-host.txt), installs official OCIO-enabled Blender,
and checks the data, training and evaluation dependencies. It then downloads the complete official
FireRed-1.1 and Pixal3D model repositories to `checkpoints/upstream/`.
Set `GATE3DGEN_UPSTREAM_ROOT` to an external directory for maintainer tests.
The constraints pin direct
dependencies, not every transitive package. The optional
[light environment](../setup/environment-lite.yml) is for CPU utilities only.

To reuse an existing official Blender, set `GATE3DGEN_BLENDER` before installation.
Its embedded Python is checked separately; WebP conversion runs in the host environment.
For a custom certificate chain, configure your trusted CA; do not disable TLS verification.

## Checkpoints

Place the self-contained inference package here:

```text
checkpoints/
├── router/checkpoint-6000/
│   ├── checkpoint_complete.json
│   └── exported_model/{config.json,model.safetensors.index.json,model-*.safetensors,...}
├── purifier/checkpoint-15000/
│   └── {checkpoint_complete.json,config.json,pytorch_model_fsdp.bin}
└── runtime/
    ├── processor/
    ├── vae/
    ├── scheduler/
    └── {runtime_manifest.json,LICENSE,NOTICE.md}
```

These contain the complete trained router and purifier plus frozen inference components.
Demo does not load the original FireRed diffusion or Qwen weights.
Keep the package's license and notices.

```bash
python setup/check.py
python setup/check.py --environment-only
python setup/check.py --weights-only
```

Checks validate dependencies and package readiness, not benchmark quality or weight hashes.
The Hugging Face release contains `PACKAGE_METADATA.json` and `MANIFEST.sha256` for
file sizes and independent verification; see [weights](weights.md). Model files are
excluded from this source repository. The setup readiness check does not hash large weights.

Custom locations use `GATE3DGEN_ROUTER_CHECKPOINT`, `GATE3DGEN_PURIFIER_CHECKPOINT`
and `GATE3DGEN_RUNTIME_PATH`. Demo uses selective routing, seed 42 and automatic CPU offload;
KEEP copies the original bytes, while PURIFY saves a one-step PNG and a decision record.
The default demo then generates before/after Pixal3D GLBs without replacing the saved image.
Install the Pixal3D backend described in [setup](../setup/README.md) first.

### Optional demo modes

These are script options; the main README shows only the default command:

```bash
python -m gate3dgen.demo assets/demo_01.png --output outputs/router --router-only
python -m gate3dgen.demo assets/demo_01.png --output outputs/purifier --purify-only
```

Router-only writes the decision JSON without loading the purifier or 3D generator.
Purify-only skips classification, but still uses the router's trained hidden-state encoder;
it saves the purified image and generates before/after GLBs. The flags are mutually exclusive.

Default output layout:

```text
outputs/demo/
├── demo_01.png                         # processed image; unchanged bytes for KEEP
├── demo_01.png.json                    # decision and generation status
├── manifest.jsonl                     # matched input/output image paths
└── generation/
    ├── conditions.jsonl
    └── pixal3d/demo_01/{degraded.glb,restored_s01.glb}
```

The router/purifier are released before Pixal3D starts. If 3D generation fails, the processed
images and decisions remain available; the command exits nonzero and records the failure.
Use a new output directory for each run to keep meshes paired with the correct images.

## Upstream models

[Setup](../setup/README.md) provides official FireRed-1.1 and Pixal3D download commands.
FireRed is needed for data synthesis and the unchanged paper-training initialization,
not released-model inference. Pixal3D's upstream code and auxiliary models need its own
compatible environment; its checkpoint alone is not enough.
The default downloads are `checkpoints/upstream/{FireRed-Image-Edit-1.1,Pixal3D}`;
maintainer tests override them with external paths to avoid accidental redistribution.

Use the prepared inference package in the layout above. Training states and inference packages
are different artifacts; training-checkpoint export and release-packaging tools are not bundled.

## Troubleshooting

If activation succeeds but imports come from a different Python, check the interpreter:

```bash
type -a python conda
python -c 'import sys, torch; print(sys.executable); print(torch.__version__)'
```

After confirming `CONDA_PREFIX` is the intended environment, bypass shell aliases with
`"$CONDA_PREFIX/bin/python" -I setup/check.py`. Do not install missing packages into another
Python to compensate for a shadowed shell command.

Examples live in [assets/examples](../assets/examples/README.md). Generated data, caches,
training states and reports belong in `outputs/` or an external work directory.
