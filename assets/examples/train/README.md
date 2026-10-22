# Real training-pair example

See [the training guide](../../../docs/training.md) and [data preparation](../../../docs/data.md).

```text
assets/examples/train/
├── pairs.jsonl
├── manifests/{train,validation,test}.jsonl
└── pairs/
    └── <pair_id>/{gt.png,degraded.png,metadata.json}
```

There are four historical FireRed pairs from the museum asset
`demo_asset_01` (source asset `0a648b5b6efa423ab84c13dc77012953`):
shadow (`006`), overexposure (`007`), glass reflection
(`000`) and underexposure (`003`). Images are CC BY 4.0; see [ATTRIBUTION.md](../ATTRIBUTION.md).
GT is the generated clean scene, not the albedo. No purifier prediction is included here.
Paths are relative to `assets/examples/train`; albedos resolve from `assets/examples/data`.
Metadata retains historical prompts, seeds and generation checks without machine-specific paths.
Example IDs use `demo_asset_01`; `source_asset_id` preserves the original asset identity,
and `parameter_sampling_asset_id` preserves the original random-parameter key for exact
procedural audit replay. Image hashes, pair IDs and historical seeds are unchanged.

All four pairs are in `train`; `validation.jsonl` and `test.jsonl` are intentionally empty.
The included pairs use `quality_status=not_audited`. Run the audit and export passing pairs
before training.

For actual example training, generate new pairs or audit the included pairs using
[the data example](../data/README.md),
then export passing pairs to `outputs/examples/manifests`. Extract and review real clean-GT RMBG
masks under `outputs/examples/masks`, build router supervision and sample weights under
`outputs/examples/router`, and extract frozen-router embeddings under `outputs/examples/cache`.
Follow the data guide; included images do not replace those steps. For included pairs, use
`assets/examples/train` as the purifier image root instead of `outputs/examples/pairs`.
Add independent validation assets before launching the released router recipe: neither
validation set can be constructed from four views of this training asset.

Copy `configs/train/router.yaml` and `configs/train/purifier.yaml` to a local runtime directory
and change **paths only**:

| Config field | Example run location |
|---|---|
| Router `data.*_manifest` | `outputs/examples/router/{train,demo_validation,validation}.jsonl` |
| Router `data.data_root` | `.` (router image paths are repository-relative) |
| Router `data.texverse_root` | `assets/examples/data` |
| Router `data.texverse_full_root` | `assets/examples/data` |
| Router `sample_weights.path` | `outputs/examples/router/weights/sample_weights.jsonl` |
| Router `paths.output_dir` | `outputs/examples/training/router` |
| Purifier `data.manifests.*` | `outputs/examples/manifests/{train,validation,test}.jsonl` |
| Purifier `data.data_root` | `outputs/examples/pairs` |
| Purifier `data.subject_mask_root` | `outputs/examples/masks` |
| Purifier `paths.cache_root` | `outputs/examples/cache/purifier` |
| Purifier `paths.output_dir` | `outputs/examples/training/purifier` |

After preparing audited data, reviewed masks, supervision/weights and independent validation,
use the existing entry points with your edited runtime configs:

```bash
python -m gate3dgen.training.router_tools.launch --config outputs/examples/router.yaml --dry-run
python -m gate3dgen.training.router_tools.launch --config outputs/examples/router.yaml
python -m gate3dgen.training.cli --config outputs/examples/purifier.yaml condition-preflight
python -m gate3dgen.training.cli --config outputs/examples/purifier.yaml check --split train
python -m gate3dgen.training.cli --config outputs/examples/purifier.yaml prepare-data
python -m gate3dgen.training.cli --config outputs/examples/purifier.yaml extract-embeddings
python -m gate3dgen.training.cli --config outputs/examples/purifier.yaml train --dry-run
```

See the training guide for the actual 32-GPU/four-node purifier launch. These calls require the
preparation above; the bundled asset alone is not a complete training dataset.

The released checkpoint pair remains the default for conditioning. External model locations are
supplied through `GATE3DGEN_MODEL_PATH`; do not copy base weights into these examples. See the
training guide for config flags, preflight, launch, export and resume. The released GPU topology
is router: **one node / 8 A800**, purifier: **four nodes / 32 A800**.
