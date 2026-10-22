# Data preparation

Run commands from the repository root with the Conda environment activated and
`GATE3DGEN_MODEL_PATH` pointing to an external FireRed-Image-Edit-1.1 download.
The full training corpus, benchmark images, mesh assets, HDRIs and private selection lists are not
distributed. One explicitly attributed CC BY 4.0 museum asset is included as a small example.
Demo inputs and workflow examples are stored under `assets/`.

The implementation is consolidated from the original pair-generation, quality-audit and router-data
tools. Train the released recipe on your own authorized data.

## 1. Supply rendered source views

Prepare four albedo views per asset under `data/TexVerse/renders_crossview/` and a JSONL inventory
`data/assets.jsonl`. The generator consumes these rendered images; it does not download third-party
assets or reconstruct missing renders. Each JSONL line has this structure:

```json
{"sha256":"your_unique_asset_id","semantic":{"primary_category":{"coarse":"object","fine":"figurine"}},"vlm":{"view_selection":{"selected_views":[{"view_id":"000","rank":0,"path":"data/TexVerse/renders_crossview/your_unique_asset_id/000.png"},{"view_id":"003","rank":1,"path":"data/TexVerse/renders_crossview/your_unique_asset_id/003.png"},{"view_id":"006","rank":2,"path":"data/TexVerse/renders_crossview/your_unique_asset_id/006.png"},{"view_id":"007","rank":3,"path":"data/TexVerse/renders_crossview/your_unique_asset_id/007.png"}]}}}
```

`sha256` is the historical asset-ID field, not an image hash. The `vlm.view_selection` field records
the chosen views; manually selected views are also accepted. View IDs are rendering metadata, not
part of an asset license. Preserve required asset attribution separately from the image inventory.

A complete small directory example is included in [assets/examples/data](../assets/examples/data/README.md),
including four real orthographic albedo views of one asset and an actual `assets.jsonl`.
See [asset attribution](../assets/examples/ATTRIBUTION.md) for CC BY 4.0 and modification notices.
`scripts/run_example_data.sh` calls the generator, actual VLM audit and accepted-pair exporter;
the four historical clean-scene GT/degraded pairs are also included under `assets/examples/train`.
The example commands use `--dataset-root assets/examples/data` and keep
generated pairs/audit outputs in `outputs/examples/`. Do not put generated data in `assets/`.
For private validation, keep real source views, inventories and their symlinks in an external
work directory, and pass its paths through `--input` and `--dataset-root`. Do not add private
data links to the source checkout; see [local-input staging](../assets/examples/README.md#maintainer-only-real-inputs).

| Location | Required content |
|---|---|
| `data/assets.jsonl` | Asset inventory and four selected-view paths per asset |
| `data/TexVerse/renders_crossview/<asset>/albedo/` | Authorized rendered source PNGs |
| `outputs/pairs/` | Generated pairs, plans, generation manifests and audit input |
| `outputs/audit/` | Actual quality checks and VLM observations/decisions |
| `data/manifests/` | Accepted train/validation/test manifests |
| `outputs/masks/` | Real clean-GT RMBG masks and explicit review records |
| `data/router/` | Expanded supervision, held-out validation and sample weights |

For the paper protocol these are **four selected views from eight rendered candidates**, not four
newly rendered arbitrary views. Preserve all eight candidate camera transforms, render settings,
asset provenance and the selection record privately. The public generator begins at the selected-view
inventory.
If using a new selector, record its model/revision, prompt hash, selected IDs and any manual exclusions.
Do not describe a manually selected four-view fixture as the original VLM selection pipeline.

The planner supports arbitrary nonempty inventories. It assigns approximately 90/5/5 percent of
assets to train/validation/test, with at least one asset per holdout for inventories of three or more
assets. All views of an asset stay in one split. Each asset receives all 12 synthesis tasks, three
per view, using a deterministic seed.

```bash
python -m gate3dgen.data.pair_generation.pipeline check \
  --input data/assets.jsonl --dataset-root data/TexVerse --gpus 0
python -m gate3dgen.data.pair_generation.pipeline plan \
  --input data/assets.jsonl --dataset-root data/TexVerse --output-root outputs/pairs

# Run a two-pair generation smoke test before committing all GPUs.
python -m gate3dgen.data.pair_generation.pipeline run \
  --scope pilot --gpus 0 --limit 2 --output-root outputs/pairs

# Resume successful pairs and generate the full plan.
python -m gate3dgen.data.pair_generation.pipeline run \
  --scope full --gpus 0,1,2,3,4,5,6,7 --steps 40 --cfg-scale 4.0 \
  --output-root outputs/pairs
python -m gate3dgen.data.pair_generation.pipeline report --output-root outputs/pairs
```

Outputs include `plans/master.jsonl`, `plans/pilot.jsonl`, individual GT/degraded pairs and
`manifests/{train,validation,test,all_pass}.jsonl`. Each pair retains image hashes, source view,
task, seed, prompts and generation status. Here generation `pass` means the generation checks
passed; it does not replace semantic quality review. Generation failures are not training samples.

The private training set also contained curated animal-motion and human-support examples. The
public synthesis planner covers the original 12 core tasks; optional reviewed additional pairs can
be included in the input manifest using the same schema. They are not silently fabricated to match
the private training counts.

The low-level degradation backend's optional occlusion selector is not used by the main 12-task
recipe. If enabling it for a new experiment, supply your real prompt JSON (`{"prompt":"..."}`)
through `GATE3DGEN_OCCLUSION_PROMPT` and a local-service YAML through `--vlm-config`, with
`qwen_vlm: {base_url: http://127.0.0.1:8100/v1, model: YOUR_SERVED_MODEL}`. Optional
`api_key_env` names an environment variable; do not put credentials in repository files.
It no longer imports a private TabletopGen installation. Failed selections retain explicit fallback
metadata and are not historical VLM judgments or automatically accepted training pairs.

## 2. Audit and publish accepted pairs

The audit supports programmatic checks, optional DINOv2/SAM diagnostics and Qwen VLM review. Models
and services are external dependencies. For an OpenAI-compatible **local VLM server** serving
Qwen3-VL-32B-Instruct, the following explicitly disables the optional vision-diagnostic stage:

```bash
python -m gate3dgen.data.quality_audit.cli all \
  --scope full --manifest outputs/pairs/audit/audit_input.jsonl \
  --pair-root outputs/pairs --dataset-root data/TexVerse --output-root outputs/audit \
  --skip-vision --vlm-backend vllm --vlm-base-url http://127.0.0.1:8100/v1 \
  --vlm-served-model Qwen/Qwen3-VL-32B-Instruct --gpus 0
```

To include the original vision diagnostics, omit `--skip-vision` and provide `--dino-repo`,
`--dino-checkpoint`, `--sam-repo`, `--sam-checkpoint` and `--sam-config`. Install these backends in an
appropriate environment before running this stage. The audit records which stages were enabled.
Review uncertain cases and record explicit decisions; do not relabel missing or pending audits as pass.

For a small offline audit without a running VLM service, the repository also includes a local
Qwen3-VL runner. Point it to an authorized, complete local Qwen3-VL-32B-Instruct snapshot:

```bash
python -m gate3dgen.data.quality_audit.cli all \
  --scope full --manifest outputs/pairs/audit/audit_input.jsonl \
  --pair-root outputs/pairs --dataset-root data/TexVerse --output-root outputs/audit_local \
  --skip-vision --vlm-all-pairs --vlm-backend local \
  --qwen-model /path/to/Qwen3-VL-32B-Instruct --gpus 0
```

This uses the current Python environment and the packaged `quality_audit/qwen_runner.py`,
not a private external script. It loads BF16 weights on one CUDA GPU, performs deterministic
decoding, and accepts only embedded images; model loading is local-only. Allocate enough memory
for the 32B model and image tokens (an 80 GB GPU is recommended); do not run it concurrently
with a full FireRed worker on that GPU. The VLM observations, programmatic checks and final
quality decision remain separate records. `--skip-vision` explicitly excludes DINO/SAM diagnostics
and must not be described as verification of those optional stages.

```bash
python -m gate3dgen.data.prepare dataset \
  --manifest outputs/pairs/manifests/all_pass.jsonl \
  --audit outputs/audit/manifests/pair_quality_audit.jsonl \
  --output data/manifests
```

This joins by `pair_id`, rejects duplicate IDs and asset leakage, and exports only `quality_status=pass`.
`data/manifests/{train,validation,test}.jsonl` retains the original relative image paths, so the
purifier config uses `data_root: outputs/pairs`. For an existing reviewed dataset, provide your own
manifests with `pair_id`, `sha256`, `split`, `task_id`, `gt_relative_path`, `degraded_relative_path`,
`source_relative_path`, `source_image_sha256`, `gt_sha256` and `degraded_sha256`.

## 3. Extract and review clean-GT subject masks

The purifier's subject losses and the router's subtle-degradation weights use **clean-GT masks**.
They do not use an albedo-derived mask. Obtain an authorized external RMBG-2.0 model and set
`RMBG_MODEL` to its local directory. Its separate terms still apply.

```bash
python -m gate3dgen.training.extract_masks \
  --manifest data/manifests/train.jsonl --data-root outputs/pairs \
  --output-root outputs/masks --model "$RMBG_MODEL" \
  --num-shards 1 --shard-index 0 --device cuda:0
python -m gate3dgen.training.extract_masks \
  --manifest data/manifests/train.jsonl --data-root outputs/pairs \
  --output-root outputs/masks --model "$RMBG_MODEL" --num-shards 1 --merge-shards
```

For multiple GPUs, run every shard index `0..N-1` with the same `--num-shards N`, then merge only
after all shards complete. Inspect masks and remove or correct ambiguous cases. After that human
review, record the reviewed training manifest explicitly:

```bash
python -m gate3dgen.data.prepare mask-review \
  --cache outputs/masks --reviewed-manifest data/manifests/train.jsonl \
  --reviewer YOUR_NAME --confirm-reviewed
```

This command records your confirmation; it does not perform review or invent a review result.
If review removed pairs, retain the original extraction manifest and pass it as `--source-manifest`
to the mask-sidecar command below.

## 4. Build router supervision and sample weights

The original single-image supervision schema is retained: degraded positives, clean-GT negatives
and deduplicated albedo negatives. The trained token remains `REPAIR`, mapped to public `PURIFY`.

```bash
python -m gate3dgen.training.router_tools.manifest \
  --train-manifest data/manifests/train.jsonl \
  --validation-manifest data/manifests/validation.jsonl \
  --data-root outputs/pairs --texverse-root data/TexVerse \
  --texverse-full-root data/TexVerse-full --output-dir data/router
python -m gate3dgen.training.router_tools.weights masks
python -m gate3dgen.training.router_tools.weights weights
```

Use `--hand-manifest` only if you have the original-format, reviewed human-support sidecar. Without
it, no human-support labels are inferred. The weight builder retains the original family multipliers,
within-family bottom-25% mask-based subtle proxy, multiplier 1.25 and cap 2.5. Missing ordinary GT
masks or mismatched hashes are errors, not fallback weights.

The primary router validation population must be independent of training **and final evaluation**.
Create `data/router_demo_cases.jsonl` and `data/final_cases.jsonl` with stable asset IDs; each demo
case also needs `case_id`, `degraded`, `gt`, `task_id` and explicit canonical `reasons`, for example
`["OPTICAL_BLUR"]`. Image paths resolve from the repository root.

```bash
python -m gate3dgen.data.prepare router-validation \
  --cases data/router_demo_cases.jsonl --final-cases data/final_cases.jsonl
```

This produces `data/router/demo_validation.jsonl` and checks asset exclusion and training-image
hash overlap. The config keeps in-domain `data/router/validation.jsonl` as a secondary diagnostic.
Do not use final benchmark labels to select a threshold or checkpoint.

Continue with [router and purifier training](training.md), then [evaluation](evaluation.md).

## Aggregate provenance without releasing private assets

```bash
python -m gate3dgen.data.statistics \
  --manifest data/manifests/train.jsonl --output outputs/data_statistics/train.json
python -m gate3dgen.data.statistics \
  --manifest data/router/train.jsonl --output outputs/data_statistics/router_train.json
```

This publishes split/task/decision counts, asset and declared-image-hash counts, and sorted-set
fingerprints. It rejects cross-split asset/image-hash overlap when multiple splits are in one input.
For separate files, assign their actual experiment roles explicitly:

```bash
python -m gate3dgen.data.statistics \
  --split-manifest train=data/router/train.jsonl \
  --split-manifest development=data/router/demo_validation.jsonl \
  --split-manifest final=data/router_final.jsonl --output outputs/data_statistics/router_isolation.json
```

The output includes source-manifest SHA256, population counts/fingerprints and pairwise intersection
counts, but no private IDs or source paths. Roles override internal source `split` labels and are
recorded as explicit assignments. Empty populations, duplicate IDs and cross-population asset or
declared-hash overlap are errors. Use the real final **router evaluation** manifest here; the source
case manifest used for restoration has a different population.
This tool does not redistribute
asset IDs/images or recompute image hashes from file bytes. Original training aggregates are in
[docs/metadata](metadata). The purifier aggregate has 43,936 reviewed pairs and the router aggregate
104,528 single-image samples (44,588 REPAIR / 59,940 KEEP), both covering 3,961 assets.
These files contain training split statistics.
Set fingerprints cover declared image hashes, not a fresh image-byte scan. Private IDs and paths
are not distributed by this tool.

## Benchmark construction boundary

The original benchmark generation used shared source-camera/clean-GT anchors, authorized DTC
meshes and external HDRI, Flare7K++ and SIR2/reflection resources. Retain seeds, camera matrices,
lighting/environment identities, resource identifiers and degradation parameters per candidate.
Do not put those private resources or lists under `assets/`. Each evaluated case needs `case_id`,
`asset_id`, `task_id`, `degraded`, `gt` and, for geometry, `gt_mesh`; preserve rendering anchors for
the [shared-camera protocol](evaluation.md#shared-camera-pbr-evaluation).

The source evaluation lineage distinguishes 362 legacy cases plus 730 extension **candidates**
(1,092) from 362 plus 720 accepted extension cases (**1,082 canonical cases**). Preserve the ten
extension exclusions and reasons in your private selection manifest, freeze it before method
evaluation, and report population/failure counts. Never exclude a case because one model scored
poorly. Training-pair synthesis and benchmark rendering are separate workflows.

## Evaluation manifest adapters

Use `gate3dgen.evaluation.manifest` for **training-format pair JSONL**: it resolves GT/degraded/albedo
relative paths under declared roots and selects cases by asset/task. Use
`gate3dgen.evaluation.benchmark_manifest` for **rendered benchmark JSON/JSONL**: it joins
`asset_index` to source-camera anchors and the GLB catalog, validates mesh mapping and preserves
benchmark metadata. These are different input adapters, not alternative metric protocols. Both
feed the common case schema in [evaluation](evaluation.md#purification-and-2d-metrics); use
`--help` to choose the adapter matching your source, not to silently convert a different population.
