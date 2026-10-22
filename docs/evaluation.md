# Evaluation

## Quick start

```bash
bash scripts/evaluate.sh
```

By default this scores the retained Pixal3D `degraded.glb` and `restored_s01.glb` under
`assets/examples/eval/generation/`, along with their original images and GT mesh.
It computes 2D metrics, full-budget visible-shell geometry and four-view texture metrics.
Outputs go to `outputs/eval/`; inputs and historical meshes are never overwritten.
This reuses historical reconstructions, not a fresh inference run. The 2D entry reports
full-image diagnostics; foreground paper metrics require the masks described below.
The geometry/PBR validators remain strict. Pretrained AlexNet LPIPS must be available.

For a different population set `EVAL_MANIFEST`, `EVAL_GENERATION_DIR`, `EVAL_CAMERA_PLAN`
and `EVAL_OUTPUT_ROOT`. All images, meshes and cameras must refer to the same cases.

### Generate new before/after GLBs

Download Pixal3D through [setup](../setup/README.md) and install its official backend in
a compatible environment. Install this checkout there with `python -m pip install -e . --no-deps`,
then set `PIXAL3D_REPO` and `PIXAL3D_PYTHON` to that installation.
Auxiliary DINOv3, MoGe and RMBG weights remain upstream dependencies;
set `GATE3DGEN_RMBG_MODEL` to an authorized local RMBG-2.0 checkpoint.
`PIXAL3D_MODEL` defaults to the in-repository `checkpoints/upstream/Pixal3D` download;
set it to an external model directory for maintainer tests.

```bash
# Regenerate both branches from the shipped before/after images.
bash scripts/generate_glb.sh

# Or use the manifest written by a fresh Gate3DGen inference run.
bash scripts/generate_glb.sh --manifest outputs/eval/routed/manifest.jsonl --output outputs/eval/fresh_generation
```

The script writes `conditions.jsonl` and
`pixal3d/<case_id>/{degraded.glb,restored_s01.glb}` to a new output directory.
It preserves matched case IDs, uses the same shared-reference camera policy as the existing
generator, and does not generate or substitute a clean-image oracle for mesh GT.
`--dry-run` prints the generation call without creating output files.
Point `EVAL_GENERATION_DIR` to that output root and `EVAL_MANIFEST` to the corresponding
image manifest before scoring. Never pair new images with historical GLBs.

Run from the repository root. Use the self-contained router-6000 / purifier-15000 package and its
frozen `checkpoints/runtime/`, or pass explicit checkpoint/runtime overrides. Original FireRed
diffusion/Qwen weights are not needed for Gate3DGen inference. Keep training, checkpoint/threshold-selection
validation, and final evaluation populations separate at the **asset** level.

## Input and output directories

[assets/examples/eval](../assets/examples/eval/README.md) contains one real DTC airplane case with motion blur
and uneven lighting, no SIR² layer, three 2D images, the original DTC GT mesh, historical Pixal3D
before/after meshes and a four-view evaluation camera plan. Its separate DTC share-alike/no-sale
terms are documented in [the example license](../assets/examples/eval/LICENSE.md); it is not Apache-2.0.
The Pixal3D clean-oracle mesh from the PPT is not GT and is excluded. Included predictions are
checkpoint-15000 outputs. Keep the full private benchmark and all other images, meshes,
camera plans and their symlinks
in an external work directory. Point the CLI to its manifest/generation/render paths explicitly.
If reusing historical generated GLBs, record that reuse rather than calling them new reconstructions;
see [local-input staging](../assets/examples/README.md#maintainer-only-real-inputs).

| Location | Required content / role |
|---|---|
| `data/final_cases.jsonl` | Stable case/asset IDs, degraded/GT image paths and optional `gt_mesh` |
| `data/final/<case>/` | Authorized evaluation inputs and references |
| `data/router_final.jsonl` | Held-out routing images and labels |
| `outputs/eval/{routed,purifier}/` | Newly inferred images and recorded inference manifest |
| `outputs/eval/generation/<backend>/<case>/` | Generated `restored_s01.glb` and other requested variants |
| Camera-plan JSON supplied via `--camera-plan` | Four per-case camera matrices and source anchor |
| `outputs/eval/geometry/` | Per-case geometry JSONL and coverage summary |
| `outputs/eval/renders/` | GT/predicted PBR channels and recorded camera contracts |
| `outputs/eval/{2d,pbr}/` | Metric records and per-metric coverage summaries |

All paths in public fixtures are repository-relative. Models, upstream 3D backend installations,
new runtime outputs and private benchmark populations are not bundled into the source release.
The explicitly attributed historical outputs for the single public example are the exception.

## Paper table protocols

| Table | Required flow / reported quantities |
|---|---|
| 1: router | Fixed development-calibrated routing → Recall, Accuracy, H; H is not F1. |
| 2: purifier | Forced purification/native baselines → foreground AlexNet LPIPS, full-image PSNR/SSIM/Grad, independent alpha IoU. |
| 3: TRELLIS.2 | Frozen TRELLIS.2 generation → geometry evaluation → four-view normal/shaded/albedo rendering metrics. |
| 4: wild | FD-DINOv2 distributions, recorded VLM judgments and blinded human preferences; no clean reference GT. |
| 5: purifier ablations | A/B/Full training → identical held-out purification/downstream protocol; see [training](training.md#controlled-paper-ablations). |
| 6: router baselines | Language-only/vision-LoRA/MLP → the same fixed-threshold router protocol, with actual budgets documented. |
| 7: Pixal3D | Frozen Pixal3D generation → geometry evaluation → four-view normal/shaded/albedo rendering metrics. |

**Tables 3 and 7 both require geometry and rendering.** Table 7 is not a materials-only/PBR table:
its columns are MD, F1, CD, normal PSNR, shaded LPIPS and albedo LPIPS. Metallic, roughness and alpha
channels are extra diagnostics, not extra paper columns. Keep generator-specific runs and four-view
camera plans separate. Do not pool generators, routed/force-purify protocols, view counts, reduced
sampling runs or different references. Keep native metric units; declare any presentation scaling.
New geometry runs now compute all-surface MD/F1 and genuine depth-visible outer-surface CD/F1
for Tables 3, 5 and 7.

## Router decisions

Provide a labeled JSONL manifest with `image` and `label` (0 = KEEP, 1 = PURIFY), or use the
expanded router-supervision schema produced by the data tools. Image paths resolve from
`--data-root` (the working directory by default). Identical image bytes are deduplicated; conflicting
labels for identical bytes are errors.

```bash
python -m gate3dgen.evaluation.router \
  --manifest data/router_final.jsonl --output-dir outputs/eval/router
```

Outputs include predictions, the confusion matrix, accuracy, precision, recall, specificity, F1,
and **H = 2 × Recall × Accuracy / (Recall + Accuracy)**, used in Table 1. H is not standard F1.
`paper_percent` reports Recall, Accuracy and H on the paper's percentage scale. Additional outputs are
AUROC, AUPRC and calibration error. Undefined ranking metrics for one-class populations are null.
The released threshold is fixed at 0.006440580124035478. This evaluator does not fit a threshold;
never select it on final benchmark labels.

To calibrate a new router:

```bash
python -m gate3dgen.evaluation.router \
  --manifest data/router_development.jsonl --output-dir outputs/eval/router_development
python -m gate3dgen.evaluation.router_metrics \
  --development-predictions outputs/eval/router_development/predictions.jsonl \
  --final-manifest data/router_final.jsonl --output outputs/eval/new_threshold.json
```

Both populations must declare stable asset IDs and be disjoint. Candidates are 0, 1 and midpoints
of consecutive unique development probabilities. Maximize balanced accuracy, break ties by greater
recall, then lower threshold. Freeze the resulting value before final scoring; pass it explicitly
with `--threshold`. Never overwrite the released calibration with a new experiment's selection.

## Purification and 2D metrics

Prepare `data/final_cases.jsonl`, one case per line:

```json
{"case_id":"example_001","asset_id":"owned_asset_001","task_id":"optical_blur","degraded":"data/final/example_001/input.png","gt":"data/final/example_001/clean.png"}
```

The `gt` field is optional for qualitative in-the-wild cases, but required for full-reference metrics.
Use stable unique case IDs and repository-root-relative or absolute paths. Ground truth is never
passed into the purifier. Retain benchmark metadata such as `gt_mesh` when evaluating 3D geometry.

```bash
# Evaluate the complete selective system (KEEP is copied byte-for-byte).
python -m gate3dgen.evaluation.infer \
  --manifest data/final_cases.jsonl --output-dir outputs/eval/routed \
  --protocol routed --cpu-offload

# Evaluate purification quality independently of routing.
python -m gate3dgen.evaluation.infer \
  --manifest data/final_cases.jsonl --output-dir outputs/eval/purifier \
  --protocol force-purify --cpu-offload

python -m gate3dgen.evaluation.metrics2d \
  --manifest outputs/eval/purifier/manifest.jsonl \
  --output-dir outputs/eval/2d --lpips --device cuda \
  --foreground-model "$RMBG_MODEL" --paper-protocol
```

Use a new empty inference output directory for each run. The manifest records the router decision,
whether purification was invoked, protocol and seed (default 1234). Do not pool routed and
force-purify results as if they were the same experiment. Metrics include PSNR, SSIM and image/edge
errors, with optional LPIPS and DINO. Prediction and GT dimensions must match; this evaluator rejects
shape mismatches rather than silently resizing. Define and record any benchmark preprocessing before
scoring. Use the full environment with scikit-image for quantitative SSIM; the dependency-light
fallback is only for smoke tests. Do not report full-reference scores for cases without clean GT.

The Table 2 contract is explicit: LPIPS = `object_lpips_alex`, computed on a tight **GT** foreground
bbox with all pixels outside GT support black; PSNR = `psnr_db`, SSIM = `ssim`, Grad = `gradient_mae`
remain full-image; IoU = `foreground_mask_iou`, comparing independently extracted prediction/GT
RMBG-2.0 alpha masks thresholded at 0.8. Full-image `lpips_alex` is an additional diagnostic, not
the paper's foreground LPIPS column. Save the generated masks and recorded model identity.
For previously extracted masks each manifest row can declare:

```json
{"foreground_masks":{"gt":"data/masks/clean.png","degraded":"data/masks/input.png","restored":"data/masks/output.png"}}
```

Use `--foreground-mask-model <immutable-RMBG-revision>` instead of loading RMBG again. Cached masks
must come from that model; this argument records provenance, not a verification of their origin.
`--paper-protocol` fails without foreground masks/model, LPIPS or quantitative SSIM support.

## Frozen image-to-3D generation

Create the shared condition manifest:

```bash
python -m gate3dgen.evaluation.official_manifest \
  --input-manifest outputs/eval/purifier/manifest.jsonl \
  --output outputs/eval/conditions.jsonl --path-root "$PWD" \
  --steps 1 --dataset-kind benchmark_with_glb_gt --validate
```

For cases without mesh GT use `--dataset-kind wild_no_gt`; these cannot produce reference geometry
scores. Install the upstream Pixal3D and/or TRELLIS.2 repositories and their authorized model weights
in their own compatible environments. These large, mutually specialized stacks are not bundled in
the Gate3DGen Conda environment. Keep this package importable in the selected environment and run
from this repository root. The wrappers below expose those backends without retraining them:

```bash
python -m gate3dgen.evaluation.generate_pixal3d \
  --manifest outputs/eval/conditions.jsonl --output-dir outputs/eval/generation \
  --repo "$PIXAL3D_REPO" --model "$PIXAL3D_MODEL" \
  --rembg-model "$RMBG_MODEL" --dino-model "$DINO_MODEL" --no-renders

python -m gate3dgen.evaluation.generate_trellis2 \
  --manifest outputs/eval/conditions.jsonl --output-dir outputs/eval/generation \
  --repo "$TRELLIS2_REPO" --model "$TRELLIS2_MODEL" \
  --rembg-model "$RMBG_MODEL" --no-renders
```

Record each backend's camera and segmentation policy. Pixal3D defaults to a GT-shared camera for
paired evaluation and independent segmentation; select an appropriate supported policy for wild
inputs. Do not equate different camera policies. Each backend writes an artifact manifest, consumed
by geometry evaluation:

```bash
python -m gate3dgen.evaluation.metrics3d_official \
  --manifest outputs/eval/conditions.jsonl --generation-dir outputs/eval/generation \
  --output-dir outputs/eval/3d --resume
```

The default schema is `geometry_visible_shell_v3`. It computes the two distinct
geometry populations in the paper, rather than resampling the raw mesh twice:

| Paper metric | Output | Computation |
|---|---|---|
| MD | `mesh_distance` | Sample 1M points on each full mesh, including hidden internal surfaces. Average the two directional **squared point-to-triangle** distances, using a complete R-tree triangle search. |
| F1all | `mesh_distance_fscore` | Precision/recall from those full-surface distances; squared-distance threshold 1e-8. |
| CD | `outer_shell_chamfer_distance` | Obtain first-hit depth maps from 100 cameras per aligned mesh, unproject valid pixels, randomly sample 1M points without replacement per mesh, then average the two directional **squared nearest-point** distances. |
| F1outer | `outer_shell_fscore` | Precision/recall on those shell point clouds; squared-distance threshold 1e-6. |

CD is `0.5 * (mean(d_pred_to_gt**2) + mean(d_gt_to_pred**2))`. F1 uses
`2*precision*recall/(precision+recall)`, or zero if both are zero. Returned distances
are **unscaled**. Gate3DGen Tables 3/5/7 display MD, CD and F1all in units of 1e-3
(multiply raw values by 1e3), and F1outer in units of 1e-1 (multiply by 10).
Do not reuse TRELLIS.2's different table-display scaling. The explicit
`*_l1` diagnostics use unsquared distances, not the paper MD/CD. The `*_l2` fields
are squared-distance aliases.

The visible-shell pipeline follows the 100-depth-view / 1M-point definition in
[TRELLIS.2 Appendix D.1](https://arxiv.org/html/2512.14692v1#A4.SS1).
The implementation pins details not specified there: Fibonacci-sphere directions;
512×512 orthographic pixel-center rays; one bounding sphere around the **aligned pair**;
camera distance 3 times that radius; projection half-extent 1.05 times the radius.
The pair shares center, projection and views. Embree finds the first positive triangle
intersection for each ray; this is the depth-map unprojection, not visible-face area
resampling. Both face sides are opaque for geometry, ignoring texture alpha/transmission.
Completely enclosed surfaces are excluded, but visible concavities and openings remain.
This is a finite-view **visible** shell, not a certified topological exterior; an interior
wall visible through an opening is intentionally included.

Uniform random-priority streaming samples valid depth pixels without replacement, with
seeds `seed+2` / `seed+3`. It bounds memory without storing all 100 dense clouds.
Insufficient valid depth pixels cause failure, never duplicated fill points: increase
`--shell-resolution` for exploratory runs. Strict paper mode pins 512; such a case
cannot enter its summary under that protocol. Independent sampled clouds mean even
identical meshes can have nonzero finite-sampling CD; MD remains zero.

Install the geometry backend (already included in the one-click conda environment):

```bash
python -m pip install -e '.[eval]'
python -m gate3dgen.evaluation.metrics3d_official \
  --manifest outputs/eval/conditions.jsonl --generation-dir outputs/eval/generation \
  --output-dir outputs/eval/visible_shell --paper-protocol --resume
```

`--paper-protocol` requires 1M samples, 100 views, resolution 512, genuine shell,
mesh GT and complete finite MD/F1/CD/F1 for every planned case/baseline/variant.
It rejects proxies, scaled demos, `--allow-missing`, duplicate identities and a
pre-existing summary; use a new output directory. Failed runs may retain per-case
checkpoints for diagnosis/resume but do not publish a formal summary.
For smoke tests use `--scaled-demo`; view/resolution overrides are exploratory only.
The protocol stores shared cameras, backend/version, dense/selected point counts,
thresholds, seeds and distance kernels. Configuration hashes prevent reusing old
proxy or different-camera shards on `--resume`.

Keep sample count, alignment, reference and failure counts with scores. For routed
comparisons use a separate condition manifest from `routed/manifest.jsonl`.

For legacy raw-surface diagnostic settings, use `--historical-report-protocol`.
This mode records its field mapping separately from the visible-shell metrics.

## Shared-camera PBR evaluation

The default `--frame-mode fit` fits the cameras to the **reference mesh only** and
reuses the resulting actual matrices for GT and all predictions. It does not refit each
prediction independently. `camera.json` records those matrices, `frame_policy` and the actual
Blender version; changing the frame policy/matrices invalidates camera matching. Previous
renderer-v2 metadata stored planned cameras while independently fitting each mesh, so those
cached renders must be regenerated for this shared-camera protocol. `--frame-mode plan`
uses the supplied matrices verbatim; both sides must use the same choice.
Each rendered asset/variant gets its own `camera.json` and source-mesh SHA256. Cached
images are reused only when that full render contract still matches; otherwise rendering fails
with an instruction to use `--overwrite` or a fresh render root. New metadata must never
recertify images made with an old camera or mesh.

PBR evaluation requires GT meshes, Blender, and the benchmark's source-camera anchors. Prepare a
camera-anchor JSON with a `cases` list; each case records `case_id` and
`source_anchor.expected_camera_matrix_world` (a 4×4 matrix). These are your rendering metadata, not estimated
from benchmark labels. Freeze one plan and use it for GT and all predictions:

```bash
python -m gate3dgen.evaluation.camera_plan \
  --manifest outputs/eval/conditions.jsonl --camera-plan-v1 data/camera_anchors.json \
  --workspace "$PWD" --output outputs/eval/camera_plan.json --views 4
python -m gate3dgen.evaluation.render_pbr \
  --metrics outputs/eval/3d/per_case.jsonl --render-root outputs/eval/pbr_renders \
  --camera-plan outputs/eval/camera_plan.json --mode gt
python -m gate3dgen.evaluation.render_pbr \
  --metrics outputs/eval/3d/per_case.jsonl --render-root outputs/eval/pbr_renders \
  --camera-plan outputs/eval/camera_plan.json --mode prediction
python -m gate3dgen.evaluation.pbr_metrics \
  --manifest outputs/eval/conditions.jsonl --render-root outputs/eval/pbr_renders \
  --output-dir outputs/eval/pbr_paper --views 4 --lpips --paper-protocol
```

Set `GATE3DGEN_BLENDER` if Blender is not on PATH. `render_pbr --dry-run` prints its Blender command.
Without an explicit override, the wrapper prefers `blender` beside the current Python executable
before searching PATH, so invoking an absolute Conda Python does not accidentally select a broken
system Blender from an unactivated shell. `--threads N` bounds CPU-render workers when sharding cases.
Blender uses its **own Python**, which can differ from the conda Python. Do not add conda
site-packages to Blender's `PYTHONPATH`: binary NumPy modules must match Blender's Python ABI.
If Blender is installed when `setup/setup.sh` runs, the setup checks it and installs ABI-matched
NumPy into a separate target under the conda prefix when necessary. For an existing environment:

```bash
python -m gate3dgen.evaluation.blender_runtime --install
python setup/check.py --environment-only --blender "${GATE3DGEN_BLENDER:-blender}"
```

The self-check covers dependency consistency, installed command entry points, training-module
imports, WebP encoding/decoding, a small exact-MD/visible-shell fixture, LPIPS forward execution,
and a small CUDA operation when a GPU is available. The offline LPIPS dependency check uses
a random backbone. To run the regression suite
including an actual WebP GLB → Blender → seven-channel render integration test:

```bash
GATE3DGEN_TEST_BLENDER="${GATE3DGEN_BLENDER:-blender}" python -m pytest
```

To validate a generated data plan instead of the default `outputs/pairs` plan,
set `GATE3DGEN_TEST_PAIR_ROOT` to that pair-output
directory when running pytest. The test checks the actual plan summary, per-asset task/view
balance and split isolation, not hard-coded private corpus counts.

This does not install Blender itself, its glTF add-on or Cycles resources. The automatic NumPy
bootstrap supports Linux x86_64/aarch64 and downloads a binary wheel for Blender's Python, not
the host interpreter. A custom target can be passed through `--runtime-path` during setup and
`--blender-python-path` during rendering, or `GATE3DGEN_BLENDER_PYTHONPATH`.

Blender must be built with **OpenColorIO enabled** and have its complete **OCIO configuration and LUTs**. A trimmed system
package can start successfully while printing `Color management: using fallback mode` and
writing nearly black images. `bpy.app.build_options.opencolorio` must be true; adding LUTs
cannot fix a binary compiled without OCIO. The runtime check now rejects this state. Use a complete Blender
distribution, or set `OCIO=/path/to/matching/colormanagement/config.ocio` with all its referenced
resources intact. An installed Blender-3.0.1 configuration under
`$CONDA_PREFIX/share/gate3dgen/blender-color-3.0.1/config.ocio` is also reused when `OCIO` is unset.
Keep the OCIO version/config hash with the run; rerender every GT/prediction channel after fixing
color management (`--overwrite` or a new render root), rather than reusing invalid images.
The WebP integration test checks foreground pixels, not just file existence, and paired metrics
reject empty or near-black silhouette masks instead of reporting artificial perfect scores.

`EXT_texture_webp` GLBs are converted to PNG-textured compatibility copies automatically for
both prediction and GT. The packaged `glb_compat.py` runs in the **host conda interpreter**, where
Pillow/WebP is checked; Blender does not need Pillow. Geometry buffers, texture pixels including
alpha, and canonical source files are preserved. Copies are cached under
`render-root/.compatibility_cache` (override with `--compatibility-cache-root`). Blender is launched
with `--python-exit-code 1`, so import/conversion failures propagate as failed render commands.

The renderer exports shaded, normal, mask, base-color, metallic, roughness and alpha channels.
Camera provenance is validated by default; do not disable it for reported comparisons. An optional
12-view plan is available through `camera_plan --views 12`; pass the same view count to both rendering
and metric commands and report it as a distinct protocol.

`--paper-protocol` requires a fresh output directory, exactly four views, verified shared cameras,
and real GLB GT. Its default channels are `normal`, `shaded` and `base_color` (albedo). It checks
normal PSNR and shaded/albedo LPIPS values, rejects missing/nonfinite learned metrics, duplicate
case/view identities and incomplete views, and requires the **same case/view set across every
generator, variant and required channel**. Failed cases are never intersected away. An LPIPS
dependency or computation failure marks the record `partial`, not `ok`. A failed paper run exits
nonzero and writes `per_case.jsonl` and `validation_failure.json` for diagnosis, **no `summary.json`**.
Run geometry metrics separately from rendering metrics.

Without `--paper-protocol`, partial diagnostic summaries are allowed. Every metric has
`metric_coverage.<name>.{valid_count,expected_count,coverage}`; expected counts include failed/missing
records. `metrics.<name>` is a mean over valid observations only, or null when none are valid.
The group-level `coverage` is explicitly labeled as complete-record status coverage, **not** a
guarantee that every metric is complete. Compare methods only with complete metric coverage on the
same population. Perfect PSNR is represented by the JSON string `"Infinity"` and accompanied by a
`positive_infinity_count`, rather than silently converted to null.

## Native purification baselines

`baseline` replaces the historical multi-purpose baseline launcher. It uses the same packaged
purification instruction for all methods, native samplers, 1024×1024 inputs and original per-case
seed policy: `(base_seed + SHA256(case_id)[:4 bytes]) mod 2**31`. Each run records the prompt hash,
model path, sampler settings and output manifest. Use a separate compatible environment for each
upstream backend; installing all of them into the Gate3DGen environment is not supported.

| `--backend` | Native sampler |
|---|---|
| `firered_image_edit_1_1` / `qwen_image_edit_2511` | 40 steps, true CFG 4, guidance 1, negative prompt one space |
| `mage_flow_edit` | 30 steps, CFG 5, VL encoder long edge 384 |
| `step1x_edit` | v1p2, 50 steps, true CFG 6, thinking/reflection disabled |
| `omnigen2` | 50 steps, text CFG 5, image CFG 2, CFG interval [0,1] |
| `flux_kontext` | Kontext-dev, 30 steps, guidance 2.5 |
| `gpt_image_2_5_flare` | Import original provider outputs; no reconstructed/API-inferred sampler |

```bash
python -m gate3dgen.evaluation.baseline \
  --manifest data/final_cases.jsonl --backend firered_image_edit_1_1 \
  --model "$GATE3DGEN_MODEL_PATH" --output-dir outputs/baselines/firered --dry-run
# Remove --dry-run to run. For external source implementations additionally pass --code-root.
```

The closed-source baseline needs `baseline_restored` on every manifest row. Keep its original
request/response records and provider revision privately; imported outputs alone do not reconstruct
the historical service. Evaluate every baseline manifest through the same 2D/downstream protocols.

## Wild FD-DINOv2 (Table 4)

Declare a wild manifest with one original photo and four shaded renders per generator/method:

```json
{"case_id":"wild_001","original":"data/wild/input.png","renders":{"pixal3d":{"ours":["data/pixal/0.png","data/pixal/1.png","data/pixal/2.png","data/pixal/3.png"]},"trellis2":{"ours":["data/trellis/0.png","data/trellis/1.png","data/trellis/2.png","data/trellis/3.png"]}}}
```

Add all comparison methods to both dictionaries. All cases and generators must contain the same
methods; each requires exactly four distinct views. No failed/missing case is silently dropped.

```bash
python -m gate3dgen.evaluation.wild \
  --manifest data/wild_fd.jsonl --dino-hub-dir "$DINO_HUB_REPO" \
  --dino-weights "$DINO_VITL14_REG_WEIGHTS" --output outputs/eval/wild_fd.json
```

The paper population is 96 photographs versus 384 **individual** rendered-view samples per method
and generator. The backbone is DINOv2 ViT-L/14 **with registers**, L2-normalized CLS, 224×224 ImageNet
preprocessing. This is Fréchet distance between feature distributions, not paired cosine similarity
or a four-view averaged feature. The weight SHA256 is recorded; a different backbone is not an
interchangeable metric. Smaller experiments require explicit `--expected-cases` and are marked
as not using the paper population. Original photographs are distribution references, not clean GT.

## VLM scores and blinded human preferences

For each case/method, supply one 2D score row and eight 3D score rows, four from each generator:

```json
{"case_id":"wild_001","method":"ours","domain":"2d","identity":8,"purification":7}
{"case_id":"wild_001","method":"ours","domain":"3d","generator":"pixal3d","view":0,"identity":8,"purification":7,"quality":8}
```

Scores must be finite in [0,10]. Identity and Purification average the 2D score and the mean of
eight 3D views with **equal domain weight**; Quality averages only the eight 3D views. All methods
must cover the same population. Supply the prompt actually used, not an asset-captioning prompt:

```bash
python -m gate3dgen.evaluation.preferences vlm \
  --scores data/judge_scores.jsonl --judge-model YOUR_ACTUAL_MODEL \
  --judge-revision IMMUTABLE_REVISION --prompt data/actual_judge_prompt.txt \
  --output outputs/eval/vlm.json
```

Human assignment input uses `{"case_id":"wild_001","methods":{"ours":"opaque/result1.png",
"baseline_name":"opaque/result2.png"}}`. Serve media at opaque filenames so paths cannot reveal
method identities. Preparation deterministically randomizes A/B positions and presentation order;
keep `private_assignments.jsonl` hidden from raters.

```bash
python -m gate3dgen.evaluation.preferences prepare \
  --manifest data/human_cases.jsonl --output-dir outputs/human/assignments
python -m gate3dgen.evaluation.preferences human \
  --assignments outputs/human/assignments/private_assignments.jsonl \
  --ratings data/human_ratings.jsonl --output outputs/eval/human.json
```

Ratings require `participant_id`, `assignment_id`, explicit `valid:true`, and `identity`,
`purification`, `quality` each equal to `A`, `B` or `tie`. Mark excluded responses `valid:false`
with `exclusion_reason`. Duplicate participant/assignment ratings are rejected. Reports preserve
ours/baseline/tie counts, valid denominators and exclusions. Ties stay in the denominator and are
not ours wins; do not compare different tie policies without disclosing them.

## Runtime profiling

```bash
python -m gate3dgen.evaluation.infer \
  --manifest data/profile_cases.jsonl --output-dir outputs/eval/profile \
  --protocol routed --cpu-offload --profile
```

`profile.json` separates model loading from router/purifier per-case timings and records GPU name,
peak allocated/reserved VRAM, process peak RSS, Python/package versions, per-case input/output
resolution and offload setting. Timers synchronize CUDA; context initialization precedes load timing.
Purifier timing includes saving its output. Run both KEEP and PURIFY inputs and report warm/cold
conditions, resolution, software, hardware and sample count. Instrumentation is not itself a
measurement; do not quote fixture-test timings as model latency.

## Reports

```bash
python -m gate3dgen.evaluation.report \
  --metrics2d-dir outputs/eval/2d --metrics3d-dir outputs/eval/3d \
  --output-dir outputs/eval/report
```

Keep the per-case JSONL, summary files, camera plan, manifests and run metadata alongside the report.
PBR metrics have their own summaries under `outputs/eval/pbr`. Report actual evaluated counts,
failures, seeds, checkpoints and protocols; private benchmark images need not be redistributed.
