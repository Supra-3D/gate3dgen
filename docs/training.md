# Training

Run from the repository root in the full Conda environment. Complete [data preparation](data.md)
first and set `GATE3DGEN_MODEL_PATH` to the external FireRed-Image-Edit-1.1 directory.
Training uses your authorized dataset; the full private training corpus is not released.

## Input directories

See [assets/examples/train](../assets/examples/train/README.md) for actual small pair files and split manifests,
plus the path substitutions for an audited example run. Its included pairs are explicitly
four historical FireRed pairs from one CC BY 4.0 museum asset. Complete historical VLM audit
evidence is not bundled, so their public-workflow status is `not_audited`; train is populated
and validation/test are empty. Follow [data preparation](data.md)
before training; do not bypass masks, audits or sample-weight provenance.
Keep private manifests, images, subject masks, supervision/sample weights and their symlinks
outside the source checkout. Set their absolute paths in a local YAML; retain actual quality and
mask-review provenance when reusing an existing dataset. Small one-asset fixtures are not
train/holdout experiments. Private inputs and conditioning caches are not part of source archives.

| Location | Role |
|---|---|
| `data/manifests/{train,validation,test}.jsonl` | Reviewed purifier populations |
| `outputs/pairs/` | Images resolved by purifier `data.data_root` |
| `outputs/masks/` | Clean-GT subject masks and completion/review sidecars |
| `data/router/` | Router supervision, two validation sets and `weights/sample_weights.jsonl` |
| External `GATE3DGEN_MODEL_PATH` | Complete authorized FireRed base model |
| `checkpoints/router/checkpoint-6000/` | Released frozen conditioning router |
| `outputs/cache/purifier/` | Prepared data and router embeddings |
| `outputs/training/{router,purifier}/` | New training logs and complete resumable checkpoints |

Keep input fixtures in `assets/examples/` and runtime artifacts in ignored `outputs/`. Released inference
weights and resumable optimizer/RNG/dataloader training checkpoints are different artifacts.

To keep all local training artifacts outside the source checkout, explicitly authorize a dedicated
output root and set the output/cache paths accordingly:

```bash
export GATE3DGEN_ALLOWED_OUTPUT_ROOT="$PWD/../Gate3DGen-local-run/outputs"
export GATE3DGEN_OUTPUT_ROOT="$GATE3DGEN_ALLOWED_OUTPUT_ROOT/purifier"
export GATE3DGEN_TRAIN_CACHE_ROOT="$PWD/../Gate3DGen-local-run/cache/purifier"
```

Manifests, image roots and masks can also use absolute external paths in your local YAML.
Without `GATE3DGEN_ALLOWED_OUTPUT_ROOT`, the default protection still requires training outputs
under this repository's `outputs/`. Do not use the filesystem or repository root as the allowed root.

## Router

`configs/train/router.yaml` preserves the released checkpoint-6000 recipe: 8 NVIDIA A800 GPUs on one node,
BF16 FSDP, batch size 1 per rank, gradient accumulation 4 (global batch 32), 6,000 optimizer steps,
learning rate 1e-6, and 300 warmup steps. Language blocks 24–27 and the configured normalization
parameters are trainable. The last four vision blocks use rank-16 LoRA (alpha 32, dropout 0.05),
with a 10x vision learning-rate multiplier. The original weighted token loss and sample weights
are retained. The native trained decision token is `REPAIR`; the public API calls this `PURIFY`. Native KEEP is uppercase `KEEP`; appendix-style `Purify`/`Keep` must not replace trained prompt tokens.

Required inputs are `data/router/train.jsonl`, `demo_validation.jsonl`, `validation.jsonl`, and
`weights/sample_weights.jsonl`. Primary validation must exclude training and final-evaluation assets;
in-domain validation is a secondary diagnostic. Expected population sizes are not hard-coded to
the private corpus.

```bash
# Print the exact distributed command without starting training.
python -m gate3dgen.training.router_tools.launch --dry-run
python -m gate3dgen.training.router_tools.launch

# Resume a complete training checkpoint, including optimizer state.
python -m gate3dgen.training.router_tools.launch \
  --resume outputs/training/router/checkpoint-3000
```

Export the **complete** FSDP state and merge vision LoRA before using the router for inference or
purifier conditioning. The exporter rejects incomplete/rank-local states and adapter-only files.
The base argument below is the Qwen text-encoder component, not the entire diffusion pipeline:

```bash
accelerate merge-weights \
  outputs/training/router/checkpoint-6000/pytorch_model_fsdp_0 \
  outputs/training/router/checkpoint-6000/merged
python -m gate3dgen.training.router_tools.export \
  --base-model "$GATE3DGEN_MODEL_PATH/text_encoder" \
  --checkpoint outputs/training/router/checkpoint-6000 \
  --merged-fsdp-model outputs/training/router/checkpoint-6000/merged \
  --output outputs/training/router/checkpoint-6000/exported_model
```

Use the actual FSDP state directory emitted by your installed Accelerate version if its name differs.
Keep `checkpoint_complete.json`, `trainer_state.json` and `exported_model/v3_lora_export.json`:
they establish the checkpoint step, source world size and complete LoRA merge.

## Purifier

`configs/train/purifier.yaml` is the released frozen-router-conditioned checkpoint-15000 recipe,
not a baseline unconditional restoration trainer. It freezes the trained router, caches its hidden
states, and trains diffusion blocks 29–49. The released purifier was trained on 32 NVIDIA A800 GPUs
across four nodes (8 GPUs per node), with global batch 32,
15,000 steps, learning rate 5e-7, cosine scheduling and 500 warmup steps. The one-step x0 objective
uses sigma 1, latent weight 0.1, **VGG LPIPS** weight 0.5, and full-image, subject, affected-region and
subject-gradient weights of 1. Clean-GT RMBG-2.0 masks are required for training.

By default the configuration uses the packaged router checkpoint-6000. To train with your newly
trained router, set **both** paths before all preparation, caching and training commands:

```bash
export GATE3DGEN_ROUTER_CHECKPOINT="$PWD/outputs/training/router/checkpoint-6000"
export GATE3DGEN_ROUTER_EXPORT="$GATE3DGEN_ROUTER_CHECKPOINT/exported_model"
python -m gate3dgen.training.cli condition-preflight
python -m gate3dgen.training.cli check --split train
python -m gate3dgen.training.cli prepare-data
python -m gate3dgen.training.cli extract-embeddings
```

Preflight verifies the complete router export, step 6000, eight-rank provenance and model hash.
Embedding extraction runs once on an eight-GPU node. Share the resulting cache, manifests, masks,
base model and router export at identical paths on every training node. Changing the router requires
rebuilding its embedding cache. LPIPS also needs its external pretrained feature weights available
in the Torch cache; do not substitute random features when offline.

Warm the **same** Torch cache used by the workers before a multi-rank launch, so ranks do not
download VGG16 concurrently. `TORCH_HOME`, if set, must have the same value in this preparation
shell and all training workers. This downloads the official pretrained backbone when absent;
it does not change the objective or use random features:

```bash
python -c 'from torchvision.models import VGG16_Weights; VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
```

Evaluation uses pretrained AlexNet instead. Prepare that cache before an offline benchmark:

```bash
python -c 'from torchvision.models import AlexNet_Weights; AlexNet_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
```

On **each** of the four nodes, set `NODE_RANK` to a distinct value from 0 through 3, and set
`MAIN_PROCESS_IP` to the reachable address of node 0. Launch the same command on all nodes:

```bash
python -m gate3dgen.training.cli train \
  --num-processes 32 --num-machines 4 --machine-rank "$NODE_RANK" \
  --main-process-ip "$MAIN_PROCESS_IP"
```

Inspect the launch without allocating GPUs:

```bash
python -m gate3dgen.training.cli train --dry-run \
  --num-processes 32 --num-machines 4 --machine-rank 0 --main-process-ip 127.0.0.1
```

For an explicitly different, single-node experiment, use `--num-processes 8 --num-machines 1
--machine-rank 0`. This changes the global batch and is **not** the released training protocol.
To resume, add `--resume latest` or a full training checkpoint path on every node. Inference-only
release weights omit optimizer/RNG/dataloader state and cannot reproduce an exact training resume.

Reducing the rank count does not guarantee the full recipe fits the remaining GPUs: FP32 master
parameters and AdamW moments can exhaust two 80 GB GPUs at the first optimizer step even when
forward/backward fit. For an explicitly different small-rank validation, a copied YAML can set
`training.fsdp_offload_params: true` to offload FSDP parameters/gradients and optimizer work to CPU.
Provide substantial host RAM and expect slower execution. This switch defaults to false and does
not change `configs/train/purifier.yaml`, its selected blocks or loss coefficients. 
Full optimizer-state gathering can also exceed a container's **host** memory limit.
For an explicitly memory-constrained run, set `training.fsdp_state_dict_type: SHARDED_STATE_DICT`
in the copied config (the released default remains `FULL_STATE_DICT`). This saves distributed
model and optimizer state plus scheduler, per-rank RNG and dataloader state; the completion
marker is written only after every rank's shard exists. Resume with the same state-dict setting.
Sharded training checkpoints are not direct inference inputs. Merge the model after training:

```bash
accelerate merge-weights outputs/training/purifier/checkpoint-STEP/pytorch_model_fsdp_0 \
  outputs/training/purifier/checkpoint-STEP/merged
ln outputs/training/purifier/checkpoint-STEP/merged/model.safetensors \
  outputs/training/purifier/checkpoint-STEP/merged/diffusion_pytorch_model.safetensors
```

The hard link gives the merged weights the filename expected by
`gate3dgen.training.inference.load_pipeline`; pass the `merged` directory to that API.
The public demo/benchmark entry points remain pinned to the released step-15000 checkpoint
and do not accept arbitrary smoke-test checkpoints.
Merging needs sufficient host RAM and disk; optimizer shards are retained for training resume,
not shipped as inference weights. Do not mark a model-only save as a resumable checkpoint.

## Loss implementation and parameter audit

`training/purifier_losses.py` is the authoritative implementation of the released x0 objective,
ported from the original trainer. Equations 7–9 use broadcast-normalized weighted L1, independent
subject/affected-region terms, and adjacent-mask products for subject-edge differences. The affected
map is average-pooled 7×7 absolute degraded/GT RGB residual on the [0,1] pixel scale. Empty masks
contribute zero. `tests/test_purifier_losses.py` checks numerical definitions and nonzero gradients,
then invokes the real training worker's forward, backward and optimizer update with tiny components.

Both stages record trainable parameter names: router `run_config.json` (`metadata`), purifier
`trainable_parameters.json` before FSDP wrapping. All configured block indices are **zero-based**.
The released patterns select `transformer_blocks.29.` through `.49.` inclusive: 21 blocks.
The external FireRed transformer config has **60 blocks**, indexed 0–59. Blocks **0–28** and **50–59**
are frozen; the selected range is not the final 21 blocks. All parameters inside selected blocks
are trainable, including text-side
`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj` and `attn.to_add_out`. Block-external
`txt_norm`, `txt_in`, `img_in`, `time_text_embed`, `norm_out` and `proj_out` are not selected and
remain frozen. The cached router hidden states enter through the frozen `txt_norm`/`txt_in` path
before block attention; there is no separately unfrozen external h projection in this recipe.
The appendix must distinguish these zero-based module indices and internal projections from a
one-based block numbering or an external projection claim. New parameter exports audit the current
recipe; they do not substitute for a missing original run's trainable-parameter record.
The prompt libraries have one source under
`gate3dgen/resources/prompts/`; historical token labels remain fixed for checkpoint compatibility.
Legacy albedo/classifier integration remains in `training/experimental/` for compatibility;
it is not a supported released recipe. Some recipe/data contracts still depend on it, so deleting
that directory wholesale is unsafe. The independent MLP and released checkpoint root normalization
have standalone helpers; optional worker branches load legacy implementation only when enabled.

## Controlled paper ablations

Only the two main recipes live in `configs/train/`. Generate additional experiment configs explicitly:

```bash
python -m gate3dgen.training.ablations --experiment purifier-A --output outputs/configs/purifier-A.yaml
python -m gate3dgen.training.ablations --experiment purifier-B --output outputs/configs/purifier-B.yaml
python -m gate3dgen.training.ablations --experiment router-language-only --output outputs/configs/router-language-only.yaml
python -m gate3dgen.training.cli --config outputs/configs/purifier-A.yaml train --dry-run
python -m gate3dgen.training.router_tools.launch --config outputs/configs/router-language-only.yaml --dry-run
```

Table 5 A removes router conditioning and all spatial region/edge losses; B retains conditioning
but removes all spatial losses; Full is the released recipe. Global pixel L1, latent and LPIPS remain.
A/B use independent cache/output roots; prepare data and extract embeddings with the selected config
before training. Table 6 language-only disables vision LoRA while retaining the main 8-GPU/6,000-step
budget for a matched new comparison. The original historical language-only configuration used a
different budget. These generated configurations express the paper's factors, but are **not bound
to original Table 5/6 checkpoints**. In particular, the original affected/edge-only ablation retains
subject loss and must not be relabeled paper B.

## Independent VAE–MLP router baseline

`router_tools.mlp` exposes the original LayerNorm → Linear(64,128) → SiLU → Linear(128,1) head as an
independent experiment, with no purifier training. Cache frozen FireRed VAE posterior-mode latents,
normalize and pack 2×2 patches as in the trainer, then mean-pool tokens. Supply declared 1024×1024
inputs and a genuine immutable VAE identity for every split:

```bash
python -m gate3dgen.training.router_tools.mlp cache \
  --manifest data/router/train.jsonl --base-model "$GATE3DGEN_MODEL_PATH" \
  --vae-identity YOUR_VAE_REVISION_OR_HASH --output outputs/mlp/train.pt
python -m gate3dgen.training.router_tools.mlp cache \
  --manifest data/router/demo_validation.jsonl --base-model "$GATE3DGEN_MODEL_PATH" \
  --vae-identity YOUR_VAE_REVISION_OR_HASH --output outputs/mlp/development.pt
python -m gate3dgen.training.router_tools.mlp train \
  --train-cache outputs/mlp/train.pt --development-cache outputs/mlp/development.pt \
  --steps 6000 --batch-size 32 --learning-rate 1e-4 --output outputs/mlp/head.pt
python -m gate3dgen.training.router_tools.mlp eval \
  --cache outputs/mlp/development.pt --checkpoint outputs/mlp/head.pt \
  --threshold 0.5 --output-dir outputs/mlp/development_scores
```

The example MLP hyperparameters are an explicit **new baseline**, not recovered paper hyperparameters.
Select its threshold separately with `evaluation.router_metrics` on the development predictions, then
cache/score final data with that frozen threshold. The Gate3DGen router's released threshold does not
apply to this MLP. Training/evaluation reject asset or byte-hash overlap. Feature/head artifacts include
private sample/asset identities for leakage checks; they are not public aggregate metadata.

## Release and validation

The output roots are `outputs/training/router` and `outputs/training/purifier`. Retain full training
states privately for reproducibility. Training-checkpoint export tools are not included in this
source release. For the existing released models, use the prepared inference package described
in [installation](installation.md#checkpoints), not an arbitrary resumable training directory.
Inference-only packages cannot provide exact optimizer/RNG/dataloader resume. The package
includes frozen VAE, processor/tokenizer, scheduler and architecture config, but excludes original
FireRed diffusion/Qwen weights. This does not change the paper training initialization.
Derived-weight attribution and modification notices are in
[THIRD_PARTY](../THIRD_PARTY.md). Evaluate checkpoint
selection on validation data, freeze the choice and threshold, then run [final evaluation](evaluation.md).

## Validation

Training VGG LPIPS and evaluation AlexNet LPIPS intentionally use different backbones. The former
is a training loss; the latter is the Table 2 evaluation metric. Frozen feature weights are required.
Flow, foreground/background and low-frequency compatibility terms are not enabled by the released
x0/spatial recipe; their presence in imported code does not add them to this objective.

Run the regression suite with `python -m pytest -rs`.

When publishing, retain the Gate3DGen router, purifier and runtime with their applicable licenses
and notices. Exclude outputs, private data, official upstream model downloads and bytecode caches.
