# Four-view data generation example

See [the data guide](../../../docs/data.md#1-supply-rendered-source-views) for the full workflow.

```text
assets/examples/data/
├── assets.jsonl
└── renders_crossview/
    └── demo_asset_01/albedo/{000,003,006,007}.png
```

These are real orthographic albedo renders of **Porcelain figurine of a woman** by
**The Hunt Museum**, licensed under **CC BY 4.0**. See [attribution](../ATTRIBUTION.md).
Images are unchanged copies from the historical Part-A figure pack; slot order is
`006`, `007`, `000`, `003`.
`demo_asset_01` is both the example's inventory ID (`sha256`, the historical ID field)
and directory name. The original Sketchfab ID is retained as `source_asset_id` in the
inventory, training pairs and license record. Image SHA-256 checksums remain unchanged.

FireRed first creates a **clean scene GT** from an albedo, then edits that GT to create the
matched degradation. Albedo is not scene GT. The current planner assigns 12 tasks, three per
view; the four pairs in `assets/examples/train` illustrate four tasks. This asset occupies only train; validation/test remain empty.

```bash
# CPU-only: validate the source inventory and create a 12-pair plan, not images.
bash scripts/run_example_data.sh plan

# Requires authorized external FireRed weights and CUDA.
EXAMPLE_GPUS=0 bash scripts/run_example_data.sh generate

# Requires an external local VLM server. Audits every generated pair;
# optional DINO/SAM diagnostics are explicitly excluded.
EXAMPLE_VLM_BASE_URL=http://127.0.0.1:8100/v1 \
  bash scripts/run_example_data.sh audit
bash scripts/run_example_data.sh export
```

Defaults are `outputs/examples/pairs`, `outputs/examples/audit` and
`outputs/examples/manifests`. `EXAMPLE_OUTPUT_ROOT` changes this runtime root within the
repository's output boundary. Rejected, uncertain or pending audits are not accepted samples.
Generation failures must be fixed or excluded, not relabeled as passing data.

To audit the **already included** historical pairs instead of generating new ones:

```bash
EXAMPLE_PAIR_ROOT=assets/examples/train EXAMPLE_PAIR_MANIFEST=assets/examples/train/pairs.jsonl \
  bash scripts/run_example_data.sh audit
EXAMPLE_PAIR_ROOT=assets/examples/train EXAMPLE_PAIR_MANIFEST=assets/examples/train/pairs.jsonl \
  bash scripts/run_example_data.sh export
```

Runtime artifacts never belong in `assets/examples/`. Add asset-disjoint authorized data before
running a train/holdout experiment. For offline Qwen audit, vision diagnostics, masks and
router-data preparation, follow the full data guide. No pre-approved audit output is shipped.
