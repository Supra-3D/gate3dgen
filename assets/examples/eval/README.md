# Real benchmark evaluation example

Example case: **`demo_case_01`**, DTC airplane
`DTC_1_1_Airplane_B00DOTLGCU_WhiteBlack_3d-asset`.
Its degradations are **motion blur + uneven lighting**. It uses the CC0
[Hansaplatz HDRI](https://polyhaven.com/a/hansaplatz), not SIR² or Flare7K++ layers.

This replaces the procedural evaluation fixtures. Images and meshes retain the
**DTC Dataset License Agreement (CC BY-SA 4.0 with additional no-sale terms)**,
not the code's Apache-2.0 license. Read [LICENSE.md](LICENSE.md),
[asset_license.json](asset_license.json) and [attribution](../ATTRIBUTION.md).

```text
assets/examples/eval/
├── cases.jsonl
├── camera_plan.json
├── provenance.json
├── asset_license.json
├── LICENSE.md
├── demo_case_01/
│   └── {gt.png,degraded.png,restored.png,gt.glb}
└── generation/pixal3d/demo_case_01/
    └── {degraded.glb,restored_s01.glb}
```

`gt.png` is the clean 2D reference; `gt.glb` is the **original DTC asset**.
The PPT's Pixal3D `gt.glb` is a reconstruction from the clean image (an oracle),
not 3D ground truth, and is deliberately excluded. `restored.png` and both Pixal3D GLBs
are historical outputs of the checkpoint-15000 experiment, **not newly inferred results**.
Checksums and generation provenance are in `provenance.json`.
`demo_case_01` is the `case_id` in all example manifests, camera records and mesh directories.
Its example asset ID is `demo_eval_asset_01`, distinct from the training asset `demo_asset_01`.
Original benchmark IDs remain in `source_case_id` and `source_asset_id` for attribution and
traceability. Filenames and checksums are otherwise unchanged.

Only this one case and the before/after Pixal3D branches are included. Other restoration
methods, TRELLIS.2 meshes, videos, preview frames, the HDRI binary and full benchmark are not
bundled. Each mesh is below 100 MB, but the complete example is approximately 150 MB.

The four shared cameras come from the historical benchmark evaluation contract, not the
8-second qualitative PPT turntable. The PBR renderer's default GT-fitted framing is shared
by reference and predictions. Runtime artifacts stay in ignored `outputs/` or an external
work directory; never write them into this input example.

## Run the metrics

Run from the repository root:

```bash
bash scripts/evaluate.sh
```

This uses the example image manifest, retained Pixal3D meshes and shared four-view camera plan.
Outputs go to `outputs/eval/`. Pretrained AlexNet LPIPS and official OCIO-enabled Blender are required.
For 2D foreground masks, metric settings and new Pixal3D reconstructions, see the
[evaluation guide](../../../docs/evaluation.md).

To generate new before/after GLBs:

```bash
bash scripts/generate_glb.sh
```

Keep newly generated images and meshes together in a new output directory.
