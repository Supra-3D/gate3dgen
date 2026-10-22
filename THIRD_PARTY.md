# Third-party components

Gate3DGen's Apache-2.0 license covers original code that contributors have the right to license.

- FireRed-Image-Edit-1.1 supplies the base diffusion architecture, processor, VAE and scheduler.
- The released router is based on Qwen2.5-VL and includes the merged vision LoRA.
- Vendored training code retains its original FireRed copyright notices.
- PyTorch, Diffusers, Transformers and other dependencies retain their upstream licenses.
- Images named `assets/demo_*.png` are maintainer-selected demo inputs, with selected screenshot overlays cropped.
  Cropping does not change their underlying rights. They are not automatically
  relicensed as Apache-2.0. README output images are derived from those inputs.

The full private training corpus and benchmark are not bundled. The small real example under
`assets/examples/data/` and `assets/examples/train/` uses **Porcelain figurine of a woman** by **The Hunt Museum**,
asset `0a648b5b6efa423ab84c13dc77012953`, under **CC BY 4.0**. Its four albedo renders and four
FireRed-generated GT/degraded pairs are distributed under CC BY 4.0, not Apache-2.0.
See [assets/examples/ATTRIBUTION.md](assets/examples/ATTRIBUTION.md) for original source, license links,
TexVerse ODC-BY database attribution and modification notices, and
[assets/examples/asset_license.json](assets/examples/asset_license.json) for machine-readable provenance.
The real DTC airplane evaluation example under `assets/examples/eval/` retains the DTC Dataset
License Agreement (CC BY-SA 4.0 with additional no-sale terms), not Apache-2.0.
See [assets/examples/eval/LICENSE.md](assets/examples/eval/LICENSE.md) for the applicable restrictions,
Meta attribution, modifications and the CC0 Poly Haven HDRI notice. It contains no SIR² or
Flare7K++ layer. Its restored image and Pixal3D meshes are historical outputs, not fresh
checkpoint-pair inference.

Use your own authorized data for new experiments. Four rendered views are our rendering choices, not a
separate asset license.

## Pixal3D

Official checkpoints are downloaded from [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D)
by `setup/download_models.py` into ignored `checkpoints/upstream/` by default.
Maintainer test downloads use an explicit external directory. Its upstream code, models and
auxiliary dependencies retain their respective licenses and notices; see the
[official repository](https://github.com/TencentARC/Pixal3D). No upstream checkpoint is
redistributed in the Gate3DGen source archive.

## FireRed-Image-Edit-1.1

Verified on 2026-10-10: the [official model card](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1/blob/main/README.md)
lists **Apache-2.0**. The [upstream code license](https://github.com/FireRedTeam/FireRed-Image-Edit/blob/main/LICENSE)
is also Apache-2.0. The full unmodified base-model download goes to ignored `checkpoints/upstream/` for
synthesis and paper-training initialization. The inference weight package includes only its frozen
VAE, processor/tokenizer, scheduler/configuration and the separately fine-tuned router/purifier.
Original diffusion and Qwen tensor files are not duplicated. The runtime package carries an
Apache-2.0 license and modification/attribution notice; retain applicable upstream notices.

[Section 4 of Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0#redistribution)
permits redistribution, including modified versions, subject to conditions. Distributors must include
a copy of the license, identify modifications, retain applicable upstream copyright, patent,
trademark and attribution notices, and preserve applicable NOTICE content if the upstream work
includes it. Apache-2.0 does not grant general rights to upstream trademarks.

Gate3DGen modification notice: the released purifier checkpoint-15000 contains a fine-tuned
diffusion transformer initialized from FireRed-Image-Edit-1.1, conditioned on the released router's
hidden states. It is not an unmodified FireRed checkpoint or an upstream-endorsed release.
The router checkpoint-6000 is a fine-tuned Qwen2.5-VL export with merged vision LoRA.
Keeping the base model external does not remove applicable obligations for these derived weights.
When publishing weight archives, include the applicable upstream licenses and attribution notices,
as well as these modification notices. The source-code license alone does not replace that step.
