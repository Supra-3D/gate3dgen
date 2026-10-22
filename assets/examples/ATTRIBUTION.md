# Example asset attribution and licenses

## The Hunt Museum figurine

**Porcelain figurine of a woman**, by **The Hunt Museum** (Sketchfab: `TheHuntMuseum`).

- Example ID: `demo_asset_01`; source asset ID: `0a648b5b6efa423ab84c13dc77012953`.
- Original model: [Sketchfab](https://sketchfab.com/3d-models/porcelain-figurine-of-a-woman-0a648b5b6efa423ab84c13dc77012953).
- Creator: [The Hunt Museum](https://sketchfab.com/TheHuntMuseum).
- Original asset license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Verification: the [Sketchfab model API](https://api.sketchfab.com/v3/models/0a648b5b6efa423ab84c13dc77012953)
  reported `CC Attribution` and the BY 4.0 license URL on 2026-10-12.
- Machine-readable attribution, file scope and rendering information: [asset_license.json](asset_license.json).

The four orthographic albedo renders under `data/renders_crossview/` were made by the
Gate3DGen authors from this asset. The four clean-scene GT images and four matched degraded
images under `train/pairs/` were generated with FireRed-Image-Edit-1.1 from those renders.
Changes include albedo rendering, clean scene/background generation, and hard shadow,
overexposure, glass reflection or underexposure generation. These twelve example images are
distributed under CC BY 4.0, not the source code's Apache-2.0 license. Retain the original
creator attribution, source and license links, and indicate subsequent modifications.
No endorsement by The Hunt Museum or the upstream model authors is implied.

The original 3D model and textures are not bundled. View IDs `000`, `003`, `006`, `007` are
rendering metadata, not license terms.

## TexVerse database provenance

The source asset was obtained through [TexVerse](https://huggingface.co/datasets/YiboZhang2001/TexVerse).
Contains information from TexVerse, which is made available under the
[Open Data Commons Attribution License (ODC-BY 1.0)](https://opendatacommons.org/licenses/by/1-0/).
This database notice does not replace the individual model's CC BY 4.0 license.
No full TexVerse database, private training inventory or benchmark is redistributed here.

## DTC benchmark airplane

`eval/` contains one **Digital Twin Catalog (DTC)** airplane supplied by
**Meta Platforms Technologies, LLC**, example `demo_case_01` / asset `demo_eval_asset_01`.
Its source case is `dtcplus_5e236feb89f71086022f__dark`.
The original GT mesh, rendered images and historical reconstructed meshes retain the
**DTC Dataset License Agreement: CC BY-SA 4.0 with additional no-sale terms**.
See [eval/LICENSE.md](eval/LICENSE.md) and [eval/asset_license.json](eval/asset_license.json)
for source links, restrictions and modifications. These files are not Apache-2.0.

The environment is **Hansaplatz**, by **Greg Zaal / Poly Haven**, under **CC0 1.0**.
The degradations are motion blur and uneven lighting; no SIR² or Flare7K++ layer is used.
Restored images and Pixal3D meshes are historical outputs, not a new inference run.
The generated Pixal3D clean-oracle mesh is not included or used as 3D ground truth.
