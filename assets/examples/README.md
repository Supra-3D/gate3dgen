# Small workflow examples

Run commands from the repository root, not from this directory.

The data and training directories contain one real TexVerse asset: **Porcelain figurine of a
woman** by **The Hunt Museum**, licensed under **CC BY 4.0**. The evaluation example is
a DTC airplane under separate share-alike/no-sale terms; see [eval/LICENSE.md](eval/LICENSE.md).
See [ATTRIBUTION.md](ATTRIBUTION.md) and
[asset_license.json](asset_license.json). The full private corpus and benchmark are not released.

- [Data inputs](data/README.md): four real orthographic albedo views and a portable inventory.
- [Training inputs](train/README.md): four historical FireRed clean-scene GT/degraded pairs.
- [Evaluation inputs](eval/README.md): one real motion-blur/uneven-lighting case, original DTC GT,
  historical Pixal3D before/after meshes and a four-view evaluation camera plan. No SIR² layers.

`albedo` is the source render; `gt` is a generated clean scene; `degraded` is its matched
observation. Audit the pairs and prepare reviewed masks before training.
Keep train, validation and test sets asset-disjoint.
Add authorized, asset-disjoint data before training or evaluating model quality.

Generation, audit and accepted-pair export calls are in `scripts/run_example_data.sh`.
Training entry points and preparation are documented in the training example and training guide.

Generated plans, audit records, model
caches, training states, renders and metric reports belong under ignored `outputs/`, never here.

## Maintainer-only real inputs

Keep full private inputs and their symlink layout **outside** the repository. Do not add private
data/model links inside the source checkout. This separates actual validation inputs from
public examples. The explicitly attributed museum and DTC examples are exceptions, not authorization
to release other assets. An external example layout can contain `data/assets.jsonl` plus selected
albedo views, `train/manifests/` plus accepted pairs, and `eval/cases.jsonl`, reference images/meshes,
historical prediction GLBs and a camera plan. Pass these external paths explicitly to the tools.

Organizing local inputs does not grant redistribution rights. Publishing those inputs requires
explicit dataset/asset authorization; do not add private inputs to a source archive.
