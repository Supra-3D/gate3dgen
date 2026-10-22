"""Verify/reconstruct replacement tensors into the released inference layout."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3bc3f2a12722fd9883eb6357500de191d56baaf5'
BASE_REPO = 'FireRedTeam/FireRed-Image-Edit-1.1'
SPECS = (
    ('restoration', 'transformer', 'diffusion_pytorch_model.safetensors.index.json',
     'purifier/checkpoint-15000'),
    ('gateb', 'text_encoder', 'model.safetensors.index.json',
     'router/checkpoint-6000/exported_model'),
)


def reconstruct(release, firered, output_root, *, verify_only=False):
    import torch
    delta = runpy.run_path(str(ROOT / 'setup/delta_weights.py'))
    sha_file = delta['sha_file']
    release, firered, output_root = map(Path, (release, firered, output_root))
    manifests = {kind: json.loads((release / 'finetuned' / kind / 'manifest.json').read_text())
                 for kind, *_ in SPECS}
    identity = {kind: sha_file(release / 'finetuned' / kind / 'manifest.json') for kind in manifests}
    marker = output_root / '.gate3dgen-reconstruction.json'
    check = runpy.run_path(str(ROOT / 'setup/check.py'))['validate_checkpoints']

    def ready(root):
        return check(root / 'router/checkpoint-6000', root / 'purifier/checkpoint-15000', root / 'runtime')

    if not verify_only:
        if marker.is_file():
            previous = json.loads(marker.read_text())
            if previous['replacement_manifests'] != identity:
                raise ValueError('Existing outputs belong to different replacement manifests')
            for name, size in previous['files'].items():
                path = output_root / name
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError('Missing/truncated reconstructed file: ' + str(path))
            return dict(**ready(output_root), reused=True,
                        validation='manifest identity, readiness and file sizes; not a full weight rehash')
        for name in ('router', 'purifier', 'runtime'):
            if (output_root / name).exists() or (output_root / name).is_symlink():
                raise FileExistsError('Refusing to overwrite existing checkpoint component: ' + str(output_root / name))
        for kind, *_ in SPECS:
            if not (release / 'finetuned' / kind / 'assets/config.json').is_file():
                raise FileNotFoundError(release / 'finetuned' / kind / 'assets/config.json')
        for name in ('processor', 'vae', 'scheduler'):
            if not (firered / name).is_dir():
                raise FileNotFoundError(firered / name)
        output_root.mkdir(parents=True, exist_ok=True)
        runtime_bytes = sum(p.stat().st_size for name in ('processor', 'vae', 'scheduler')
                            for p in (firered / name).rglob('*') if p.is_file())
        needed = sum(m['full_tensor_bytes'] for m in manifests.values()) + runtime_bytes + 64 * 1024**2
        if shutil.disk_usage(output_root).free < needed:
            raise OSError(f'Insufficient disk space: reconstruction needs at least {needed} free bytes')

    torch.set_num_threads(4)
    if verify_only:
        reports = {}
        for kind, folder, index, _ in SPECS:
            with delta['IndexedState'](firered / folder, index) as base:
                reports[kind] = delta['verify'](base, release / 'finetuned' / kind)
            print(json.dumps(dict(kind=kind, **reports[kind])), flush=True)
        return dict(status='verified', stages=reports,
                    validation='all reconstructed tensor hashes; no complete checkpoint files written')

    # Stage all components first; never publish a half-written model as ready.
    # Keep a failed staging tree for inspection instead of silently deleting it.
    stage = Path(tempfile.mkdtemp(prefix='.reconstruct-', dir=output_root))
    for kind, folder, index, relative in SPECS:
        package, target = release / 'finetuned' / kind, stage / relative
        with delta['IndexedState'](firered / folder, index) as base:
            delta['materialize'](base, package, target, kind)
        for asset in (package / 'assets').iterdir():
            if asset.is_file() and not asset.name.startswith('.'):
                shutil.copy2(asset, target / asset.name)
        if kind == 'gateb':
            shutil.copy2(target / 'training_checkpoint_complete.json',
                         target.parent / 'checkpoint_complete.json')
    runtime = stage / 'runtime'
    for name in ('processor', 'vae', 'scheduler'):
        shutil.copytree(firered / name, runtime / name,
                        ignore=shutil.ignore_patterns('.cache', '._*', '.DS_Store'))
    (runtime / 'runtime_manifest.json').write_text(json.dumps(dict(
        schema_version='gate3dgen-runtime-v1', complete=True, source_model=BASE_REPO,
        source_revision=REVISION, components=['processor', 'vae', 'scheduler'],
        original_diffusion_weights_included=False, original_qwen_weights_included=False), indent=2)+'\n')
    ready(stage)
    files = {p.relative_to(stage).as_posix(): p.stat().st_size for p in stage.rglob('*')
             if p.is_file() and not p.name.startswith('._')}
    for name in ('router', 'purifier', 'runtime'):
        (stage / name).rename(output_root / name)
    stage.rmdir()
    marker.write_text(json.dumps(dict(replacement_manifests=identity, files=files), indent=2)+'\n')
    return dict(**ready(output_root), reused=False, validation='all reconstructed tensor hashes and readiness')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firered', type=Path, default=ROOT/'checkpoints/upstream/FireRed-Image-Edit-1.1')
    parser.add_argument('--release-dir', type=Path, default=ROOT/'checkpoints/release')
    parser.add_argument('--output-root', type=Path, default=ROOT/'checkpoints')
    parser.add_argument('--download-base', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.download_base:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=BASE_REPO, revision=REVISION, local_dir=args.firered, token=False)
    result = reconstruct(args.release_dir, args.firered, args.output_root, verify_only=args.verify_only)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
