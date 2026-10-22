"""Rebuild full original-dtype checkpoints into Gate3DGen's existing layout."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3bc3f2a12722fd9883eb6357500de191d56baaf5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firered', type=Path, default=ROOT/'checkpoints/upstream/FireRed-Image-Edit-1.1')
    parser.add_argument('--download-base', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.download_base:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='FireRedTeam/FireRed-Image-Edit-1.1', revision=REVISION, local_dir=args.firered)
    import torch
    from delta_weights import IndexedState, materialize, verify
    torch.set_num_threads(4)
    release = ROOT/'checkpoints/finetuned'
    router = ROOT/'checkpoints/router/checkpoint-6000/exported_model'
    purifier = ROOT/'checkpoints/purifier/checkpoint-15000'
    if not args.verify_only:
        assert not (purifier/'pytorch_model_fsdp.bin').exists(), 'Refuse to overwrite full purifier weights'
        assert not list(router.glob('*.safetensors')), 'Refuse to overwrite full router weights'
    for kind, subfolder, index in [('restoration','transformer','diffusion_pytorch_model.safetensors.index.json'), ('gateb','text_encoder','model.safetensors.index.json')]:
        package = release/kind
        with IndexedState(args.firered/subfolder, index) as base:
            if args.verify_only:
                print(json.dumps(dict(kind=kind, **verify(base, package))), flush=True)
                continue
            output = ROOT/'checkpoints/reconstructed_original'/kind
            materialize(base, package, output, kind)
        destination = purifier if kind == 'restoration' else router
        # Commit only completed, validated output; retain existing configs/markers.
        for path in output.iterdir():
            if path.suffix in ['.bin', '.safetensors'] or path.name == 'model.safetensors.index.json':
                path.replace(destination/path.name)
        shutil.copyfile(output/'RECONSTRUCTED.json', destination/'RECONSTRUCTED.json')
        print('RESTORED', kind, destination, flush=True)
    if not args.verify_only:
        print('Complete. Run: python setup/check.py --weights-only')


if __name__ == '__main__':
    main()
