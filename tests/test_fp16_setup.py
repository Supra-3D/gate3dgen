"""Exercise base + FP16 replacement assembly through the public setup downloader."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch
from safetensors.torch import save_file, load_file

from delta_weights import export

ROOT = Path(__file__).resolve().parents[1]


def load_setup(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'setup' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def package(tmp_path):
    base, release = tmp_path / 'base', tmp_path / 'release'
    for kind, folder, index in [('restoration', 'transformer', 'diffusion_pytorch_model.safetensors.index.json'),
                                ('gateb', 'text_encoder', 'model.safetensors.index.json')]:
        tensors = {'frozen': torch.ones(2, dtype=torch.bfloat16),
                   'trained': torch.zeros(2, dtype=torch.bfloat16)}
        json_file(base / folder / index, {'weight_map': {key: 'model.safetensors' for key in tensors}})
        save_file(tensors, base / folder / 'model.safetensors')
        target = {key: value.half() for key, value in tensors.items()}
        target['trained'] += 2
        destination = release / 'finetuned' / kind
        manifest = export(tensors, target, destination)
        manifest.update(precision_reference='FP16 reference', original_checkpoint_bit_identical=False)
        json_file(destination / 'manifest.json', manifest)
        json_file(destination / 'assets/config.json', {'fixture': True})
    json_file(release / 'finetuned/gateb/assets/training_checkpoint_complete.json',
              {'status': 'complete', 'global_step': 6000})
    json_file(release / 'finetuned/restoration/assets/checkpoint_complete.json',
              {'complete': True, 'global_step': 15000, 'condition_checkpoint_step': 6000, 'condition_role': 'B'})
    for component, names in [('processor', ['preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json']),
                             ('vae', ['config.json']), ('scheduler', ['scheduler_config.json'])]:
        for name in names:
            json_file(base / component / name, {'fixture': True})
    save_file({'w': torch.zeros(1)}, base / 'vae/diffusion_pytorch_model.safetensors')
    return base, release


def test_local_fp16_pair_roundtrip_and_resume(package, tmp_path):
    base, release = package
    output = tmp_path / 'checkpoints'
    restore = load_setup('restore_original_weights')
    result = restore.reconstruct(release, base, output)
    assert result['status'] == 'ready'
    state = torch.load(output / 'purifier/checkpoint-15000/pytorch_model_fsdp.bin', weights_only=True)
    assert state['frozen'].dtype == torch.float16
    assert torch.equal(state['trained'], torch.full((2,), 2., dtype=torch.float16))
    router = output / 'router/checkpoint-6000/exported_model'
    state = load_file(next(router.glob('*.safetensors')))
    assert state['trained'].dtype == torch.float16
    assert state['trained'].tolist() == [2., 2.]
    marker = json.loads((router / 'RECONSTRUCTED.json').read_text())
    assert marker['original_checkpoint_bit_identical'] is False
    assert restore.reconstruct(release, base, output)['reused'] is True
    (router / 'config.json').write_text('')
    with pytest.raises(ValueError, match='truncated'):
        restore.reconstruct(release, base, output)


def test_verify_only_writes_nothing_and_rejects_wrong_base(package, tmp_path):
    base, release = package
    output = tmp_path / 'no-output'
    restore = load_setup('restore_original_weights')
    report = restore.reconstruct(release, base, output, verify_only=True)
    assert report['status'] == 'verified'
    assert report['stages']['restoration']['verified_tensors'] == 2
    assert report['stages']['gateb']['original_checkpoint_bit_identical'] is False
    assert not output.exists()
    save_file({'frozen': torch.zeros(2, dtype=torch.bfloat16),
               'trained': torch.zeros(2, dtype=torch.bfloat16)}, base / 'transformer/model.safetensors')
    with pytest.raises(ValueError, match='Wrong upstream base'):
        restore.reconstruct(release, base, output, verify_only=True)


def test_existing_components_are_never_overwritten(package, tmp_path):
    base, release = package
    output = tmp_path / 'existing'
    (output / 'router').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='overwrite'):
        load_setup('restore_original_weights').reconstruct(release, base, output)


def test_no_staging_writes_when_disk_is_full(package, tmp_path, monkeypatch):
    base, release = package
    output = tmp_path / 'too-small'
    restore = load_setup('restore_original_weights')
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')(100, 100, 0)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage)
    with pytest.raises(OSError, match='Insufficient disk space'):
        restore.reconstruct(release, base, output)
    assert not list(output.glob('.reconstruct-*'))


def test_downloads_pinned_base_and_only_changed_tensors(package, tmp_path, monkeypatch):
    import huggingface_hub
    base, release = package
    output = tmp_path / 'downloaded'
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        source = base if kwargs['repo_id'].startswith('FireRedTeam/') else release
        shutil.copytree(source, kwargs['local_dir'], dirs_exist_ok=True)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    assert load_setup('download_weights').main(['--output-root', str(output)]) == 0
    assert len(calls) == 2
    assert calls[0]['allow_patterns'] == ['finetuned/**']
    assert calls[1]['revision'] == load_setup('restore_original_weights').REVISION
    assert all(call['token'] is False for call in calls)


def test_local_downloader_never_contacts_hub(package, tmp_path, monkeypatch):
    import huggingface_hub
    base, release = package
    def forbidden(**kwargs):
        pytest.fail('Local validation must not download weights')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', forbidden)
    output = tmp_path / 'local-verification'
    assert load_setup('download_weights').main([
        '--release-dir', str(release), '--firered', str(base), '--verify-only',
        '--output-root', str(output)]) == 0
    assert not output.exists()
