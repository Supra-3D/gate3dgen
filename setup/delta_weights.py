"""Lossless replacement-tensor export; never subtract or quantize weights."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(tensor):
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast('B')).hexdigest()


class IndexedState:
    def __init__(self, directory, index_name):
        self.directory = Path(directory)
        self.index_name = index_name
        self.index = json.loads((self.directory / index_name).read_text())
        self.weight_map = self.index['weight_map']
        self.stack = ExitStack()
        self.handles = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def keys(self):
        return self.weight_map.keys()

    def __getitem__(self, key):
        filename = self.weight_map[key]
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(safe_open(
                self.directory / filename, framework='pt', device='cpu'))
        return self.handles[filename].get_tensor(key)


def export(base, target, output, *, shard_bytes=512 * 1024 * 1024):
    """Compare every target tensor to the base cast to its original target dtype."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base_keys, target_keys = set(base.keys()), set(target.keys())
    if not base_keys <= target_keys:
        raise ValueError(f'Base keys absent in target: {sorted(base_keys-target_keys)[:10]}')
    output.mkdir(parents=True)
    rows, weight_map, shards = [], {}, []
    pending, pending_bytes = {}, 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f'changed-{len(shards)+1:05d}.safetensors'
        save_file(pending, output / filename, metadata={'format': 'pt', 'kind': 'replacement-tensors'})
        shards.append(dict(path=filename, bytes=(output / filename).stat().st_size,
                           sha256=sha_file(output / filename)))
        for key in pending:
            weight_map[key] = filename
        pending, pending_bytes = {}, 0

    for number, key in enumerate(sorted(target_keys), 1):
        tensor = target[key].detach().cpu().contiguous()
        baseline = base[key] if key in base_keys else None
        if baseline is not None and baseline.shape != tensor.shape:
            raise ValueError(f'Shape mismatch: {key}: {baseline.shape} != {tensor.shape}')
        base_hash = sha_tensor(baseline) if baseline is not None else None
        converted = baseline.to(dtype=tensor.dtype) if baseline is not None else None
        digest = sha_tensor(tensor)
        changed = converted is None or digest != sha_tensor(converted)
        size = tensor.numel() * tensor.element_size()
        rows.append(dict(key=key, shape=list(tensor.shape), dtype=str(tensor.dtype).split('.')[-1],
                         bytes=size, sha256=digest, base_sha256=base_hash,
                         base_dtype=str(baseline.dtype).split('.')[-1] if baseline is not None else None,
                         replacement=changed))
        if changed:
            if pending_bytes and pending_bytes + size > shard_bytes:
                flush()
            pending[key] = tensor
            pending_bytes += size
        if number % 100 == 0:
            print(json.dumps(dict(compared=number, total=len(target_keys),
                                  replacements=sum(r['replacement'] for r in rows))), flush=True)
    flush()
    manifest = dict(schema='gate3dgen-replacement-tensors-v1',
                    method='exact changed tensors, not arithmetic deltas; unchanged base cast to original target dtype',
                    tensor_count=len(rows), changed_tensor_count=len(weight_map),
                    full_tensor_bytes=sum(r['bytes'] for r in rows),
                    changed_tensor_bytes=sum(r['bytes'] for r in rows if r['replacement']),
                    unchanged_tensor_count=sum(not r['replacement'] for r in rows),
                    base_key_count=len(base_keys), tensors=rows, weight_map=weight_map, shards=shards)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_and_iter(base, package):
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['schema'] != 'gate3dgen-replacement-tensors-v1':
        raise ValueError('Unknown replacement package schema')
    if set(base.keys()) != {r['key'] for r in manifest['tensors'] if r['base_sha256'] is not None}:
        raise ValueError('Base tensor key set mismatch')
    for shard in manifest['shards']:
        path = package / shard['path']
        if path.stat().st_size != shard['bytes'] or sha_file(path) != shard['sha256']:
            raise ValueError(f'Corrupt replacement shard: {path}')
    # Read replacement shards directly; no mutable index file is needed.
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(package / name, framework='pt', device='cpu'))
                   for name in sorted(set(manifest['weight_map'].values()))}
        for row in manifest['tensors']:
            key = row['key']
            original = base[key] if row['base_sha256'] is not None else None
            if original is not None and sha_tensor(original) != row['base_sha256']:
                raise ValueError(f'Wrong upstream base tensor: {key}')
            if row['replacement']:
                tensor = handles[manifest['weight_map'][key]].get_tensor(key)
            else:
                tensor = original.to(dtype=getattr(torch, row['dtype']))
            if list(tensor.shape) != row['shape'] or str(tensor.dtype).split('.')[-1] != row['dtype']:
                raise ValueError(f'Reconstructed shape/dtype mismatch: {key}')
            if sha_tensor(tensor) != row['sha256']:
                raise ValueError(f'Reconstructed tensor is not bit-identical: {key}')
            yield key, tensor


def verify(base, package):
    count = sum(1 for _ in verify_and_iter(base, package))
    return dict(all_tensors_bit_identical=True, verified_tensors=count,
                checks='base bytes, replacement shard checksums, reconstructed tensor shape/dtype/bytes')


def materialize(base, package, output, kind):
    output = Path(output)
    marker = output / 'RECONSTRUCTED.json'
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    if kind == 'restoration':
        state = dict(verify_and_iter(base, package))
        torch.save(state, output / 'pytorch_model_fsdp.bin')
    else:
        pending, size, mapping = {}, 0, {}
        part = 0
        for key, tensor in verify_and_iter(base, package):
            if size and size + tensor.numel() * tensor.element_size() > 4 * 1024**3:
                name = f'model-part-{part:05d}.safetensors'
                save_file(pending, output / name, metadata={'format': 'pt'})
                mapping.update({key: name for key in pending})
                pending, size, part = {}, 0, part + 1
            pending[key] = tensor
            size += tensor.numel() * tensor.element_size()
        if pending:
            name = f'model-part-{part:05d}.safetensors'
            save_file(pending, output / name, metadata={'format': 'pt'})
            mapping.update({key: name for key in pending})
        (output / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}, indent=2) + '\n')
    marker.write_text(json.dumps(dict(all_tensors_bit_identical=True,
                                     replacement_manifest_sha256=sha_file(Path(package) / 'manifest.json')), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'materialize'])
    parser.add_argument('--firered', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--kind', choices=['restoration', 'gateb'], required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder, index = ('transformer', 'diffusion_pytorch_model.safetensors.index.json') if args.kind == 'restoration' else ('text_encoder', 'model.safetensors.index.json')
    torch.set_num_threads(4)
    with IndexedState(args.firered / folder, index) as base:
        if args.command == 'verify':
            print(json.dumps(verify(base, args.package)), flush=True)
        else:
            if args.output is None:
                parser.error('--output is required for materialize')
            materialize(base, args.package, args.output, args.kind)


if __name__ == '__main__':
    main()
