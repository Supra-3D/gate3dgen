from collections import OrderedDict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from delta_weights import export, verify, verify_and_iter, materialize, IndexedState


class DeltaTests(unittest.TestCase):
    def test_changed_values_dtype_cast_and_new_keys_roundtrip(self):
        base = OrderedDict(a=torch.tensor([1., 2.], dtype=torch.bfloat16),
                           b=torch.tensor([3., 4.], dtype=torch.bfloat16))
        target = OrderedDict(a=base['a'].float(), b=base['b'] + 1,
                             new=torch.tensor(7., dtype=torch.float32))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'delta'
            result = export(base, target, path, shard_bytes=2)
            self.assertEqual(result['changed_tensor_count'], 2)
            self.assertEqual(result['unchanged_tensor_count'], 1)
            self.assertTrue(verify(base, path)['all_tensors_bit_identical'])
            reconstructed = dict(verify_and_iter(base, path))
            for key, value in target.items():
                self.assertEqual(reconstructed[key].dtype, value.dtype)
                self.assertTrue(torch.equal(reconstructed[key], value))

    def test_wrong_base_and_corrupt_shard_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'delta'
            base = {'a': torch.tensor([1.])}
            export(base, {'a': torch.tensor([2.])}, path)
            with self.assertRaisesRegex(ValueError, 'Wrong upstream'):
                verify({'a': torch.tensor([8.])}, path)
            shard = next(path.glob('*.safetensors'))
            shard.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                verify(base, path)

    def test_shape_missing_base_key_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'absent'):
                export({'a': torch.ones(2)}, {}, root / 'missing')
            with self.assertRaisesRegex(ValueError, 'Shape'):
                export({'a': torch.ones(2)}, {'a': torch.ones(3)}, root / 'shape')
            with self.assertRaises(FileExistsError):
                export({}, {}, root / 'shape')

    def test_materialize_both_file_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = {'a': torch.ones(2), 'b': torch.zeros(2)}
            target = {'a': torch.ones(2), 'b': torch.ones(2)}
            export(base, target, root / 'delta')
            materialize(base, root / 'delta', root / 'restoration', 'restoration')
            state = torch.load(root / 'restoration/pytorch_model_fsdp.bin', weights_only=True)
            self.assertTrue(torch.equal(state['b'], target['b']))
            materialize(base, root / 'delta', root / 'gateb', 'gateb')
            with IndexedState(root / 'gateb', 'model.safetensors.index.json') as state:
                self.assertTrue(torch.equal(state['b'], target['b']))


if __name__ == '__main__':
    unittest.main()
