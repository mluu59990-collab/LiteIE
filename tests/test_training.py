import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import paired_paths, resolve_root
from model import LiteIE


class TrainingTests(unittest.TestCase):
    def test_batch_independence_and_checkpoint(self):
        torch.set_num_threads(1)
        net = LiteIE()
        net.load_state_dict(torch.load(ROOT/'snapshots/epoth_199.pth', weights_only=True, map_location='cpu'))
        images = torch.cat([torch.full((1,3,16,16), .03), torch.full((1,3,16,16), .5)])
        expected = torch.cat([net(image[None]) for image in images])
        self.assertTrue(torch.allclose(net(images), expected, atol=1e-6))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 58)

    def test_train_resume_and_pair_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'Real_captured'
            for split, count in [('Train', 4), ('Test', 2)]:
                for folder, prefix, value in [('Low', 'low', 30), ('Normal', 'normal', 120)]:
                    directory = root/split/folder
                    directory.mkdir(parents=True)
                    for i in range(count):
                        Image.fromarray(np.full((16,16,3), value+i, dtype=np.uint8)).save(directory/f'{prefix}{i:05}.png')
            self.assertEqual(resolve_root(tmp, 'real'), root.resolve())
            self.assertEqual(len(paired_paths(root, 'Train')), 4)
            output = Path(tmp)/'run'
            base = [sys.executable, str(ROOT/'train.py'), '--dataset-root', tmp, '--device', 'cpu',
                    '--workers', '0', '--crop-size', '16', '--batch-size', '2', '--output-dir', str(output)]
            subprocess.run(base+['--check-data'], check=True)
            subprocess.run(base+['--epochs', '1'], check=True)
            # Resume into a new output location, as when moving Kaggle sessions.
            resumed = Path(tmp)/'resumed'
            subprocess.run(base+['--output-dir', str(resumed), '--epochs', '2', '--resume', str(output/'last.pt')], check=True)
            self.assertEqual(len(json.loads((resumed/'history.json').read_text())), 2)
            self.assertTrue((resumed/'best.pth').exists())
            self.assertEqual(json.loads((resumed/'test_metrics.json').read_text())['test_images'], 2)
            (root/'Train'/'Normal'/'normal00000.png').unlink()
            with self.assertRaises(ValueError):
                paired_paths(root, 'Train')

    def test_notebook_cells_compile(self):
        notebook = json.loads((ROOT/'notebooks/train_liteie_lolv2_kaggle.ipynb').read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
