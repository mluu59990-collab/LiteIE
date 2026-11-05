import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import paired_paths, resolve_root
from model import LiteIE
from losses import ZeroDCELoss
import train


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
            root = Path(tmp)/'LOLv2'/'Real_captured'
            (Path(tmp)/'LOLv2'/'Synthetic'/'Train').mkdir(parents=True)
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
                    '--workers', '0', '--eval-every', '2', '--image-size', '16', '--batch-size', '2', '--output-dir', str(output)]
            for path in (root/'Train'/'Normal').iterdir():
                path.unlink()
            (root/'Train'/'Normal').rmdir()
            subprocess.run(base+['--check-data'], check=True)
            subprocess.run(base+['--epochs', '1'], check=True)
            # Resume into a new output location, as when moving Kaggle sessions.
            resumed = Path(tmp)/'resumed'
            subprocess.run(base+['--output-dir', str(resumed), '--epochs', '2', '--resume', str(output/'last.pt')], check=True)
            self.assertEqual(len(json.loads((resumed/'history.json').read_text())), 2)
            self.assertTrue((resumed/'best.pt').exists())
            self.assertEqual(json.loads((resumed/'test_metrics.json').read_text())['test_images'], 2)
            manifest = json.loads((resumed/'splits.json').read_text())
            self.assertEqual(len(manifest['train']), 4)
            self.assertNotIn('validation', manifest)
            metrics = json.loads((resumed/'test_metrics.json').read_text())
            evaluations = json.loads((resumed/'evaluations.json').read_text())
            self.assertEqual(metrics['best_psnr_rgb'], max(row['psnr'] for row in evaluations))
            # Resume is identical to uninterrupted CPU training at epoch boundaries.
            full = Path(tmp)/'full'
            subprocess.run(base+['--output-dir', str(full), '--epochs', '2'], check=True)
            a = torch.load(resumed/'best.pt', weights_only=True)
            b = torch.load(full/'best.pt', weights_only=True)
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
            # Force epoch 1 to win: Test must receive its weights, not epoch 2.
            selected = Path(tmp)/'selected'
            observed = []
            def metric(net, loader, device):
                observed.append({k: v.detach().cpu().clone() for k,v in net.state_dict().items()})
                return [{'psnr': 30., 'ssim': .8}, {'psnr': 20., 'ssim': .9}, {'psnr': 30., 'ssim': .8}][len(observed)-1]
            with patch.object(sys, 'argv', base[1:]+['--output-dir', str(selected), '--epochs', '2']), patch.object(train, 'evaluate', side_effect=metric):
                train.main()
            saved = torch.load(selected/'best.pt', weights_only=True)
            self.assertTrue(all(torch.equal(saved[k], observed[0][k]) for k in saved))
            self.assertTrue(all(torch.equal(observed[2][k], observed[0][k]) for k in saved))
            self.assertTrue(any(not torch.equal(observed[1][k], saved[k]) for k in saved))
            selected_metrics = json.loads((selected/'test_metrics.json').read_text())
            self.assertEqual(selected_metrics['epoch'], 1)
            self.assertEqual(selected_metrics['ssim_at_best_psnr'], .8)
            # Completed checkpoints can rerun evaluation (interrupted Kaggle export).
            subprocess.run(base+['--output-dir', str(resumed), '--epochs', '2', '--resume', str(resumed/'last.pt')], check=True)
            legacy = Path(tmp)/'legacy.pt'
            torch.save({'config': {'lr': .0001}}, legacy)
            failed = subprocess.run(base+['--output-dir', str(Path(tmp)/'reject'), '--resume', str(legacy)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Incompatible checkpoint recipe', failed.stderr)
            (root/'Test'/'Normal'/'normal00000.png').unlink()
            with self.assertRaises(ValueError):
                paired_paths(root, 'Test')

    def test_loss_constants_and_finite_gray_gradients(self):
        objective = ZeroDCELoss()
        low = torch.full((2, 3, 16, 16), .2)
        enhanced = low.clone().requires_grad_()
        curves = torch.zeros_like(low, requires_grad=True)
        total, parts = objective(low, enhanced, curves)
        self.assertAlmostEqual(parts['exposure'].item(), 1.6, places=5)
        for name in ('tv', 'spatial', 'color'):
            self.assertEqual(parts[name].item(), 0.)
        total.backward()
        self.assertTrue(torch.isfinite(enhanced.grad).all())
        self.assertTrue(torch.isfinite(curves.grad).all())

    def test_metrics_identity(self):
        image = torch.rand(1, 3, 16, 16)
        self.assertAlmostEqual(train.rgb_ssim(image, image), 1., places=10)
        self.assertLess(train.rgb_ssim(image, image*.5), 1.)
        net = torch.nn.Identity().train()
        result = train.evaluate(net, [(image, image)], torch.device('cpu'))
        self.assertTrue(net.training)
        self.assertEqual(result['psnr'], float('inf'))
        self.assertAlmostEqual(result['ssim'], 1.)

    def test_tv_channel_normalization(self):
        image = torch.full((1, 3, 16, 16), .6)
        curve = torch.arange(16.).view(1,1,16,1).expand(1,3,16,16)
        _, parts = ZeroDCELoss()(image, image, curve)
        # Three channels, vertical difference 1, TV = 2 * 3, weight = 200.
        self.assertAlmostEqual(parts['tv'].item(), 1200.)

    def test_notebook_cells_compile(self):
        notebook = json.loads((ROOT/'notebooks/train_liteie_lolv2_kaggle.ipynb').read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
