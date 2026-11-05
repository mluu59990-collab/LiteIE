"""LiteIE with author-recommended Zero-DCE training (not the paper loss)."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import LowLightImages, PairedImages, lowlight_paths, paired_paths, resolve_root
from losses import ZeroDCELoss
from model import LiteIE

RECIPE = 'liteie-zero-dce-v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def weights_init(module):
    # Zero-DCE initializes conv weights only; biases retain PyTorch defaults.
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, 0., .02)


@torch.inference_mode()
def evaluate(net, loader, device):
    net.eval()
    scores = []
    for low, normal in loader:
        prediction = net(low.to(device)).clamp(0, 1)
        mse = (prediction - normal.to(device)).square().mean().item()
        scores.append(-10 * math.log10(max(mse, 1e-12)))
    return sum(scores) / len(scores)


def save_atomic(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--variant', choices=['real', 'synthetic'], default='real')
    parser.add_argument('--output-dir', default='/kaggle/working/liteie-zero-dce-output')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.lr, args.grad_clip) <= 0 or args.image_size < 16
            or args.workers < 0 or args.weight_decay < 0):
        parser.error('Positive epochs/batch/lr/clip, image-size >= 16 and nonnegative workers/weight-decay required')
    seed_all(args.seed)
    root = resolve_root(args.dataset_root, args.variant)
    train_paths = lowlight_paths(root)
    test_pairs = paired_paths(root, 'Test')
    train_data = LowLightImages(train_paths, args.image_size)
    # Check all inputs before a long run. Train/Normal is never accessed.
    for _ in train_data:
        pass
    for _ in PairedImages(test_pairs):
        pass
    print(f'Dataset: {root}\nTrain low-only={len(train_paths)}, test={len(test_pairs)}', flush=True)
    if args.check_data:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator on Kaggle, or pass --device cpu for a smoke test')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() and not args.resume:
        raise ValueError('Output already has a run. Use --resume or a new --output-dir')
    device = torch.device(args.device)
    net = LiteIE().to(device)
    net.apply(weights_init)
    objective = ZeroDCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    manifest = {'train': [str(p.relative_to(root)) for p in train_paths],
                'test': [[str(a.relative_to(root)), str(b.relative_to(root))] for a,b in test_pairs]}
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config['recipe'] = RECIPE
    config['loss_weights'] = {'tv': 200, 'spatial': 1, 'color': 5, 'exposure': 10}
    config['exposure_target'] = .6
    start, history = 0, []
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state.get('config', {}).get('recipe') != RECIPE:
            raise ValueError('Incompatible checkpoint recipe. Old supervised L1 checkpoints cannot be resumed.')
        for key in ('variant', 'seed', 'image_size', 'batch_size', 'lr', 'weight_decay', 'grad_clip'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if state['splits'] != manifest:
            raise ValueError('Dataset manifest differs from checkpoint')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['epoch'], state['history']
    if args.epochs < start:
        raise ValueError('--epochs cannot be less than the completed checkpoint epoch')
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'splits.json').write_text(json.dumps(manifest, indent=2))
    print(f'Parameters: {sum(p.numel() for p in net.parameters())}; {RECIPE}', flush=True)
    for epoch in range(start, args.epochs):
        seed_all(args.seed + epoch)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(args.seed+epoch),
                            pin_memory=device.type == 'cuda')
        net.train()
        totals = dict.fromkeys(['total', 'tv', 'spatial', 'color', 'exposure'], 0.)
        for low in loader:
            low = low.to(device)
            optimizer.zero_grad(set_to_none=True)
            enhanced, curves = net(low, return_curves=True)
            loss, parts = objective(low, enhanced, curves)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, value in {'total': loss, **parts}.items():
                totals[key] += value.item()*len(low)
        row = {'epoch': epoch+1, **{f'train_{k}': v/len(train_data) for k,v in totals.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_atomic({'epoch': epoch+1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': history, 'config': config, 'splits': manifest}, output/'last.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    # Evaluate the final epoch, never choose a checkpoint using Test/Normal.
    save_atomic(net.state_dict(), output/'final.pth')
    score = evaluate(net, DataLoader(PairedImages(test_pairs), batch_size=1, num_workers=args.workers), device)
    result = {'recipe': RECIPE, 'checkpoint': 'final.pth', 'epoch': args.epochs,
              'test_psnr_rgb': score, 'test_images': len(test_pairs)}
    (output/'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
