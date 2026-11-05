"""Supervised L1 baseline for LiteIE, not the paper's training recipe."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PairedImages, paired_paths, resolve_root
from model import LiteIE


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
    parser.add_argument('--output-dir', default='/kaggle/working/liteie-output')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.crop_size, args.lr) <= 0 or args.workers < 0 or not 0 < args.val_fraction < 1:
        parser.error('Positive epochs/batch/crop/lr, nonnegative workers and 0 < val-fraction < 1 required')
    seed_all(args.seed)
    root = resolve_root(args.dataset_root, args.variant)
    pairs = paired_paths(root, 'Train')
    test_pairs = paired_paths(root, 'Test')
    if len(pairs) < 2:
        raise ValueError('At least two training pairs are required')
    random.Random(args.seed).shuffle(pairs)
    count = min(len(pairs)-1, max(1, round(len(pairs)*args.val_fraction)))
    val_pairs, train_pairs = pairs[:count], pairs[count:]
    # Validate every image before launching a long GPU run.
    for split_pairs in (train_pairs, val_pairs, test_pairs):
        for low, normal in PairedImages(split_pairs):
            if split_pairs is train_pairs and min(low.shape[-2:]) < args.crop_size:
                raise ValueError('Training image is smaller than --crop-size')
    print(f'Dataset: {root}\nTrain={len(train_pairs)}, validation={len(val_pairs)}, test={len(test_pairs)}', flush=True)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    manifest = {name: [[str(a.relative_to(root)), str(b.relative_to(root))] for a,b in items]
                for name, items in [('train', train_pairs), ('validation', val_pairs), ('test', test_pairs)]}
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    start, best, history, best_weights = 0, -float('inf'), [], None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key in ('variant', 'seed', 'val_fraction', 'crop_size', 'batch_size', 'lr'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if state['splits'] != manifest:
            raise ValueError('Dataset split differs from checkpoint')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, history = state['epoch'], state['best_psnr'], state['history']
        best_weights = state['best_model']
        save_atomic(best_weights, output/'best.pth')
    if args.epochs <= start:
        raise ValueError('--epochs must exceed the completed checkpoint epoch')
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'splits.json').write_text(json.dumps(manifest, indent=2))
    val_loader = DataLoader(PairedImages(val_pairs), batch_size=1, num_workers=args.workers)
    print(f'Parameters: {sum(p.numel() for p in net.parameters())}; supervised L1 baseline', flush=True)
    for epoch in range(start, args.epochs):
        seed_all(args.seed + epoch)
        loader = DataLoader(PairedImages(train_pairs, args.crop_size), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(args.seed+epoch),
                            pin_memory=device.type == 'cuda')
        net.train()
        total = 0.
        for low, normal in loader:
            low, normal = low.to(device), normal.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(net(low), normal)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(low)
        psnr = evaluate(net, val_loader, device)
        row = {'epoch': epoch+1, 'train_l1': total/len(train_pairs), 'val_psnr_rgb': psnr}
        history.append(row)
        print(json.dumps(row), flush=True)
        if psnr > best:
            best = psnr
            best_weights = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
            save_atomic(best_weights, output/'best.pth')
        save_atomic({'epoch': epoch+1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'best_model': best_weights, 'best_psnr': best, 'history': history,
                     'config': config, 'splits': manifest}, output/'last.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    net.load_state_dict(best_weights)
    score = evaluate(net, DataLoader(PairedImages(test_pairs), batch_size=1, num_workers=args.workers), device)
    result = {'best_val_psnr_rgb': best, 'test_psnr_rgb': score, 'test_images': len(test_pairs)}
    (output/'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
