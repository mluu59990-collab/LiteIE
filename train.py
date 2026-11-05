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

RECIPE = 'liteie-zero-dce-multinex-best-v3'


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


def rgb_ssim(prediction, target):
    # Standard valid 11x11 Gaussian window, sigma=1.5, RGB channel mean.
    prediction, target = prediction.double(), target.double()
    coordinate = torch.arange(11, device=prediction.device, dtype=torch.float64)-5
    gaussian = torch.exp(-coordinate.square()/(2*1.5**2))
    gaussian /= gaussian.sum()
    window = (gaussian[:, None]*gaussian[None, :]).expand(3, 1, 11, 11)
    conv = lambda x: torch.nn.functional.conv2d(x, window, groups=3)
    mu1, mu2 = conv(prediction), conv(target)
    v1 = conv(prediction.square())-mu1.square()
    v2 = conv(target.square())-mu2.square()
    covariance = conv(prediction*target)-mu1*mu2
    return (((2*mu1*mu2+.01**2)*(2*covariance+.03**2)) /
            ((mu1.square()+mu2.square()+.01**2)*(v1+v2+.03**2))).mean().item()


@torch.inference_mode()
def evaluate(net, loader, device):
    was_training = net.training
    net.eval()
    psnrs, ssims = [], []
    try:
        for low, normal in loader:
            # LiteIE's curve output remains in [0,1]; evaluate float RGB, not PNG.
            prediction = net(low.to(device))
            normal = normal.to(device)
            mse = (prediction.double()-normal.double()).square().mean().item()
            psnrs.append(-10*math.log10(mse) if mse > 0 else float('inf'))
            ssims.append(rgb_ssim(prediction, normal))
    finally:
        net.train(was_training)
    return {'psnr': sum(psnrs)/len(psnrs), 'ssim': sum(ssims)/len(ssims)}


def save_atomic(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--variant', choices=['real', 'synthetic'], default='real')
    parser.add_argument('--output-dir', default='/kaggle/working/liteie-real-best-output')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.lr, args.grad_clip) <= 0 or args.image_size < 16
            or args.workers < 0 or args.weight_decay < 0 or args.eval_every < 1):
        parser.error('Positive epochs/batch/lr/clip, image-size >= 16 and nonnegative workers/weight-decay required')
    seed_all(args.seed)
    root = resolve_root(args.dataset_root, args.variant)
    train_paths = lowlight_paths(root)
    test_pairs = paired_paths(root, 'Test')
    train_data = LowLightImages(train_paths, args.image_size)
    # Train/Normal is never read. Test supplies selection metrics, as in Multinex.
    for _ in train_data:
        pass
    for low, _ in PairedImages(test_pairs):
        if min(low.shape[-2:]) < 11:
            raise ValueError("Test images must be at least 11x11 for SSIM")
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
    val_loader = DataLoader(PairedImages(test_pairs), batch_size=1, num_workers=args.workers)
    manifest = {'train': [str(p.relative_to(root)) for p in train_paths],
                'test': [[str(a.relative_to(root)), str(b.relative_to(root))] for a,b in test_pairs]}
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config['recipe'] = RECIPE
    config['loss_weights'] = {'tv': 200, 'spatial': 1, 'color': 5, 'exposure': 10}
    config['exposure_target'] = .6
    start, history, evaluations, iteration = 0, [], [], 0
    best_ssim, best_iteration = None, 0
    best_psnr, best_epoch, best_weights = -float("inf"), 0, None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state.get('config', {}).get('recipe') != RECIPE:
            raise ValueError('Incompatible checkpoint recipe. Start a new Multinex-selection run; older recipes cannot be resumed.')
        for key in ('variant', 'seed', 'image_size', 'batch_size', 'lr', 'weight_decay', 'grad_clip', 'eval_every'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume setting differs: {key}')
        if state['splits'] != manifest:
            raise ValueError('Dataset manifest differs from checkpoint')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, history = state['epoch'], state['history']
        best_psnr, best_epoch, best_weights = state['best_psnr'], state['best_epoch'], state['best_model']
        best_ssim, best_iteration = state['best_ssim'], state['best_iteration']
        evaluations, iteration = state['evaluations'], state['iteration']
        if best_weights is not None:
            save_atomic(best_weights, output/'best.pt')
    if args.epochs < start:
        raise ValueError('--epochs cannot be less than the completed checkpoint epoch')
    (output/'config.json').write_text(json.dumps(config, indent=2))
    (output/'splits.json').write_text(json.dumps(manifest, indent=2))
    print(f'Parameters: {sum(p.numel() for p in net.parameters())}; {RECIPE}', flush=True)
    def validate_and_select(epoch):
        nonlocal best_psnr, best_ssim, best_epoch, best_iteration, best_weights
        metrics = evaluate(net, val_loader, device)
        if math.isnan(metrics['psnr']) or not math.isfinite(metrics['ssim']):
            raise RuntimeError('Invalid evaluation metrics')
        improved = metrics['psnr'] > best_psnr
        if improved:
            best_psnr, best_ssim = metrics['psnr'], metrics['ssim']
            best_epoch, best_iteration = epoch, iteration
            best_weights = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
            save_atomic(best_weights, output/'best.pt')
        row = {'epoch': epoch, 'iteration': iteration, **metrics, 'updated_best': improved}
        evaluations.append(row)
        (output/'evaluations.json').write_text(json.dumps(evaluations, indent=2))
        print(json.dumps(row), flush=True)

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
            iteration += 1
            if iteration % args.eval_every == 0:
                validate_and_select(epoch+1)
            for key, value in {'total': loss, **parts}.items():
                totals[key] += value.item()*len(low)
        # Short smoke runs still need one candidate if they never reach 500 steps.
        if epoch+1 == args.epochs and best_weights is None:
            validate_and_select(epoch+1)
        row = {'epoch': epoch+1, 'iteration': iteration,
               **{f'train_{k}': v/len(train_data) for k,v in totals.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        save_atomic({'epoch': epoch+1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': history, 'config': config, 'splits': manifest,
                     'best_psnr': best_psnr, 'best_epoch': best_epoch, 'best_model': best_weights,
                     'best_ssim': best_ssim, 'best_iteration': best_iteration,
                     'evaluations': evaluations, 'iteration': iteration}, output/'last.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    # Reload the PSNR-selected model. best.pt is a plain state dict.
    net.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True))
    score = evaluate(net, val_loader, device)
    result = {'recipe': RECIPE, 'checkpoint': 'best.pt', 'epoch': best_epoch, 'iteration': best_iteration, 'selection_split': 'Test',
              'selection_metric': 'psnr', 'best_psnr_rgb': best_psnr, 'ssim_at_best_psnr': best_ssim,
              'test_psnr_rgb': score['psnr'], 'test_ssim_rgb': score['ssim'], 'test_images': len(test_pairs)}
    (output/'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
