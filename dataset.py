"""Paired LOLv2 loading; never pair images by sorted position."""
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def child(path, name):
    matches = [p for p in path.iterdir() if p.is_dir() and p.name.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} directory under {path}')
    return matches[0]


def resolve_root(root, variant):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset directory does not exist: {root}')
    target = 'real_captured' if variant == 'real' else 'synthetic'
    candidates = [p for p in [root, *root.rglob('*')] if p.is_dir() and p.name.lower() == target]
    if root.name.lower() not in {'real_captured', 'synthetic'}:
        try:
            child(root, 'Train')
            candidates.append(root)
        except ValueError:
            pass
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} {variant} dataset roots in {root}; pass the exact subset directory.')
    return candidates[0]


def paired_paths(root, split):
    directory = child(root, split)
    def index(folder):
        result = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            key = re.sub(r'^(low|normal)[_-]?', '', path.stem.lower())
            if key in result:
                raise ValueError(f'Duplicate image key {key} in {folder}')
            result[key] = path
        return result
    low, normal = index(child(directory, 'Low')), index(child(directory, 'Normal'))
    if not low or low.keys() != normal.keys():
        raise ValueError(f'Empty or unmatched pairs in {directory}: low-only={sorted(low.keys()-normal.keys())[:5]}, normal-only={sorted(normal.keys()-low.keys())[:5]}')
    return [(low[k], normal[k]) for k in sorted(low)]


def lowlight_paths(root):
    folder = child(child(root, 'Train'), 'Low')
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No training images in {folder}')
    return paths


class LowLightImages(Dataset):
    """Zero-DCE preprocessing: resize, RGB [0,1], no crop/flip or GT access."""
    def __init__(self, paths, image_size=256):
        self.paths, self.image_size = paths, image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return torch.from_numpy(np.asarray(image)/255.0).float().permute(2, 0, 1)


class PairedImages(Dataset):
    """Full-resolution evaluation only."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        paths = self.pairs[index]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        if images[0].size != images[1].size:
            raise ValueError(f'Pair dimensions differ: {paths}')
        return tuple(torch.from_numpy(np.array(im, dtype=np.float32)/255).permute(2, 0, 1) for im in images)
