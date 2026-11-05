"""Paired LOLv2 loading; never pair images by sorted position."""
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image, ImageOps
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


class PairedImages(Dataset):
    def __init__(self, pairs, crop_size=None):
        self.pairs, self.crop_size = pairs, crop_size

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
        size = self.crop_size
        if size:
            w, h = images[0].size
            if min(w, h) < size:
                raise ValueError(f'Image smaller than crop {size}: {paths[0]} ({w}x{h})')
            x, y = random.randint(0, w-size), random.randint(0, h-size)
            images = [im.crop((x, y, x+size, y+size)) for im in images]
            if random.random() < .5:
                images = [ImageOps.mirror(im) for im in images]
            if random.random() < .5:
                images = [ImageOps.flip(im) for im in images]
        return tuple(torch.from_numpy(np.array(im, dtype=np.float32)/255).permute(2, 0, 1) for im in images)
