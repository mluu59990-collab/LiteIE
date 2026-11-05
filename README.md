# LiteIE: Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices

[![Paper](https://img.shields.io/badge/arXiv-2507.04277-b31b1b.svg)](https://arxiv.org/abs/2507.04277)
[![Elsevier Journal](https://img.shields.io/badge/Expert%20Systems%20with%20Applications-In%20Press-blue.svg)](https://doi.org/10.1016/j.eswa.2025.129125)

LiteIE is an ultra-lightweight, real-time low-light image enhancement framework specifically designed for mobile and embedded devices. This repository provides the official resources for the paper [Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices](https://arxiv.org/abs/2507.04277).

---

## Training on LOLv2 (Kaggle)

Author-recommended Zero-DCE training for the LiteIE model: see [TRAINING.md](TRAINING.md) and [Kaggle notebook](notebooks/train_liteie_lolv2_kaggle.ipynb). Training uses low-light inputs only. This follows the temporary guidance in [upstream issue #1](https://github.com/mubaisam/LiteIE/issues/1), not the unpublished LiteIE paper loss. See [source audit](TRAINING_SOURCES.md).

## Highlights

1. **Ultra-light backbone:**
   We design a network-agnostic feature extractor that uses only two convolutions (58 weights in total) to produce a compact enhancement tensor.

2. **Parameter-free Iterative Restoration Module (IRM):**
   IRM reuses previously extracted features to progressively recover fine details lost in earlier enhancement steps, without introducing any additional learnable parameters.

3. **Multi-Scale Color Consistency Loss:**
   To ensure stable and natural enhancement, we design MSCol Loss, which preserves local color structure while maintaining global color balance.

4. **Extensive validation:**
   LiteIE generalizes well across diverse low-light datasets and runs efficiently on mobile SoCs, enabling fast inference on resource-limited devices.

---

## Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (GPU required)
- PyTorch 1.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/mubaisam/LiteIE.git
cd LiteIE

# Install dependencies
pip install torch torchvision
pip install opencv-python pillow numpy
pip install lpips  # Required for eval_en.py
```

---

## Quick Start

### Data Preparation

Create the following directory structure:
```
LiteIE/
├── data/
│   ├── test_data/          # Input low-light images
│   ├── LOL/                # Reference images (for evaluation)
│   └── result/             # Output enhanced images
├── snapshots/
│   └── epoth_199.pth       # Pre-trained model
└── ...
```

Place your test images in `data/test_data/` and reference images in `data/LOL/` or organize them in corresponding subdirectories within `data/test_data/`.

### Running Inference

#### Option 1: Inference Only (test.py)
Run inference and save enhanced results without computing metrics:

```bash
python test.py
```

**Output:**
- Enhanced images saved to `data/result/`
- Average inference time per image displayed

#### Option 2: Inference with Evaluation (eval_en.py)
Run inference and compute quality metrics (PSNR, SSIM, LPIPS):

```bash
python eval_en.py
```

**Output:**
- Enhanced images saved to `data/result/`
- Metrics computed for each image against reference:
  - **PSNR** (Peak Signal-to-Noise Ratio) - Higher is better
  - **SSIM** (Structural Similarity Index) - Higher is better
  - **LPIPS** (Learned Perceptual Image Patch Similarity) - Lower is better
- Average metrics and statistics displayed

---

## Usage Details

### test.py
Simple inference script that enhances low-light images and measures inference time.

**Configuration:**
- `TEST_DATA_DIR`: Path to input images (default: `data/test_data/`)
- `RESULT_BASE_DIR`: Path to save results (default: `data/result/`)
- `MODEL_PATH`: Path to pre-trained model (default: `snapshots/epoth_199.pth`)

**Features:**
- Automatic GPU warm-up to avoid timing overhead
- Batch processing of all images in the directory
- Displays average inference time per image
- Preserves original image resolution

### eval_en.py
Evaluation script that performs inference and computes perceptual quality metrics.

**Configuration:**
- `TEST_DATA_DIR`: Path to input images (default: `data/test_data/`)
- `REFERENCE_DIR`: Path to reference images (default: `data/LOL/`)
- `RESULT_BASE_DIR`: Path to save results (default: `data/result/`)
- `MODEL_PATH`: Path to pre-trained model (default: `snapshots/epoth_199.pth`)

**Features:**
- Two-phase processing: inference followed by metric computation
- Supports SSIM, PSNR, and LPIPS metrics
- Automatic reference image matching
- Detailed per-image metric output
- Summary statistics for batch evaluation

---

## Note on Code Release

While keeping the model extremely lightweight (58 parameters), this version achieves a slight improvement in quantitative metrics compared with the paper-reported implementation.

---

## Paper Info

- **Title:** Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices
- **Authors:** Guangrui Bai, Hailong Yan, Wenhai Liu, Yahui Deng, Erbao Dong
- **Journal:** Expert Systems with Applications (Elsevier), In Press
- **arXiv:** [2507.04277](https://arxiv.org/abs/2507.04277)
- **DOI:** [10.1016/j.eswa.2025.129125](https://doi.org/10.1016/j.eswa.2025.129125)

---

## Citation

If you find this project helpful in your research, please cite our paper:

```bibtex
@article{LiteIE2025,
  title={Towards Lightest Low-Light Image Enhancement Architecture for Mobile Devices},
  author={Guangrui Bai and Hailong Yan and Wenhai Liu and Yahui Deng and Erbao Dong},
  journal={Expert Systems with Applications},
  year={2025},
  doi={10.1016/j.eswa.2025.129125},
  eprint={2507.04277},
  archivePrefix={arXiv}
}
```

---

## Contact

- Author: [mubaisam](https://github.com/mubaisam)
- Issues: [LiteIE Issues](https://github.com/mubaisam/LiteIE/issues)
- Paper: [arXiv](https://arxiv.org/abs/2507.04277)

---

## License

This project is licensed under the Apache-2.0 License.

---

Star, fork, or open an issue to help us improve LiteIE!
