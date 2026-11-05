# Audit of LiteIE training sources (2026-11-05)

## Verified sources

- LiteIE upstream commit: `67addb2f9e42db5de15eddb4783ab2a5e1b75b45`.
  https://github.com/mubaisam/LiteIE/tree/67addb2f9e42db5de15eddb4783ab2a5e1b75b45
  All eight commits inspected: no training script, loss implementation, or training configuration was published in this history.
- Author's replies, 2026-03-12 and 2026-03-16:
  https://github.com/mubaisam/LiteIE/issues/1
  The author says training code is still being organized, recommends Zero-DCE's training script, specifies Adam, and says brightness loss should dominate with color as an auxiliary loss. No numeric LiteIE loss weights are supplied.
- Another request for training/loss code and the training dataset remains unanswered in the inspected issue:
  https://github.com/mubaisam/LiteIE/issues/2
- Supplied paper `2507.04277v3.pdf`, sections 3.3 and 4.2, pages 5 and 8:
  https://arxiv.org/abs/2507.04277v3
  LiteIE is unsupervised. Equations (4)-(7) describe exposure, edge-aware TV and multi-scale color consistency losses. Alpha=0.8 and beta=0.4 are reported. These are NOT the fixed exposure target and TV coefficient from Zero-DCE.

## What cannot currently be reproduced exactly

The paper does not provide a complete executable training recipe: the MSCol reference target, local pooling scales, reduction/normalization conventions, numerical local/global coefficients and precise training settings are not fully specified. The released model is also explicitly described by the upstream README as differing from the paper implementation. The released checkpoint filename `epoth_199.pth` is not proof of a complete 200-epoch training configuration.

Do not label an inferred implementation of the paper loss as official. Do not label supervised L1 training as the LiteIE author's training method.

## Author-recommended fallback: Zero-DCE training

Source commit: `e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd`.

- https://github.com/Li-Chongyi/Zero-DCE/blob/e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd/Zero-DCE_code/lowlight_train.py
- https://github.com/Li-Chongyi/Zero-DCE/blob/e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd/Zero-DCE_code/dataloader.py
- https://github.com/Li-Chongyi/Zero-DCE/blob/e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd/Zero-DCE_code/Myloss.py

| Setting | Published Zero-DCE code |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4, constant |
| Weight decay | 1e-4 |
| Epochs | 200 |
| Training batch size | 8 |
| Training image processing | Resize to 256x256, PIL ANTIALIAS (modern name LANCZOS) |
| Random crop / flip | None |
| Gradient clip norm | 0.1 |
| DataLoader workers | 4 |
| Convolution weights | Normal(mean=0, std=0.02); bias left at PyTorch initialization |
| Loss | 200 * curve TV + spatial + 5 * color + 10 * exposure |
| Exposure | Grayscale 16x16 average pooling, target 0.6 |
| Spatial | Grayscale 4x4 average pooling, four directional differences |
| Training targets | None; only low-light inputs |

These are verified **Zero-DCE defaults**, recommended as a runnable fallback by the LiteIE author. They are not verified hyperparameters used to obtain the LiteIE paper results. Training on LOLv2 Real is the user's requested dataset adaptation, not a verified choice for the author's released checkpoint.

## Adaptations in this repository

- Keep the published LiteIE network (58 parameters), expose its one shared RGB curve for TV loss; do not duplicate it into Zero-DCE's 24-channel curve tensor.
- Compute iteration counts per image instead of using the upstream batch-wide brightness mean.
- Use all LOLv2 Real Train/Low for optimization; never read Train/Normal. No held-out split or paired training loss.
- Select best.pt by strictly increasing Test RGB PSNR every 500 optimizer steps, as requested following Multinex. Log SSIM of the same checkpoint; it is not a selection criterion. Test is therefore not independent of selection.
- Add seed=42, epoch-based resume, recipe checks, per-epoch checkpoints and logs. These are engineering additions, not claimed author settings.
- Modernize PIL ANTIALIAS to LANCZOS and device placement. Color loss uses vector_norm of squared color differences, mathematically equivalent to the upstream expression with finite gradients at exactly gray inputs.

Validation: six CPU tests passed, including all-low training, PSNR-only selection and resume equivalence, legacy-checkpoint rejection and notebook syntax. Separately compared all four weighted loss components and gradients against the pinned upstream Myloss.py on random 2x3x32x32 tensors; all matched within torch.testing.assert_close tolerances. Only upstream hardcoded CUDA allocation was changed to CPU for that comparison. Full LOLv2 training on Kaggle has not been run here.

## Multinex checkpoint selection audit

Verified against public code on 2026-11-05:
- https://github.com/albrateanu/multinex/blob/main/Options/Multinex_LOL-v2-real.yaml : validation uses Real_captured/Test; val_freq=500, only PSNR enabled, crop_border=0, test_y_channel=false, use_image=false.
- https://github.com/albrateanu/multinex/blob/main/basicsr/train.py : update only when current metric exceeds best PSNR.
- https://github.com/albrateanu/multinex/blob/main/basicsr/models/image_restoration_model.py : delete previous best and save best weights. Its validation returns the last metric; this implementation explicitly selects metrics['psnr'] so logging SSIM cannot change selection.

Local adaptations: fixed output name best.pt (plain state dict); preserve resume checkpoint each epoch; SSIM is additionally logged using RGB Gaussian 11x11 sigma=1.5, valid region. If a short run never reaches an evaluation interval, evaluate once at its final step so a best.pt exists. Other runs select only on scheduled evaluation steps. Training/loss remains Zero-DCE, not Multinex.
