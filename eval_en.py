import os
import time
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
import lpips
import model


TEST_DATA_DIR = "data/test_data/"
REFERENCE_DIR = "data/LOL/"
RESULT_BASE_DIR = "data/result/"
MODEL_PATH = "snapshots/epoth_199.pth"


def ssim(prediction, target):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)])
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    return 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def iter_image_paths(root_dir):
    img_paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_paths.append(os.path.join(r, f))
    return img_paths


def make_result_path(image_path):
    if "test_data" in image_path:
        return image_path.replace("test_data", "result")
    filename = os.path.basename(image_path)
    return os.path.join(RESULT_BASE_DIR, filename)


def resolve_reference_path(result_path):
    ref_path = result_path.replace("result", "reference")
    if os.path.exists(ref_path):
        return ref_path
    filename = os.path.basename(result_path)
    flat = os.path.join(REFERENCE_DIR, filename)
    if os.path.exists(flat):
        return flat
    return None


def to_tensor_rgb_01(pil_img):
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def process_image(net, image_path, warmup_state):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    rw = (w // 16) * 16
    rh = (h // 16) * 16
    img_resized = img.resize((rw, rh))
    data = to_tensor_rgb_01(img_resized).cuda(non_blocking=True)

    if warmup_state["first"]:
        _ = net(data)
        torch.cuda.synchronize()
        warmup_state["first"] = False

    start = time.time()
    enhanced = net(data)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    result_path = make_result_path(image_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    enhanced_resized = F.interpolate(
        enhanced, size=(h, w), mode="bilinear", align_corners=False
    )
    torchvision.utils.save_image(enhanced_resized, result_path)

    return elapsed, result_path


def main():
    print("🚀 [Phase 1] Starting inference...")

    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Error: input directory does not exist: {TEST_DATA_DIR}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this script requires a GPU.")

    LiteIE = model.LiteIE().cuda()
    state = torch.load(MODEL_PATH)
    LiteIE.load_state_dict(state)
    LiteIE.eval()

    total_params = count_params(LiteIE)
    print(f"✅ Model parameters: {total_params:,}")

    img_list = iter_image_paths(TEST_DATA_DIR)
    total_time = 0.0
    processed_count = 0
    generated_files = []
    warmup_state = {"first": True}

    with torch.inference_mode():
        for img_path in img_list:
            print(f"Processing: {img_path}")
            one_time, res_path = process_image(LiteIE, img_path, warmup_state)
            total_time += one_time
            processed_count += 1
            generated_files.append(res_path)

    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"\n✅ Inference finished. Processed {len(img_list)} images.")
        print(f"⏱️ Average runtime: {avg_time:.4f} s/image")
    else:
        print("\n⚠️ No images were processed.")

    print("\n📊 [Phase 2] Computing metrics (PSNR / SSIM / LPIPS)...")

    lpips_fn = lpips.LPIPS(net="alex").cuda()
    lpips_fn.eval()

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    metric_count = 0

    with torch.inference_mode():
        for res_path in generated_files:
            ref_path = resolve_reference_path(res_path)
            if ref_path is None:
                print(f"⚠️ Skipped (reference not found): {os.path.basename(res_path)}")
                continue

            im_res = Image.open(res_path).convert("RGB")
            im_ref = Image.open(ref_path).convert("RGB")

            w, h = im_ref.size
            im_res = im_res.resize((w, h))

            np_res = np.array(im_res)
            np_ref = np.array(im_ref)

            score_psnr = calculate_psnr(np_res, np_ref)
            score_ssim = calculate_ssim(np_res, np_ref)

            tens_res = lpips.im2tensor(np_res).cuda(non_blocking=True)
            tens_ref = lpips.im2tensor(np_ref).cuda(non_blocking=True)
            score_lpips = lpips_fn.forward(tens_ref, tens_res).item()

            print(
                f"[{metric_count+1}] {os.path.basename(res_path)} | "
                f"PSNR: {score_psnr:.2f} | SSIM: {score_ssim:.4f} | LPIPS: {score_lpips:.4f}"
            )

            avg_psnr += score_psnr
            avg_ssim += score_ssim
            avg_lpips += score_lpips
            metric_count += 1

    if metric_count > 0:
        print("\n" + "=" * 30)
        print(f"🎉 Final results ({metric_count} images):")
        print(f"===> Avg. PSNR : {avg_psnr / metric_count:.4f} dB")
        print(f"===> Avg. SSIM : {avg_ssim / metric_count:.4f}")
        print(f"===> Avg. LPIPS: {avg_lpips / metric_count:.4f}")
        print("=" * 30)
    else:
        print("\n❌ No metrics were computed. Please check your reference images.")


if __name__ == "__main__":
    main()
