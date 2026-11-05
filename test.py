import os
import time
import glob
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F
import model

TEST_DATA_DIR = "data/test_data/"
RESULT_BASE_DIR = "data/result/"
MODEL_PATH = "snapshots/epoth_199.pth"


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


def to_tensor_rgb_01(pil_img):
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def process_image(net, image_path, warmup_state):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    rw = max(16, (w // 16) * 16)
    rh = max(16, (h // 16) * 16)

    img_resized = img.resize((rw, rh))
    data = to_tensor_rgb_01(img_resized).cuda(non_blocking=True)

    # warmup：避免第一次耗时偏大
    if warmup_state["first"]:
        _ = net(data)
        torch.cuda.synchronize()
        warmup_state["first"] = False

    start = time.time()
    enhanced = net(data)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    enhanced_resized = F.interpolate(
        enhanced, size=(h, w), mode="bilinear", align_corners=False
    )

    result_path = make_result_path(image_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_resized, result_path)

    return elapsed, result_path


def main():
    print("🚀 Starting inference (save results only)...")

    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Error: input directory does not exist: {TEST_DATA_DIR}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this script requires a GPU.")

    # Load model
    net = model.LiteIE().cuda()
    state = torch.load(MODEL_PATH, map_location="cuda")
    net.load_state_dict(state)
    net.eval()

    total_params = count_params(net)
    print(f"✅ Model parameters: {total_params:,}")

    img_list = iter_image_paths(TEST_DATA_DIR)
    if not img_list:
        print("⚠️ No images found.")
        return

    warmup_state = {"first": True}
    total_time = 0.0
    processed = 0

    with torch.inference_mode():
        for img_path in img_list:
            print(f"Processing: {img_path}")
            one_time, out_path = process_image(net, img_path, warmup_state)
            total_time += one_time
            processed += 1
            print(f"  -> saved: {out_path} | {one_time:.4f}s")

    avg_time = total_time / max(1, processed)
    print("\n" + "=" * 30)
    print(f"✅ Done. Processed {processed} images.")
    print(f"⏱️ Avg runtime: {avg_time:.4f} s/image")
    print(f"📁 Results saved under: {RESULT_BASE_DIR}")
    print("=" * 30)


if __name__ == "__main__":
    main()
