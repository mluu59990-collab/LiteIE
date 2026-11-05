# LiteIE + Zero-DCE trên LOLv2 Real / Kaggle

Dùng **model LiteIE + training/loss Zero-DCE theo hướng dẫn tạm của tác giả**. Đây không phải loss LiteIE mới trong paper. Nguồn, commit và phần chưa công bố: [TRAINING_SOURCES.md](TRAINING_SOURCES.md).

## Chạy Kaggle

1. Import `notebooks/train_liteie_lolv2_kaggle.ipynb`.
2. Bật GPU và Internet; Add Input LOLv2.
3. Sửa duy nhất `DATASET_ROOT`, rồi Run All.

Notebook clone https://github.com/mluu59990-collab/LiteIE.git (repo cần truy cập được từ Kaggle), cập nhật fast-forward nếu đã có checkout, in commit sử dụng và kiểm tra dataset trước khi train.

`DATASET_ROOT` có thể là thư mục chứa LOLv2 hoặc trực tiếp `Real_captured`:

```
LOLv2/Real_captured/
  Train/Low/low00001.png
  Test/Low/low00690.png
  Test/Normal/normal00690.png
```

**Train/Normal không cần có và không được đọc.** Tất cả Train/Low được dùng để train. Không random crop/flip, không tách validation. Test không được dùng chọn checkpoint; chỉ đánh giá model ở epoch cuối. Tên cặp Test được ghép theo tên (bỏ tiền tố low/normal), không theo vị trí sắp xếp.

## Cấu hình mặc định lấy từ Zero-DCE

| Tham số | Mặc định |
|---|---|
| `--epochs` | 200 |
| `--batch-size` | 8 |
| `--image-size` | 256: resize Lanczos thành 256x256 |
| Optimizer | Adam, betas=(0.9,0.999), eps=1e-8 |
| `--lr` | 1e-4, cố định |
| `--weight-decay` | 1e-4 |
| `--grad-clip` | 0.1 |
| `--workers` | 4 |
| Khởi tạo convolution weight | Normal(0,0.02), bias mặc định PyTorch |
| Loss | 200 × TV(curve) + spatial + 5 × color + 10 × exposure |
| Exposure | Gray average pool 16x16, target 0.6 |
| Spatial | Gray average pool 4x4, sai khác 4 hướng |

`--dataset-root` bắt buộc. `--variant real` mặc định, có hỗ trợ `synthetic`. `--seed 42` là bổ sung để tái lập, không phải seed training chính thức LiteIE. `--device cuda` mặc định; CPU dùng cho smoke test. Không nạp pretrained khi bắt đầu run mới.

Giữ cách chuẩn hóa TV của Zero-DCE: cộng theo channel và chuẩn hóa spatial/batch. LiteIE có **một RGB curve dùng chung**; dùng trực tiếp tensor 3-channel này, không tự nhân thành 24 channel. Cấu trúc LiteIE không bị đổi thành Zero-DCE. `net(x)` vẫn trả ảnh; `net(x, return_curves=True)` trả thêm curve phục vụ loss. Số vòng enhancement tính riêng từng ảnh để không phụ thuộc ảnh khác trong batch; đây là điều chỉnh của repo này so với upstream.

```bash
python train.py --dataset-root /kaggle/input/lolv2 --check-data
python train.py --dataset-root /kaggle/input/lolv2
python train.py --dataset-root /kaggle/input/lolv2 --resume /kaggle/working/liteie-zero-dce-output/last.pt
```

## Kết quả và resume

Mặc định ghi vào `/kaggle/working/liteie-zero-dce-output`:

- `last.pt`: lưu sau mỗi epoch, gồm model/optimizer, epoch, recipe, lịch sử và manifest.
- `final.pth`: state dict ở epoch cuối, dùng được trong inference gốc bằng cách đổi `MODEL_PATH`.
- `config.json`, `splits.json`, `history.json`: cấu hình, tên ảnh và loss có trọng số theo epoch.
- `test_metrics.json`: RGB PSNR của checkpoint cuối, trung bình theo ảnh nguyên kích thước [0,1]; không chỉnh brightness bằng GT. Chưa có SSIM/LPIPS trong pipeline train.

Notebook tạo ZIP kết quả. Lưu output trước khi session kết thúc. Để resume sau khi đổi session, Add Input output cũ và đặt `RESUME` tới `last.pt`. Đường dẫn dataset có thể đổi, tên ảnh/manifest và cấu hình training phải giữ nguyên. `--epochs` là tổng epoch, không phải số epoch bổ sung. Resume checkpoint đã hoàn tất cho phép tạo lại final weights và đánh giá nếu session bị ngắt sau epoch cuối.

**Không resume checkpoint từ bản supervised L1 cũ.** Recipe khác sẽ báo lỗi; bắt đầu run mới trong thư mục output mới. Thí nghiệm cũ không được xem là kết quả của cấu hình này.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Các test CPU nhỏ kiểm tra loss, gradient, train không cần Train/Normal, resume, từ chối checkpoint L1 và tương thích checkpoint/model. Không thay thế training đầy đủ trên LOLv2/GPU Kaggle.
