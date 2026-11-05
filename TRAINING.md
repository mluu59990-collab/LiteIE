# Train LiteIE trên LOLv2 / Kaggle

Đây là **supervised L1 baseline bổ sung**, không phải code training chính thức hay cam kết tái lập số liệu paper. Giữ nguyên 58 tham số của model, sửa quy tắc số vòng enhancement để tính riêng từng ảnh trong batch. Không dùng pretrained weights khi bắt đầu run mới.

## Kaggle

1. Upload `notebooks/train_liteie_lolv2_kaggle.ipynb` vào Kaggle.
2. Bật GPU và Internet, Add Input dataset LOLv2.
3. Sửa `DATASET_ROOT` và Run All. Repo GitHub cần truy cập được từ Kaggle.

Notebook clone https://github.com/mluu59990-collab/LiteIE.git và in commit để truy vết. Nếu thư mục clone đã tồn tại, notebook sử dụng checkout hiện tại.

Có thể truyền thư mục chứa LOLv2 hoặc trực tiếp `Real_captured`. Loader tìm đúng một subset, không phân biệt hoa thường ở tên thư mục:

```
LOLv2/
  Real_captured/
    Train/
      Low/low00001.png
      Normal/normal00001.png
    Test/
      Low/low00690.png
      Normal/normal00690.png
```

Ảnh ghép theo tên (bỏ tiền tố `low`/`normal`), không ghép theo thứ tự. Tên trùng, thiếu cặp, lỗi ảnh và sai kích thước sẽ báo lỗi. Crop không resize ảnh; crop quá lớn sẽ báo lỗi.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--dataset-root` | bắt buộc | Đường dẫn dataset |
| `--variant` | `real` | `real` hoặc `synthetic` |
| `--epochs` | 200 | Tổng epoch, kể cả khi resume |
| `--batch-size` | 8 | Số cặp ảnh trong batch |
| `--crop-size` | 256 | Crop vuông, đồng bộ low/normal |
| `--lr` | 0.0001 | Adam, learning rate cố định |
| `--seed` | 42 | Seed split, khởi tạo và augmentation |
| `--val-fraction` | 0.1 | Tách 10% Train để chọn best checkpoint |
| `--workers` | 2 | DataLoader workers |
| `--device` | `cuda` | `cpu` dành cho smoke test |
| `--output-dir` | `/kaggle/working/liteie-output` | Thư mục ghi kết quả |
| `--resume` | không | Đường dẫn `last.pt` do pipeline tạo |
| `--check-data` | tắt | Chỉ kiểm tra dataset rồi thoát |

Ví dụ:

```bash
python train.py --dataset-root /kaggle/input/lolv2 --check-data
python train.py --dataset-root /kaggle/input/lolv2
python train.py --dataset-root /kaggle/input/lolv2 --resume /kaggle/working/liteie-output/last.pt
```

Train dùng L1 trên RGB [0,1], random crop và flip đồng bộ, clip gradient norm 1.0, FP32, một GPU. Không dùng Test để chọn checkpoint. Validation và Test chạy từng ảnh nguyên kích thước. PSNR được tính trên RGB [0,1], trung bình theo ảnh; không chỉnh brightness bằng ground truth. Chưa tính SSIM/LPIPS trong pipeline này.

## Kết quả / resume

- `best.pth`: state dict tương thích `model.LiteIE`, dùng được bằng cách đổi `MODEL_PATH` trong inference gốc.
- `last.pt`: model, optimizer, best model, epoch, lịch sử và cấu hình.
- `config.json`, `splits.json`, `history.json`, `test_metrics.json` lưu cấu hình và kết quả.

Notebook zip output thành `/kaggle/working/liteie-results.zip`. Lưu output trước khi session Kaggle kết thúc. Khi chuyển session, Add Input output cũ rồi đặt `RESUME` tới `last.pt`; đường dẫn dataset được phép đổi nhưng tên cặp và split phải giữ nguyên. Resume giữ cấu hình train và seed theo epoch; không cam kết bitwise giống giữa các GPU/phần mềm khác nhau.

Các mặc định là điểm khởi đầu, cần tune bằng validation. Để so sánh ba method công bằng, dùng cùng split, cùng protocol metric và cùng chính sách chọn checkpoint.
