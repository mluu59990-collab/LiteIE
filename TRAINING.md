# LiteIE: training Zero-DCE, chọn best theo Multinex

Model LiteIE 58 tham số, training/loss Zero-DCE theo hướng dẫn tạm của tác giả LiteIE. Không phải loss LiteIE mới trong paper. Nguồn: [TRAINING_SOURCES.md](TRAINING_SOURCES.md).

## Chạy Kaggle

1. Import `notebooks/train_liteie_lolv2_kaggle.ipynb`, bật GPU + Internet, Add Input LOLv2.
2. Mặc định `DATASET_ROOT="/kaggle/input"`: tự tìm duy nhất thư mục `Real_captured`, bỏ qua `Synthetic`. Nếu có nhiều bản dataset, truyền đường dẫn cụ thể tới `LOLv2/Real_captured`.
3. Run All. Notebook clone/cập nhật repo GitHub và cho tải riêng `best.pt`.

```
LOLv2/Real_captured/
  Train/Low/
  Test/Low/
  Test/Normal/
```

Dùng toàn bộ Train/Low, không đọc Train/Normal, không tách validation. Test/Low và Test/Normal ghép theo tên bỏ tiền tố low/normal; ảnh sai cặp/kích thước sẽ báo lỗi. SSIM yêu cầu ảnh Test ít nhất 11x11.

## Training theo Zero-DCE

| Tham số | Mặc định |
|---|---|
| Epoch / batch | 200 / 8 |
| Image size | Resize Lanczos 256x256, không crop/flip |
| Optimizer | Adam, betas=(0.9,0.999), eps=1e-8 |
| LR / weight decay | 1e-4 / 1e-4 |
| Gradient clip | 0.1 |
| Workers | 4 |
| Conv weights | Normal(0,0.02); bias mặc định PyTorch |
| Loss | 200 × TV(curve) + spatial + 5 × color + 10 × exposure |
| Exposure | Pool gray 16x16, target 0.6 |

Seed 42, ghi log và resume là bổ sung để chạy thí nghiệm. Bắt đầu run mới không nạp pretrained. LiteIE dùng một RGB curve chung; TV giữ phép chuẩn hóa của Zero-DCE trên curve này, không nhân thành 24 channels. Số vòng enhancement tính riêng từng ảnh, không theo trung bình cả batch.

## Chọn best theo Multinex

- Mỗi **500 optimizer steps** (`--eval-every 500`), tính PSNR và SSIM trên toàn bộ `Real_captured/Test`.
- Chỉ khi **PSNR > best PSNR** thì ghi đè `best.pt`. Bằng nhau không cập nhật; SSIM tăng/giảm không thay đổi quyết định.
- Báo cáo SSIM của chính checkpoint có best PSNR, không ghép với SSIM cực đại của checkpoint khác.
- Dùng RGB float [0,1], PSNR trung bình theo ảnh, không crop border, không chỉnh sáng theo GT. SSIM Gaussian 11x11 sigma=1.5, valid region, trung bình RGB/theo ảnh.
- Sau train, nạp lại best.pt để đánh giá. Không xuất final.pth.
- Run ngắn chưa đủ 500 steps sẽ đánh giá một lần lúc kết thúc để có best.pt. Các run khác chỉ chọn tại mốc định kỳ.

**Test được dùng để chọn weights**, giống cấu hình Multinex; không xem kết quả này là đánh giá trên tập Test độc lập. Chỉ mượn cách chọn best, không thay optimizer/loss LiteIE bằng training Multinex.

## Artifacts và resume

Output mặc định `/kaggle/working/liteie-real-best-output`:

- **best.pt**: file duy nhất cần tải cho inference, plain state dict tương thích `model.LiteIE`.
- `last.pt`: lưu sau mỗi epoch để resume; gồm optimizer, iteration, best weights và metrics.
- `evaluations.json`: PSNR/SSIM theo iteration, có cờ updated_best.
- `history.json`, `config.json`, `splits.json`, `test_metrics.json`: loss, cấu hình, manifest và kết quả best.

```bash
python train.py --dataset-root /kaggle/input --variant real --check-data
python train.py --dataset-root /kaggle/input --variant real
python train.py --dataset-root /kaggle/input --resume /kaggle/working/liteie-real-best-output/last.pt
```

Sau khi đổi session, Add Input output cũ và đặt biến `RESUME` trong notebook tới last.pt. Resume ở ranh giới epoch; nếu ngắt giữa epoch, tiến độ từ checkpoint epoch cuối sẽ được chạy lại. `--epochs` là tổng epoch. Không resume recipe cũ (L1, final-only hoặc validation split); code sẽ báo lỗi. Đường dẫn dataset có thể đổi nhưng manifest và cấu hình training phải giữ nguyên.

```python
net.load_state_dict(torch.load('best.pt', map_location='cpu', weights_only=True))
net.eval()
```

Kiểm thử CPU: `python -m unittest discover -s tests -v`. Chưa chạy đầy đủ trên LOLv2/GPU Kaggle.
