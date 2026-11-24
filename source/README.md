# Mã nguồn Lab 02

Chạy từ project root theo [README](../README.md). `config.py` đã triển khai và kiểm thử;
`experiments/00_dataset_exploration.ipynb` chứa EDA thật nhưng chưa chạy trọn vẹn do thiếu dữ liệu.
Các module khác chỉ có signature, hợp đồng input/output và TODO; không thể huấn luyện ở giai đoạn này.
`requirements.txt` tham chiếu bản ghim duy nhất ở root.

M1 bắt đầu tại `dataset.py`, `preprocessing.py`, `tests/test_dataset.py`:

1. Đọc CSV bất biến, đường dẫn từ `Config.root`; horses=0, humans=1.
2. Chuyển mọi ảnh sang RGB, resize 64×64, tensor float32 trong [0,1].
3. Dataset trả ảnh [3,64,64], nhãn float32 [1] để batch nhãn [B,1].
4. DataLoader batch=32, workers=0; chỉ shuffle train, seed 42.
5. Flatten trong model M5 thành [B,12288], không làm mất chiều batch.
6. Bỏ skip test_dataset khi có implementation; kiểm tra shape, dtype, range, nhãn,
   toàn bộ CSV hợp lệ, batch cuối và dữ liệu validation/test xác định.

BCEWithLogitsLoss nhận logits [B,1] và target cùng shape; num_classes=2 không đồng nghĩa
cần 2 output logits trong cấu hình này. Không sigmoid trong MLP trước loss.
Không tự triển khai thuật toán trong bước chuẩn bị.
