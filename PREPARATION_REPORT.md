# Báo cáo chuẩn bị Lab 02

Cập nhật: 2026-11-24 sau khi người dùng hoàn tất chuẩn bị trên máy.
**Điều kiện môi trường, dữ liệu, split và hình EDA của M0 đã đạt; đủ đầu vào để bắt đầu M1.**
Chưa đóng toàn bộ checklist M0: artifact notebook chưa lưu bằng chứng Run All, và Git
trong workspace hiển thị cho phiên này chưa phải repository hợp lệ.
Không huấn luyện mô hình, không tạo ảnh/số liệu giả. Đã đọc toàn bộ PDF 2 trang sau khi
file xuất hiện trong workspace khi phiên làm việc khởi động lại. File PDF được giữ nguyên.

## A. Environment

- Ubuntu 26.04 LTS, kernel Linux 7.0.0-43-generic x86_64.
- CPU Intel Core i3-1305U: 5 core / 6 logical CPU; Intel UHD Graphics.
- RAM 6,9 GiB; thời điểm khảo sát còn khoảng 3,7 GiB available; swap 26 GiB.
- Phân vùng workspace 395 GiB, còn khoảng 293 GiB trống tại thời điểm khảo sát.
- Python hệ thống 3.14.4; Conda base 3.14.6; có Python 3.11 trong nhiều Conda env.
- Env đã tìm thấy: base, agent, arxiv-pipeline, csAI, cv-env, diffusion, hugging,
  img_th, khdl, lab-day, pytorch, test_th_img. Không có env đang active qua VIRTUAL_ENV/CONDA_PREFIX.
- `.venv` tạo từ Conda pytorch Python 3.11.15 với `--system-site-packages`.
  Venv riêng nhưng dependencies vẫn phụ thuộc Conda gốc, chưa phải môi trường độc lập hoàn toàn.
- Thư viện có sẵn được tái sử dụng; pytest 9.2.0, pluggy 1.7.0, iniconfig 2.3.1 được
  sao chép từ package đã cài trong cv-env vào venv. Không sửa môi trường hệ thống/Conda.
- Pip không lấy được pytest từ index; phương án offline ở trên đã import và `pip check` thành công.
- PyTorch build CUDA 13.0 có sẵn, chạy **CPU**, `cuda.is_available() == False`.
  Không cài lại CPU wheel vì bản hiện có thực hiện phép toán CPU thành công.
- Kernel `lab02` đăng ký tại `.venv/share/jupyter/kernels/lab02`.
  Phiên trước bị sandbox chặn socket. Hiện đã có hình EDA do người dùng tạo trên máy;
  notebook lưu trên đĩa chưa có execution count/output để xác nhận Run All.

| Thư viện | Phiên bản import thành công |
|---|---|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| Pillow | 12.3.0 |
| scikit-learn | 1.9.0 |
| tqdm | 4.70.0 |
| pytest | 9.2.0 |
| jupyter | 1.1.1 |
| ipykernel | 7.3.0 |

Minh chứng: `results/metrics/environment.json`, `environment_check.log`, `pip_check.log`,
`system_inventory.json`. `requirements.txt` ghim đúng các phiên bản kiểm tra tại máy;
chưa kiểm thử cài mới từ internet hoặc trên Colab/Kaggle.

## B. File Changes

Danh sách dưới đây mô tả cấu trúc đã chuẩn bị trong workspace (không có Lab_02 lồng nhau).
Lần cập nhật này chỉ sửa tài liệu và xác minh chỉ đọc; không tải/chia lại dữ liệu,
không cài môi trường và không triển khai M1. PDF, CSV, metadata, notebook và ảnh được giữ nguyên.

| File/nhóm | Chức năng |
|---|---|
| `.gitignore` | Bỏ qua venv, raw/ZIP, cache, checkpoint, secrets |
| `requirements.txt`, `source/requirements.txt` | Phiên bản đã xác minh; source tham chiếu root |
| `source/__init__.py`, `scripts/__init__.py` | Package marker |
| `source/config.py` | Cấu hình đầy đủ, pathlib, tự chọn CPU/CUDA |
| `scripts/check_environment.py` | Import, tensor CPU, config, lưu kết quả JSON |
| `scripts/download_dataset.py` | ZIP chính thức, validate, tổ chức ảnh, nguồn/hash/nhãn |
| `scripts/prepare_dataset.py` | Audit ảnh, hash, stratified split, giữ test và khóa split |
| `source/dataset.py`, `source/preprocessing.py` | Skeleton M1 |
| `source/augmentation.py` | Skeleton M2 |
| `source/activations.py` | Skeleton M3 |
| `source/initialization.py` | Skeleton M4 |
| `source/model.py` | Skeleton M5 |
| `source/metrics.py`, `source/trainer.py`, `source/train.py`, `source/predict.py` | Skeleton M6 |
| `source/visualization.py` | Skeleton M8 |
| `source/experiments/00_dataset_exploration.ipynb` | Mã EDA, thống kê/hash và xuất 4 hình |
| `source/experiments/01_initialization.ipynb`, `02_activation.ipynb`, `03_augmentation.ipynb` | Notebook TODO, chưa chạy thí nghiệm |
| `source/tests/test_dataset.py`, `test_activations.py`, `test_initialization.py`, `test_metrics.py`, `test_model.py` | 5 test skeleton được skip rõ ràng |
| `data/splits/train.csv`, `val.csv`, `test.csv` | 845 / 182 / 256 mẫu; đường dẫn tương đối |
| `data/splits/split_metadata.json`, `data/source_manifest.csv` | Seed, fingerprint, nguồn/hash và split chính thức |
| `results/figures/*.png` | Đủ 4 hình EDA, đã kiểm tra đọc được |
| `README.md`, `source/README.md` | Hướng dẫn môi trường, dữ liệu, Jupyter, cloud và M1 |
| `PLAN.md` | Checklist M0–M8, dependencies, Definition of Done theo PDF |
| `doc/DATASET_SOURCE.md` | Nguồn, attribution, giấy phép và giới hạn |
| `PREPARATION_REPORT.md` | Báo cáo này |
| `results/metrics/*.json`, `*.log` | Minh chứng thực thi thực tế |
| Các `.gitkeep` | Giữ các thư mục chưa có artifact |

`.venv`, cache Python/Matplotlib/Jupyter là artifact cục bộ và được ignore.

## C. Dataset Summary

Đã kiểm tra trực tiếp mọi ảnh bằng Pillow (verify và load), SHA-256, CSV và manifest
trong lượt này; không gọi lại splitter hoặc ghi vào artifact dữ liệu.

| Dataset | Horse | Human | Total | Tỷ lệ thực tế |
|---|---:|---:|---:|---:|
| Train | 411 | 434 | 845 | 65,8613% |
| Validation | 89 | 93 | 182 | 14,1855% |
| Test | 128 | 128 | 256 | 19,9532% |
| Total | 628 | 655 | 1283 | 100% |

- 1.283 ảnh đọc được; 0 corrupt images; 0 duplicate files theo SHA-256.
- Toàn bộ ảnh 300×300, PNG, RGBA, 4 kênh; chưa khảo sát giá trị kênh Alpha.
- CSV đúng thứ tự cột `image_path,label,class_name`; mọi đường dẫn tương đối tồn tại.
- Nhãn horses=0, humans=1. Các tập không giao nhau cả đường dẫn lẫn hash.
- Hợp của các split khớp toàn bộ raw và source manifest; hash từng ảnh khớp manifest.
- Test khớp membership test chính thức; train/val chỉ lấy từ train gốc.
- Seed 42; giữ 256 ảnh test chính thức, chia phần còn lại Train:Validation = 70:15.
  Vì giữ test chính thức, tỷ lệ toàn bộ không phải 70/15/15.
- Người dùng xác nhận `python scripts/prepare_dataset.py --check-only`: PASS;
  `dataset_audit.json` hiện ghi PASS và khớp kiểm tra chỉ đọc độc lập trong lượt này.
  Không chạy lại prepare/check-only vì script này vẫn ghi audit/inventory và gọi logic split.
- Metadata lưu fingerprint `2a9b0122ffa19fccbfc72208e7927263299af903a2cfae5c999aa67efb1197db`.
  Tiếp tục giữ nguyên CSV/metadata giữa các thực nghiệm.

Nguồn: [TFDS](https://www.tensorflow.org/datasets/catalog/horses_or_humans),
[tác giả Laurence Moroney](https://laurencemoroney.com/datasets.html), giấy phép CC BY 2.0.
Dữ liệu CGI; hash file không phát hiện near-duplicate hoặc sự phụ thuộc nhân vật/cảnh.

Bốn hình PNG đã đọc thành công:

| Hình | Kích thước artifact |
|---|---|
| dataset_samples.png | 1500×1500 |
| class_distribution.png | 900×600 |
| split_distribution.png | 1050×600 |
| image_size_distribution.png | 1500×600 |

## D. Verification

Phân biệt bằng chứng kiểm tra trước đây, xác nhận của người dùng và artifact kiểm tra trực tiếp.

| Nội dung kiểm tra | Trạng thái | Minh chứng |
|---|---|---|
| Python, 11 imports, tensor CPU | PASS (lượt chuẩn bị trước) | environment.json, environment_check.log |
| Dependency compatibility | PASS (lượt trước) | pip_check.log |
| Cấu hình mặc định | PASS (lượt trước) | static_checks.log |
| CUDA | Không khả dụng; dùng CPU | environment.json |
| Kernel registration | PASS (lượt trước) | kernelspec trong venv |
| Dataset readability / properties | PASS | Đọc lại đủ 1.283 ảnh; khớp dataset_audit.json và image_inventory.csv |
| CSV schema / relative paths / labels | PASS | Kiểm tra từng hàng CSV và file thực tế |
| Split coverage / official test | PASS | Hợp split=raw=manifest; membership official test được giữ |
| Duplicate / split overlap checking | PASS | 1.283 hash duy nhất, không giao giữa các tập |
| Split verification bằng --check-only | PASS (người dùng xác nhận) | Metadata/audit hiện có khớp kiểm tra độc lập; không chạy lại splitter |
| Bốn hình EDA | PASS | Đủ 4 PNG, Pillow load thành công |
| Notebook JSON/Python syntax | PASS (lượt trước) | static_checks.log |
| EDA notebook Run All | CHƯA XÁC MINH từ artifact | 6 code cells đều execution_count=null, outputs=[] |
| Python compilation | PASS (lượt trước) | static_checks.log |
| Thuật toán/unit tests | CHƯA TRIỂN KHAI | pytest.log: 5 skipped |
| .gitignore patterns | PASS (lượt trước) | gitignore_check.log; kiểm tra bằng metadata Git tạm |
| Git repository/tracking | CHƯA ĐẠT trong phiên này | git status/rev-parse/ls-files đều báo not a git repository; .git rỗng |

Log download.log, preparation.log, notebook_execution.log và git_init.log còn giữ lỗi
của lần chuẩn bị trước. Chúng là lịch sử, không thay thế audit dữ liệu hiện tại hoặc chứng
minh EDA mới thất bại. Không xóa hay ghi đè các log này. Có đủ hình EDA không tự chứng minh
mọi cell notebook đã chạy. Không khởi tạo Git hay thay đổi branch/remote/index trong lượt này.
Nếu repository đã được tạo ở terminal máy thật, trạng thái đó chưa hiển thị qua workspace
của phiên hiện tại; không suy diễn rằng repository ngoài phiên đã bị mất.

## E. Remaining Tasks

- M0: lưu notebook sau Run All để có bằng chứng toàn bộ cell chạy không lỗi;
  kiểm tra lại Git/tracking từ terminal máy thật. Chưa đánh dấu hai mục này hoàn tất.
- **Google Drive Backup: bắt buộc hoàn thành trước khi nộp bài**, thuộc checklist M8,
  không chặn chuẩn bị dữ liệu cục bộ hoặc bắt đầu M1. Upload bản dữ liệu cố định, ghi link
  public trong báo cáo, thử truy cập/tải không đăng nhập, giữ nguyên sau deadline.
  Chưa có link/backup được xác minh trong artifact hiện tại.
- M1–M8 còn lại chưa triển khai. Không chạy training hay thí nghiệm trong lượt này.
- Venv hiện dùng lại Conda dependencies; tạo môi trường độc lập nếu cần chuyển máy,
  nhưng không cài lại trong lượt xác minh này.

## F. Next Step — M1 (chỉ ghi yêu cầu, chưa triển khai)

Điều kiện môi trường/dữ liệu để bắt đầu M1 đã đạt. Các file cần chỉnh ở bước tiếp theo:
`source/dataset.py`, `source/preprocessing.py`, `source/tests/test_dataset.py`.

1. Kiểm tra kênh Alpha: thống kê min/max hoặc histogram, số ảnh/pixel có alpha < 255;
   xác định có transparency thực hay toàn bộ opaque. Hiện chỉ xác nhận mode RGBA,
   không kết luận Alpha luôn 255.
2. Quy định chuyển RGBA sang RGB: nếu opaque có thể bỏ Alpha; nếu có transparency,
   chọn và ghi rõ màu nền/compositing trước RGB, tránh vô tình giữ RGB ẩn dưới pixel trong suốt.
   Dùng cùng chính sách cho train/val/test/inference; không sửa raw images.
3. RGB → resize 64×64 → float32 [0,1], ảnh [3,64,64], target float32 [1].
4. Đọc CSV cố định; DataLoader batch 32, workers=0, chỉ shuffle train, seed 42.
5. Test shape/dtype/range, nhãn, đường dẫn, batch cuối, tính xác định của eval và
   xử lý ảnh RGBA có/không transparency. Test thật phải PASS thay cho skip.

Flatten [B,12288] thực hiện ở MLP M5. BCEWithLogitsLoss dùng logits [B,1], không sigmoid
trước loss. Không triển khai bất kỳ bước M1 nào trong lượt cập nhật tài liệu này.
