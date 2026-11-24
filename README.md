# Lab 02 — Phân lớp ảnh nhị phân với PyTorch

Horse = 0, Human = 1. Mục tiêu cuối cùng: MLP nhận vector ảnh 64×64×3,
so sánh activation, initialization và augmentation theo đề `XLAV-CQ-LQN-TH02-Lab02-v1.0.pdf`.
Hiện đã có preprocessing M1 bằng OpenCV; Dataset/Loader đang viết dở. Các module
thuật toán khác vẫn là skeleton `NotImplementedError`.
Xem trạng thái thực tế trong [PREPARATION_REPORT.md](PREPARATION_REPORT.md) và [PLAN.md](PLAN.md).

## Cấu trúc

Thư mục hiện tại chính là root Lab 02, không có thư mục Lab_02 lồng nhau.

```text
.
├── data/
│   ├── raw/{horses,humans}/
│   ├── downloads/                 # ZIP gốc (không đưa vào Git)
│   └── splits/{train,val,test}.csv
├── source/
│   ├── config.py                 # cấu hình đã triển khai
│   ├── dataset.py, preprocessing.py, augmentation.py
│   ├── activations.py, initialization.py, model.py
│   ├── metrics.py, trainer.py, visualization.py, train.py, predict.py
│   ├── experiments/00_data_pipeline.ipynb    # kiểm tra OpenCV preprocessing
│   ├── experiments/00_dataset_exploration.ipynb
│   ├── experiments/01_initialization.ipynb
│   ├── experiments/02_activation.ipynb
│   ├── experiments/03_augmentation.ipynb
│   ├── tests/                    # preprocessing tests + 5 skeleton SKIP
│   ├── requirements.txt          # tham chiếu requirements ở root
│   └── README.md
├── scripts/{download_dataset,prepare_dataset,check_environment}.py
├── results/{figures,metrics,histories,checkpoints}/
├── doc/
├── requirements.txt
├── README.md, PLAN.md, PREPARATION_REPORT.md
└── .gitignore
```

## Môi trường đã chuẩn bị trên máy này

```bash
source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export IPYTHONDIR="$PWD/.cache/ipython"
export JUPYTER_DATA_DIR="$PWD/.venv/share/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter/runtime"
python scripts/check_environment.py
python -m pip check
```

`.venv` dùng Python 3.11.15 và `--system-site-packages` từ Conda `pytorch` đã có.
Đây là venv riêng nhưng **chưa độc lập hoàn toàn về dependencies**: các thư viện lớn
được đọc từ Conda, không cài lại hoặc thay đổi môi trường đó. Pytest, pluggy, iniconfig
được sao chép từ bản cài sẵn `cv-env` vào venv vì mạng không tải được gói thiếu.
Không di chuyển nguyên `.venv` sang máy khác; tạo môi trường mới từ requirements.
PyTorch có sẵn là bản build CUDA 13.0 nhưng đã kiểm thử chạy CPU;
không tải lại bản CPU chỉ để thay thế thư viện đang hoạt động.

## Cài mới độc lập khi có mạng (Python 3.11 hoặc 3.12)

Dùng tên mới `.venv-clean` để không ghi đè môi trường đã chuẩn bị:

```bash
python3.11 -m venv .venv-clean
source .venv-clean/bin/activate
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name lab02 --display-name 'Python (Lab 02)'
export JUPYTER_DATA_DIR="$VIRTUAL_ENV/share/jupyter"
python scripts/check_environment.py
```

Requirements ghim phiên bản đã import và kiểm tra dependency tại máy này; không phải lock
cho mọi hệ điều hành. Cách chọn cặp torch/torchvision: [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/).
Không dùng `sudo pip` hay cài vào Python hệ thống.

## Tải và chuẩn bị dữ liệu

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --check-only
```

Nếu tải tự động lỗi, tải nguyên hai ZIP rồi đặt trong `data/downloads/`:

- [horse-or-human.zip](https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip)
- [validation-horse-or-human.zip](https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip)
- Nguồn dự phòng của tác giả: [training ZIP](https://storage.googleapis.com/learning-datasets/horse-or-human.zip), [validation ZIP](https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip).

Sau đó chạy lại downloader; không cần tự giải nén. Có thể dùng `--archive-dir PATH` cho ZIP
đã tải ở nơi khác. Script đọc ảnh bằng Pillow, kiểm tra ZIP CRC và lưu SHA-256, nguồn,
nhãn và split gốc trong `data/source_manifest.csv`. Khi manifest đã tồn tại và hợp lệ,
script bỏ qua download. Không ghi đè ảnh khác nội dung.

`prepare_dataset.py` kiểm tra mọi ảnh, thống kê width/height/format/mode/channels,
phát hiện trùng file SHA-256 và xung đột nhãn. Giữ bản đầu tiên theo thứ tự xác định,
ưu tiên ảnh test khi trùng với train; giữ nguyên file gốc và ghi danh sách loại khỏi CSV.
Nếu trùng nội bộ test chính thức, dừng để kiểm tra thay vì tự sửa holdout.
Hash file không phát hiện ảnh gần giống nhau hoặc ảnh cùng pixel nhưng mã hóa khác nhau.

Theo [TFDS](https://www.tensorflow.org/datasets/catalog/horses_or_humans), có 1.027 ảnh train
và 256 ảnh test. Tập tác giả gọi là validation được TFDS sử dụng làm test; dự án giữ nguyên
holdout này, chia phần train theo 70:15 với stratification, seed 42.
Số lượng thực tế đã xác nhận là 845/182/256
(~65,86%/14,19%/19,95%). Khi chỉ có ảnh tự cung cấp và không có manifest nguồn chính thức,
script dùng 70/15/15. Tỷ lệ thực tế được lưu sau khi chạy thành công.

CSV chứa đường dẫn tương đối từ project root, nhãn và tên lớp. Metadata lưu fingerprint
nội dung dữ liệu và seed. Các lần chạy sau phải khớp CSV/fingerprint đã đóng băng;
script từ chối thay đổi dữ liệu chia. Không chạy Python với `-O` vì script dùng assertions
để kiểm tra bất biến. Không augmentation trước khi chia, không dùng test để chọn cấu hình.
CSV đã có đủ 845/182/256 mẫu; giữ nguyên khi triển khai M1.

## Notebook và kiểm tra

```bash
python -m jupyterlab source/experiments/00_dataset_exploration.ipynb
# Chọn kernel Python (Lab 02), Run All sau khi tải/chia dữ liệu thành công.
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=lab02 source/experiments/00_dataset_exploration.ipynb
python -m pytest source/tests -q
```

Notebook tạo `dataset_samples.png`, `class_distribution.png`, `split_distribution.png`,
`image_size_distribution.png` trong `results/figures/`. Không tạo hình nếu dữ liệu thiếu.
Các test thuật toán hiện được skip có chủ đích; skip không có nghĩa thuật toán PASS.
Sandbox hiện chặn socket của kernel; cần chạy Jupyter từ terminal bình thường có quyền mở
socket localhost. Kernel đã đăng ký trong venv, không thay đổi kernel toàn hệ thống.

## M1 — OpenCV Data Pipeline

Dùng OpenCV để thống nhất công cụ với môn Xử lý ảnh/Computer Vision. Hợp đồng callable:
`NumPy uint8 HWC BGR/BGRA → RGB Tensor float32 CHW [0,1]`, contiguous, deterministic.
`build_preprocessing(image_size=(height,width))` đổi màu một lần, dùng
`cv2.resize(..., (width,height), interpolation=cv2.INTER_LINEAR)`, rồi /255.
Alpha khác 255 được ghép lên nền trắng trước resize, giữ chính sách Pillow trước refactor.
Dữ liệu hiện tại có 1.283 ảnh opaque; không có transparency. Không flatten hoặc augmentation.
Input grayscale, rỗng, None, dtype khác uint8 hoặc sai số kênh sẽ báo lỗi rõ ràng.

Dataset sau khi người học hoàn thiện sẽ đọc lazy bằng `cv2.imread(path, cv2.IMREAD_UNCHANGED)`;
kiểm tra None và báo đường dẫn lỗi, rồi truyền thẳng BGR/BGRA cho transform. Không đổi màu
lần nữa ở Dataset. Hiện __init__ mới đọc CSV; __len__, __getitem__, build_loaders vẫn TODO.
Nhãn float32 [1], transform=None policy và DataLoader chưa được kiểm thử vì chưa triển khai.

Đã cài một biến thể duy nhất `opencv-python-headless==4.14.0.94` (cv2 4.14.0, GUI NONE)
trong .venv từ wheel cache cục bộ; pip online không tìm được gói trong phiên này.
Import với NumPy 2.4.6 và pip check thành công. Không gỡ Pillow, không cài lại torch/NumPy.
`source/requirements.txt` tiếp tục tham chiếu requirements ở root.

```bash
python -c "import cv2; print(cv2.__version__)"
python -m pytest source/tests -q
python -m jupyterlab source/experiments/00_data_pipeline.ipynb
```

Notebook kiểm tra alpha, channels, tensor, non-square resize, deterministic và ảnh tổng hợp.
Khi vẽ raw dùng cvtColor; khi vẽ tensor RGB chỉ permute CHW→HWC. Cell so sánh giữ Pillow
chỉ làm baseline cũ, không thuộc pipeline mới. Hình mới `results/figures/preprocessing_opencv_comparison.png`
không ghi đè `preprocessing.png` của người học. Pillow/OpenCV bilinear có thể khác từng pixel.
Kiểm tra refactor: 26 tests PASS, 5 skeleton SKIP. Bốn code cells notebook đã chạy
tuần tự bằng Python exec và lưu stdout thật; execution_count để null, metadata ghi rõ
phương thức kiểm tra. Run All qua Jupyter kernel bị sandbox chặn socket, chưa xác minh
trong phiên này. Hình so sánh đã được mở kiểm tra; chênh lệch pixel trung bình trên
ảnh mẫu là 0,02068 trong thang [0,1], không dùng làm ngưỡng chất lượng mô hình.

## Chuyển sang Colab/Kaggle

Chuyển toàn bộ source, scripts, requirements, CSV và metadata đã đóng băng; không chuyển venv,
cache hoặc checkpoint không cần thiết. Đặt project ở thư mục có quyền ghi và đặt working directory
vào root. Trên Kaggle, sao chép mã từ input chỉ đọc sang working trước khi chạy script.
Dùng torch/torchvision GPU có sẵn nếu import thành công, kiểm tra `torch.cuda.is_available()`;
không chạy lệnh cài CPU ở trên trong runtime GPU. Chỉ cài các thư viện thiếu, ghi lại phiên bản
thực tế bằng `check_environment.py`. Khi đổi phiên bản dependency, kiểm tra lại split đã lưu,
không tự tạo split mới nếu khác kết quả. Tải lại đúng dataset bằng script hoặc mang theo raw data;
chạy `prepare_dataset.py --check-only` để kiểm tra đường dẫn và hash. `Config` tự chọn CUDA nếu có;
khi huấn luyện M6 phải chuyển cả model và batch lên `config.device`.

## Yêu cầu bài nộp từ PDF

Tối thiểu 2 augmentation, 2 activation tự cài kèm đạo hàm, 2 initialization ngoài baseline
phân phối chuẩn; accuracy (%) và ít nhất 2 metric khác tự cài. Cần hình trước/sau augmentation,
sơ đồ mạng, loss train/val, và so sánh hội tụ initialization.
Cuối dự án tạo `MSSV_th02.zip`, `doc/MSSV_report_th02.pdf` cùng bản chỉnh sửa được
`.doc/.docx/.odt/.latex`, bảng tự đánh giá 0–100%, hướng dẫn cài/chạy và hình minh chứng.
Bổ sung Google Drive public dự phòng cho dữ liệu và giữ nguyên sau deadline.
Chưa tạo hoặc công khai liên kết Drive trong giai đoạn này. Khi đóng gói, đưa cả scripts và cấu
hình phụ trợ vào phần mã nguồn với hướng dẫn đường dẫn phù hợp; không chỉ sao chép riêng package source.
