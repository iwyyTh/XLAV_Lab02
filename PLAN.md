# Kế hoạch Lab 02 — 3 ngày

Đã đọc đầy đủ 2 trang PDF `XLAV-CQ-LQN-TH02-Lab02-v1.0.pdf`.
M0 đã đạt điều kiện môi trường/dữ liệu/split/hình EDA để bắt đầu M1.
Còn xác minh bằng chứng Run All và Git trong phiên hiện tại; chưa triển khai M1–M8.
Google Drive Backup là yêu cầu trước khi nộp bài, không chặn bắt đầu M1.


## Quyết định sau khi đọc mã giảng viên

- [x] Đọc toàn bộ 32 file Python của 4 project; đối chiếu PDF, công thức và skeleton hiện tại.
- [x] Lập [LECTURER_REFERENCE_GUIDE.md](LECTURER_REFERENCE_GUIDE.md): bảng nguồn → đích M1–M8, lỗi/hạn chế và phương án thích nghi.
- Mới hoàn tất **phân tích**, chưa triển khai hoặc chạy các thuật toán tham khảo.
- Thiết kế đề xuất: nn.Module + nn.Linear; custom ReLU/Sigmoid qua autograd.Function
  có backward tự cài; custom initialization; Adam + BCEWithLogitsLoss.
- Không sao chép toàn bộ framework layer/backprop thủ công, không nạp pretrained demo.
- Sigmoid của ffnn2 simple không nhất quán với ActivationLayer; dùng hợp đồng derivative
  nhận preactivation, đối chiếu bản ffnn2_image đã sửa. Fan-in Normal không phải He Normal.

## M0 — Environment & Project Preparation

Dependencies: Python có sẵn, mạng để tải dữ liệu, quyền mở socket Jupyter/quyền ghi Git.
Files: scripts/*.py, config.py, requirements.txt, EDA notebook, README, báo cáo chuẩn bị.

- [x] Kiểm tra OS, phần cứng, Python/Conda, torch/torchvision, CUDA và workspace.
- [x] Tạo venv riêng dùng lại dependencies; import 11 thư viện, CPU tensor, pip check.
- [x] Đăng ký kernel nội bộ venv; cấu trúc và cấu hình project.
- [x] Viết downloader, audit/hash, split stratified/freeze và notebook EDA.
- [x] Đọc PDF và ghi yêu cầu bài nộp.
- [x] Dataset thật: 1.283 ảnh, 0 corrupt, 0 duplicate SHA-256; toàn bộ PNG 300×300 RGBA.
- [x] Split cố định seed 42: 845 train / 182 val / 256 test; nhãn/đường dẫn/membership/hash hợp lệ.
- [x] Người dùng xác nhận --check-only PASS; audit/metadata khớp kiểm tra chỉ đọc độc lập.
- [x] Đủ 4 hình EDA, đã kiểm tra đọc được.
- [ ] Lưu bằng chứng notebook Run All: 6 code cells trong file hiện tại chưa có execution count/output.
- [ ] Xác minh Git/tracking từ terminal máy thật; trong phiên này .git rỗng, Git báo not a git repository.

Definition of Done: môi trường/dữ liệu/split/hình EDA PASS, notebook chạy toàn bộ và Git/tracking xác minh được.
Hiện các điều kiện kỹ thuật về dữ liệu đã đạt, đủ để bắt đầu M1; chưa đóng toàn bộ M0 vì
hai mục bằng chứng notebook và Git còn mở. Không tải/chia lại dữ liệu để xử lý hai mục này.

## M1 — Dataset & Preprocessing

Dependencies: M0 có dữ liệu/split hợp lệ. Files: dataset.py, preprocessing.py, tests/test_dataset.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:forward`
(flatten giữ batch), `references/lecturer/pytorch_ffnn2_image/network_image/network.py:fit`
(flatten từng ảnh), `references/lecturer/pytorch_ffnn_image/test_ffnn_image.py` (NCHW demo).
Áp dụng vào `source/dataset.py`, `source/preprocessing.py`, `source/tests/test_dataset.py`;
không có Dataset/DataLoader thật trong reference, phải tự triển khai. Flatten mạng để M5.

- [ ] Custom PyTorch Dataset đọc CSV, kiểm tra đường dẫn/nhãn.
- [ ] Kiểm tra giá trị Alpha trên ảnh RGBA: min/max hoặc histogram, tỷ lệ pixel/ảnh có transparency.
- [ ] Quy định RGBA → RGB: bỏ alpha nếu opaque; nếu có transparency, compositing lên nền đã chọn và ghi rõ. Không sửa raw.
- [ ] RGB, resize 64×64, float32 [0,1]; target float [1]; cùng chính sách Alpha cho train/val/test/inference.
- [ ] DataLoader seed 42, batch 32, workers=0, shuffle chỉ train.
- [ ] Kiểm tra shape/dtype/range, nhãn, độ dài, batch cuối và eval deterministic.

Definition of Done: batch ảnh [B,3,64,64], nhãn [B,1], toàn bộ mẫu đọc được;
không thay đổi split; xác minh xử lý RGBA có/không transparency; test_dataset không còn skip và PASS.

## M2 — Data Augmentation

Dependencies: M1. Files: augmentation.py, experiments/03_augmentation.ipynb.

Nguồn tham khảo: **không có thuật toán augmentation** trong 4 project.
`references/lecturer/pytorch_ffnn_image/visualize_ffnn.py:visualize` chỉ tham khảo subplot.
Đích: `source/augmentation.py`, `source/experiments/03_augmentation.ipynb`.
Tự cài flip/rotation và fill policy phù hợp sau xử lý Alpha; cùng ảnh cho hình trước/sau.

- [ ] Cài ít nhất 2 kỹ thuật (đề xuất horizontal flip và random rotation).
- [ ] Chỉ áp dụng train; kiểm tra kích thước/range và tái lập seed.
- [ ] Lưới ảnh trước/sau và thí nghiệm có/không augmentation.

Definition of Done: 2 kỹ thuật hoạt động, hình minh chứng thật, val/test không augmentation.

## M3 — Custom Activation Functions

Dependencies: M0; M5/M6 để chạy so sánh. Files: activations.py, test_activations.py,
experiments/02_activation.ipynb.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn/nn_simple/ffnn.py:sigmoid_derivative,backward`;
`references/lecturer/pytorch_ffnn2/function_simple/activation.py`, `activation_prime.py`,
`references/lecturer/pytorch_ffnn2/layer_simple/act_layer.py:ActivationLayer.backward`;
`references/lecturer/pytorch_ffnn2_image/function_image/activation_prime.py` (đã sửa Sigmoid).
Đích: `source/activations.py`, `source/tests/test_activations.py`, `source/experiments/02_activation.ipynb`.
Bản simple truyền x vào hàm yêu cầu sigmoid(x); không sao chép lỗi hợp đồng này.

- [ ] Tự cài ít nhất 2 activation (đề xuất ReLU và sigmoid) cùng đạo hàm.
- [ ] Kiểm tra reference, numerical gradient, điểm không khả vi và độ ổn định số.
- [ ] Kết nối custom forward/backward bằng torch.autograd.Function + nn.Module wrapper; derivative(x) nhận preactivation.
- [ ] Sigmoid ổn định với trị tuyệt đối lớn; gradcheck float64 (ReLU tránh 0); kiểm tra gradient end-to-end, không chỉ gọi hàm đạo hàm riêng.

Definition of Done: công thức và đạo hàm đúng, test PASS, không chỉ wrapper hàm có sẵn.

## M4 — Weight Initialization

Dependencies: M3 để chọn scaling phù hợp; M5/M6 để so sánh.
Files: initialization.py, test_initialization.py, experiments/01_initialization.ipynb.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn/nn_simple/ffnn.py:FFNeuralNetwork.__init__`,
`references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:FFNeuralNetwork.__init__`,
`references/lecturer/pytorch_ffnn2/layer_simple/fc_layer.py:FCLayer.__init__` (Normal std=1),
`references/lecturer/pytorch_ffnn2_image/layer_image/fc_layer.py:FCLayer.__init__`
(thực tế std=1/sqrt(fan_in), comment Xavier/He không chính xác).
Đích: `source/initialization.py`, `source/tests/test_initialization.py`, `source/experiments/01_initialization.ipynb`.

- [ ] Baseline random normal theo đề; thêm ít nhất He Normal và Xavier.
- [ ] Phân biệt variance: Normal=1, Fan-in=1/fan_in, Xavier=2/(fan_in+fan_out) với gain=1, He ReLU=2/fan_in.
- [ ] Tính fan đúng cho nn.Linear.weight[out,in]; bias=0, no_grad, seed, kiểm tra mean/std. Default nn.Linear không phải Normal baseline.
- [ ] Protocol: đổi init hidden, giữ output Xavier gain=1 và bias=0; ghi rõ baseline hidden Normal. Fan-in là nhánh tùy chọn.
- [ ] So sánh training loss/hội tụ với baseline giữ nguyên mọi yếu tố khác.

Definition of Done: 2 phương pháp ngoài baseline, test PASS và dữ liệu so sánh hội tụ thật.

## M5 — MLP Architecture

Dependencies: M1, M3, M4. Files: model.py, test_model.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:forward`;
`references/lecturer/pytorch_ffnn2/layer_simple/fc_layer.py:FCLayer.forward,backward`;
`references/lecturer/pytorch_ffnn2_image/layer_image/fc_layer.py:FCLayer`;
`references/lecturer/pytorch_ffnn2_image/network_image/network.py:Network.add,predict`.
Đích: `source/model.py`, `source/tests/test_model.py`. Dùng nn.Linear/Sequential hoặc
ModuleList để đăng ký parameters; không giữ raw Tensor weights như demo, không ghi đè train().

- [ ] Flatten [B,3,64,64] thành [B,12288].
- [ ] MLP 12288 → 128 → 64 → 1; hidden activation configurable.
- [ ] Logits đầu ra, registered parameters và gradients hợp lệ; kiểm tra B=1/B=32, CPU và CUDA nếu có. Không dùng view(1,-1) cho cả batch.
- [ ] Vẽ sơ đồ kiến trúc, ghi số tham số.

Definition of Done: logits [B,1], gradient hợp lệ, test PASS, tương thích BCEWithLogitsLoss.

## M6 — Training & Evaluation

Dependencies: M1–M5. Files: trainer.py, metrics.py, train.py, predict.py, test_metrics.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn2/network_simple/network.py:Network.fit`;
`references/lecturer/pytorch_ffnn2_image/network_image/network.py:Network.fit`;
`references/lecturer/pytorch_ffnn2_image/function_image/loss.py`, `loss_prime.py` (MSE);
`references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:train,predict,save_weights,load_weights`.
Đích: `source/trainer.py`, `source/train.py`, `source/predict.py`, `source/metrics.py`, `source/tests/test_metrics.py`.
Reference chưa có BCE/validation/test/metrics; tự triển khai pipeline, không dùng manual update.

- [ ] Adam 0.001, 30 epochs; mini-batch 32, zero_grad/backward/step; giữ train()/eval() chuẩn, không manual weight update.
- [ ] BCEWithLogitsLoss với logits/target float [B,1], không sigmoid trước loss; epoch loss có trọng số theo số mẫu thực tế.
- [ ] Validation no_grad mỗi epoch; test chỉ sau khi chốt mô hình; aggregate confusion counts toàn tập.
- [ ] Tự cài accuracy (%) và ít nhất 2 metric khác (precision, recall, F1 đề xuất).
- [ ] Kiểm tra metric với ví dụ biên/reference, quy định zero division.
- [ ] Lưu history/config/seed, checkpoint tensor-only state_dict; chọn bằng validation; đánh giá test cuối cùng.
- [ ] Kiểm tra save/load round-trip bằng weights_only=True + load_state_dict; không nạp whole-model pickle của reference.
- [ ] Inference cùng preprocessing, sigmoid/threshold sau logits.

Definition of Done: chạy end-to-end, test metric PASS, checkpoint load được, metric test thật.

## M7 — Comparative Experiments

Dependencies: M2–M6. Files: 01/02/03 notebooks, results/{histories,metrics,checkpoints}.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn/test_ffnn.py`,
`references/lecturer/pytorch_ffnn2/test_network_simple.py`,
`references/lecturer/pytorch_ffnn_image/test_ffnn_image.py`,
`references/lecturer/pytorch_ffnn2_image/test_network_image.py` (chỉ demo cấu hình/chạy).
Đích: `source/experiments/01_initialization.ipynb`, `02_activation.ipynb`, `03_augmentation.ipynb`,
`results/histories/`, `results/metrics/`. Không dùng loss trên random data làm kết quả đồ án.

- [ ] Bảng so sánh baseline/2 initialization, >=2 activation, có/không augmentation.
- [ ] Giữ nguyên split/seed/kiến trúc/budget khi so sánh từng yếu tố.
- [ ] Khoảng 5 cấu hình dùng chung mốc: Normal/Xavier/He với ReLU; ReLU/Sigmoid cùng Xavier; có/không augmentation cùng ReLU+He. Không full factorial khi chưa đủ thời gian.
- [ ] Ghi phiên bản runtime, cấu hình, thời gian và kết quả; chỉ chọn bằng validation.

Definition of Done: kết quả tái lập, có loss curves và phân tích hội tụ, không bịa số liệu.

## M8 — Visualization & Report

Dependencies: M0–M7. Files: visualization.py, results/figures, doc/.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn_image/visualize_ffnn.py:visualize`,
`references/lecturer/pytorch_ffnn2_image/visualize_ffnn2.py:visualize` (subplot/colorbar/output).
Đích: `source/visualization.py`, `results/figures/`, `doc/`. Không chạy nhánh load checkpoint;
heatmap vector hidden không phải giải thích không gian ảnh và không thay thế các hình PDF yêu cầu.

- [ ] Lưới trước/sau augmentation, sơ đồ MLP, loss Train/Validation theo epoch.
- [ ] Báo cáo chỉnh sửa được và PDF; bảng tự đánh giá từng yêu cầu 0–100%.
- [ ] Mô tả thuật toán bằng công thức/mã giả, chức năng, cài đặt/cách chạy và hình minh chứng.
- [ ] **Google Drive Backup — hoàn thành trước khi nộp bài:** upload dataset cố định, ghi link public, thử tải không đăng nhập; không thay đổi sau deadline. Chưa xác minh backup.
- [ ] Đóng gói MSSV_th02.zip với doc và toàn bộ mã nguồn chạy được; bỏ venv/cache/raw lớn.

Definition of Done: đủ hai định dạng báo cáo, link dữ liệu, source sạch, chạy được từ bản giải nén.

## Phân bổ thời gian

- Ngày 1: lưu bằng chứng Run All/kiểm tra Git, hoàn thành M1–M3; không tải/chia lại dataset.
- Ngày 2: M4–M6; chuyển GPU nếu CPU không đủ thời gian.
- Ngày 3: M7–M8, kiểm tra bài nộp và đóng gói.

Không chuyển sang M1 tự động trong phiên phân tích tham khảo này. Các checklist thuật toán vẫn chưa hoàn thành.
Chỉ chạy unit tests dưới `source/tests`; `references/**/test_*.py` là demo có side effects, không collect như tests.
