# Kế hoạch Lab 02 — 3 ngày

Đã đọc đầy đủ 2 trang PDF `XLAV-CQ-LQN-TH02-Lab02-v1.0.pdf`.
M0 đã đạt điều kiện môi trường/dữ liệu/split/hình EDA để bắt đầu M1.
M1 hoàn thành kỹ thuật theo kiểm thử độc lập: 54 PASS, 0 FAIL/SKIP/ERROR; xem báo cáo M1.
M2 đã đạt kiểm chứng kỹ thuật (59 PASS); notebook thực nghiệm còn TODO.
M3 hoàn thành kỹ thuật: 100 PASS trên CPU float32/float64; notebook lưu trên đĩa còn TODO.
M4 hoàn thành implementation/testing: 128 PASS; so sánh training loss/hội tụ để M7. M5 hoàn thành kỹ thuật: 58 PASS; regression M1–M5: 425 PASS. M6 hoàn thành kỹ thuật: 104 PASS; regression M1–M6: 529 PASS. M7 đã nghiệm thu artifact; M8 chưa thực hiện. Git repository đã truy cập được ở lượt refactor M1.
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
- [x] Git repository truy cập được trong lượt refactor M1; đã đọc status/diff, không commit/push.

Definition of Done: môi trường/dữ liệu/split/hình EDA PASS, notebook chạy toàn bộ và Git/tracking xác minh được.
Hiện các điều kiện kỹ thuật về dữ liệu đã đạt, đủ để bắt đầu M1; chưa đóng toàn bộ M0 vì
mục bằng chứng Run All notebook EDA vẫn cần xác minh riêng. Không tải/chia lại dữ liệu.

## M1 — Dataset & Preprocessing

Dependencies: M0 có dữ liệu/split hợp lệ. Files: dataset.py, preprocessing.py, tests/test_dataset.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:forward`
(flatten giữ batch), `references/lecturer/pytorch_ffnn2_image/network_image/network.py:fit`
(flatten từng ảnh), `references/lecturer/pytorch_ffnn_image/test_ffnn_image.py` (NCHW demo).
Áp dụng vào `source/dataset.py`, `source/preprocessing.py`, `source/tests/test_dataset.py`;
không có Dataset/DataLoader thật trong reference, phải tự triển khai. Flatten mạng để M5.
Quyết định M1: dùng OpenCV NumPy uint8 BGR/BGRA → RGB float32 CHW,
alpha trong suốt composite nền trắng; INTER_LINEAR, deterministic, không sửa input.
Người học đã triển khai Dataset/Loader; lượt này chỉ bổ sung test, không sửa thuật toán.

- [x] Preprocessing P01–P12: shape/dtype/range/finite, resize vuông/không vuông, black/white,
  RGB/BGRA đúng pixel, alpha, deterministic, invalid input và không mutation.
- [x] Dataset D01–D21: CSV validation, counts 845/182/256, target float32 [1], nhãn đúng,
  missing/corrupt files, default/custom transform, relative paths độc lập cwd và lazy loading.
- [x] DataLoader L01–L18: samplers, 27/6/8 batches, tails 13/22/32, không bỏ/trùng mẫu,
  seeded shuffle, different seeds, eval deterministic, custom transforms và generator độc lập.
- [x] Duyệt cả ba DataLoader qua toàn bộ 1.283 ảnh thật đúng một lượt trong test tích hợp.
- [x] `.venv/bin/python -m pytest source/tests/test_dataset.py -v --tb=short --durations=10`:
  **54 PASS, 0 FAIL, 0 SKIP, 0 ERROR**, 6.71s. Số lượng từ pytest thực tế.

Definition of Done: **M1 hoàn thành về mặt kỹ thuật** với cấu hình CPU hiện tại,
batch 32 / workers 0 / seed 42, theo phạm vi kiểm tra P01–P12, D01–D21, L01–L18.
Chi tiết: `results/metrics/m1_test_report.txt`, `results/metrics/m1_test_summary.json`.
Không có lỗi bắt buộc cần sửa trước khi đóng M1. Kiểm tra repeated eval/shuffle/custom
transforms dùng dữ liệu tổng hợp nhỏ để tránh lặp lại toàn bộ dataset thật.
Chưa xác minh GPU/multiprocessing/custom root độc lập; đây không phải yêu cầu mặc định M1.
P2 tùy chọn: dọn docstring/imports cũ; quy định config.root tùy biến; sửa cell notebook
đang lấy loaders["train"] cho mọi vòng lặp thành loader. Không sửa các file đó trong lượt này.

## M2 — Data Augmentation

Dependencies: M1. Files: augmentation.py, experiments/03_augmentation.ipynb.

Nguồn tham khảo: **không có thuật toán augmentation** trong 4 project.
`references/lecturer/pytorch_ffnn_image/visualize_ffnn.py:visualize` chỉ tham khảo subplot.
Đích: `source/augmentation.py`, `source/experiments/03_augmentation.ipynb`.
Tự cài flip/rotation và fill policy phù hợp sau xử lý Alpha; cùng ảnh cho hình trước/sau.

- [x] Cài và kiểm thử horizontal_flip/random_horizontal_flip, rotate_image/random_rotation và build_augmentation.
- [x] Ánh xạ pixel flip/rotation 0°, ±90°/180°; ảnh không vuông, border REFLECT_101,
  INTER_LINEAR, shape/dtype, opaque alpha=255 và không mutation.
- [x] Xác suất/góc hợp lệ và invalid/NaN/Inf; kiểm thử nhánh bằng mock, không dùng test xác suất flaky.
- [x] Pipeline Flip → Rotation → Preprocessing; RGB tensor float32 [3,64,64], finite [0,1].
- [x] Tái lập bằng Python random.seed cùng seed shuffle của DataLoader; không nhầm hai RNG.
- [x] Integration batch thật: augmentation chỉ Train, Val/Test deterministic, custom transforms
  đúng tuyến, nhãn không đổi, shape ảnh [32,3,64,64], target [32,1].
- [x] Hình `results/figures/augmentation_comparison.png` tồn tại và đã kiểm tra trực quan.
- [x] Pytest độc lập: **59 PASS, 0 FAIL, 0 SKIP, 0 ERROR**, 2.60s.
- [ ] Đưa code minh họa vào `03_augmentation.ipynb` (hiện còn TODO) để tái lập artifact.
- [ ] Thí nghiệm huấn luyện có/không augmentation: thực hiện ở M7 sau M3–M6.

Definition of Done: **M2 đạt kiểm chứng kỹ thuật** cho hai kỹ thuật, pipeline và tích hợp
DataLoader CPU/num_workers=0; đã có hình minh chứng, Val/Test không bị augmentation.
Chưa khẳng định hiệu quả accuracy/hội tụ hoặc hoàn thành notebook thực nghiệm.
Báo cáo: `results/metrics/m2_test_report.txt`, `results/metrics/m2_test_summary.json`.
Không có lỗi thuật toán bắt buộc cần sửa qua bộ test này. Trước các run training, seed
Python random một lần đầu run; config.seed trong build_loaders chỉ seed torch shuffle.
Giới hạn: integration kiểm tra các batch đầu trên dữ liệu thật, chưa kiểm tra multiprocessing
hoặc ảnh có transparency; không sửa augmentation.py/preprocessing.py/dataset.py trong lượt này.

## M3 — Custom Activation Functions

Dependencies: M0; M5/M6 để chạy so sánh. Files: activations.py, test_activations.py,
experiments/02_activation.ipynb.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn/nn_simple/ffnn.py:sigmoid_derivative,backward`;
`references/lecturer/pytorch_ffnn2/function_simple/activation.py`, `activation_prime.py`,
`references/lecturer/pytorch_ffnn2/layer_simple/act_layer.py:ActivationLayer.backward`;
`references/lecturer/pytorch_ffnn2_image/function_image/activation_prime.py` (đã sửa Sigmoid).
Đích: `source/activations.py`, `source/tests/test_activations.py`, `source/experiments/02_activation.ipynb`.
Bản simple truyền x vào hàm yêu cầu sigmoid(x); không sao chép lỗi hợp đồng này.

- [x] ReLU và stable Sigmoid tự cài; derivative nhận preactivation x; ReLU derivative(0)=0.
- [x] Forward/reference tests: scalar/vector/matrix/batch, CPU float32/float64, shape/dtype/device,
  không mutation; sigmoid(0)=0.5, range, symmetry và monotonicity.
- [x] Đạo hàm đối chiếu PyTorch autograd và numerical derivative; sigmoid derivative(0)=0.25.
- [x] Custom backward thực sự gọi activation_derivative; chain rule với upstream khác 1.
- [x] Gradcheck float64 PASS cho ReLU và Sigmoid; ReLU tránh điểm không khả vi x=0.
- [x] Forward và gradient hữu hạn tại ±100/±1000, cả direct forward autograd và custom Function.
- [x] nn.Sequential + nn.Linear: gradient truyền tới weight/bias; optimizer cập nhật tham số.
- [x] Unsupported names và non-Tensor inputs được từ chối trong các test đã chạy.
- [x] Pytest độc lập: **100 PASS, 0 FAIL, 0 SKIP, 0 ERROR**, 1.59s (lượt cuối).
- [ ] Lưu notebook trực quan hóa/gradcheck đã làm: `02_activation.ipynb` hiện còn TODO.

Definition of Done: **M3 hoàn thành kỹ thuật** theo hợp đồng CPU float32/float64 và gradient
bậc một đã kiểm chứng; không dùng built-in thay implementation, chỉ dùng làm reference tests.
Sai số analytical/numerical Sigmoid tối đa: float32 4.109e-08,
float64 3.494e-11, finite-difference epsilon=1e-6.
Báo cáo: `results/metrics/m3_test_report.txt`, `results/metrics/m3_test_summary.json`.
Không có lỗi bắt buộc cần sửa qua bộ test. CUDA không khả dụng nên chưa kiểm thử GPU;
không tuyên bố hỗ trợ float16/integer/complex, input NaN/Inf, double backward hoặc compile.
Cần lưu đúng notebook để hoàn thiện minh chứng, tách biệt với kết quả pytest độc lập.
Không sửa activations.py hoặc triển khai M4 trong lượt nghiệm thu này.

## M4 — Weight Initialization

Dependencies: M3 để chọn scaling phù hợp; M5/M6 để so sánh.
Files: initialization.py, test_initialization.py, experiments/01_initialization.ipynb.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn/nn_simple/ffnn.py:FFNeuralNetwork.__init__`,
`references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:FFNeuralNetwork.__init__`,
`references/lecturer/pytorch_ffnn2/layer_simple/fc_layer.py:FCLayer.__init__` (Normal std=1),
`references/lecturer/pytorch_ffnn2_image/layer_image/fc_layer.py:FCLayer.__init__`
(thực tế std=1/sqrt(fan_in), comment Xavier/He không chính xác).
Đích: `source/initialization.py`, `source/tests/test_initialization.py`, `source/experiments/01_initialization.ipynb`.

- [x] Bốn phương pháp Normal/Fan-in/Xavier/He đúng variance 1, 1/fan_in, 2/(fan_in+fan_out), 2/fan_in.
- [x] Mean/std trên 6 kích thước (512→512, 512→128, 128→512, 12288→128, 128→64, 64→1),
  CPU float32/float64; tolerance theo số mẫu; matched-draw scaling xác minh chiều fan của nn.Linear.
- [x] Bias=0, bias=False, giữ Parameter/storage/shape/dtype/device/requires_grad; no_grad và leaf Parameters.
- [x] Cùng seed tái lập, khác seed khác weight, không reset seed nội bộ; layer liên tiếp không trùng weight.
- [x] Unsupported/empty/invalid hashable method bị từ chối; non-Linear no-op và không tự đệ quy.
- [x] Signal propagation 10×Linear(512,512)+Custom ReLU: Normal tăng mạnh, He giữ moment tương đối ổn định.
- [x] Fan-in và Xavier trùng chính xác weight và moment với layer vuông, cả float32/64 (max difference=0).
  Layer không vuông khác đúng tỷ lệ moment Xavier/Fan-in: 1.6 (512→128), 0.4 (128→512).
- [x] model.apply + Sequential + CustomActivation: forward/backward, gradient weight/bias và optimizer
  tạo trước initialization vẫn cập nhật đúng Parameter.
- [x] Pytest độc lập: **128 PASS, 0 FAIL, 0 SKIP, 0 ERROR**, 3.68s.
- [ ] M7: so sánh training loss/hội tụ; đổi init hidden, giữ output Xavier gain=1 và bias=0,
  baseline hidden Normal, giữ nguyên các yếu tố khác. Chưa chạy training trong nghiệm thu M4.

Definition of Done: **M4 hoàn thành implementation và kiểm thử độc lập** trong phạm vi
CPU float32/float64; so sánh training loss và hội tụ thuộc M7 sau M5/M6.
Không phát hiện lỗi thuật toán bắt buộc sửa. CUDA không khả dụng nên GPU chưa kiểm chứng.
Float32 seed42: input moment=1.00917; sau 10 layer Normal=1.13814e24,
Fan-in=Xavier=0.000919381, He=0.941446; đây là signal probe, không phải kết quả training.
Báo cáo: `results/metrics/m4_test_report.txt`, `results/metrics/m4_test_summary.json`
(chứa mean/std expected/actual cho từng trường hợp và toàn bộ moment theo layer).
Notebook đã có code/output minh họa nhưng execution_count=None và tiêu đề còn TODO;
không chạy lại notebook, không dùng output notebook làm bằng chứng PASS của pytest.
Không sửa initialization.py, activations.py, config, dữ liệu hoặc reference; không triển khai M5.

## M5 — MLP Architecture

Dependencies: M1, M3, M4. Files: model.py, test_model.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:forward`;
`references/lecturer/pytorch_ffnn2/layer_simple/fc_layer.py:FCLayer.forward,backward`;
`references/lecturer/pytorch_ffnn2_image/layer_image/fc_layer.py:FCLayer`;
`references/lecturer/pytorch_ffnn2_image/network_image/network.py:Network.add,predict`.
Đích: `source/model.py`, `source/tests/test_model.py`. Dùng nn.Linear/Sequential hoặc
ModuleList để đăng ký parameters; không giữ raw Tensor weights như demo, không ghi đè train().

- [x] MLP 12288 → 128 → 64 → 1, registered Sequential/Parameters; hidden ReLU/Sigmoid và custom widths.
- [x] Flatten giữ batch 1/2/32, noncontiguous input, không trộn mẫu/mutation; output logits [B,1] hữu hạn, không sigmoid output.
- [x] 1.581.313 trainable parameters; state_dict đủ sáu tensors, save/load weights_only=True cho forward tương đương.
- [x] initialize_model dispatch đủ bốn hidden methods, output luôn Xavier; bias=0, Parameter/storage identity và seed giữ đúng.
- [x] CustomActivationFunction.backward và activation_derivative thực sự được gọi; BCE gradients tới mọi weight/bias; Adam cập nhật.
- [x] Batch dữ liệu thật, augmentation chỉ Train; Val/Test deterministic; Forward → Loss → Backward → Step hoạt động.
- [x] Single-batch learning: 100 Adam steps, batch32 ảnh thật cố định, seed42. Float32 loss 1.038946 → 0.059332;
  float64 0.792980 → 0.011564. Cả hai giảm >90%; không suy ra generalization từ phép kiểm tra này.
- [x] Pytest M5: **58 PASS, 0 FAIL/SKIP/ERROR**, 9.50s; regression sáu file M1–M5: **425 PASS**, 17.77s.
- [ ] Review hình sơ đồ kiến trúc hiện có để dùng trong báo cáo M8; lượt này kiểm chứng cấu trúc bằng code/test.

Definition of Done: **M5 hoàn thành kỹ thuật** trong phạm vi CPU float32/float64 đã kiểm thử;
CUDA không khả dụng nên GPU chưa kiểm chứng. Không có lỗi bắt buộc sửa của kiến trúc mặc định.
P2 tùy chọn: yêu cầu hidden width >0; validate activation/method ngay entry nếu muốn từ chối
invalid names cả khi hidden_dims rỗng. Hiện zero-width được nn.Linear chấp nhận; tên không dùng
ở mạng không hidden bị bỏ qua. Không thay đổi thuật toán để áp đặt hợp đồng mới.
Báo cáo: `results/metrics/m5_test_report.txt`, `results/metrics/m5_test_summary.json`.
Không huấn luyện toàn dataset, không triển khai M6; notebook/hình hiện có được giữ nguyên.

## M6 — Training & Evaluation

Dependencies: M1–M5. Files: trainer.py, metrics.py, train.py, predict.py, test_metrics.py.

Nguồn tham khảo: `references/lecturer/pytorch_ffnn2/network_simple/network.py:Network.fit`;
`references/lecturer/pytorch_ffnn2_image/network_image/network.py:Network.fit`;
`references/lecturer/pytorch_ffnn2_image/function_image/loss.py`, `loss_prime.py` (MSE);
`references/lecturer/pytorch_ffnn_image/nn_image/ffnn_image.py:train,predict,save_weights,load_weights`.
Đích: `source/trainer.py`, `source/train.py`, `source/predict.py`, `source/metrics.py`, `source/tests/test_metrics.py`.
Reference chưa có BCE/validation/test/metrics; tự triển khai pipeline, không dùng manual update.

- [x] train_epoch/evaluate: train/eval mode, zero_grad/backward/step, custom activation,
  finite loss, empty/nonfinite errors, device transfer, detached scalar results.
- [x] Sample-weighted BCE loss và global metrics; không trung bình F1 theo batch; validation no_grad, deterministic, không cập nhật weights.
- [x] binary_metrics: accuracy/precision/recall/F1 tỷ lệ [0,1], counts TP/TN/FP/FN,
  Human positive, threshold >=, zero division=0; oracle sklearn và invalid inputs.
- [x] fit: đúng thứ tự train/val, đủ history, best theo val loss giảm nghiêm ngặt (giữ earliest tie),
  tạo directories, checkpoint metadata/optimizer state và JSON hợp lệ; epochs invalid bị từ chối.
- [x] CLI: arguments/defaults/validation, seed Python/NumPy/torch, augmentation toggle,
  immutable CONFIG, Adam/BCE, run paths và experiment config; không chọn model bằng Test.
- [x] Inference: config portable theo current root, khôi phục architecture/image size,
  weights_only=True, eval/no_grad, RGB preprocessing, sigmoid threshold0.5, class mapping, lỗi file/split.
- [x] Visualization: dữ liệu curves/marker và confusion [[TN,FP],[FN,TP]], trục True/Predicted,
  PNG hợp lệ, tạo directory và đóng figure; dùng Agg.
- [x] Smoke pipeline: 8 ảnh Train + 6 Validation thật, 2 epoch, test-local config nhỏ;
  restore validation loss khớp history **0.6937192678451538**, có prediction/metrics/hình trong tmp_path.
- [x] M6 pytest: **104 PASS, 0 FAIL/SKIP/ERROR**, 11.26s.
- [x] Regression toàn bộ source/tests/: **529 PASS, 0 FAIL/SKIP/ERROR**, 20.38s;
  gồm M6, không chạy demo giảng viên.
- [ ] Chạy thực nghiệm chính thức/so sánh ở M7; không dùng smoke để kết luận chất lượng mô hình.
- [ ] Đánh giá Test thật chỉ sau khi chốt mô hình bằng Validation; chưa thực hiện trong nghiệm thu này.

Definition of Done: **M6 hoàn thành kỹ thuật trong phạm vi đã kiểm chứng CPU**, gồm
training/evaluation/inference/checkpoint/visualization. Chưa hoàn thành đánh giá chất lượng
cuối đồ án trên Test, chủ động để sau M7. GPU chưa kiểm chứng vì CUDA không khả dụng.
Không phát hiện lỗi bắt buộc sửa. P2 tùy chọn: ghi rõ criterion mean-reduced BCE/model đã ở
đúng device/fixed inference threshold và contract validation cho dữ liệu visualization ngoài fit().
Báo cáo: `results/metrics/m6_test_report.txt`, `results/metrics/m6_test_summary.json`.
Không sửa thuật toán/config/dataset hoặc artifacts cũ; không chạy M7/30 epoch.
Regression M1–M5 đọc ảnh Test để kiểm tra pipeline cấu trúc; không dùng prediction/metric
trên Test thật để lựa chọn hoặc điều chỉnh model. Smoke M6 chỉ tạo loader Train/Val.

## M7 — Comparative Experiments

Dependencies: M0–M6. Nguồn nguyên lý: LECTURER_REFERENCE_GUIDE.md; không chạy demo giảng viên.
Files: experiments/06_comparative_experiments.ipynb, scripts/verify_experiments.py,
results/{configs,histories,metrics,figures,checkpoints}.

- [x] Đủ E1–E5, mỗi run 30 epoch Train/Val; fixed seed42/batch32/Adam0.001/MLP12288→128→64→1.
- [x] Config/history/checkpoint metadata nhất quán; metrics finite, confusion counts đúng845/182.
- [x] Khôi phục năm checkpoint và đối chiếu Validation. E1 loss lệch1.82e-6 do số học float32/runtime;
  tolerance loss abs/rel1e-5, confusion counts khớp. Không ghi đè loss lịch sử.
- [x] Bảng comparison CSV/JSON lấy train loss và mọi validation metric tại cùng best epoch.
- [x] Protocol: E1/E2/E3 đổi init; E2/E4 đổi activation; E3/E5 đổi augmentation;
  output Xavier cố định. Augmentation chỉ Train theo config/current CLI/regression.
- [x] E5 được chọn bằng Validation Loss0.1031503833 tại epoch24, trước Test; threshold0.5 giữ nguyên.
- [x] Final Test256 ảnh: TN99/FP29/FN12/TP116; accuracy83.984375%, precision0.8,
  recall0.90625, F1=0.8498168498. Giữ nguyên mọi metric Test đã lưu, chỉ bổ sung metadata.
- [x] CSV đủ256 dự đoán, đối chiếu loss/counts; 16 ảnh gốc chọn seed42 theo TN/TP/FP/FN.
- [x] Đủ8 hình220DPI, màu nhất quán, epoch integer, log loss initialization,
  best epoch marker và chữ confusion tương phản; đã mở kiểm tra trực quan.
- [x] Notebook12 section/cell code chạy tuần tự trong tiến trình .venv mới và lưu outputs/hình;
  dùng IPython in-process vì sandbox cấm socket Jupyter, chưa xác minh kernel transport thông thường.
- [x] scripts/verify_experiments.py: metadata/protocol/comparison/final/predictions/seed selection/
  hashes/figures/notebook PASS. Manifest lưu SHA256 inputs/outputs và môi trường kiểm chứng.
- [x] Regression source/tests/: **529 PASS, 0 FAIL/SKIP/ERROR**, 44.29s (pytest terminal).

Definition of Done: **M7 hoàn thành100% về artifact và nghiệm thu kỹ thuật trong workspace**.
Giới hạn: một seed; không có training duration/software version ghi lúc chạy; không có
execution trace lịch sử của augmentation. Không khẳng định nguyên nhân vanishing gradient,
overfitting hoặc distribution shift. Không huấn luyện lại và không tối ưu theo Test.
Báo cáo: results/metrics/m7_verification_report.txt và m7_verification_summary.json.
Chạy lại kiểm chứng: `.venv/bin/python -m pytest source/tests/ -q --junitxml=/tmp/m7-regression.xml`
và `.venv/bin/python scripts/verify_experiments.py --regression-xml /tmp/m7-regression.xml`.
Notebook: `.venv/bin/python scripts/execute_m7_notebook.py` (fresh-process IPython fallback).
Checkpoint được giữ local, không đưa vào Git theo .gitignore; cần cung cấp riêng khi đóng gói.
Git commit/push: **BLOCKED** — .git read-only, không tạo được index.lock; chưa stage/commit/push.
Các thay đổi M5–M6 có sẵn chưa được đưa vào commit M7. Xem báo cáo nghiệm thu để bàn giao.

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
- [ ] Chuẩn bị source code, cập nhật README còn mô tả trạng thái skeleton cũ; xử lý các thay đổi M5–M6 chưa commit.
- [ ] Đóng gói MSSV_th02.zip với doc và toàn bộ mã nguồn chạy được; bỏ venv/cache/raw lớn.
- [ ] Kiểm tra cấu trúc ZIP, đường dẫn tương đối và chạy từ bản giải nén.

Definition of Done: đủ hai định dạng báo cáo, link dữ liệu, source sạch, chạy được từ bản giải nén.

## Phân bổ thời gian

- Ngày 1: lưu bằng chứng Run All/kiểm tra Git, hoàn thành M1–M3; không tải/chia lại dataset.
- Ngày 2: M4–M6; chuyển GPU nếu CPU không đủ thời gian.
- Ngày 3: M7–M8, kiểm tra bài nộp và đóng gói.

Không chuyển sang M1 tự động trong phiên phân tích tham khảo này. Các checklist thuật toán vẫn chưa hoàn thành.
Chỉ chạy unit tests dưới `source/tests`; `references/**/test_*.py` là demo có side effects, không collect như tests.
