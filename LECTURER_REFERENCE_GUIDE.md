# Hướng dẫn sử dụng mã tham khảo giảng viên — Lab 02

Ngày đối chiếu: 2026-11-24. Đã đọc toàn bộ **32 file Python** trong bốn project,
bao gồm test/demo, visualization và `__init__.py`; đối chiếu PDF 2 trang, PLAN.md và
skeleton/config hiện tại. Đây là phân tích tĩnh, không phải kết quả chạy thử các demo.
Không import/chạy mã tham khảo, không nạp `weights/NN_weight`, không deserialize checkpoint.
Không sửa `source/`, dữ liệu, split, môi trường hoặc mã tham khảo.

Kết luận: dùng các demo để hiểu affine transform, chain rule, flatten và tổ chức layer;
triển khai bài bằng `nn.Module` + `nn.Linear`, **tự cài activation/backward và initialization**.
Không cần viết lại một engine backprop/optimizer hoặc sao chép cả project demo.
PDF yêu cầu tự cài activation/đạo hàm và metrics, nhưng không yêu cầu tự cài Linear/Adam/BCE.
Custom backward qua `torch.autograd.Function` là phương án đề xuất để đạo hàm tự cài
thực sự được dùng khi training, không chỉ tồn tại dưới dạng một hàm không được gọi.

## Cập nhật M1: OpenCV thay Pillow

Người học đã viết preprocessing closure và bắt đầu Dataset.__init__. Refactor giữ thiết kế
closure, chỉ chuyển hợp đồng sang NumPy uint8 BGR/BGRA → RGB tensor float32 CHW [0,1],
contiguous. Dùng cvtColor một lần, resize INTER_LINEAR với (width,height); image_size
của project vẫn (height,width). Alpha trong suốt ghép nền trắng để giữ hành vi trước đây.
Lý do: thống nhất công cụ xử lý ảnh với kiến thức OpenCV trong môn học.

Dataset.__getitem__ và build_loaders vẫn chưa triển khai: TODO đọc cv2.imread IMREAD_UNCHANGED,
báo lỗi None kèm path; không convert màu ở Dataset. Dataset.__init__/__len__ giữ nguyên
mức hoàn thiện của người học. Không suy diễn preprocessing tests PASS là toàn M1 PASS.
Pillow vẫn là dependency cho chuẩn bị dữ liệu/EDA/torchvision và baseline so sánh trong notebook.
Không sửa framework giảng viên hoặc thay đổi kiến trúc MLP vì thay thư viện đọc ảnh.

## 1. Chức năng của bốn project

Đường dẫn trong bảng đều tương đối với `references/lecturer/`.

| Project | File/class/function chính | Nguyên lý và giá trị tham khảo | Hạn chế |
|---|---|---|---|
| pytorch_ffnn | `nn_simple/ffnn.py`: `FFNeuralNetwork`, `sigmoid`, `sigmoid_derivative`, `forward`, `backward`, `train`, `predict`, `save_weights`, `load_weights`; `test_ffnn.py` | Mạng 3→4→1, sigmoid hai tầng, nhân ma trận và backprop thủ công toàn bộ 3 mẫu; input số và target điểm số chia 100; in MSE | Không bias, weights chỉ là Tensor; train ghi đè API PyTorch; normalization dự đoán khác training; không split/metrics |
| pytorch_ffnn2 | `layer_simple`: `BaseLayer`, `FCLayer`, `ActivationLayer`; `function_simple`: `Activation`, `ActivationPrime`, `Loss`, `LossPrime`; `network_simple/network.py`: `Network.add/use/predict/predicts/fit`; `test_network_simple.py` | Tách affine/activation/loss/network; forward theo thứ tự layer, backward ngược thứ tự; demo XOR 2→3→1 với tanh, MSE, SGD từng mẫu | Sigmoid derivative sai hợp đồng đầu vào; bias backward chỉ phù hợp batch=1; test chính là dữ liệu train |
| pytorch_ffnn_image | `nn_image/ffnn_image.py`: `FFNeuralNetwork`; `test_ffnn_image.py`; `visualize_ffnn.py:visualize` | Mở rộng bản đầu cho ảnh, flatten giữ batch; 4096→16→1; demo 10 ảnh random grayscale 64×64, target liên tục random; hình input/hidden/output | Không phải phân loại ảnh thật; Normal std=1 dễ bão hòa sigmoid; load không gán weights; heatmap hidden không phải bản đồ không gian ảnh |
| pytorch_ffnn2_image | `layer_image`, `function_image`, `network_image/network.py`; `test_network_image.py`; `visualize_ffnn2.py:visualize` | Bản modular cho ảnh, sửa sigmoid derivative nhận preactivation, scale weight theo fan-in; train từng mẫu; demo 4096→16→1 | Comment initialization không đúng tên; flatten predict gộp batch; bias update vẫn không hỗ trợ mini-batch; code tạo dữ liệu/mạng bị lặp; visualization dùng `weights_only=False` |

### Những chi tiết quan trọng từ mã thật

- `FFNeuralNetwork` kế thừa `nn.Module` nhưng `W1/W2` là `torch.randn(...)`, không phải
  `nn.Parameter` hoặc buffer. `parameters()`/`state_dict()` không quản lý chúng;
  `.to(device)` không tự chuyển chúng. Không thể chỉ thêm Adam mà giữ nguyên thiết kế.
- Hai `FFNeuralNetwork.train(X,y,rate)` che `nn.Module.train(mode=True)`.
  `model.eval()` vốn gọi `train(False)` sẽ không hoạt động với signature này.
  Đưa training sang trainer.py, giữ nguyên API train/eval chuẩn.
- FFNN cập nhật `W += rate * X.T @ delta` với error `y-o`: dấu cộng ở đây phù hợp gradient
  descent cho **half sum squared error**, không phải tự nó sai dấu. MSE được in dùng mean;
  gradient demo thiếu hệ số `2/N` (một output/mẫu) so với loss được in. Batch lớn làm thay đổi
  scale bước cập nhật. Không mang công thức này sang BCE.
- Hai bản modular: `Loss.mse` và `LossPrime.mse_prime` nhất quán nếu tensor cùng shape:
  derivative `2*(prediction-target)/target.numel()`. Cần tránh broadcasting target sai shape.
- `BaseLayer.__index__` ở hai bản modular không phải constructor `__init__`, và `pass`
  không tạo abstract interface bắt buộc. Chỉ tham khảo ý tưởng, không cần kế thừa lớp này.
- `test_ffnn.py:10–20` dùng max từng feature khi train nhưng max của cả vector mới khi predict;
  scaler không nhất quán, có thể chia 0 nếu feature có max=0. Với ảnh dùng quy tắc cố định /255.
- `test_network_image.py:8–53` khai báo dữ liệu/mạng hai lần; lần sau thay thế lần trước.
  Tên biến batch_size=10 ở đây là **số mẫu demo**, không có DataLoader mini-batch.
- `test_ffnn_image.py` comment 8×8 nhưng config là 64×64; `visualize_ffnn.py:41` nhãn 8×8
  cũng không khớp. Không suy luận kích thước từ comment.
- Các file `test_*.py` là script demo có side effect ở top level, không phải pytest unit tests.
  Không chạy pytest toàn repo để tránh collect/train/save các demo; dùng `python -m pytest source/tests`.

## 2. Bảng đối chiếu M1–M8: nguồn → đích

Tiền tố `R/` dưới đây viết tắt cho `references/lecturer/`; đích tính từ root project.

| Module | Nguồn chính xác và thành phần | Đích hiện tại | Tham khảo / cần tự làm |
|---|---|---|---|
| M1 | `R/pytorch_ffnn_image/nn_image/ffnn_image.py:57–60` (`forward`); `R/pytorch_ffnn2_image/network_image/network.py:49–52` (`fit`); `R/pytorch_ffnn_image/test_ffnn_image.py:8–18` | `source/dataset.py`, `source/preprocessing.py`, `source/tests/test_dataset.py` | Hiểu NCHW/flatten, thay random tensors bằng CSV/OpenCV/Dataset/DataLoader; kiểm tra Alpha, RGB, resize, /255. Flatten chính thức để ở model M5 |
| M2 | Không project nào có augmentation; `R/pytorch_ffnn_image/visualize_ffnn.py:36–48` (`visualize`) chỉ gợi ý bố cục hình | `source/augmentation.py`, `source/experiments/03_augmentation.ipynb` | Tự cài ít nhất flip và rotation; train-only, seed, fill policy và lưới cùng ảnh trước/sau |
| M3 | `R/pytorch_ffnn/nn_simple/ffnn.py:8–14,67–73`; `R/pytorch_ffnn2/function_simple/activation.py`, `activation_prime.py`; `R/pytorch_ffnn2/layer_simple/act_layer.py:12–19`; `R/pytorch_ffnn2_image/function_image/activation.py`, `activation_prime.py`; `R/pytorch_ffnn2_image/layer_image/act_layer.py` | `source/activations.py`, `source/tests/test_activations.py`, `source/experiments/02_activation.ipynb` | Chain rule, phân biệt x và sigmoid(x); tự cài ReLU/Sigmoid, custom Function.backward, gradcheck |
| M4 | `R/pytorch_ffnn/nn_simple/ffnn.py:29–31`; `R/pytorch_ffnn_image/nn_image/ffnn_image.py:29–31`; `R/pytorch_ffnn2/layer_simple/fc_layer.py:11–12`; `R/pytorch_ffnn2_image/layer_image/fc_layer.py:11–13` (`FCLayer.__init__`) | `source/initialization.py`, `source/tests/test_initialization.py`, `source/experiments/01_initialization.ipynb` | Normal baseline và fan-in scaling; tự cài Xavier/He đúng công thức, thống nhất bias, gain và chiều weight |
| M5 | `R/pytorch_ffnn2/layer_simple/fc_layer.py:14–26`; `R/pytorch_ffnn2_image/layer_image/fc_layer.py:15–27`; `R/pytorch_ffnn2_image/network_image/network.py:11–25`; `R/pytorch_ffnn_image/nn_image/ffnn_image.py:57–68` | `source/model.py`, `source/tests/test_model.py` | Affine + activation và thứ tự lớp; chuyển sang nn.Linear/Sequential hoặc ModuleList; 12288→128→64→1 logits |
| M6 | `R/pytorch_ffnn2/network_simple/network.py:38–60`; `R/pytorch_ffnn2_image/network_image/network.py:42–66`; `R/pytorch_ffnn2_image/function_image/loss.py`, `loss_prime.py`; `R/pytorch_ffnn_image/nn_image/ffnn_image.py:87–110` | `source/trainer.py`, `source/train.py`, `source/predict.py`, `source/metrics.py`, `source/tests/test_metrics.py` | Hiểu cycle forward/loss/backward/update; tự viết mini-batch loop/val/test và metrics; BCEWithLogitsLoss+Adam thay MSE/update thủ công; state_dict checkpoint |
| M7 | `R/pytorch_ffnn/test_ffnn.py:25–30`; `R/pytorch_ffnn_image/test_ffnn_image.py:23–30`; `R/pytorch_ffnn2/test_network_simple.py:12–24`; `R/pytorch_ffnn2_image/test_network_image.py:49–58` | `source/experiments/01_initialization.ipynb`, `02_activation.ipynb`, `03_augmentation.ipynb`, `results/histories/`, `results/metrics/` | Demo cách cấu hình/chạy; chưa có comparative experiment hợp lệ. Tự làm kiểm soát seed/split/budget, lưu history và so sánh validation |
| M8 | `R/pytorch_ffnn_image/visualize_ffnn.py:25–60`; `R/pytorch_ffnn2_image/visualize_ffnn2.py:41–91` (`visualize`) | `source/visualization.py`, `results/figures/`, `doc/` | Tham khảo subplot/colorbar, không chạy load model; tự vẽ trước/sau augmentation, sơ đồ MLP và train/val loss; hidden heatmap tùy chọn |

M0 giữ nguyên: 845 train (411/434), 182 val (89/93), 256 test (128/128).
Không có project tham khảo nào thay thế được pipeline Horse or Human đã chuẩn bị.

## 3. Activation và đạo hàm: kiểm tra đúng đối số

Ký hiệu `x` là preactivation, `a = sigmoid(x)` là output.

- Sigmoid: `a = 1/(1+exp(-x))`.
- Derivative theo preactivation: `da/dx = sigmoid(x)*(1-sigmoid(x))`.
- Nếu đã có output `a`: tính tương đương `a*(1-a)`. Đây không phải công thức áp trực tiếp lên x.
- Tanh: `d tanh(x)/dx = 1-tanh(x)^2`; nếu lưu output a thì dùng `1-a^2`.
- ReLU cần tự bổ sung: `max(0,x)`, derivative 1 khi x>0, 0 khi x<0;
  chọn quy ước derivative tại 0 bằng 0 và ghi rõ.

| Vị trí | Hàm derivative nhận gì khi được gọi? | Kết luận |
|---|---|---|
| `pytorch_ffnn/nn_simple/ffnn.py:13–14,69,73` | Output o hoặc z2 sau sigmoid | `s*(1-s)` đúng trong các call site này; tên s chưa rõ |
| `pytorch_ffnn_image/nn_image/ffnn_image.py:13–14,77,81` | Output o hoặc z2 sau sigmoid | Nhất quán như bản số |
| `pytorch_ffnn2/function_simple/activation_prime.py:7–8` qua `layer_simple/act_layer.py:19` | self.in_data, tức x trước sigmoid | **Sai khi chọn Sigmoid**: đang trả `x*(1-x)` |
| `pytorch_ffnn2/function_simple/activation_prime.py:12–13` qua cùng ActivationLayer | x trước tanh | Tanh đúng; demo XOR chỉ dùng tanh nên không bộc lộ lỗi Sigmoid |
| `pytorch_ffnn2_image/function_image/activation_prime.py:7–9` qua `layer_image/act_layer.py:19` | x trước sigmoid, hàm tự tính sigmoid(x) | Đúng; đã sửa so với bản simple |

Phản ví dụ đại số cho lỗi simple: x=0 thì derivative đúng là 0,25 nhưng hàm trả 0;
x=2 thì derivative Sigmoid phải dương nhưng `x*(1-x)=-2`. Đây là suy luận từ công thức,
không phải tuyên bố đã chạy test giảng viên.

Hợp đồng đề xuất cho `source/activations.py`: `activation_derivative(x, name)` luôn nhận
preactivation. Custom Function có thể lưu output sigmoid để backward dùng `a*(1-a)`;
tên biến/docstring phải nói rõ khác biệt. Backward trả `grad_output * derivative`, cùng
shape/device/dtype, không tự chia batch lần nữa, không cập nhật weights ở activation.

Sigmoid trực tiếp `exp(-x)` có thể overflow khi x âm lớn. Khi tự cài, dùng cách ổn định
`z=exp(-abs(x))`, chọn `1/(1+z)` nếu x>=0, `z/(1+z)` nếu x<0. Tránh cách torch.where
với hai nhánh exp không an toàn vì cả hai nhánh có thể được tính. Đây là đề xuất công thức,
chưa sửa implementation. Kiểm tra ±100, 0, đầu vào thông thường, không NaN/Inf;
so sánh forward/gradient với torch reference và gradcheck float64, tránh x=0 của ReLU.
Kiểm tra cả loss.backward end-to-end để chắc custom backward thực sự được gọi.

## 4. Initialization: phân biệt bằng phương sai

Ký hiệu `Z ~ N(0,1)`, fan_in là số đầu vào, fan_out là số đầu ra. Mọi phương pháp dưới đây
có mean 0; tham số thứ hai trong N(0, variance) là **variance**, không phải std.

| Phương pháp | Công thức weights | Std | Variance | Có trong reference? |
|---|---|---|---|---|
| Random Normal baseline | `W=Z` | 1 | 1 | Hai FFNeuralNetwork và FCLayer simple |
| Fan-in Normal (LeCun-style) | `W=Z/sqrt(fan_in)` | sqrt(1/fan_in) | 1/fan_in | FCLayer image |
| Xavier Normal, gain=1 | `W=Z*sqrt(2/(fan_in+fan_out))` | sqrt(2/(fan_in+fan_out)) | 2/(fan_in+fan_out) | Chưa có công thức tổng quát này |
| He Normal cho ReLU, fan_in | `W=Z*sqrt(2/fan_in)` | sqrt(2/fan_in) | 2/fan_in | Chưa có |

Vì thế comment `# Xavier/He initialization` trong `pytorch_ffnn2_image/layer_image/fc_layer.py:11`
không mô tả đúng công thức dòng 12. Fan-in Normal chỉ trùng Xavier gain=1 khi fan_in=fan_out;
không phải He cho ReLU. Với gain khác 1, Xavier variance nhân thêm gain²; phải ghi gain
trong cấu hình thí nghiệm. Tanh có thể dùng gain 5/3, nhưng không gọi mọi scaling là Xavier.

`nn.Linear.weight` có shape `[fan_out,fan_in]`, tính `X @ W.T + b`; reference lưu
`[fan_in,fan_out]`, tính `X @ W + b`. Khi chuyển, không suy fan_in từ chiều đầu như reference.
Với Linear dùng in_features/out_features hoặc weight.shape[1]/shape[0].

Default `nn.Linear` trong bản PyTorch cài sẵn dùng Uniform với biên ±1/sqrt(fan_in)
(thực hiện qua kaiming_uniform_ với a=sqrt(5)); **không phải** Random Normal std=1 và
không phải He Normal cho ReLU. Baseline theo đề phải được gán rõ `N(0,1)`, không dùng
constructor mặc định rồi đặt tên “random normal”.

Đề xuất M4: tự tính std rồi lấy mẫu normal trong `torch.no_grad()`; giữ Parameter hiện có,
không thay bằng Tensor mới và không dùng `.data`. Bias=0 cho mọi phương pháp để so sánh
chỉ thay weight scaling (reference modular dùng random bias, hai bản FFNN không có bias).
Đây là thích nghi có chủ đích, không tuyên bố tái hiện chính xác demo.

Mặc định thực nghiệm: He cho hidden ReLU, Xavier gain=1 cho output tuyến tính. Trong
thí nghiệm initialization, đổi hidden initialization giữa Normal/Xavier/He, giữ output
Xavier và bias=0; ghi rõ phạm vi để baseline không bị hiểu nhầm là mọi layer đều Normal.
Fan-in Normal là nhánh thêm nếu còn thời gian, không thay thế yêu cầu hai phương pháp mới.
Nếu chọn so sánh toàn bộ layer thay vì hidden-only, phải khai báo riêng protocol trước khi chạy.
Test cần kiểm tra mean/std với tolerance thống kê phù hợp, seed, bias, shape, dtype/device;
so sánh reference torch.nn.init để đối chiếu, không dùng nó thay toàn bộ phần tự cài.

## 5. Kiến trúc đề xuất với custom backward

Luồng forward dự kiến:

```text
Ảnh RGBA 300×300 → xử lý Alpha → RGB → augmentation chỉ train
→ resize/tensor [B,3,64,64] trong [0,1]
→ flatten(start_dim=1) [B,12288]
→ Linear(12288,128) → CustomActivation
→ Linear(128,64) → CustomActivation
→ Linear(64,1) → logits [B,1] → BCEWithLogitsLoss
```

- MLP là nn.Module, chứa nn.Linear và activation nn.Module qua Sequential/ModuleList
  hoặc thuộc tính trực tiếp. Không lưu module trong list Python thường nếu cần đăng ký parameters.
- Mỗi activation module gọi `CustomFunction.apply(x)`; `torch.autograd.Function.forward`
  lưu tensors cần thiết bằng `ctx.save_for_backward`, backward nhận gradient từ lớp kế tiếp.
- Chỉ custom backward của activation; autograd xử lý Linear, BCE và nối chain rule;
  Adam xử lý cập nhật parameters. Không đồng thời gọi backward/update thủ công của reference.
- Với Sigmoid lưu a; với ReLU lưu x hoặc mask. Chỉ cam kết first-order gradient ở giai đoạn
  này; cần thêm kiểm thử riêng nếu muốn hỗ trợ đạo hàm bậc cao.
- Giữ các hàm forward/derivative công khai trong skeleton để học/kiểm thử công thức,
  chia sẻ logic với Function nhằm tránh hai bản công thức sai khác.
- Đầu ra một logit, không activation cuối. `num_classes=2` vẫn đúng cho bài toán nhị phân
  dùng BCE; không tạo hai logits rồi đưa vào BCE với target [B,1].
- Không ghi đè `.train()`; dùng `train_epoch()` trong trainer.py.
- Khởi tạo weights một lần trước training, tạo optimizer sau khi model/init/device đã sẵn sàng.

Điều kiện nghiệm thu M5: registered parameters không rỗng, gradient finite qua toàn mạng,
shape [B,1] với cả B=1 và B=32; flatten không gộp các ảnh. MLP này có 1.581.313 parameters
(kể cả bias), tính từ kích thước layer, không phải kết quả benchmark. Với Adam cần thêm
weights/gradients và hai moment buffers; không hứa thời gian huấn luyện khi chưa đo CPU.

## 6. Chuyển sang mini-batch, BCE, Validation và Test

### Mini-batch và gradient

`Network.fit` modular hiện update từng mẫu theo thứ tự cố định. Nếu chỉ thay input thành
batch B, `bias -= rate*out_error` ở FCLayer sẽ cố đưa tensor [B,out] vào bias [1,out] và
không hợp lệ với B>1. Công thức affine thủ công đúng (W theo chiều reference) là:

- `dX = G @ W.T`
- `dW = X.T @ G`
- `db = G.sum(dim=0, keepdim=True)`

Nếu loss đã mean theo batch, G đã chứa hệ số 1/B; không chia thêm ở mỗi layer.
Với nn.Linear, autograd làm đúng phép cộng gradient bias này. `Network.predict` bản image
sử dụng `view(1,-1)` cho input dim>2, nên gộp cả batch nếu truyền B>1. Dùng flatten giữ
chiều đầu; xử lý ảnh đơn bằng thêm batch dimension. `flatten`/`reshape` phù hợp hơn `view`
khi tensor có thể không contiguous sau biến đổi.

Trong M6: lặp DataLoader batch_size=32 (845 ảnh → 27 batches, batch cuối 13 nếu drop_last=False),
train shuffle với generator seed 42. Mỗi batch: chuyển device → zero_grad → forward → loss
→ backward → optimizer.step. Epoch loss = tổng `loss.item()*actual_batch_size` chia tổng mẫu,
không lấy trung bình không trọng số của batch losses. Không retain_graph giữa các batch.

### Binary Cross Entropy

Với y∈{0,1}, p=sigmoid(z), BCE là `-[y*log(p)+(1-y)*log(1-p)]`.
Dạng logits ổn định theo từng mẫu là `max(z,0)-z*y+log(1+exp(-abs(z)))`.
Với mean trên B mẫu một output, derivative theo logit là `(sigmoid(z)-y)/B`.
Không nhân thêm sigmoid derivative như công thức MSE của demo.

Dùng `nn.BCEWithLogitsLoss` để tính ổn định thay vì tự log(sigmoid(z)). Target float32 [B,1]
phải cùng shape logits; không squeeze mất chiều batch khi B=1. Chỉ sigmoid cho probability
khi metrics/inference; threshold mặc định p>=0,5 (tương đương z>=0).
Custom Sigmoid trong bài là activation **hidden** ở một nhánh so sánh, không buộc thêm
custom Sigmoid trước BCEWithLogitsLoss. Không cần pos_weight mặc định khi phân bố lớp gần cân bằng;
nếu thay đổi phải ghi rationale và cấu hình, không chọn dựa vào test.

### Validation/Test và metric tự cài

- Không tạo lại split. Train 845, validation 182, test 256; giữ test nguồn chính thức.
- Mỗi epoch: model.train() trên train; model.eval() + no_grad trên validation;
  không optimizer step, không augmentation ngẫu nhiên ở validation/test.
- Chọn checkpoint bằng validation loss thấp nhất, test chỉ đánh giá sau khi chốt phương án.
  Không chọn activation/initialization hoặc threshold từ kết quả test.
- Tự tính TP/TN/FP/FN trên toàn tập; accuracy=100*(TP+TN)/N, precision=TP/(TP+FP),
  recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Quy định denominator=0 trả 0 và ghi rõ.
  Human=positive=1. Không trung bình F1 theo batch; aggregate counts rồi tính.
- Reference chưa có các metric này. Dùng sklearn làm oracle trong tests, không thay phần tự cài.
- Tránh giữ graph khi lưu history; lưu số scalar, config, seed, epoch, phiên bản và thời gian.

### Lưu/nạp model

`FFNeuralNetwork.save_weights` hai bản lưu cả model; `load_weights` bỏ kết quả torch.load,
không trả object hoặc gán state vào instance. Vì vậy visualization bản image có thể vẽ model
vừa random-init dù in thông báo đã nạp thành công. Cơ chế allowlist của bản image không sửa
được lỗi bỏ kết quả này. Không đánh giá chất lượng pretrained file hiện có từ mã nguồn alone.

`visualize_ffnn2.py:32` gọi `torch.load(..., weights_only=False)`; không chạy script đó.
Trong dự án mới, lưu tensor-only model.state_dict(), lưu config/history riêng JSON,
khởi tạo đúng architecture rồi load_state_dict với `torch.load(..., weights_only=True,
map_location=device)`. Không allowlist các class demo hoặc dùng weights_only=False để
“sửa lỗi” nạp. Không sử dụng checkpoint giảng viên: khác input channels/kích thước/kiến trúc
và chưa xác minh nguồn serialization. Toàn bộ phân tích này không nạp bất kỳ weights nào.

## 7. Từ dữ liệu giả lập sang Horse or Human

| Demo | Dữ liệu/project thật | Thay đổi cần làm |
|---|---|---|
| 3 mẫu điểm số, XOR hoặc 10 random grayscale images | 1.283 ảnh PNG RGBA 300×300, nhãn 0/1 | Dataset đọc CSV, mở file lazy và xử lý lỗi rõ ràng |
| torch.rand labels liên tục độc lập input | Horse/Human có ý nghĩa ngữ nghĩa | Không coi loss demo giảm là bằng chứng phân lớp hoặc tổng quát hóa |
| 1×64×64 = 4096 features | 3×64×64 = 12288 sau chuyển RGB | Kiểm tra Alpha trước; không giữ 4 channels thành 16384 hoặc đọc grayscale |
| Không có Alpha | RGBA, đã xác nhận 1.283 ảnh alpha=255 | OpenCV đọc BGRA; nếu gặp transparency mới thì composite nền trắng trước resize; thống nhất train/eval/inference |
| Random tensor đã [0,1] | PNG thường đọc uint8 | RGB/resize/CHW/float32 /255; không /255 hai lần |
| Tensor liên tục và đặt sẵn toàn bộ trong RAM | Pipeline OpenCV, augmentation và batching | Lazy load, workers=0, batch=32, kiểm tra batch cuối và tensor không contiguous |
| Random Normal std=1 trên input 4096 | Input chiều lớn, sigmoid có thể bão hòa | Theo dõi saturation/gradient, initialization đúng; không coi baseline kém là bug tất yếu |
| Không split/augmentation, predict train hoặc noise | CSV cố định + validation + official test | Không leakage, không sửa raw; tối thiểu hai augmentation chỉ train |
| Hidden 16 reshape 4×4 | Hidden 128/64 | Nếu vẽ activation, dùng bar chart hoặc grid ghi rõ là vector neuron, không phải spatial explanation |
| .numpy() trực tiếp CPU | Có thể chạy CUDA | .detach().cpu().numpy(), không giữ graph để vẽ |

Ảnh CGI có thể cùng nhân vật/tư thế/nền tương tự; không duplicate file không có nghĩa độc lập
ngữ nghĩa hoặc tương đương ảnh chụp thực. Không sửa split đã chốt để giải quyết nhận xét này;
ghi giới hạn trong báo cáo. MLP flatten không khai thác cấu trúc không gian như CNN,
nhưng phù hợp yêu cầu bài; không đổi sang CNN/pretrained backbone để tiết kiệm thời gian.

## 8. Nội dung bắt buộc tự triển khai và lịch 3 ngày

| Ngày | Ưu tiên | Điều kiện nghiệm thu |
|---|---|---|
| 1 | M1 pipeline/Alpha, M2 flip+rotation, M3 ReLU+Sigmoid/custom backward | Batch RGB đúng, cùng hợp đồng derivative, gradcheck/reference tests; không sửa split |
| 2 | M4 Normal/Xavier/He, M5 nn.Linear MLP, M6 Adam/BCE/validation/metrics | Init đúng variance, parameters/gradients hợp lệ, train/val history và checkpoint round-trip |
| 3 | M7 so sánh có kiểm soát, M8 hình/báo cáo/đóng gói/Drive Backup | Kết quả thật, test cuối, báo cáo editable+PDF, link dữ liệu dự phòng truy cập được |

Để giảm số lần huấn luyện: dùng một cấu hình chuẩn (ReLU + He hidden + output Xavier,
không augmentation) làm mốc dùng chung; so Normal/Xavier/He với cùng activation, so
ReLU/Sigmoid với **cùng initializer Xavier** để chỉ thay activation, so có/không
augmentation với cùng model/init. Nhánh ReLU+Xavier dùng lại nếu mọi điều kiện giống hệt.
Có thể khoảng 5 cấu hình độc lập thay vì full factorial. Giữ 30 epochs/budget thống nhất;
1 seed chỉ là so sánh giới hạn, không khẳng định ý nghĩa thống kê. Không giả lập kết quả.

PDF yêu cầu tối thiểu: 2 augmentation; 2 activation tự cài kèm đạo hàm; 2 initialization
ngoài baseline; accuracy (%) + ít nhất 2 metric khác; grid trước/sau augmentation,
sơ đồ mạng, loss train/validation; báo cáo có tự đánh giá 0–100%, editable+PDF và backup
Google Drive public cố định sau deadline. Reference chưa cung cấp đủ các phần này.

Cần tự giải thích được: vì sao gradient đi ngược lớp; khác x và sigmoid(x); khác std và
variance; vì sao fan-in lớn cần scaling; vì sao bias gradient cộng theo batch; vì sao
BCE nhận logits; vì sao validation được dùng để chọn nhưng test không được dùng để tune.
Ưu tiên các hiểu biết/kiểm thử này hơn xây dựng framework tùy biến hoặc vẽ hidden heatmap.

## 9. Phạm vi đọc và nguồn kiểm chứng

- pytorch_ffnn: 3 file Python (ffnn, demo, init rỗng).
- pytorch_ffnn2: 12 file Python (network + init; 3 layer + init; 4 function + init; demo).
- pytorch_ffnn_image: 4 file Python (ffnn_image + init rỗng; demo; visualize).
- pytorch_ffnn2_image: 13 file Python (network + init; 3 layer + init; 4 function + init;
  demo; visualize). Các init không rỗng chỉ re-export các lớp, không có thuật toán riêng.
- Artifact `pytorch_ffnn_image/weights/NN_weight` được giữ nguyên; không được load/unpickle.
- Nguồn API đối chiếu: bản torch cài sẵn, `torch/nn/modules/linear.py:117–124`,
  `torch/nn/init.py:xavier_normal_,kaiming_normal_`, `torch/nn/modules/loss.py:BCEWithLogitsLoss`,
  `torch/autograd/function.py:Function,FunctionCtx.save_for_backward`.
  Tra cứu web tài liệu version 2.13 không truy cập được trong phiên này; không coi trang
  redirect stable là bằng chứng về version máy. Công thức/nhận xét chính dựa trên source đã đọc.
- Không báo cáo PASS runtime cho bất kỳ reference nào. Các phép đối chiếu công thức là
  phân tích tĩnh; test đề xuất ở trên thuộc những module tiếp theo, chưa được triển khai.
