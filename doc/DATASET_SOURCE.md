# Nguồn và giới hạn dữ liệu

- Dataset: Horse or Human, Laurence Moroney, 2019.
- Catalog: https://www.tensorflow.org/datasets/catalog/horses_or_humans
- Tác giả và giấy phép CC BY 2.0: https://laurencemoroney.com/datasets.html
- Builder xác nhận nguồn ZIP, nhãn horses/humans và official test:
  https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py
- URL tải và cách tải thủ công: README ở root.
- Đây là ảnh CGI. Theo tác giả, ảnh 300×300; đây là thông tin nguồn, chưa phải kết quả
  kiểm kê tại máy. Không mặc định mode RGB cho file gốc; script kiểm tra channels/mode thực tế.
- Dữ liệu thực tế hiện chưa tải được do DNS. Chưa có thống kê kích thước/mode/format đã xác minh.
- Chỉ SHA-256 file: không kiểm tra near-duplicate, nhân vật/cảnh chung, hoặc leakage ngữ nghĩa.
- Khi sử dụng/phân phối dữ liệu, ghi công Laurence Moroney và ghi giấy phép CC BY 2.0.
- TODO theo PDF: link Google Drive public dự phòng, không thay đổi sau deadline. Chưa upload.
