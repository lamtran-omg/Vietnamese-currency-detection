# Vietnamese Currency Detector

Dự án phát hiện và nhận diện các mệnh giá tiền Việt Nam sử dụng YOLOv11.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (để sử dụng chức năng detect realtime)
- Windows/Linux/MacOS

## Hướng dẫn cài đặt

### 1. Tạo môi trường ảo (Virtual Environment)

Môi trường ảo giúp cách ly các thư viện của dự án này với các dự án Python khác trên máy bạn.

#### Trên Windows:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
venv\Scripts\activate
```

#### Trên Linux/MacOS:

```bash
# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate
```

**Lưu ý:** Sau khi kích hoạt thành công, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

### 2. Cài đặt các thư viện cần thiết

Sau khi đã kích hoạt môi trường ảo, cài đặt các thư viện:

```bash
# Cập nhật pip
python -m pip install --upgrade pip


# Cài đặt Ultralytics (YOLOv11)
pip install ultralytics
```

**Hoặc** bạn có thể tạo file `requirements.txt` với nội dung:

```txt
ultralytics>=8.0.0
```

Sau đó cài đặt tất cả bằng một lệnh:

```bash
pip install -r requirements.txt
```

### 3. Kiểm tra cài đặt

Kiểm tra xem các thư viện đã được cài đặt thành công:

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics installed successfully')"
python -c "import torch; print('PyTorch:', torch.__version__)"
```

## Cấu trúc dự án

```
Face_Mask_Detection/
├── main.py                              # File chính chạy detection
├── Currency_detection.ipynb             # Jupyter notebook
├── models                               # Các model của dự án
│  ├── best_v1.pt                        # Model YOLOv11(v1) đã train
│  ├── best_v2.pt                        # Model YOLOv11(v2) đã train
│  └── best_v3.pt                        # Model YOLOv11(v3) đã train
├── README.md                            # File hướng dẫn này
└──  requirements.txt                    # Danh sách thư viện 
```

## Cách sử dụng

### Chạy detection qua webcam:

```bash
# Đảm bảo đã kích hoạt môi trường ảo
python main.py
```

Hoặc:

```bash
py main.py
```

- Nhấn phím `q` để thoát chương trình
- Chương trình sẽ tự động tính tổng giá trị các tờ tiền được phát hiện

### Chạy với Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

## Các lệnh thường dùng

### Tắt môi trường ảo:

```bash
deactivate
```

### Xem danh sách thư viện đã cài:

```bash
pip list
```

### Xuất danh sách thư viện ra file:

```bash
pip freeze > requirements.txt
```

## Xử lý lỗi thường gặp

### Lỗi: "Python not found" hoặc "pip not found"

- Đảm bảo Python đã được cài đặt và thêm vào PATH
- Thử dùng `py` thay vì `python` trên Windows

### Lỗi: "No module named 'cv2'" hoặc "No module named 'ultralytics'"

- Đảm bảo bạn đã kích hoạt môi trường ảo
- Cài lại thư viện: `pip install opencv-python ultralytics`

### Lỗi: "Camera not found" hoặc không mở được camera

- Kiểm tra camera đã được kết nối
- Thử thay đổi số camera trong code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Lỗi: "CUDA not available" (GPU)

- Nếu muốn dùng GPU, cài PyTorch với CUDA support
- Nếu chỉ dùng CPU, bỏ qua lỗi này, model vẫn chạy được

## Thông tin thêm

- Model được train để nhận diện các mệnh giá tiền Việt Nam: 1.000đ, 2.000đ, 5.000đ, 10.000đ, 20.000đ, 50.000đ, 100.000đ, 200.000đ, 500.000đ
- Confidence threshold: 0.5
- IOU threshold: 0.8
