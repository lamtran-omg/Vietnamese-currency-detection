import cv2
import os
from ultralytics import YOLO

# --- CẤU HÌNH ---
camera = 1  # 0: Chạy ảnh trong thư mục /data | 1: Chạy Webcam
model_path = "models/best_v3.pt"
image_folder = "data" 
target_width = 1280  # Độ rộng mong muốn khi hiển thị

# 1. Load model
model = YOLO(model_path)

# 2. Định nghĩa mệnh giá
class_values = {
    "1000": 1000, "2000": 2000, "5000": 5000,
    "10000": 10000, "20000": 20000, "50000": 50000,
    "100000": 100000, "200000": 200000, "500000": 500000
}

def draw_ui(img, total_amount):
    """Hàm chuyên vẽ giao diện Tổng tiền để đảm bảo chữ luôn đẹp"""
    # Tính toán kích thước chữ dựa trên độ rộng ảnh để không bị quá to/nhỏ
    font_scale = img.shape[1] / 1000.0 
    thickness = max(2, int(font_scale * 2))
    text = f"TONG: {total_amount:,} VND"
    
    # Lấy kích thước khung chữ để vẽ nền đen vừa vặn
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Vẽ nền đen (bo góc nhẹ nếu muốn, ở đây vẽ hình chữ nhật)
    cv2.rectangle(img, (10, 10), (20 + text_w, 20 + text_h + 20), (0, 0, 0), -1)
    # Vẽ chữ xanh
    cv2.putText(img, text, (20, 10 + text_h + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    return img

def predict_and_show(frame, window_name):
    # 1. Dự đoán trên khung hình gốc để giữ độ chính xác
    results = model(frame, conf=0.5, iou=0.8)
    
    # 2. Tính tổng tiền
    current_total = 0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = r.names[cls_id]
            if name in class_values:
                current_total += class_values[name]
    
    # 3. Lấy ảnh đã vẽ các box (Annotated frame)
    annotated_frame = results[0].plot()

    # 4. PHÓNG TO ẢNH TRƯỚC KHI VẼ UI
    h, w = annotated_frame.shape[:2]
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    resized_img = cv2.resize(annotated_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 5. VẼ UI LÊN ẢNH ĐÃ PHÓNG TO (Giúp chữ sắc nét, không bị đè)
    final_img = draw_ui(resized_img, current_total)
    
    cv2.imshow(window_name, final_img)

# --- CHẠY CHƯƠNG TRÌNH ---
if camera == 0:
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]
    
    cv2.namedWindow("Folder Mode", cv2.WINDOW_NORMAL)
    for img_name in images:
        frame = cv2.imread(os.path.join(image_folder, img_name))
        if frame is None: continue
        
        predict_and_show(frame, "Folder Mode")
        if cv2.waitKey(0) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

elif camera == 1:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)  # Lật ngang camera (như gương)
        predict_and_show(frame, "Webcam Mode")
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()