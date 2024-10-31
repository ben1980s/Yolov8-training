import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加載模型
model = YOLO("best.pt")  # 加載你訓練好的 YOLOv8 模型

# 加載圖像
image_path = "1..png"  # 替換為你的圖像路徑
frame = cv2.imread(image_path)

# 在圖像上進行推理
results = model([frame])

for result in results:
    # 獲取邊界框
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # 僅顯示標籤為 "mature" 的框，並將標籤顯示為 "rot"
            if model.names[cls] == "mature":
                label = f"rot {conf:.2f}"
                
                # 繪製邊界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 將 BGR 圖像轉換為 RGB，適用於 Matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 顯示帶有檢測結果的圖像
plt.imshow(frame_rgb)
plt.axis('off')
plt.show()
