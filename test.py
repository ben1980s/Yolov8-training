# 導入所需的模組
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# 初始化 YOLO 模型，並載入你訓練的模型檔案（例如 best.pt）
def load_yolo_model(model_path):
    return YOLO(model_path)

# 進行預測，設定圖片大小 imgsz=640，並將結果串流保存
def predict_with_yolo(model, image_path):
    result = model.predict(image_path, imgsz=640, stream=True, save=True)
    return result

# 顯示預測結果的圖片
def show_predicted_image(predicted_image_path):
    predicted_image = Image.open(predicted_image_path)
    plt.imshow(predicted_image)
    plt.axis('off')  # 隱藏坐標軸
    plt.show()

# 主程式
if __name__ == "__main__":
    model_path = "best.pt"
    source_image_path = "C:/Users/Benson-NB/Desktop/tech/180.jpg"

    model = load_yolo_model(model_path)
    result = predict_with_yolo(model, source_image_path)
    predicted_result = next(result)
    # 顯示預測結果的圖片
    show_predicted_image(predicted_result[0])
