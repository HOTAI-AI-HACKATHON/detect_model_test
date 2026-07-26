import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI(title="車損辨識與照片品質檢測 API")

# ==========================================
# 1. 載入 YOLO 模型
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained.pt")  # 請確保 trained.pt 與此 py 檔在同一目錄

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型檔案: {MODEL_PATH}")

print("正在載入 YOLO 車損模型...")
model = YOLO(MODEL_PATH)
print("✅ 模型載入完成！")


# ==========================================
# 2. OpenCV 品質檢測邏輯
# ==========================================
def inspect_image_quality(img, blur_thresh=35.0, min_bright=40, max_bright=220):
    """
    對影像進行清晰度與亮度過濾
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯降噪後計算拉普拉斯變異數 (過濾背景雜訊/紋理騙分)
    denoised_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_score = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()

    # 計算平均亮度
    mean_brightness = float(np.mean(gray))

    metrics = {
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2)
    }

    if blur_score < blur_thresh:
        return False, f"照片過度模糊 (得分: {blur_score:.1f})，請重新對焦", metrics

    if mean_brightness < min_bright:
        return False, f"照片光線太暗 (亮度: {mean_brightness:.1f})，請至光線充足處拍攝", metrics

    if mean_brightness > max_bright:
        return False, f"照片過曝或反光過強 (亮度: {mean_brightness:.1f})，請調整拍攝角度", metrics

    return True, "照片品質良好", metrics


# ==========================================
# 3. API 路由端點 (/predict)
# ==========================================
@app.post("/predict")
async def predict_car_damage(file: UploadFile = File(...)):
    """
    接收 App 上傳的照片 -> 品質檢測 -> YOLO 車損辨識 -> 回傳結果
    """
    # A. 驗證檔案格式
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上傳的檔案必須是圖片格式 (.jpg, .png)")

    # B. 將圖片 Read 進記憶體並轉為 OpenCV 格式
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼圖片，檔案可能損毀")

    # C. 第一關：OpenCV 照片品質檢測
    is_quality_pass, quality_msg, metrics = inspect_image_quality(img)
    if not is_quality_pass:
        return {
            "status": "rejected",
            "message": quality_msg,
            "quality_metrics": metrics,
            "damage_count": 0,
            "predictions": []
        }

    # D. 第二關：YOLO 車損推論
    results = model(
        source=img,
        imgsz=1280,
        conf=0.15,
        iou=0.45,
        augment=True,
        save=False
    )

    result = results[0]
    detections = []

    # E. 解析推論結果 (標籤、信心度、Bounding Box 座標)
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]

        detections.append({
            "label": class_name,
            "confidence": round(confidence, 4),
            "box": [round(x, 2) for x in xyxy]
        })

    # F. 將畫好車損框的圖片轉換為 Base64 字串 (方便前端 App 直接顯示)
    annotated_img = result.plot()
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # G. 回傳成功結果 JSON
    return {
        "status": "success",
        "message": "辨識成功",
        "quality_metrics": metrics,
        "damage_count": len(detections),
        "predictions": detections,
        "annotated_image_base64": f"data:image/jpeg;base64,{base64_image}"
    }


# ==========================================
# 4. 啟動進入點
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # 監聽在 0.0.0.0:8000，讓區網內的所有裝置 (如手機) 都能連進來
    uvicorn.run(app, host="0.0.0.0", port=8000)