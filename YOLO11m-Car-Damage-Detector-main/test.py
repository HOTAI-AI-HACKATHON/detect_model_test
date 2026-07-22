import os
import cv2
from ultralytics import YOLO

# 取得目前 test.py 檔案所在的資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 載入模型
model_path = os.path.join(BASE_DIR, "trained.pt")
model = YOLO(model_path)

# 2. 設定輸入與輸出檔名
input_path = os.path.join(BASE_DIR, "test1.jpg")
output_path = os.path.join(BASE_DIR, "result.jpg")

# 3. 進行預測（設置 save=False，由我們自己控制儲存）
results = model(
    source=input_path,
    imgsz=1280,             # 高解析度推論
    conf=0.15,              # 信心度門檻（調低以捕捉微小車損）
    iou=0.45,               # 重疊過濾門檻
    augment=True,           # 開啟 TTA 測試時增強，大幅提升精確度
    save=False              # 關閉內建自動儲存，避免覆寫原檔
)

# 4. 取得畫好 Bounding Box 的 numpy 圖像陣列
# results 是一個 list，單張圖片取第一個元素 results[0]
annotated_frame = results[0].plot()

# 5. 將結果圖片另存為 result.jpg
cv2.imwrite(output_path, annotated_frame)

print(f"偵測完成！原始圖片未受影響，結果已另存為：{output_path}")