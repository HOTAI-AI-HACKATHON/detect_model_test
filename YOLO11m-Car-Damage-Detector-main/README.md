# 🚗 車損辨識與照片品質檢測 API (Car Damage Detector)

本專案提供基於 **FastAPI** 與 **YOLO11m** 的車損辨識 API 服務。結合了 OpenCV 的照片品質預審機制（過濾模糊、過暗或過曝照片），以及 Fine-tuned YOLO11m 模型，用於自動偵測車輛外觀損傷並回傳標註圖像。

專案全面使用 **`uv`** 進行 Python 套件與環境管理，實現極速、一致且跨平台的開發體驗。

---

## 📋 目錄
- [核心功能與特色](#-核心功能與特色)
- [辨識類別](#-辨識類別)
- [環境需求與 uv 安裝](#-環境需求與-uv-安裝)
- [快速開始](#-快速開始)
- [API 使用說明](#-api-使用說明)
- [專案結構](#-專案結構)

---

## 🎯 核心功能與特色

1. **二階段檢測機制**：
   - **第一關（照片品質預審）**：採用 OpenCV 計算拉普拉斯變異數 (`blur_score`) 與平均亮度 (`mean_brightness`)，自動駁回模糊、光線不足或過曝反光的照片，節省模型推論資源。
   - **第二關（YOLO11m 推論）**：通過品質初審的照片將送入車損模型，標記損傷類別、信心度及 Bounding Box 座標。
2. **前端友善**：API 回傳結果包含畫有標記框的圖片之 **Base64** 字串，方便 App / Web 前端直接呈現。
3. **`uv` 套件管理**：免除繁瑣的 Python 虛擬環境手動設定，一鍵同步依賴並鎖定套件版本。

---

## 🏷️ 辨識類別

模型支援以下 6 種常見車損種類：

| 類別標籤 (Label) | 中文說明 |
| :--- | :--- |
| `dent` | 凹痕 |
| `scratch` | 刮痕 |
| `crack` | 裂痕 |
| `shattered_glass` | 車窗/擋風玻璃破碎 |
| `broken_lamp` | 車燈破損 |
| `flat_tire` | 爆胎 |

---

## ⚙️ 環境需求與 uv 安裝

本專案建議使用 **Python 3.10+** 及 **`uv`** 套件管理器。

### 1. 安裝 `uv`

若你的電腦尚未安裝 `uv`，請根據作業系統選擇以下指令安裝：

* **Windows (PowerShell)**：
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **macOS / Linux**：
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **透過 pip 安裝**：
  ```bash
  pip install uv
  ```

> 💡 安裝完成後，請重啟終端機並輸入 `uv --version` 確認安裝成功。

---

## 🚀 快速開始

### 1. 取得專案並進入目錄
```bash
git clone <YOUR_REPOSITORY_URL>
cd YOLO11m-Car-Damage-Detector-main
```

### 2. 安裝與同步專案環境
使用 `uv` 建立虛擬環境並安裝所有依賴套件：
```bash
uv sync
```
*(若為新建專案或加入套件，也可執行 `uv add fastapi uvicorn ultralytics opencv-python numpy pydantic python-multipart`)*

---

### 3. 執行單機測試腳本
在啟動 API 伺服器前，可先執行單機測試確認模型推論正常：
```bash
uv run test.py
```
> 執行後會讀取目錄下的 `test4.jpg` 並輸出繪製標記後的 `result4.jpg`。

---

### 4. 啟動 API 伺服器
執行以下指令啟動 FastAPI 服務：

```bash
uv run server.py
```

或使用 uvicorn 啟動熱重載 (Hot-reload) 模式：
```bash
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

伺服器啟動後，將監聽在 `http://localhost:8000`。

---

## 📡 API 使用說明

### 1. Swagger 互動式 API 文件
啟動服務後，開啟瀏覽器造訪：
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

### 2. 車損預測端點 `/predict`

* **HTTP 方法**：`POST`
* **Content-Type**：`multipart/form-data`
* **請求參數**：
  * `file` (File, 必填)：要檢測的照片（支援 `.jpg`, `.png` 格式）。

---

### 3. 回應範例

#### 成功辨識 (`status: success`)
```json
{
  "status": "success",
  "message": "辨識成功",
  "quality_metrics": {
    "blur_score": 145.2,
    "mean_brightness": 120.5
  },
  "damage_count": 1,
  "predictions": [
    {
      "label": "dent",
      "confidence": 0.8842,
      "box": [125.4, 340.2, 450.1, 610.8]
    }
  ],
  "annotated_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

#### 品質未通過駁回 (`status: rejected`)
```json
{
  "status": "rejected",
  "message": "照片過度模糊 (得分: 18.2)，請重新對焦",
  "quality_metrics": {
    "blur_score": 18.2,
    "mean_brightness": 110.0
  },
  "damage_count": 0,
  "predictions": []
}
```

---

### 4. cURL 與 Python 請求範例

#### cURL：
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test1.jpg"
```

#### Python (`requests`)：
```python
import requests

url = "http://localhost:8000/predict"
file_path = "test1.jpg"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print("辨識結果：", result["message"])
print("發現車損數量：", result["damage_count"])
```

---

## 📁 專案結構

```text
YOLO11m-Car-Damage-Detector-main/
├── server.py              # FastAPI 伺服器主程式 (API 入口)
├── test.py                # 單機推論測試腳本
├── trained.pt             # Fine-tuned YOLO11m 模型權重檔
├── pyproject.toml         # uv 專案設定檔與套件依賴清單
├── uv.lock                # uv 版本鎖定檔
├── README.md              # 專案說明文件
└── public/                # 範例圖片目錄
```

---

## 🛠️ 開發說明與補充

* **照片品質評估標準**（可在 [server.py](file:///c:/Users/jammy/Downloads/detect_model_test/YOLO11m-Car-Damage-Detector-main/server.py) 中微調）：
  * 清晰度門檻 (`blur_thresh`)：預設 `35.0`
  * 最小亮度 (`min_bright`)：預設 `40`
  * 最大亮度 (`max_bright`)：預設 `220`
