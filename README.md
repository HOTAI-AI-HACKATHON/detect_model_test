# 🔍 物件偵測與車損辨識測試專案 (detect_model_test)

本儲存庫整合了兩個核心的機器學習與電腦視覺專案：基於 **Python FastAPI 的 YOLO11m 車損辨識服務**，以及基於 **C++17/ggml 的 locate-anything 開放詞彙物件定位工具**。

---

## 📁 專案總覽

| 專案名稱 | 核心技術 | 主要功能 | 適用場景 / 目標 |
| :--- | :--- | :--- | :--- |
| 🚗 **[YOLO11m 車損辨識服務](./YOLO11m-Car-Damage-Detector-main)** | Python 3.12+, FastAPI, YOLO11m, OpenCV, `uv` | 照片品質初審 + 6 種車損類別辨識 (`dent`, `scratch`, `crack`, `shattered_glass`, `broken_lamp`, `flat_tire`) | 車險理賠自動化、租車還車車況點交檢測、維修接待 API |
| 🎯 **[locate-anything.cpp](./locate-anything.cpp)** | C++17, ggml, CMake, C++ CLI | 輸入任意文字 Prompt (例如: `"person"`, `"car"`) 進行開放詞彙物件定位與 Bounding Box 輸出 | 高效能 CPU/GPU 邊緣端推論、無須 Python/PyTorch 依賴之嵌入式部署 |

---

## 1. 🚗 YOLO11m 車損辨識與照片品質檢測 API

基於 **FastAPI** 構建的 REST API 服務，整合 OpenCV 照片品質檢測與 YOLO11m 深度學習車損辨識模型。

### 🌟 核心特色
- **二階段檢測**：先以 OpenCV 計算模糊度 (`blur_score`) 與亮度 (`mean_brightness`) 進行品質初審，未通過直接駁回；通過後才進行 YOLO11m 模型推論。
- **全 `uv` 套件管理**：使用現代化 Python 工具 `uv` 管理套件依賴 (`pyproject.toml` / `uv.lock`)，提供極速且一致的開發體驗。
- **前端友善**：API 回傳包含 Bounding Box 座標、類別信心度以及繪製標記後的 **Base64** 影像。

### 🚀 快速啟動 (Quick Start with `uv`)

1. **安裝 `uv`** (若電腦尚未安裝)：
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **進入目錄並同步環境**：
   ```bash
   cd YOLO11m-Car-Damage-Detector-main
   uv sync
   ```

3. **執行單機測試腳本**：
   ```bash
   uv run test.py
   ```

4. **啟動 API 伺服器**：
   ```bash
   uv run server.py
   ```
   *造訪 [http://localhost:8000/docs](http://localhost:8000/docs) 即可使用 Swagger UI 測試 API。*

👉 詳細說明與 API 規格請參閱：[YOLO11m README.md](./YOLO11m-Car-Damage-Detector-main/README.md)

---

## 2. 🎯 locate-anything.cpp (C++ 開放詞彙物件定位)

移植自 NVIDIA 的 `LocateAnything-3B` 模型，使用 C++17 及輕量化推論引擎 **ggml** 所構建的高效能 Open-Vocabulary 物件偵測工具。

### 🌟 核心特色
- **開放詞彙偵測 (Open-Vocabulary)**：不限於特定訓練好的類別，可透過自然語言 Prompt 自由指定要定位的目標物件。
- **獨立高效率**：推論時完全不依賴 Python 或 PyTorch 執行環境，支援 CPU 與 GPU (ggml Backend) 加速推論。

### 🚀 快速編譯與使用範例

1. **編譯專案** (需要 CMake 與 C++17 編譯器)：
   ```bash
   cd locate-anything.cpp
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```

2. **執行命令列檢測 (CLI)**：
   ```bash
   ./locate-anything-cli detect \
     --model models/locate-anything-q8_0.gguf \
     --input test.jpg \
     --prompt "Locate all the instances that matches the following description: person</c>car." \
     --annotated out.png
   ```

👉 詳細說明請參閱：[locate-anything.cpp README.md](./locate-anything.cpp/README.md)

---

## 📂 專案目錄結構

```text
detect_model_test/
├── README.md                           # 本檔案 (專案總覽說明)
├── YOLO11m-Car-Damage-Detector-main/   # Python / FastAPI 車損辨識服務
│   ├── server.py                       # FastAPI API 服務主程式
│   ├── test.py                         # 單機測試腳本
│   ├── trained.pt                      # 車損辨識 YOLO11m 權重檔
│   ├── pyproject.toml                  # uv 依賴設定檔
│   ├── uv.lock                         # uv 版本鎖定檔
│   └── README.md                       # YOLO 專案詳細說明
└── locate-anything.cpp/                # C++ / ggml 開放詞彙物件定位工具
    ├── CMakeLists.txt                  # CMake 建置設定
    ├── include/ & src/                 # C++ 原始碼與標頭檔
    └── README.md                       # locate-anything 詳細說明
```
