# Hướng dẫn Đóng Gói Windows EXE & Tối Ưu GPU Dòng Card 5x (RTX 50-Series)

Tài liệu này hướng dẫn chi tiết cách build toàn bộ dự án thành bộ công cụ độc lập **`ThumbnailPipeline.exe`** trên Windows, tối ưu hóa tốc độ tối đa và đảm bảo tương thích 100% với các dòng card đồ họa mới nhất của NVIDIA, đặc biệt là **GeForce RTX 50-series (RTX 5060, 5070, 5080, 5090 kiến trúc Blackwell)**.

---

## ⚡ CÁCH NHANH NHẤT (1-CLICK BUILD VỚI FILE .BAT)

Trên máy Windows:
1. Sao chép toàn bộ thư mục dự án này sang máy Windows.
2. Click đúp chuột vào file **`build_windows.bat`**.
3. File `.bat` sẽ tự động:
   - Kiểm tra Python.
   - Tạo môi trường ảo `.venv`.
   - Cài đặt đầy đủ thư viện `requirements.txt` và `onnxruntime-directml` (tối ưu 100% cho dòng card RTX 50x / mọi GPU).
   - Tự động copy `ffmpeg.exe` & `ffprobe.exe` nếu tìm thấy trong máy.
   - Chạy PyInstaller đóng gói toàn bộ model AI, template prompt và file launcher.
   - Tạo sẵn các thư mục `input/`, `output/`, `net/` trong bản xuất bản.

---

## 1. Cơ Chế Tương Thích GPU & Dòng Card 5x (Blackwell)

Dòng card **RTX 50-series** sử dụng vi kiến trúc mới **Blackwell (Compute Capability 12.0 / sm_120)**. 
Nếu dùng các phiên bản AI / CUDA cũ (như CUDA 11.x hoặc CUDA 12.1/12.2), bạn sẽ gặp lỗi:
```text
CUDA error: no kernel image is available for execution on the device
```

### Dự án đã được tối ưu hóa như sau:
1. **Kiểm tra thông minh & Tự động Fallback**: 
   Cả `extract_char.py` và `net.py` đều tự động chạy thử một dummy tensor trên GPU ngay lúc khởi động. Nếu CUDA gặp lỗi kiến trúc chưa tương thích, hệ thống **tự động chuyển sang DirectML hoặc CPU mà không bị crash**.
2. **Hỗ trợ DirectML (DirectX 12 Machine Learning)**:
   Đây là giải pháp **hoàn hảo và ổn định nhất** cho dòng card 5x trên Windows. DirectML giao tiếp trực tiếp với phần cứng qua DirectX 12, tận dụng toàn bộ sức mạnh GPU RTX 50x mà không phụ thuộc vào việc CUDA Toolkit đã hỗ trợ `sm_120` hay chưa.
3. **Chạy theo Batch (Batch Inference)**:
   `net.py` được nâng cấp để gom các ô ảnh thành batch (4–8 ô cùng lúc), tăng tốc độ làm nét ảnh trên GPU từ **2x đến 4x** so với bản cũ.
4. **Bộ đệm nạp trước hình ảnh (Prefetching)**:
   `extract_char.py` sử dụng luồng đọc ngầm (background thread) để nạp trước keyframe từ ổ cứng, loại bỏ hoàn toàn hiện tượng nghẽn I/O khi quét mặt nhân vật.

---

## 2. Các Bước Build Bằng Tay (Nếu Không Dùng File .bat)

### Bước 1: Cài đặt Python trên Windows
- Tải và cài đặt **Python 3.12 trở lên** từ [python.org](https://www.python.org/downloads/).
  (Bắt buộc 3.12+ vì `requirements.txt` ghim `numpy==2.5.2`, mà numpy 2.5.x yêu
  cầu `Python >=3.12`. Dùng 3.11 thì `pip install` đổ ngay.)
- ⚠️ **Rất quan trọng**: Nhớ tích chọn ô **`Add python.exe to PATH`** trong lúc cài.

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
Mở PowerShell / CMD tại thư mục dự án:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pyinstaller onnxruntime-directml
```

### Bước 3: Tải `ffmpeg.exe` và `ffprobe.exe`
1. Tải bản FFmpeg static cho Windows từ: [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip).
2. Lấy 2 file **`ffmpeg.exe`** và **`ffprobe.exe`** copy đặt vào cùng thư mục với `main.py`.

### Bước 4: Chạy lệnh đóng gói EXE
```powershell
pyinstaller ThumbnailPipeline.spec
```

---

## 3. Kết Quả & Cách Sử Dụng Bản Build

Sau khi build xong, kết quả nằm tại thư mục: **`dist\ThumbnailPipeline\`**
File chạy chính: **`dist\ThumbnailPipeline\main.exe`** (hoặc `ThumbnailPipeline.exe`)

### Cách sử dụng:
1. **Cách 1 (Giao diện dòng lệnh trực quan)**:
   - Click đúp vào **`main.exe`**.
   - Một menu bằng tiếng Việt sẽ hiện ra cho phép chọn:
     - Phím 1: Tải video & phụ đề Google Drive (`tai.py`).
     - Phím 2: Trích xuất nhân vật & tạo Prompt (`make.py`).
     - Phím 3: Làm nét ảnh trong `net/` (`net.py`).
     - Phím 4: Ghép tiêu đề (`title.py`).
     - Phím 5: Kiểm tra thông tin GPU & bo tăng tốc (`gpu`).
2. **Cách 2 (Chạy qua dòng lệnh / Batch script)**:
   - `main.exe gpu`
   - `main.exe tai https://drive.google.com/...`
   - `main.exe make ten_phim --cast 3`
   - `main.exe net --rong 1280 --jpg`
   - `main.exe title "output/anh.png" "Tieu De Phim"`

Khi chia sẻ cho người khác, bạn chỉ cần nén toàn bộ thư mục `dist\ThumbnailPipeline\` thành file `.zip`. Người dùng cuối giải nén ra là chạy được ngay, không cần cài đặt Python!

---

## 4. Phương Án Build Tự Động Bằng GitHub Actions (Từ Máy Mac)

Workflow tại `.github/workflows/build_windows_exe.yml` đã được thiết lập để **tự động tải 100% các dependency**:
- **FFmpeg & FFprobe**: Tự động tải bản Windows và tích hợp vào bộ cài.
- **InsightFace Model (`buffalo_l`)**: Tự động tải trọn bộ 5 model ONNX nhận diện khuôn mặt (~288MB).
- **Real-ESRGAN Model**: Tự động kiểm tra và tích hợp model làm nét ảnh (`realesrgan_x4.onnx`).
- **GPU Accelerator**: Tự động cài DirectML (tương thích mọi card NVIDIA RTX 50x/40x/30x, AMD, Intel).
- **Trọn gói phân phối**: Tạo sẵn `main.exe`, `CHAY_TOOL.bat`, các thư mục `input/`, `output/`, `net/` và tài liệu hướng dẫn nhanh.

### Các bước thực hiện:
1. Đẩy mã nguồn lên GitHub:
   ```bash
   git add .
   git commit -m "Auto download dependencies for standalone main.exe"
   git push origin main
   ```
2. Mở repository trên trình duyệt, chuyển vào tab **Actions**.
3. Chọn workflow **Build Windows EXE** ở cột bên trái và nhấn **Run workflow**.
4. Chờ build xong (~3-5 phút) và tải file **`ThumbnailPipeline-Windows.zip`** về từ mục **Artifacts** (hoặc tab **Releases** nếu gắn tag `v*`).
5. Người dùng cuối chỉ cần giải nén file zip và click đúp vào **`main.exe`** là chạy được ngay, hoạt động **hoàn toàn offline** mà không cần cài đặt gì thêm!

