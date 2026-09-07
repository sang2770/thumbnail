# Thumbnail pipeline — cài trên macOS / Windows / Ubuntu

Toàn bộ tool viết bằng Python thuần, chạy được cả ba hệ. Khác nhau đúng ba chỗ:
lệnh gọi Python, cách cài `ffmpeg`, và công cụ clipboard.

## Cài

**Chung cho cả ba hệ** — cần **Python 3.12 trở lên** (`requirements.txt` ghim
`numpy==2.5.2`, mà numpy 2.5.x khai báo `Requires-Python >=3.12`; trên 3.11 thì
`pip install` đổ ngay ở bước này):

```
python3 -m venv .venv
```

| | macOS | Windows | Ubuntu |
|---|---|---|---|
| Chạy python | `.venv/bin/python` | `.venv\Scripts\python` | `.venv/bin/python` |
| ffmpeg | `brew install ffmpeg` | `winget install Gyan.FFmpeg` | `sudo apt install ffmpeg` |
| clipboard | có sẵn | có sẵn | `sudo apt install xclip` |
| font tiếng Việt (chỉ `title.py`) | có sẵn | có sẵn | `sudo apt install fonts-dejavu` |

Rồi cài thư viện:

```
.venv/bin/python -m pip install -r requirements.txt
```

`ffmpeg` phải gọi được từ dòng lệnh (`ffmpeg -version` chạy được) — tool tìm nó
trong PATH chứ không dùng file `.exe` kèm trong thư mục.

## Khác biệt cần biết

**Tăng tốc `net.py` (làm nét ảnh).** Tool tự chọn bộ nhanh nhất có sẵn và in ra
tên bộ đang chạy:

| Máy | Bộ tăng tốc | Cài thêm |
|---|---|---|
| Mac Apple Silicon | CoreML | không cần |
| Windows/Linux có card NVIDIA | CUDA | `pip install onnxruntime-gpu` |
| Windows máy nào cũng được | DirectML | `pip install onnxruntime-directml` |
| Còn lại | CPU | không cần, chỉ chậm hơn |

**Clipboard trên Ubuntu.** Chưa cài `xclip` (hoặc `wl-clipboard` nếu dùng
Wayland) thì tool vẫn chạy bình thường, chỉ là không tự chép prompt — lấy tay ở
`output/<tên>/prompt.txt`. Trên Windows dùng UTF-16 nên chữ Hán và dấu tiếng
Việt không bị vỡ.

**Tăng tốc nhận dạng khuôn mặt & Làm nét.** Cả `extract_char.py` và `net.py` đều
tự động nhận diện GPU (CUDA, DirectML trên Windows, CoreML trên macOS) và hỗ trợ
100% dòng card mới nhất **NVIDIA RTX 50-series (Blackwell)** cùng cơ chế fallback
tự động nếu thiếu driver.

👉 **Hướng dẫn đóng gói bản `.exe` cho Windows**: Xem chi tiết tại [HUONG_DAN_BUILD_EXE.md](HUONG_DAN_BUILD_EXE.md).

**Đường dẫn.** Mọi đường dẫn trong code đều dùng `pathlib`, không có dấu `/` hay
`\` viết cứng, nên không cần sửa gì khi đổi hệ.

## Tốc độ trên máy khỏe

Tool tự đo và tự chọn, không cần cấu hình gì. Muốn can thiệp tay thì có ba cờ:

```
python make.py <tên> --luong 32      # số khung quét mặt cùng lúc
python net.py --o 384                # ép cỡ ô đưa vào Real-ESRGAN
python net.py --batch 4              # ép số ô chạy cùng lúc
```

**Quét mặt (`make.py`) — song song theo khung hình.** Các model của InsightFace
đều tí hon, nên tăng số luồng *bên trong* một model lại làm chậm đi (đo được:
1 luồng 1095ms/khung, 16 luồng 1224ms). Trục đúng là chạy nhiều khung cùng lúc —
ONNX Runtime nhả GIL khi chạy nên ăn thật: 8 luồng 1.60x, 16 luồng 2.24x,
32 luồng 2.58x. Mặc định lấy `số nhân × 2` (tối đa 32) khi chạy CPU, và chỉ vài
luồng khi có GPU vì lúc đó GPU mới là chỗ nghẽn.

**Quét mặt — bỏ sớm những mặt sẽ bị loại.** Trước đây mọi mặt vừa phát hiện đều
bị chạy đủ 4 model phụ (70.1ms/mặt, riêng nhận dạng đã 46.1ms) rồi phần lớn mới
bị lọc bỏ. Giờ chia hai cổng: mặt quá nhỏ/quá mờ bị chặn ngay sau bước phát hiện
(tiết kiệm trọn 70.1ms), mặt quay nghiêng quá bị chặn sau khi có góc đầu (tiết
kiệm 79%). Phim đầy cảnh toàn cảnh và cảnh qua vai nên phần bỏ được không ít.
Kết quả đã đối chiếu là **trùng khớp từng con số** với đường cũ.

**Làm nét (`net.py`) — ô to hơn.** File model gốc khai báo shape cố định cứng
`[1,3,128,128]`, nên đường chạy batch có sẵn trong code **chưa bao giờ hoạt
động**. Graph này thuần convolution nên nới shape thành động là đúng về toán học
(đã kiểm: đầu ra giống hệt đến từng bit). Mở ra được ô to hơn, mà ô to thì bớt
phần chồng lấn bị tính lại — với ô 128 thì mỗi điểm ảnh bị tính 1.78 lần, ô 384
chỉ còn 1.19 lần. Thời gian trên mỗi pixel hữu ích: 111.8µs (ô 128) → 69.3µs
(ô 384). Cỡ ô tốt nhất **phụ thuộc kích thước ảnh** (ô to phải đệm nhiều hơn),
nên tool tự tính số pixel phải chạy cho từng ảnh rồi lấy cỡ rẻ nhất.

Bản model đã nới shape (`models/realesrgan_x4_dong.onnx`) được sinh tự động lần
chạy đầu, và bản `.exe` thì có sẵn từ lúc build.

## Chạy

```
.venv/bin/python main.py           # menu tổng hợp trực quan (hoặc click đúp file exe trên Windows)
.venv/bin/python tai.py            # kéo video + srt từ link Google Drive trong clipboard
.venv/bin/python make.py <tên>     # cắt ảnh tham chiếu + dựng prompt
.venv/bin/python net.py --rong 1280 --jpg   # làm nét ảnh trong net/
```

Windows đổi `.venv/bin/python` thành `.venv\Scripts\python` ở mọi lệnh.

**Thứ tự bắt buộc:** đọc hết phụ đề và viết `input/<tên>.moments.txt` TRƯỚC, rồi
mới chạy `make.py`. Chạy ngược thì ảnh chỉ là "mặt to mặt nét", không dính gì
tới cốt truyện.

## Khi tool nhận nhầm người

Mỗi lần chạy, `make.py` in ra **bảng mặt** — những người xuất hiện nhiều nhất,
kèm số thứ tự, số lần, giới tính~tuổi và đoạn phim họ xuất hiện:

```
  bang mat (ghim bang '#so Ten Nguoi' o dong CAST/PHU neu khop sai):
    #1    130 lan  M~22   2m -> 116m
    #2    130 lan  F~24   0m -> 117m
    #3     42 lan  F~19   0m -> 115m
```

Đối chiếu bảng này với ảnh trong `output/<tên>/`. Hai kiểu sai hay gặp:

**Sai 1 — hai người bị gộp làm một.** Dấu hiệu: một cụm có số lần lớn bất
thường và trải từ đầu đến cuối phim, còn vai chính thứ hai thì gần như không
có cụm nào. Chữa bằng cách nâng ngưỡng nhận dạng trong file khoảnh khắc:

```
SIM: 0.50
```

Mặc định 0.38 hợp phim hiện đại. Nâng dần 0.42 → 0.45 → 0.48 → 0.50 đến khi họ
tách ra mà vai chính vẫn còn nguyên một khối. Phim cổ trang (tóc dài đen, áo
bào giống nhau) và phim có hai nam chính-phụ cùng lứa tuổi hầu như luôn cần nâng.

**Sai 2 — tuổi đoán lệch nên khớp nhầm.** InsightFace đoán tuổi rất thô: một
thằng bé 5 tuổi có thể ra "nữ 19 tuổi", hai diễn viên 23 và 33 tuổi đều ra 22.
Lúc đó `nam 33` vô dụng — ghim thẳng số thứ tự trong bảng:

```
CAST: #2 On Du, #1 Ky Huu, #3 Ky Tinh Nhiem
PHU: #4 Ong Noi
```

Ghim và mô tả trộn chung một dòng cũng được: `CAST: nu 27 Manh Tinh Hoa, #4 Le Tu Nhien`.
Số ghim luôn tính theo bảng vừa in ra, không đổi khi thêm/bớt vai.

**Sai 3 — nhận dạng mặt không dùng được, chấm hết.** Ba dấu hiệu:

- nhân vật chính là **con vật** (phim trùng sinh thành hổ) — không có mặt người
  nào để mà nhận;
- nhân vật chỉ hiện đúng **một hai khung** trong cả phim (người hoá thú, chỉ trở
  lại hình người ở cảnh cuối) — không đọng nổi một cụm để lọt vào bảng;
- cụm trộn hai diễn viên mà **nâng SIM đến 0.52 vẫn không tách** — phim dựng mặt
  bằng AI thì hai người khác hẳn nhau vẫn nằm sát nhau trong không gian embedding.

Lúc đó mở video ra xem, tìm khung đúng, rồi ghim thẳng mốc thời gian bằng `@`:

```
CAST: @2:18:57 Trieu Hoanh, @0:16:19 Quan Nhu
PHU:  @0:01:55 Luu Ha, @0:16:34 Nuu Nuu
```

Vai ghim `@` không đi qua nhận dạng: tool lấy keyframe gần mốc đó nhất và đặt tên
theo **thứ tự viết trong dòng CAST**, nên `@` viết đầu dòng vẫn ra `chinh_1`.
Trộn chung với vai khớp bằng mặt cũng được — `CAST: nu 26 A, @0:10:00 B, nam 30 C`
ra đúng `chinh_1_A`, `chinh_2_B`, `chinh_3_C`.

Lệch quá 3 giây so với mốc yêu cầu thì tool in `! LECH` — nghĩa là đoạn đó không
có keyframe nào, khung lấy về gần như chắc chắn sai cảnh. Chọn mốc khác.

## Con vật trung tâm và ảnh chung chọn tay

Dòng `THEM:` cắt thêm ảnh tham chiếu không đi qua nhận dạng mặt:

```
THEM: vat   0:15:19  Bach Ho
THEM: chung 0:16:14  Chan ho chan nua khung, dan lang ngoi bet duoi tuyet
```

- `vat` → `vat_1_BachHo_15m19.jpg`. Con vật/quái vật trung tâm không có mặt người
  nên không đường nào lọt qua nhận dạng, mà với thumbnail nó lại là vật hút mắt
  lớn nhất. Không đưa ảnh thì AI vẽ con mặc định trong đầu nó — phim bạch hổ ra
  con hổ vàng.
- `chung` → `canh_chung_1_16m14.jpg`, và **tắt hẳn ảnh chung tự động**. Cần khi
  phim chỉ có một vai khớp được bằng mặt: lúc đó ảnh chung tự động thoái hoá
  thành một khung cận mặt, chẳng cho biết ai đứng cạnh ai. Chọn khung thấy được
  tỉ lệ to nhỏ giữa các nhân vật.

Tên đặt trong `THEM:` thành tên file nên để ngắn — AI nhìn ảnh là biết chi tiết.

Sửa xong chạy lại `make.py <tên>` — lần hai chỉ mất vài giây vì đã có cache.
