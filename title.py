#!/usr/bin/env python3
"""Ghep tieu de len anh thumbnail + xuat ban thu nho de kiem tra.

    .venv/bin/python title.py anh.png "Tieu De Cua Ban"
    .venv/bin/python title.py anh.png "Tieu De" --goc duoi-trai
    .venv/bin/python title.py anh.png "Tieu De" --co 0.9      # chu to hon

Xuat ra hai file canh anh goc:
    <ten>_title.jpg     ban 1280x720 de tai len YouTube
    <ten>_check.jpg     ban 210x118 - dung co no hien tren dien thoai

Ban _check moi la ban dang tin. Xem anh to thi cai gi cung dep; neu o co 210px
ma van doc duoc chu va van thay duoc chuyen gi dang xay ra thi thumbnail dat.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720          # kich thuoc chuan YouTube
CHECK_W = 210             # co thumbnail thuc te tren dien thoai
# Arial Black trong dam nhat nhung THIEU dau tieng Viet - chu se ra o vuong.
# Ba font duoi day deu co du dau, xep theo do dam giam dan.
FONT_CANDS = (
    # macOS
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    # Windows
    r"C:\Windows\Fonts\verdanab.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    # Linux (apt install fonts-dejavu fonts-liberation)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
)
VIET_TEST = "ịảỉọợồắễựỹ"
MARGIN = 0.045            # le, tinh theo chieu rong
GOC = ("tren-trai", "tren-phai", "duoi-trai", "duoi-phai")


def pick_font():
    """Chon font dau tien ve duoc HET dau tieng Viet.

    Glyph thieu van tra ve be rong nen khong the kiem bang getbbox. Cach dung
    la render thu roi so voi ky tu chac chan khong ton tai.
    """
    for path in FONT_CANDS:
        if not Path(path).is_file():
            continue
        f = ImageFont.truetype(path, 48)

        def render(ch):
            im = Image.new("L", (72, 72), 0)
            ImageDraw.Draw(im).text((4, 4), ch, font=f, fill=255)
            return im.tobytes()

        missing = render("")
        if all(render(c) != missing for c in VIET_TEST):
            return path
    sys.exit("Khong tim thay font nao co du dau tieng Viet.")


def wrap(text, font, max_w, draw):
    """Ngat dong theo be rong thuc te, uu tien ngat sau dau phay."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        thu = f"{cur} {w}".strip()
        if draw.textlength(thu, font=font) <= max_w or not cur:
            cur = thu
        else:
            lines.append(cur)
            cur = w
        if cur.endswith(",") and len(lines) == 0:   # ngat ngay sau dau phay dau tien
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def fit(text, draw, max_w, max_h, start, font_path):
    """Giam co chu cho den khi khoi chu lot vao vung cho phep."""
    for size in range(start, 18, -2):
        font = ImageFont.truetype(font_path, size)
        lines = wrap(text, font, max_w, draw)
        line_h = int(size * 1.18)
        if len(lines) * line_h <= max_h and \
                all(draw.textlength(l, font=font) <= max_w for l in lines):
            return font, lines, line_h
    font = ImageFont.truetype(font_path, 20)
    return font, wrap(text, font, max_w, draw), 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anh")
    ap.add_argument("tieude")
    ap.add_argument("--goc", choices=GOC, default="tren-trai")
    ap.add_argument("--co", type=float, default=1.0, help="he so co chu")
    a = ap.parse_args()

    src = Path(a.anh)
    if not src.is_file():
        sys.exit(f"Khong thay {src}")

    img = Image.open(src).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    m = int(W * MARGIN)
    box_w, box_h = int(W * 0.52), int(H * 0.34)     # chu chiem toi da nua khung
    font_path = pick_font()
    font, lines, line_h = fit(a.tieude, draw, box_w, box_h,
                              int(96 * a.co), font_path)

    block_h = len(lines) * line_h
    top = m if a.goc.startswith("tren") else H - m - block_h
    right = a.goc.endswith("phai")

    # Vien den day de chu noi tren moi nen. Thumbnail bi doi nen lien tuc
    # trong dong de xuat nen khong the trong cho vao mau nen cu the nao.
    stroke = max(3, font.size // 11)
    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        x = (W - m - tw) if right else m
        draw.text((x, top + i * line_h), line, font=font, fill=(255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0))

    out = src.with_name(f"{src.stem}_title.jpg")
    img.save(out, quality=92)

    check = img.resize((CHECK_W, int(CHECK_W * H / W)), Image.LANCZOS)
    chk = src.with_name(f"{src.stem}_check.jpg")
    check.save(chk, quality=92)

    print(f"font: {font_path.split('/')[-1]}")
    print(f"chu: {len(a.tieude)} ky tu, {len(a.tieude.split())} tieng, "
          f"{len(lines)} dong, co chu {font.size}px")
    for l in lines:
        print(f"   | {l}")
    print(f"-> {out}   (1280x720, tai len YouTube)")
    print(f"-> {chk}   ({CHECK_W}x{int(CHECK_W * H / W)}, mo file nay ra kiem tra)")


if __name__ == "__main__":
    main()
