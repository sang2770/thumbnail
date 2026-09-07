#!/usr/bin/env python3
"""Mot lenh duy nhat: input/ co video + phu de -> output/<ten>/ co du anh + prompt.

THU TU DUNG (quan trong):
    1. Doc het phu de, nam cot truyen, viet ra input/<ten>.moments.txt
       gom 10 khoanh khac chinh kem moc thoi gian.
    2. Chay make.py. No doc file do TRUOC, roi moi di lay khung hinh:
       moi khoanh khac lay mot anh dep nhat quanh moc thoi gian.
Khong co file do thi van chay duoc, nhung anh chi la "mat to, mat net"
rai deu thoi luong - dep ma khong dinh gi toi cot truyen.


    .venv/bin/python make.py                # lam tat ca video trong input/
    .venv/bin/python make.py tt041108       # chi lam mot video
    .venv/bin/python make.py --cast 3       # ep so nhan vat chinh (mac dinh: tu doan)
    .venv/bin/python make.py --rescan       # quet lai mat (sau khi doi nguong)
    .venv/bin/python make.py --rebuild      # trich lai keyframe tu video

Ket qua gom vao MOT thu muc cho moi video:

    output/<ten video>/
    ├── canh_chung_1_28m22.jpg      khung co DU vai chinh, tham chieu trang phuc
    ├── chinh_1_LamTriHa_21m41.jpg  mat tung vai chinh, de AI khoa dung danh tinh
    ├── phu_1_VanKimBao_55m22.jpg   vai phu lien quan cot truyen (dong PHU:)
    ├── prompt.txt                  DAN vao o chat  (4 muc: tom tat, tinh huong,
    │                               prompt thumbnail, 5 tieu de)
    └── <ten video>.txt             DINH KEM file nay (thoai da bo moc thoi gian)

    title.txt / thumb*.png nguoi dung tu bo vao output/<ten>/ (tieu de da chot,
    thumbnail da ung y) duoc tool GIU NGUYEN va dung lam mau few-shot.

Chi con MOT ban prompt. Truoc day phai giu them prompt_ngan.txt vi nhet ca
thoai vao prompt thi vuot gioi han do dai tin nhan cua ChatGPT - gio thoai luon
nam ngoai nen khong con canh do.

Chi ra bay nhieu anh: mot anh chung + moi vai chinh mot anh rieng. Dua nhieu
hon la AI nhan tin hieu danh tinh mau thuan roi tron ra mot khuon mat khong
giong ai - phim ngan Trung Quoc dung mat bang AI nen moi canh mat mot kieu.

Phu de ghep cap voi video theo TEN FILE: a.mp4 <-> a.srt
"""

import argparse
import shutil
import sys
from pathlib import Path

import build_prompt
import extract_char

INPUT_DIR = Path("input")
OUT_DIR = Path("output")
SUB_EXTS = (".srt", ".txt")


def find_pairs(only=None):
    """-> [(video, phu de hoac None)]. only = danh sach ten (khong can duoi)."""
    if not INPUT_DIR.is_dir():
        sys.exit(f"Khong tim thay thu muc {INPUT_DIR}/")
    videos = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in extract_char.VIDEO_EXTS)
    if only:
        want = {Path(n).stem for n in only}
        videos = [v for v in videos if v.stem in want]
        if not videos:
            sys.exit(f"Khong tim thay video nao ten {', '.join(sorted(want))} "
                     f"trong {INPUT_DIR}/")
    if not videos:
        sys.exit(f"Khong tim thay video nao trong {INPUT_DIR}/")

    pairs = []
    for v in videos:
        sub = next((v.with_suffix(e) for e in SUB_EXTS if v.with_suffix(e).is_file()),
                   None)
        if sub is None:                       # chi co 1 phu de thi khoi doi ten
            # .moments.txt cung duoi .txt nhung la file cot truyen, khong phai thoai
            subs = [p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in SUB_EXTS
                    and not p.name.endswith(".moments.txt")]
            sub = subs[0] if len(subs) == 1 and len(videos) == 1 else None
        pairs.append((v, sub))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", type=int, default=0,
                    help="so nhan vat chinh (bo trong = tu doan)")
    ap.add_argument("--rebuild", action="store_true", help="trich lai keyframe")
    ap.add_argument("--rescan", action="store_true", help="quet lai mat")
    ap.add_argument("ten", nargs="*",
                    help="chi lam nhung video nay (bo trong = lam tat ca)")
    ap.add_argument("--no-copy", action="store_true", help="khong chep clipboard")
    ap.add_argument("--luong", type=int,
                    help="so khung quet mat cung luc (bo trong = tu chon theo may)")
    a = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("Chua co ffmpeg. Cai bang: brew install ffmpeg")
    if not Path(build_prompt.TEMPLATE).is_file():
        sys.exit(f"Thieu {build_prompt.TEMPLATE}")

    pairs = find_pairs(a.ten)
    extract_char.CACHE_DIR.mkdir(exist_ok=True)

    for video, sub in pairs:
        out = OUT_DIR / video.stem

        # Buoc 1: cot truyen. File nay do Claude viet ra sau khi doc het phu de.
        mo_file = INPUT_DIR / f"{video.stem}.moments.txt"
        moments = extract_char.read_moments(mo_file) if mo_file.is_file() else None
        cast_ids = extract_char.read_cast(mo_file)
        sim = extract_char.read_sim(mo_file)
        phu = extract_char.read_phu(mo_file)
        the_loai = extract_char.read_the_loai(mo_file)
        them = extract_char.read_them(mo_file)
        if moments:
            print(f"khoanh khac chinh: {len(moments)} moc  <- {mo_file}")

        # Buoc 2: lay khung hinh, bam theo cac moc o buoc 1.
        extract_char.process(video, a.cast, a.rebuild, a.rescan,
                             out_dir=out, moments=moments, cast_ids=cast_ids,
                             sim=sim, phu=phu, them=them, luong=a.luong)

        if sub is None:
            print(f"  ! Khong thay phu de cho {video.name} -> bo qua prompt.")
            print(f"    Dat file .srt cung ten: input/{video.stem}.srt")
            continue

        # Mot prompt duy nhat, khong nhet thoai vao trong. Thoai nam o file .txt
        # rieng mang ten video de dinh kem thang vao ChatGPT/Gemini.
        thoai_name = f"{video.stem}.txt"
        # Moc dau tien trong moments.txt la canh dung lam thumbnail. Truyen thang
        # vao prompt, khong thi AI tu chon canh khac va lech voi anh tham chieu.
        canh = moments[0]["label"] if moments else None
        # Cac moc con lai lam mach truyen: chi dua moc dau thi AI ve dung mot
        # khoanh khac le, nguoi luot khong hieu chuyen gi dang xay ra.
        mach = [m["label"] for m in moments[1:6]] if moments else None
        # Ten anh vua cat ra -> prompt noi ro anh nao la vai nao.
        anh = sorted(p.name for p in out.glob("*.jpg"))
        prompt, n_line, body = build_prompt.make_prompt(sub, attach_name=thoai_name,
                                                       scene=canh, stem=video.stem,
                                                       the_loai=the_loai,
                                                       mach=mach, anh=anh)
        (out / "prompt.txt").write_text(prompt, encoding="utf-8")
        (out / thoai_name).write_text(body, encoding="utf-8")
        print(f"  prompt.txt               {len(prompt):,} ky tu  <- dan vao o chat")
        print(f"  {thoai_name:<24} {len(body):,} ky tu, {n_line} cau  <- dinh kem file nay")

        if not a.no_copy and len(pairs) == 1 and build_prompt.to_clipboard(prompt):
            print("  -> da chep prompt vao clipboard")

        print(f"  => {out}/")

    print("=" * 58)


if __name__ == "__main__":
    main()
