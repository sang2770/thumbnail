#!/usr/bin/env python3
"""Kiem tra tieu de YouTube: do dai that + phan thuc su hien ra tren dien thoai.

    python3 check_titles.py tieude.txt     # moi dong mot tieu de
    pbpaste | python3 check_titles.py      # dan tu clipboard

100 ky tu chi la gioi han NHAP LIEU. Thuc te bi cat som hon nhieu:
    app iOS      48-52 ky tu
    app Android  50-55
    mobile web   55-60
    desktop      60-70
Nen yeu to gay to mo phai nam tron trong ~50 ky tu dau.
"""

import re
import sys

MOBILE = 50          # nguong cat cua app dien thoai
GOOD_MAX = 60        # hien tron tren hau het thiet bi
SOFT_MAX = 70        # tren muc nay chac chan bi cat kha nhieu
HARD_MAX = 100       # gioi han nhap lieu cua YouTube

NUMBER = re.compile(r"^\s*(?:\d+[.)]|[-*•👉])\s*")
BRACKET = re.compile(r"^\s*\[[^\]]*\]\s*$")
TRAILING_COUNT = re.compile(r"\s*[(\[]\s*\d+\s*[)\]]\s*$")

# dong cua khoi "Chon tieu de tot nhat" o cuoi muc 4, khong phai tieu de
SKIP = re.compile(r"^\s*(?:```|#{1,6}\s|(?:CHỌN|Vì sao|Rủi ro|Á quân)\s*:)", re.I)


def main():
    raw = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 \
        else sys.stdin.read()

    titles = []
    for line in raw.splitlines():
        if BRACKET.match(line) or SKIP.match(line):   # "[Toi uu tu so 1]", "CHON: 3"...
            continue
        s = TRAILING_COUNT.sub("", NUMBER.sub("", line)).strip()
        if s and s not in titles:            # khoi dem 2 lan cau duoc chep lai o khoi CHON
            titles.append(s)
    if not titles:
        sys.exit("Khong tim thay tieu de nao.")

    good = 0
    for i, t in enumerate(titles, 1):
        n = len(t)
        if n > HARD_MAX:
            flag = "VUOT 100, YouTube tu cat"
        elif n > SOFT_MAX:
            flag = "qua dai, bi cat nhieu"
        elif n > GOOD_MAX:
            flag = "hoi dai, cat tren mobile"
        else:
            flag, good = "ok", good + 1
        seen = t[:MOBILE]
        cut = "…" if n > MOBILE else ""
        print(f"{i:>2}  {n:>3} ky tu  {flag:<26} {seen}{cut}")

    print(f"\n{good}/{len(titles)} tieu de hien tron (<= {GOOD_MAX} ky tu)")
    print(f"Cot ben phai la phan app dien thoai thuc su hien ra ({MOBILE} ky tu dau).")


if __name__ == "__main__":
    main()
