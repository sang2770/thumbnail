#!/usr/bin/env python3
"""Lam net anh bang Real-ESRGAN (ONNX) - dung cho anh bi AI ca di ca lai nen vo net.

    python3 net.py                 # lam het anh trong net/, ghi ra net/xong/
    python3 net.py anh.jpg         # lam mot file
    python3 net.py --scale 4       # giu nguyen 4x thay vi thu ve 2x
    python3 net.py --thumuc /duong/dan

Khac unsharp mask o cho: unsharp chi keo tuong phan o vien san co, anh nhoe bet
thi ra anh nhoe bet co vien. Real-ESRGAN DUNG LAI chi tiet - toc, mi mat, thoi
vai - nen anh qua tay AI nhieu lan moi lay lai duoc do gion.

Model nhan dung 128x128, tra ve 512x512. Anh that thi to hon nhieu nen phai cat
o va ghep lai. Cho hai o ke nhau chong len nhau roi tron theo do doc tuyen tinh:
cat sat mep khong chong lan thi cho noi hien ra mot duong ke doc anh, vi moi o
duoc model doan sang toi hoi khac nhau.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL = Path("models/realesrgan_x4.onnx")
THU_MUC = Path("net")
DUOI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

O = 128          # canh o dau vao model
CHONG = 16       # so pixel hai o ke nhau chong len nhau
PHONG = 4        # model phong 4 lan


def tao_session_options():
    """Toi uu cac thiet lap da luong va do thi tinh toan cho ONNX Runtime."""
    import os
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    cpu_count = os.cpu_count() or 4
    opts.intra_op_num_threads = min(8, cpu_count)
    return opts


def phien():
    """Chon bo tang toc nhanh nhat co san tren may nay.

    Thu tu uu tien: CUDA -> DirectML (cuc chuan cho dong card 5x tren Windows) -> CoreML -> CPU.
    Tu dong kiem tra va fallback neu driver CUDA tren dong card 5x chua ho tro kernel sm_120.
    """
    if not MODEL.is_file():
        sys.exit(f"Thieu {MODEL}. Tai lai bang lenh trong README.")

    co = ort.get_available_providers()
    providers_cand = []

    if "CUDAExecutionProvider" in co:
        providers_cand.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }))
    if "DmlExecutionProvider" in co:
        providers_cand.append(("DmlExecutionProvider", {
            "device_id": 0,
        }))
    if "CoreMLExecutionProvider" in co:
        providers_cand.append("CoreMLExecutionProvider")
    providers_cand.append("CPUExecutionProvider")

    opts = tao_session_options()

    for i in range(len(providers_cand)):
        cur_list = providers_cand[i:]
        first_prov = cur_list[0]
        pname = first_prov[0] if isinstance(first_prov, tuple) else first_prov
        try:
            sess = ort.InferenceSession(str(MODEL), sess_options=opts, providers=cur_list)
            # Test inference tren o 128x128 de chac chan khong bi loi kernel tren card 5x
            ten_vao = sess.get_inputs()[0].name
            ten_ra = sess.get_outputs()[0].name
            dummy = np.zeros((1, 3, O, O), dtype=np.float32)
            sess.run([ten_ra], {ten_vao: dummy})

            # Kiem tra xem model co ho tro chay batch (gom nhieu o cung luc) khong
            batch_size = 1
            if any(g in pname for g in ("CUDA", "Dml", "CoreML")):
                try:
                    test_b = np.zeros((4, 3, O, O), dtype=np.float32)
                    sess.run([ten_ra], {ten_vao: test_b})
                    batch_size = 8 if "CUDA" in pname else 4
                except Exception:
                    batch_size = 1

            act_prov = sess.get_providers()[0]
            return sess, act_prov, batch_size
        except Exception as e:
            print(f"  ! {pname} gap loi khoi tao ({e}). Chuyen sang provider tiep theo...")
            continue

    # Fallback ve CPU
    sess = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
    return sess, "CPUExecutionProvider", 1


def mat_na(canh, vien):
    """Trong so cho mot o: bang 1 o giua, vuot deu ve 0 o ria.

    Cong tat ca trong so lai roi chia se ra anh lien mach. Neu dung trong so 1
    dong deu thi cho chong lan bi cong hai lan -> sang gap doi thanh vet ke o.
    """
    w = np.ones(canh, np.float32)
    if vien > 0:
        doc = np.linspace(0, 1, vien + 2, dtype=np.float32)[1:-1]
        w[:vien], w[-vien:] = doc, doc[::-1]
    return (w[:, None] * w[None, :])[:, :, None]


def lam_net(img, sess, ten_vao, ten_ra, batch_size=1):
    h, w = img.shape[:2]
    buoc = O - 2 * CHONG
    # Do them vien kieu guong cho anh phu kin so o nguyen. Do bang mau den thi
    # model coi vien den la chi tiet that va ve ra khung toi quanh anh.
    nh = max(O, -(-max(0, h - O) // buoc) * buoc + O)
    nw = max(O, -(-max(0, w - O) // buoc) * buoc + O)
    pad = cv2.copyMakeBorder(img, 0, nh - h, 0, nw - w, cv2.BORDER_REFLECT_101)

    rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    acc = np.zeros((nh * PHONG, nw * PHONG, 3), np.float32)
    tong = np.zeros_like(acc)
    mn = mat_na(O * PHONG, CHONG * PHONG)

    ys = list(range(0, nh - O + 1, buoc))
    xs = list(range(0, nw - O + 1, buoc))
    coords = [(y, x) for y in ys for x in xs]
    tong_o = len(coords)

    # Chay theo Batch de tan dung toi da cong suat GPU (CUDA/DirectML/CoreML)
    for idx in range(0, tong_o, batch_size):
        b_coords = coords[idx:idx + batch_size]
        try:
            if len(b_coords) == 1:
                y, x = b_coords[0]
                o = rgb[y:y + O, x:x + O].transpose(2, 0, 1)[None]
                ra_tile = sess.run([ten_ra], {ten_vao: o})[0][0].transpose(1, 2, 0)
                Y, X = y * PHONG, x * PHONG
                acc[Y:Y + O * PHONG, X:X + O * PHONG] += ra_tile * mn
                tong[Y:Y + O * PHONG, X:X + O * PHONG] += mn
            else:
                patches = [rgb[y:y + O, x:x + O].transpose(2, 0, 1) for y, x in b_coords]
                b_tensor = np.stack(patches, axis=0)
                ra_b = sess.run([ten_ra], {ten_vao: b_tensor})[0]
                for j, (y, x) in enumerate(b_coords):
                    ra_tile = ra_b[j].transpose(1, 2, 0)
                    Y, X = y * PHONG, x * PHONG
                    acc[Y:Y + O * PHONG, X:X + O * PHONG] += ra_tile * mn
                    tong[Y:Y + O * PHONG, X:X + O * PHONG] += mn
        except Exception:
            # Fallback ve single tile neu batch loi
            for y, x in b_coords:
                o = rgb[y:y + O, x:x + O].transpose(2, 0, 1)[None]
                ra_tile = sess.run([ten_ra], {ten_vao: o})[0][0].transpose(1, 2, 0)
                Y, X = y * PHONG, x * PHONG
                acc[Y:Y + O * PHONG, X:X + O * PHONG] += ra_tile * mn
                tong[Y:Y + O * PHONG, X:X + O * PHONG] += mn

        print(f"\r    o {min(tong_o, idx + len(b_coords))}/{tong_o} (batch={batch_size})", end="", flush=True)

    print("\r" + " " * 32 + "\r", end="")

    ra = np.clip(acc / np.maximum(tong, 1e-6), 0, 1)
    ra = (ra[:h * PHONG, :w * PHONG] * 255).round().astype(np.uint8)
    return cv2.cvtColor(ra, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anh", nargs="*", help="file anh cu the; bo trong thi lam ca thu muc")
    ap.add_argument("--thumuc", default=str(THU_MUC), help=f"thu muc anh (mac dinh {THU_MUC}/)")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="do phong cuoi cung so voi anh goc (mac dinh 2; model chay 4 roi thu ve)")
    ap.add_argument("--rong", type=int,
                    help="ep chieu rong cuoi cung, vd --rong 1280 cho thumbnail YouTube")
    ap.add_argument("--jpg", action="store_true",
                    help="ghi .jpg chat luong 95 thay vi .png (nhe hon nhieu lan)")
    a = ap.parse_args()

    if a.anh:
        files = [Path(p) for p in a.anh]
    else:
        d = Path(a.thumuc)
        if not d.is_dir():
            sys.exit(f"Chua co thu muc {d}/ - tao roi bo anh vao do.")
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in DUOI and p.parent.name != "xong")
    if not files:
        sys.exit(f"Khong co anh nao trong {a.thumuc}/  (nhan {', '.join(sorted(DUOI))})")

    sess, prov, batch_size = phien()
    ten_vao = sess.get_inputs()[0].name
    ten_ra = sess.get_outputs()[0].name
    print(f"Real-ESRGAN x4 | {prov} (batch={batch_size}) | {len(files)} anh\n")

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  bo qua {p.name} (khong doc duoc)")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # PNG co alpha: ep len nen TRANG. Vut kenh alpha di thi vung trong
            # suot hien mau RGB an ben duoi (thuong la den) - anh ChatGPT xuat
            # nen trong suot se ra vien den loang lo.
            a_ch = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3].astype(np.float32) * a_ch + 255.0 * (1 - a_ch))
            img = img.round().astype(np.uint8)
        h, w = img.shape[:2]
        t = time.time()
        ra = lam_net(img, sess, ten_vao, ten_ra, batch_size=batch_size)
        # Thu nho SAU khi phong 4x moi la cach lam net that su: model dung lai
        # chi tiet o co lon roi ep xuong, chi tiet do don lai thanh net. Thu nho
        # thang tu anh goc thi khong them duoc gi.
        if a.rong:
            dich = (a.rong, round(h * a.rong / w))
        elif a.scale != PHONG:
            dich = (round(w * a.scale), round(h * a.scale))
        else:
            dich = None
        if dich and dich != (ra.shape[1], ra.shape[0]):
            # INTER_AREA chi dung khi THU NHO; --rong lon hon ban 4x thi phai
            # dung Lanczos, INTER_AREA phong to ra anh nhoa.
            interp = (cv2.INTER_AREA if dich[0] < ra.shape[1]
                      else cv2.INTER_LANCZOS4)
            ra = cv2.resize(ra, dich, interpolation=interp)
        out_dir = p.parent / "xong"
        out_dir.mkdir(exist_ok=True)
        if a.jpg:
            dest = out_dir / f"{p.stem}_net.jpg"
            cv2.imwrite(str(dest), ra, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            dest = out_dir / f"{p.stem}_net.png"
            cv2.imwrite(str(dest), ra)
        mb = dest.stat().st_size / 1e6
        print(f"  {p.name:38s} {w}x{h} -> {ra.shape[1]}x{ra.shape[0]}"
              f"  {mb:.1f}MB  {time.time() - t:.0f}s  -> {dest}")
        if mb > 2:
            print(f"     ! qua 2MB, YouTube khong nhan. Chay lai them:"
                  f"  --rong 1280 --jpg")


if __name__ == "__main__":
    main()
