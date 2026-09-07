#!/usr/bin/env python3
"""Lam net anh bang Real-ESRGAN (ONNX) - dung cho anh bi AI ca di ca lai nen vo net.

    python3 net.py                 # lam het anh trong net/, ghi ra net/xong/
    python3 net.py anh.jpg         # lam mot file
    python3 net.py --scale 4       # giu nguyen 4x thay vi thu ve 2x
    python3 net.py --thumuc /duong/dan

Khac unsharp mask o cho: unsharp chi keo tuong phan o vien san co, anh nhoe bet
thi ra anh nhoe bet co vien. Real-ESRGAN DUNG LAI chi tiet - toc, mi mat, thoi
vai - nen anh qua tay AI nhieu lan moi lay lai duoc do gion.

File model goc khai bao shape CO DINH CUNG [1,3,128,128] -> [1,3,512,512]. Anh
that thi to hon nhieu nen phai cat o va ghep lai. Cho hai o ke nhau chong len
nhau roi tron theo do doc tuyen tinh: cat sat mep khong chong lan thi cho noi
hien ra mot duong ke doc anh, vi moi o duoc model doan sang toi hoi khac nhau.

    python3 net.py --o 384          # o to hon nua (nhanh hon, ton VRAM hon)
    python3 net.py --o 128 --batch 1  # ve dung hanh vi cu

TOC DO - vi sao noi shape thanh dong:
Graph nay THUAN convolution (Conv/LeakyRelu/Concat/Add/Mul/Clip + 2 Resize dung
he so ti le), khong co Reshape nao ghim cung batch. Nen sua khai bao shape thanh
dong la dung ve mat toan hoc - da do: dau ra giong het den tung bit (sai lech
tuyet doi 0.0).

Mo ra hai thu:

1. O TO HON. Voi o 128 va vien chong 16 thi buoc nhay chi 96 -> moi diem anh bi
   tinh (128/96)^2 = 1.78 lan. O cang to phan lam thua cang nho. Do thuc te,
   thoi gian tren moi pixel HUU ICH:
       o=128  111.8 us   (lam thua 1.78x)
       o=256   75.3 us   (lam thua 1.31x)   <- mac dinh moi
       o=384   69.3 us   (lam thua 1.19x)
   Tuc chi doi mot con so la nhanh gap 1.5 lan, khong danh doi gi ve chat luong
   (vien chong van 16 pixel nen cho noi khong xau di).

2. BATCH THAT. Code cu co san duong batch nhung file model ghim batch=1 nen phep
   thu batch LUON nem loi -> batch_size luon roi ve 1, ca nhanh batch la code
   chet. Gio moi chay that, GPU khong con ngoi cho tung o mot.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

def get_model_path():
    possible_dirs = []
    if getattr(sys, "frozen", False):
        possible_dirs.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            possible_dirs.append(Path(sys._MEIPASS))
    possible_dirs.append(Path(__file__).resolve().parent)
    possible_dirs.append(Path.cwd().resolve())

    for d in possible_dirs:
        cand = d / "models" / "realesrgan_x4.onnx"
        if cand.is_file():
            return cand
    return Path("models/realesrgan_x4.onnx")

MODEL = get_model_path()
# Ban da noi shape thanh dong. Workflow build san file nay va dong goi kem, nen
# ban .exe khong can toi thu vien onnx luc chay.
MODEL_DONG = MODEL.with_name(MODEL.stem + "_dong.onnx")

THU_MUC = Path("net")
DUOI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

O = 256          # canh o "tieu bieu", dung khi phai chot mot con so
O_MAX = 384      # canh o to nhat duoc phep dung
O_TINH = 128     # canh o BAT BUOC khi chi co file model shape co dinh
CHONG = 16       # so pixel hai o ke nhau chong len nhau
PHONG = 4        # model phong 4 lan


def tao_session_options():
    """Toi uu cac thiet lap da luong va do thi tinh toan cho ONNX Runtime."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    cpu_count = os.cpu_count() or 4
    # Day la mot model to chay mot o moi lan, khac han cac model ti hon cua
    # InsightFace: o day tang luong CO an. Nhung tren 16 luong thi chi phi dong
    # bo bat dau an lai phan thang.
    opts.intra_op_num_threads = min(16, cpu_count)
    return opts


def _noi_shape_dong(src, dest):
    """Nap cham noi_shape.py. De rieng file vi buoc build chi can onnx, khong
    can cv2/onnxruntime ma net.py keo theo (xem docstring cua noi_shape.py)."""
    try:
        from noi_shape import noi_shape_dong
    except ImportError:
        return False
    return noi_shape_dong(src, dest)


def duong_model():
    """-> (duong dan model, co_shape_dong). Tao ban dong mot lan roi dung mai."""
    if MODEL_DONG.is_file():
        return MODEL_DONG, True
    if not MODEL.is_file():
        sys.exit(f"Thieu {MODEL}. Tai lai bang lenh trong README.")
    # Ghi canh model goc. Neu thu muc chi doc (vd cai trong Program Files) thi
    # thoi, chay tiep bang model tinh.
    try:
        if _noi_shape_dong(MODEL, MODEL_DONG):
            print(f"  [1 lan] da tao ban model shape dong: {MODEL_DONG.name}")
            return MODEL_DONG, True
    except OSError as e:
        print(f"  ! khong ghi duoc {MODEL_DONG.name} ({e}) -> dung o {O_TINH}")
    return MODEL, False


def phien(o_muon=None, batch_muon=None):
    """Chon bo tang toc nhanh nhat co san tren may nay.

    Thu tu uu tien: CUDA -> DirectML (cuc chuan cho dong card 5x tren Windows) -> CoreML -> CPU.
    Tu dong kiem tra va fallback neu driver CUDA tren dong card 5x chua ho tro kernel sm_120.

    -> (sess, ten_provider, canh_o, batch). Canh o va batch deu duoc DO THU that
    su bang mot lan chay: het VRAM thi tu tut xuong muc thap hon chu khong chet.
    """
    model_path, dong = duong_model()

    co = ort.get_available_providers()
    providers_cand = []

    if "CUDAExecutionProvider" in co:
        providers_cand.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            # Anh nao cung chay hang tram o CUNG MOT shape, nen bo tien do mot
            # lan tim kernel nhanh nhat roi huong loi ca chang duong con lai.
            # HEURISTIC chi loi khi shape doi lien tuc - khong phai o day.
            "cudnn_conv_algo_search": "EXHAUSTIVE",
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
            sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                        providers=cur_list)
            ten_vao = sess.get_inputs()[0].name
            ten_ra = sess.get_outputs()[0].name

            def chay_thu(n, canh):
                x = np.zeros((n, 3, canh, canh), dtype=np.float32)
                sess.run([ten_ra], {ten_vao: x})

            # Model shape co dinh thi khong co gi de chon: 1 o 128, batch 1.
            if not dong:
                chay_thu(1, O_TINH)
                return sess, sess.get_providers()[0], O_TINH, 1

            # Provider THAT SU dang chay, khong phai provider ta YEU CAU.
            #
            # ONNX Runtime khong nem loi khi provider GPU khoi tao that bai - no
            # in "EP Error ... Falling back to CPUExecutionProvider and retrying"
            # roi tao session bang CPU va tra ve BINH THUONG. Nen try/except o
            # day khong bao gio bat duoc, va neu doc ten provider tu cai da yeu
            # cau thi ta tuong dang o GPU trong khi that ra dang o CPU.
            #
            # Hau qua da thay tren GitHub Actions: DirectML co trong danh sach
            # nhung khong co card do hoa ("Specified display adapter handle is
            # invalid"), code tuong la GPU nen di do o 384 roi do batch 2 va 4 -
            # tat ca tren CPU. Mot phep thu 45 giay thanh 10 phut 24.
            # Chuyen nay xay ra ca voi nguoi dung that: ban .exe luon kem
            # onnxruntime-directml, nen chay qua Remote Desktop / trong may ao /
            # driver loi la dung canh nay.
            that_su = sess.get_providers()[0]
            gpu = any(g in that_su for g in ("CUDA", "Dml", "CoreML"))

            # Canh o TOI DA con chay duoc. Chon canh cho tung anh de sau (chon_o).
            if o_muon:
                chay_thu(1, o_muon)
                o_toi_da = o_muon
            elif gpu:
                # Het VRAM la loi that, phai do. Thu tu to xuong nho: cai nao
                # chay duoc thi moi cai nho hon deu chay duoc.
                o_toi_da = None
                for c in (O_MAX, 256, 192, O_TINH):
                    try:
                        chay_thu(1, c)
                        o_toi_da = c
                        break
                    except Exception as e:
                        print(f"  ! o {c} khong chay duoc ({str(e)[:60]})"
                              f" -> thu o nho hon")
                if o_toi_da is None:
                    raise RuntimeError("khong o nao chay duoc")
            else:
                # CPU thi RAM du dat, do tung co o rat cham (o 384 mat ~8s) ma
                # gan nhu chac chan chay duoc. Kiem mot lan o nho cho chac roi
                # mo tran len het.
                chay_thu(1, O_TINH)
                o_toi_da = O_MAX

            # Batch: CPU da bao hoa san nen batch khong giup gi (da do). Chi GPU
            # moi loi, vi no dang phai cho tung o mot.
            if batch_muon:
                batch = batch_muon
                try:
                    chay_thu(batch, min(o_toi_da, O))
                except Exception as e:
                    print(f"  ! batch {batch} khong chay duoc ({str(e)[:70]}) -> ve 1")
                    batch = 1
            elif gpu:
                batch = 1
                for b in (2, 4):
                    try:
                        chay_thu(b, o_toi_da)
                        batch = b
                    except Exception:
                        break
            else:
                batch = 1

            return sess, sess.get_providers()[0], o_toi_da, batch
        except Exception as e:
            print(f"  ! {pname} gap loi khoi tao ({e}). Chuyen sang provider tiep theo...")
            continue

    # Fallback ve CPU
    sess = ort.InferenceSession(str(MODEL), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return sess, "CPUExecutionProvider", O_TINH, 1


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


def _luoi(n, canh):
    """So o va be rong sau khi dem, cho mot truc."""
    buoc = canh - 2 * CHONG
    n_pad = max(canh, -(-max(0, n - canh) // buoc) * buoc + canh)
    return len(range(0, n_pad - canh + 1, buoc)), n_pad


def chon_o(h, w, o_toi_da, ung_vien=(128, 192, 256, 384)):
    """Chon canh o tinh it pixel nhat cho DUNG anh nay.

    O to thi bot duoc phan chong lan bi tinh lai, nhung lai phai DEM anh ra cho
    du so o nguyen - anh nho ma o to thi phan dem lai thanh lang phi chinh. Hai
    xu huong nguoc nhau nen khong co canh o nao tot nhat moi luc, va ket qua
    KHONG don dieu theo kich thuoc o. Vi du anh 1280x720:
        o=128 -> 104 o, 1.70M pixel
        o=192 ->  40 o, 1.47M pixel
        o=256 ->  24 o, 1.57M pixel   <- to hon ma lai ton hon
        o=384 ->   8 o, 1.18M pixel   <- tot nhat
    Nen cu tinh thang so pixel phai chay roi lay cai nho nhat.
    """
    tot, re_nhat = O_TINH, None
    for c in ung_vien:
        if c > o_toi_da:
            continue
        ny, _ = _luoi(h, c)
        nx, _ = _luoi(w, c)
        px = ny * nx * c * c
        if re_nhat is None or px < re_nhat:
            tot, re_nhat = c, px
    return tot


def _tong_trong_so(n_ra, moc_ra, canh_ra, w1):
    """Cong don trong so 1 chieu cho mot truc.

    Mat na la tich cua hai vector giong nhau (w[:,None] * w[None,:]) va cac o
    nam tren luoi deu, nen TONG trong so cung tach thanh tich hai vector: cong
    theo tung truc roi nhan ngoai. Truoc day code cong ca mat na 2 chieu cho
    tung o vao mot mang bang co anh - dung ket qua nhung ton them mot mang
    float32 to bang anh 4x va rat nhieu bang thong bo nho.
    """
    acc = np.zeros(n_ra, np.float32)
    for m in moc_ra:
        acc[m:m + canh_ra] += w1
    return acc


def lam_net(img, sess, ten_vao, ten_ra, batch_size=1, canh_o=O):
    h, w = img.shape[:2]
    O = canh_o
    buoc = O - 2 * CHONG
    # Do them vien kieu guong cho anh phu kin so o nguyen. Do bang mau den thi
    # model coi vien den la chi tiet that va ve ra khung toi quanh anh.
    nh = max(O, -(-max(0, h - O) // buoc) * buoc + O)
    nw = max(O, -(-max(0, w - O) // buoc) * buoc + O)
    pad = cv2.copyMakeBorder(img, 0, nh - h, 0, nw - w, cv2.BORDER_REFLECT_101)

    rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    acc = np.zeros((nh * PHONG, nw * PHONG, 3), np.float32)
    mn = mat_na(O * PHONG, CHONG * PHONG)

    ys = list(range(0, nh - O + 1, buoc))
    xs = list(range(0, nw - O + 1, buoc))
    coords = [(y, x) for y in ys for x in xs]
    tong_o = len(coords)

    # Batch khong duoc lon hon so o co thuc. O dem la o RONG nhung van bi model
    # chay day du, chi de ket qua bi nem di - anh nho ma batch to thi phan dem
    # thanh phan chinh. Da thay tren log: 2 o that voi batch=4 -> chay 4 o, tuc
    # gap doi cong viec can thiet.
    batch_size = max(1, min(batch_size, tong_o))

    def ghep(ra_tile, y, x):
        Y, X = y * PHONG, x * PHONG
        acc[Y:Y + O * PHONG, X:X + O * PHONG] += ra_tile * mn

    def mot_o(y, x):
        o = rgb[y:y + O, x:x + O].transpose(2, 0, 1)[None]
        ghep(sess.run([ten_ra], {ten_vao: o})[0][0].transpose(1, 2, 0), y, x)

    # Chay theo Batch de tan dung toi da cong suat GPU (CUDA/DirectML/CoreML)
    for idx in range(0, tong_o, batch_size):
        b_coords = coords[idx:idx + batch_size]
        try:
            if batch_size == 1:
                mot_o(*b_coords[0])
            else:
                patches = [rgb[y:y + O, x:x + O].transpose(2, 0, 1)
                           for y, x in b_coords]
                # Dem cho du batch bang o rong. Giu shape dau vao KHONG DOI suot
                # ca luot: doi shape la ONNX Runtime phai do lai kernel, dung
                # dung cai ta vua bo tien mua bang EXHAUSTIVE. O dem khong duoc
                # ghep vao ket qua nen khong lam sai gi.
                thieu = batch_size - len(patches)
                if thieu:
                    patches += [np.zeros((3, O, O), np.float32)] * thieu
                ra_b = sess.run([ten_ra], {ten_vao: np.stack(patches, axis=0)})[0]
                for j, (y, x) in enumerate(b_coords):
                    ghep(ra_b[j].transpose(1, 2, 0), y, x)
        except Exception:
            # Fallback ve single tile neu batch loi
            for y, x in b_coords:
                mot_o(y, x)

        print(f"\r    o {min(tong_o, idx + len(b_coords))}/{tong_o}"
              f" (o={O}, batch={batch_size})", end="", flush=True)

    print("\r" + " " * 40 + "\r", end="")

    # Tong trong so tach thanh tich hai vector - xem _tong_trong_so.
    w1 = mat_na(O * PHONG, CHONG * PHONG)[:, 0, 0]
    hang = _tong_trong_so(nh * PHONG, [y * PHONG for y in ys], O * PHONG, w1)
    cot = _tong_trong_so(nw * PHONG, [x * PHONG for x in xs], O * PHONG, w1)
    tong = (hang[:, None] * cot[None, :])[:, :, None]

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
    ap.add_argument("--o", type=int,
                    help=f"canh o dua vao model (mac dinh {O}; to hon = nhanh hon,"
                         f" ton VRAM hon)")
    ap.add_argument("--batch", type=int,
                    help="so o chay cung luc (mac dinh tu do tren GPU)")
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

    sess, prov, o_toi_da, batch_size = phien(o_muon=a.o, batch_muon=a.batch)
    ten_vao = sess.get_inputs()[0].name
    ten_ra = sess.get_outputs()[0].name
    # "<=" ca hai con so: canh o chon rieng cho tung anh (chon_o), con batch bi
    # kep lai theo so o co thuc cua anh do (trong lam_net).
    print(f"Real-ESRGAN x4 | {prov} | o<={o_toi_da} batch<={batch_size}"
          f" | {len(files)} anh\n")

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
        # Kep theo o_toi_da: chi co file model shape tinh thi o BUOC phai la 128,
        # nguoi dung go --o 256 ma cu the dua vao la sess.run nem loi.
        canh_o = min(a.o or chon_o(h, w, o_toi_da), o_toi_da)
        ra = lam_net(img, sess, ten_vao, ten_ra, batch_size=batch_size,
                     canh_o=canh_o)
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
              f"  o={canh_o}  {mb:.1f}MB  {time.time() - t:.0f}s  -> {dest}")
        if mb > 2:
            print(f"     ! qua 2MB, YouTube khong nhan. Chay lai them:"
                  f"  --rong 1280 --jpg")


if __name__ == "__main__":
    main()
