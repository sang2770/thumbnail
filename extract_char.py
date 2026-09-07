#!/usr/bin/env python3
"""Chon keyframe co nhan vat chinh -> anh tham chieu cho AI gen thumbnail.

KHONG cat xen, xuat nguyen keyframe goc. Anh co the co nhieu nhan vat.

Nhan dang bang InsightFace (ArcFace) - embedding danh tinh that, khong doi khi
nhan vat thay trang phuc / boi canh / anh sang. Kem doan gioi tinh + tuoi.

Chay:
    .venv/bin/python extract_char.py               # 10 anh
    .venv/bin/python extract_char.py --cast 3      # so nhan vat chinh
    .venv/bin/python extract_char.py --rescan      # quet lai mat (bo cache)
    .venv/bin/python extract_char.py --rebuild     # trich lai keyframe

Ket qua:
    refs/<ten video>/ref_01.jpg ...   nguyen keyframe
    refs/<ten video>/cast/            mat dai dien + gioi tinh/tuoi moi nhan vat
"""

import argparse
import os
import pickle
import queue
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("input")
REFS_DIR = Path("refs")
CACHE_DIR = Path(".cache")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv"}

# --- rules: khuon mat "dep va day du" ---
GOOD_FACE_W = 0.25     # mat rong bang 1/4 khung tro len la du to cho thumbnail
MIN_FACE_W = 0.12      # be ngang mat >= 12% khung hinh
MIN_DET = 0.60         # do tin cay phat hien
MAX_YAW = 25.0         # do, quay trai/phai. Duoi muc nay thi hai tai con thay ro
MAX_ROLL = 20.0        # do, nghieng dau
MAX_PITCH = 22.0       # do, ngua len / cui xuong. Vuot muc nay la khong nhin thang
MIN_LIP_GAP = 0.15     # vien moi trong / be ngang mieng. Duoi muc nay la cuoi ho rang
EDGE_MARGIN = 0.015    # mat phai nam tron trong khung

# Do net vung mat, chuan hoa ve 128x128 truoc khi do nen khong phu thuoc mat to
# hay nho. BAT BUOC phai co: det_score chi la do tin cay PHAT HIEN, mat mo nhoe
# van dat 0.85 nhu thuong - no khong he do do net.
MIN_SHARP = 100.0      # ~phan vi 50% cua phim nay
LOOSE_SHARP = 40.0
SHARP_FULL = 250.0     # tren muc nay coi nhu net toi da khi cham diem

# Nguong noi long cho nhan vat THU HAI: canh doi thoai thi nguoi kia thuong
# nho hon va quay nghieng, bat dat cung chuan thi khong bao gio co anh cap doi.
LOOSE_FACE_W = 0.06
LOOSE_DET = 0.45
LOOSE_YAW = 70.0

SAME_PERSON = 0.38     # cosine similarity toi thieu de coi la cung mot nguoi
CAST_SIZE = 4          # tran tren khi tu dong doan so nhan vat chinh
CAST_DROP = 0.35       # diem tut duoi ti le nay so voi nguoi truoc -> het vai chinh
MAIN_BONUS = 1.6       # uu tien khung co nhan vat chinh
DUO_BONUS = 1.0       # cong them moi nhan vat chinh nua trong cung khung
MIN_GAP_SEC = 45
MIN_HAMMING = 12       # khac canh: dHash 64 bit phai lech it nhat bay nhieu


# ---------- video ----------

def video_fps(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True).stdout.strip()
    try:
        num, den = out.split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return 30.0


def extract_keyframes(video, cache, rebuild=False):
    """Chi giai ma keyframe -> rat nhanh. Ten file = so thu tu frame."""
    if cache.is_dir() and any(cache.glob("*.jpg")) and not rebuild:
        return sorted(cache.glob("*.jpg"), key=lambda p: int(p.stem))
    if cache.is_dir():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey",
         "-i", str(video), "-fps_mode", "passthrough", "-frame_pts", "1",
         "-q:v", "2", str(cache / "%d.jpg")],
        check=False, stderr=subprocess.DEVNULL)
    return sorted(cache.glob("*.jpg"), key=lambda p: int(p.stem))


def face_sharpness(img, bbox, size=128):
    """Phuong sai Laplacian tren vung mat, chuan hoa kich thuoc. Cang cao cang net."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0.0
    g = cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def lip_gap(face):
    """Do day cua vien moi trong, chia cho be ngang mieng -> do "cuoi ho rang".

    Khong do mau rang: duoi den vang am cua phim ngan, rang van bao hoa mau cao
    nen loc theo mau truot han. Do hinh hoc thi mien nhiem anh sang.

    Cuoi ho rang thi moi bi keo NGANG, vien moi trong det lai. Do tren anh that:
        cuoi ho rang        0.071  0.089   <- qua NHO
        moi khep / chum moi 0.276  0.301   <- dat
        ha mieng noi/quat   0.401  0.446   <- qua LON

    Phai chan CA HAI dau. Chan moi dau duoi thi lot het khung dang noi giua cau:
    mieng ha to cung ho ca ham rang, ma phim thi dien vien noi gan nhu lien tuc
    nen loai nay chiem rat nhieu. Do la ly do co MAX_LIP_GAP.

    Tra ve 1.0 neu khong doc duoc moc - gia tri nay nam ngoai [0.15, 0.35] nen
    khung do bi day xuong noi long thay vi duoc coi la dat.
    """
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None or len(lm) < 72:
        return 1.0
    ml, mr = face.kps[3], face.kps[4]
    w = float(np.linalg.norm(mr - ml))
    if w < 1e-3:
        return 1.0
    inner = lm[64:72]                      # vien moi trong cua model 2d106
    return float(inner[:, 1].max() - inner[:, 1].min()) / w


def mouth_open(face):
    """Do ha mieng = chieu cao vien moi NGOAI / be ngang mieng. Cang lon cang ha to.

    KHONG dung lam bo loc cung duoc: do tren 9 khung da gan nhan cua 4 phim thi
    moi khep trai 0.378-0.670 con ha mieng 0.445-0.720, hai nhom chen nhau.
    Cach do theo mau cung truot y het, vi mo hinh 106 moc bam sai vien moi khi
    mieng ha to - khung quat to nhat lai cho vien trong co lai.

    Nhung dung de XEP HANG trong cung mot nguoi thi van co ich: giua cac khung
    cua chinh nguoi do, khung nao so nho hon thi mieng khep hon. So sanh tuong
    doi mien nhiem voi sai so he thong cua tung khuon mat.
    """
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None or len(lm) < 64:
        return 0.5
    ml, mr = face.kps[3], face.kps[4]
    w = float(np.linalg.norm(mr - ml))
    if w < 1e-3:
        return 0.5
    outer = lm[52:64]
    return float(outer[:, 1].max() - outer[:, 1].min()) / w


def dhash(img, size=8):
    """Van tay canh phim, de tranh chon 2 anh cung mot canh."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).flatten()


# ---------- nhan dang ----------

def get_face_providers():
    """Chon danh sach Execution Providers toi uu nhat theo thu tu uu tien:
    1. CUDAExecutionProvider (GPU NVIDIA, ho tro ca dong 50-series Blackwell neu du driver/CUDA)
    2. DmlExecutionProvider (DirectML tren Windows - chay 100% moi GPU NVIDIA/AMD/Intel, cuc chuan cho RTX 50x)
    3. CoreMLExecutionProvider (Mac Apple Silicon)
    4. CPUExecutionProvider (Fallback cuoi cung)
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = []

    if "CUDAExecutionProvider" in avail:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            # Mot phim la hang nghin khung, ma moi model chay o DUNG MOT co
            # (detector 640x640, nhan dang 112x112...). Bo tien do kernel mot lan
            # roi an lai suot chang duong con lai. HEURISTIC chi hon khi shape
            # doi lien tuc - khong phai truong hop nay.
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }))
    if "DmlExecutionProvider" in avail:
        providers.append(("DmlExecutionProvider", {
            "device_id": 0,
        }))
    if "CoreMLExecutionProvider" in avail:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def provider_thuc_su(app):
    """Provider THAT SU dang chay, khong phai provider ta YEU CAU.

    Khong duoc dua vao ort.get_available_providers(): ham do liet ke provider
    duoc BIEN DICH VAO goi onnxruntime, khong phai provider dung duoc tren may
    nay. Ban .exe luon kem onnxruntime-directml nen DirectML LUC NAO cung co
    trong danh sach do.

    Va ONNX Runtime khong nem loi khi provider GPU khoi tao that bai - no in
    "EP Error ... Falling back to CPUExecutionProvider and retrying" roi tra ve
    session CPU binh thuong. Nen chi co session moi biet su that.

    Doc sai cho nay thi tren may DirectML hong (Remote Desktop, may ao, driver
    loi) ta tuong dang o GPU va chi mo 2-6 luong, trong khi that ra dang o CPU
    va dang bo mat phan tang toc 2.6x cua 32 luong.
    """
    try:
        return app.det_model.session.get_providers()[0]
    except Exception:
        return "khong ro"


def so_luong(gpu):
    """So khung hinh quet CUNG LUC.

    Cac model cua InsightFace deu ti hon (112x112 den 192x192). Do thuc te thi
    tang so luong BEN TRONG mot model lai lam CHAM di - chi phi dong bo lon hon
    phan tinh toan:
        intra_op=1  ->  1095 ms/khung
        intra_op=16 ->  1224 ms/khung
    Nhung song song theo KHUNG HINH thi an that, vi ONNX Runtime nha GIL khi
    chay va cv2.imread cung vay:
        1 luong  1095 ms/khung
        8 luong   686 ms/khung   (1.60x)
        16 luong  489 ms/khung   (2.24x)
        32 luong  424 ms/khung   (2.58x)
    Duoi tuyen tinh vi con nghen bang thong bo nho, nhung 2.5x la mien phi.

    Tren GPU thi chinh GPU la cho nghen, chi can vai luong de no khong phai ngoi
    cho giai ma JPEG la du.
    """
    lam = os.cpu_count() or 4
    if gpu:
        return min(6, max(2, lam // 2))
    return min(32, max(4, lam * 2))


def build_app(intra_op=None):
    from insightface.app import FaceAnalysis
    prov_list = get_face_providers()

    # Kiem tra neu co san model buffalo_l offline trong models/buffalo_l thi uu tien dung
    possible_roots = []
    if getattr(sys, "frozen", False):
        possible_roots.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            possible_roots.append(Path(sys._MEIPASS))
    possible_roots.append(Path(__file__).resolve().parent)
    possible_roots.append(Path.cwd().resolve())

    model_root = "~/.insightface"
    for r in possible_roots:
        if (r / "models" / "buffalo_l").is_dir():
            model_root = str(r)
            break

    # Khi quet nhieu khung song song thi moi khung da chiem mot nhan; de model tu
    # bung luong ben trong nua la cac luong danh nhau, cham hon han.
    kw = {}
    if intra_op:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        kw["sess_options"] = so

    # Thu lan luot tung provider: neu CUDA loi tren dong card 5x thi tu dong nhay sang DirectML / CPU
    for i in range(len(prov_list)):
        candidate_providers = prov_list[i:]
        first_prov = candidate_providers[0]
        prov_name = first_prov[0] if isinstance(first_prov, tuple) else first_prov
        ctx_id = 0 if any(g in prov_name for g in ("CUDA", "Dml")) else -1

        try:
            try:
                app = FaceAnalysis(name="buffalo_l", root=model_root,
                                   providers=candidate_providers, **kw)
            except TypeError:
                # Ban insightface khac khong nhan sess_options -> thoi, chay mac dinh
                app = FaceAnalysis(name="buffalo_l", root=model_root,
                                   providers=candidate_providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            # Test thu tren dummy image de bat loi CUDA sm_120/kernel tren RTX 50x ngay lap tuc
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            app.get(dummy)
            # In provider THAT SU, khong phai cai vua yeu cau: ONNX Runtime co
            # the da am tham roi ve CPU (xem provider_thuc_su). In ten yeu cau
            # thi log bao "DirectML thanh cong" trong khi dang chay bang CPU.
            that_su = provider_thuc_su(app)
            if that_su != prov_name:
                print(f"  [GPU/AI] Da yeu cau {prov_name} nhung ONNX Runtime"
                      f" roi ve {that_su}")
            else:
                print(f"  [GPU/AI] InsightFace khoi tao thanh cong:"
                      f" {that_su} (ctx={ctx_id})")
            return app
        except Exception as e:
            print(f"  ! {prov_name} gap loi khoi tao hoac thieu kernel ({e}). Dang chuyen sang fallback...")
            continue

    # Fallback cuoi cung
    app = FaceAnalysis(name="buffalo_l", root=model_root, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  [GPU/AI] InsightFace fallback ve CPUExecutionProvider")
    return app


def prefetch_frames(frame_paths, max_prefetch=32):
    """Doc truoc anh tu o dia trong background thread de loai bo nghen I/O khi quet mat."""
    q = queue.Queue(maxsize=max_prefetch)
    SENTINEL = object()

    def loader():
        for p in frame_paths:
            img = cv2.imread(str(p))
            q.put((p, img))
        q.put(SENTINEL)

    th = threading.Thread(target=loader, daemon=True)
    th.start()

    while True:
        item = q.get()
        if item is SENTINEL:
            break
        yield item


def chuan_bi_hai_pha(app):
    """Do nghe de quet HAI PHA. Tra ve None neu ban insightface khac cau truc.

    FaceAnalysis.get() chay CA BON model phu cho MOI mat vua phat hien, roi ta
    moi loc va nem di phan lon. Ma bon model do khong he re - do tren may nay:

        detect         47.7 ms/khung  (mot lan, khong tranh duoc)
        landmark_3d_68 14.8 ms/mat    <- chi de lay goc quay dau
        landmark_2d_106 7.7 ms/mat
        genderage       1.5 ms/mat
        recognition    46.1 ms/mat    <- nang nhat, 66% chi phi moi mat
        --------------------------------
        moi mat        70.1 ms

    Trong khi mot mat bi LOAI o buoc loc chi can bbox + det_score + do net, deu
    co san ngay sau detect. Nen tach ra hai cong:

      Cong 1 (sau detect): mat qua nho / qua mo / do tin cay thap -> bo luon,
              tiet kiem TRON 70.1 ms.
      Cong 2 (sau pose):   mat quay nghieng qua -> bo, tiet kiem 55.3 ms (79%).

    Phim thi day canh toan canh va canh qua vai, nen phan bi loai khong he it.
    Da doi chieu voi app.get(): ket qua giong het tung con so.
    """
    try:
        from insightface.app.common import Face
    except ImportError:
        return None
    if not hasattr(app, "det_model") or not isinstance(getattr(app, "models", None), dict):
        return None
    phu = {t: m for t, m in app.models.items() if t != "detection"}
    if not phu:
        return None
    return {
        "Face": Face,
        "pose": phu.get("landmark_3d_68"),
        "con_lai": [m for t, m in phu.items() if t != "landmark_3d_68"],
    }


def _dac_trung(f, w, det, sh, W, H):
    """Doc cac so do can thiet tu mot khuon mat da chay du model."""
    x1, y1, x2, y2 = f.bbox
    pitch, yaw, roll = (f.pose if f.pose is not None else (0.0, 0.0, 0.0))
    pitch, yaw, roll = abs(float(pitch)), abs(float(yaw)), abs(float(roll))
    m = EDGE_MARGIN
    inside = (x1 > m * W and y1 > m * H
              and x2 < (1 - m) * W and y2 < (1 - m) * H)
    return {
        "bbox": [float(v) for v in f.bbox],
        "w": float(w),
        "det": float(det),
        "sharp": sh,
        "yaw": yaw,
        "pitch": pitch,
        "lip_gap": lip_gap(f),
        "mouth": mouth_open(f),
        "emb": np.asarray(f.normed_embedding, dtype=np.float32),
        "sex": getattr(f, "sex", None),
        "age": int(f.age) if getattr(f, "age", None) is not None else None,
    }, roll, inside


def _xep_gio(d, roll, inside, strict, extra, tally):
    """Phan mot mat vao gio chuan / noi long / loai. Y het logic goc."""
    w, det, sh = d["w"], d["det"], d["sharp"]
    yaw, pitch, gap = d["yaw"], d["pitch"], d["lip_gap"]
    # Chuan chat: mat ro, hai tai con thay (yaw nho), mat nhin thang
    # (pitch nho), mieng khong ho rang.
    if (w >= MIN_FACE_W and det >= MIN_DET and sh >= MIN_SHARP
            and yaw <= MAX_YAW and roll <= MAX_ROLL
            and pitch <= MAX_PITCH and gap >= MIN_LIP_GAP and inside):
        strict.append(d)
        tally["chuan"] += 1
        return
    if (w >= LOOSE_FACE_W and det >= LOOSE_DET
            and sh >= LOOSE_SHARP and yaw <= LOOSE_YAW):
        extra.append(d)
        tally["noi_long"] += 1
        if gap < MIN_LIP_GAP:            # dem ly do bi day xuong noi long
            tally["ho_rang"] += 1
        elif pitch > MAX_PITCH or yaw > MAX_YAW:
            tally["nghieng"] += 1
        return
    tally["loai"] += 1
    if sh < LOOSE_SHARP:
        tally["mo"] += 1


def _quet_khung(app, hp, p, img, fps):
    """Quet mot khung -> (record hoac None, tally rieng, so mat da thay).

    Chay duoc tu nhieu luong: session cua ONNX Runtime an toan da luong, va moi
    doi tuong Face o day la cua rieng lan goi nay.
    """
    tally = {"chuan": 0, "noi_long": 0, "loai": 0, "mo": 0,
             "nghieng": 0, "ho_rang": 0}
    H, W = img.shape[:2]
    strict, extra = [], []
    seen = 0

    # Nguong cho hai cong bo som: lay muc DE NHAT trong hai gio, de cong chi bo
    # nhung mat ma CA HAI gio deu khong nhan.
    #
    # Khong duoc viet thang LOOSE_* vao day. Hien tai gio chuan chat hon gio noi
    # long o moi tieu chi, nen dung LOOSE_* thi tinh co ra dung. Nhung day la su
    # trung hop cua bo so hien tai, khong phai tinh chat cua thuat toan - ma
    # README thi huong nguoi dung tu tinh chinh nguong. Ha MIN_SHARP xuong duoi
    # LOOSE_SHARP mot cai la cong bo oan nhung mat ma gio chuan van nhan, va
    # khong co dau hieu gi bao loi. Da bi dinh dung loi nay khi doi chieu.
    g_face_w = min(MIN_FACE_W, LOOSE_FACE_W)
    g_det = min(MIN_DET, LOOSE_DET)
    g_sharp = min(MIN_SHARP, LOOSE_SHARP)
    g_yaw = max(MAX_YAW, LOOSE_YAW)

    if hp is None:                       # duong cu: chay het model roi moi loc
        for f in app.get(img):
            seen += 1
            sh = face_sharpness(img, f.bbox)
            x1, _, x2, _ = f.bbox
            d, roll, inside = _dac_trung(f, (x2 - x1) / W, f.det_score, sh, W, H)
            _xep_gio(d, roll, inside, strict, extra, tally)
    else:
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric="default")
        for i in range(bboxes.shape[0]):
            seen += 1
            bbox = bboxes[i, 0:4]
            det = float(bboxes[i, 4])
            x1, _, x2, _ = bbox
            w = (x2 - x1) / W
            sh = face_sharpness(img, bbox)

            # Cong 1: chua chay model phu nao ca.
            if w < g_face_w or det < g_det or sh < g_sharp:
                tally["loai"] += 1
                if sh < LOOSE_SHARP:
                    tally["mo"] += 1
                continue

            f = hp["Face"](bbox=bbox, kps=kpss[i] if kpss is not None else None,
                           det_score=det)
            if hp["pose"] is not None:
                hp["pose"].get(img, f)
                # Cong 2: quay nghieng qua thi ca hai gio deu khong nhan.
                yaw = abs(float(f.pose[1])) if f.pose is not None else 0.0
                if yaw > g_yaw:
                    tally["loai"] += 1
                    continue
            for m in hp["con_lai"]:
                m.get(img, f)
            d, roll, inside = _dac_trung(f, w, det, sh, W, H)
            _xep_gio(d, roll, inside, strict, extra, tally)

    rec = None
    if strict:
        rec = {"path": p, "sec": int(p.stem) / fps,
               "faces": strict, "extra": extra, "hash": dhash(img)}
    return rec, tally, seen


def scan(frames, fps, cache_file, rescan=False, luong=None):
    """Quet mat tren tung keyframe. Ket qua duoc cache vi buoc nay cham."""
    if cache_file.is_file() and not rescan:
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    # intra_op phai chot LUC TAO session, tuc truoc khi biet provider thuc su.
    # Khong sao: do duoc thi o muc 32 luong, intra_op gan nhu khong anh huong
    # (=1 cho 424ms, =2 cho 417ms, =4 cho 432ms), con khi that su chay GPU thi
    # GPU moi la noi lam viec. Nen cu dat 1 khi may du nhan de chay nhieu khung
    # song song - tranh 32 luong x 16 nhan moi luong danh nhau.
    lam = os.cpu_count() or 4
    app = build_app(intra_op=1 if lam > 4 else None)

    that_su = provider_thuc_su(app)
    gpu = any(g in that_su for g in ("CUDA", "Dml", "CoreML"))
    nl = luong or so_luong(gpu)

    hp = chuan_bi_hai_pha(app)
    if hp is None:
        print("  ! ban insightface nay khac cau truc -> quet mot pha nhu cu")
    print(f"  quet {nl} khung song song tren {that_su}"
          f"{' (GPU la cho nghen nen khong can nhieu hon)' if gpu else ''}")

    records, seen = [], 0
    tally = {"chuan": 0, "noi_long": 0, "loai": 0, "mo": 0,
             "nghieng": 0, "ho_rang": 0}

    def lam(p):
        img = cv2.imread(str(p))
        if img is None:
            return None
        return _quet_khung(app, hp, p, img, fps)

    # ex.map tra ve ket qua THEO DUNG THU TU dau vao, nen 'records' xep y het ban
    # chay mot luong - cac buoc sau (chong trung canh, chon theo moc thoi gian)
    # deu dua vao thu tu nay.
    with ThreadPoolExecutor(max_workers=nl) as ex:
        for i, kq in enumerate(ex.map(lam, frames)):
            if i % 200 == 0:
                print(f"    ...{i}/{len(frames)}", flush=True)
            if kq is None:
                continue
            rec, t_khung, n = kq
            seen += n
            for k, v in t_khung.items():
                tally[k] += v
            if rec is not None:
                records.append(rec)

    data = (records, seen, tally)
    with open(cache_file, "wb") as fh:
        pickle.dump(data, fh)
    return data


def cluster_faces(records, thresh=SAME_PERSON):
    """Gom mat theo danh tinh that: cosine similarity tren embedding ArcFace."""
    clusters = []
    for rec in records:
        for f in rec["faces"]:
            f["path"] = rec["path"]
            e = f["emb"]
            best, best_sim = None, -1.0
            for c in clusters:
                sim = float(np.dot(e, c["centroid"]))
                if sim > best_sim:
                    best, best_sim = c, sim
            if best is not None and best_sim >= thresh:
                best["members"].append(f)
                best["sum"] += e
                best["centroid"] = best["sum"] / np.linalg.norm(best["sum"])
            else:
                clusters.append({"centroid": e.copy(), "sum": e.copy(),
                                 "members": [f]})

    for ci, c in enumerate(clusters):
        for f in c["members"]:
            f["cid"] = ci
        c["n"] = len(c["members"])
        c["w_avg"] = sum(f["w"] for f in c["members"]) / c["n"]
        c["score"] = c["n"] * c["w_avg"]
        c["rep"] = max(c["members"],
                       key=lambda f: f["det"] * (0.5 + f["w"])
                       * min(f.get("sharp", 0.0) / SHARP_FULL, 1.0))
        sexes = [f["sex"] for f in c["members"] if f["sex"]]
        ages = [f["age"] for f in c["members"] if f["age"]]
        c["sex"] = max(set(sexes), key=sexes.count) if sexes else "?"
        c["age"] = int(np.median(ages)) if ages else 0

    order = sorted(range(len(clusters)), key=lambda i: clusters[i]["score"],
                   reverse=True)
    return clusters, order, {cid: pos for pos, cid in enumerate(order)}


def auto_cast_size(clusters, order, cap=CAST_SIZE, drop=CAST_DROP):
    """Doan so nhan vat chinh: cat o cho diem tut manh nhat so voi nguoi truoc.

    Vai chinh xuat hien day dac va deu; vai phu tut han mot bac. Vi du phim
    hop dong hon nhan chi co 2 vai chinh (83, 69 | 13, 12) thi cat sau nguoi
    thu hai, con phim co them dua tre thi giu lai ba.
    """
    scores = [clusters[cid]["score"] for cid in order[:cap]]
    for i in range(1, len(scores)):
        if scores[i] < drop * scores[i - 1]:
            return max(2, i)
    return max(2, len(scores))


def assign_extras(records, clusters, thresh=SAME_PERSON):
    """Gan mat noi long vao nhan vat da biet, de dem duoc canh nhieu nguoi."""
    n = 0
    for rec in records:
        for f in rec["extra"]:
            best, best_sim = None, -1.0
            for ci, c in enumerate(clusters):
                sim = float(np.dot(f["emb"], c["centroid"]))
                if sim > best_sim:
                    best, best_sim = ci, sim
            if best is not None and best_sim >= thresh:
                f["cid"] = best
                n += 1
    return n


# ---------- khoanh khac chinh doc tu cot truyen ----------

N_GROUP = 2            # so phuong an anh chung xuat ra de nguoi chon
MOMENT_WINDOW = 120    # giay: khung phai nam trong khoang nay quanh moc thoi gian
MOMENT_GAP = 20        # hai khoanh khac ke nhau chi can cach bay nhieu giay
DUO_PREFER = 0.75      # khung co du hai vai chinh thang khung don neu diem >= ti le nay

TIME_LABEL = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s+(.+?)\s*$")
# KHONG chap nhan '#' dau dong: comment mot dong CAST/PHU/SIM di la phai TAT duoc no.
CAST_LINE = re.compile(r"^\s*CAST\s*:\s*(.+?)\s*$", re.I)
PHU_LINE = re.compile(r"^\s*PHU\s*:\s*(.+?)\s*$", re.I)
# 'nu 30' hoac 'nu 30 Lam Tri Ha' - ten dat sau tuoi thi duoc dung lam ten file.
CAST_SPEC = re.compile(r"^\s*(nam|nu|nữ|[mf])\s*~?\s*(\d{1,2})\s*(.*?)\s*$", re.I)
# '#4 Le Tu Nhien' - ghim thang so thu tu trong BANG MAT o tren, dung khi gioi
# tinh + tuoi khong tach noi hai nguoi (vd hai nam chinh/phu deu duoc doan ~23).
CAST_PIN = re.compile(r"^\s*#\s*(\d{1,2})\s*(.*?)\s*$")
# '@2:18:57 Trieu Hoanh' - ghim thang MOC THOI GIAN, bo qua nhan dang mat.
# Loi ra khi nhan dang khong the dung duoc, khong phai vi chinh sai nguong:
#   - nhan vat chinh la CON VAT (phim trung sinh thanh ho) - khong co mat nguoi;
#   - anh ta chi hien nguyen hinh nguoi dung 1-2 khung trong ca phim, khong the
#     dong noi mot cum de lot vao bang;
#   - cum bi tron hai dien vien ma nang SIM den 0.52 van khong tach (phim dung
#     mat sinh bang AI, hai phu nu khac nhau van gan nhau trong khong gian embedding).
# Da nhin tan mat khung do roi thi ghim thang, dung hon moi thuat toan.
CAST_GHIM = re.compile(r"^\s*@\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s+(.*?)\s*$")
AGE_TOL = 10           # tuoi InsightFace doan rat tho, cho lech bay nhieu nam


SIM_LINE = re.compile(r"^\s*SIM\s*:\s*([01]?\.\d+)\s*$", re.I)
# 'THE_LOAI: hd' - de len tien to ten file khi ten dat khong khop noi dung.
GENRE_LINE = re.compile(r"^\s*THE[_ ]?LOAI\s*:\s*([a-z0-9]{1,4})\s*$", re.I)
# 'THEM: vat 0:15:19 Bach Ho' - anh tham chieu KHONG di qua nhan dang mat.
#   vat   -> vat_1_BachHo_15m19.jpg      con vat / quai vat / do vat trung tam
#   chung -> canh_chung_1_16m39.jpg      khung rong, thay ca canh lan ti le
# Co dong 'THEM: chung' thi anh chung TU DONG bi bo: da chon tay thi khong con
# ly do de tool doan, ma phim mot vai chinh thi anh chung tu dong chi la mot
# khung can mat - vo nghia, khong cho biet ai dung canh ai.
THEM_LINE = re.compile(
    r"^\s*THEM\s*:\s*(vat|chung)\s+(?:(\d+):)?(\d{1,2}):(\d{2})\s+(.+?)\s*$", re.I)


def _giay(h, m, s):
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def read_them(path):
    """Doc cac dong 'THEM: <vai> <moc> <ten>' -> [{'vai','sec','ten'}]."""
    if not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = THEM_LINE.match(unicodedata.normalize("NFC", line))
        if m:
            out.append({"vai": m.group(1).lower(),
                        "sec": _giay(m.group(2), m.group(3), m.group(4)),
                        "ten": m.group(5).strip()})
    return out


def ghim_khung(frames, fps, sec):
    """Keyframe gan moc thoi gian nhat. Tra ve (path, giay that, do lech)."""
    if not frames:
        return None, 0.0, 0.0
    p = min(frames, key=lambda f: abs(int(f.stem) / fps - sec))
    that = int(p.stem) / fps
    return p, that, abs(that - sec)


def read_the_loai(path):
    """Doc dong 'THE_LOAI: hd' trong file khoanh khac.

    Tien to ten file khong phai luc nao cung dung the loai: mot lo phim co the
    duoc dat ten theo ngay/dot chu khong theo noi dung, luc do 'tt031308' lai la
    phim tham hoa tren bien chu khong phai tong tai. Doc nham the loai la prompt
    bat AI dung dai sanh biet thu cho mot phim ca map.
    """
    if not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = GENRE_LINE.match(line)
        if m:
            return m.group(1).lower()
    return None


def read_sim(path):
    """Doc dong 'SIM: 0.48' - nguong cosine de coi hai khuon mat la mot nguoi.

    Mac dinh 0.38 dung cho phim hien dai. Phim CO TRANG thi moi nam nhan vat deu
    toc dai den, rau nhan giong nhau, ao bao cung kieu - embedding cua ho gan
    nhau den muc bi gop lam mot nguoi. Luc do phai nang nguong len.

    Cach biet la can nang: mot cum nam co so lan xuat hien lon bat thuong, va
    tra nguoc lai vai moc thoi gian thi thay hai NHAN VAT KHAC NHAU cung roi vao
    cum do. Nang dan 0.42 -> 0.45 -> 0.48 den khi ho tach ra ma vai chinh nu van
    con nguyen mot cum.
    """
    if not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = SIM_LINE.match(line)
        if m:
            return float(m.group(1))
    return None


def read_cast(path):
    """Doc dong 'CAST: nam 26, nu 27' - chi ro ai la vai chinh.

    Vai chinh duoc ta bang GIOI TINH + TUOI doc duoc tu thoai, khong phai bang
    so thu tu tan suat. Co vay moi viet duoc file nay TRUOC khi chay tool - doc
    phu de xong la biet phim co may vai chinh, nam hay nu, chung bao nhieu tuoi.

    Can dong nay khi tan suat xuat hien KHONG phan anh do quan trong. Vi du phim
    tho san: nam chinh vao rung mot minh gan het phim nen vo anh chi hien 26 lan,
    thua ca gay chu no 36 lan. May dem mat khong biet ai quan trong.

    Dat them TEN sau tuoi thi ten do thanh ten file anh:
        CAST: nu 30 Lam Tri Ha, nu 35 Ha My Lien
    -> chinh_1_LamTriHa_21m41.jpg. Khong dat ten thi lay 'F30'.

    Van chap nhan dang cu 'CAST: 1,4' (so thu tu) de khong pha file da viet.
    """
    return _doc_vai(path, CAST_LINE, cho_so=True)


def read_phu(path):
    """Doc dong 'PHU: nu 45 Hoang Xuan Chi' - nhan vat PHU co lien quan cot truyen.

    Tach rieng khoi CAST vi hai loai duoc dung khac nhau: vai chinh phai cung ro
    mat trong ANH CHUNG (dieu kien rat kho, cang nhieu nguoi cang hiem khung dat),
    con vai phu chi can mot anh rieng de AI biet mat. Nhet vai phu vao CAST thi
    anh chung tut chat luong ma chang duoc gi.
    """
    r = _doc_vai(path, PHU_LINE, cho_so=False)
    return (r or {}).get("specs") or []


def _doc_vai(path, mau, cho_so):
    if not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # macOS/vai editor luu tieng Viet dang NFD (dau tach roi) - 'nữ' NFD
        # khong khop nhanh 'nu|nữ' cua CAST_SPEC va spec bi bo qua im lang.
        m = mau.match(unicodedata.normalize("NFC", line))
        if not m:
            continue
        parts = [p for p in m.group(1).split(",") if p.strip()]
        if cho_so and parts and all(p.strip().isdigit() for p in parts):
            return {"ids": [int(p) for p in parts]}
        specs, hong = [], []
        for p in parts:
            s = CAST_SPEC.match(p)
            g = CAST_PIN.match(p)
            t = CAST_GHIM.match(p)
            if t:
                # sex='@' la co hieu "so o giua la GIAY, khong phai tuoi" -
                # vai nay khong di qua nhan dang mat, cat thang keyframe o moc do.
                specs.append(("@", _giay(t.group(1), t.group(2), t.group(3)),
                              t.group(4) or ""))
            elif s:
                sex = "M" if s.group(1).lower() in ("nam", "m") else "F"
                specs.append((sex, int(s.group(2)), s.group(3) or ""))
            elif g:
                # sex=None la co hieu "so o giua la SO THU TU trong bang, khong
                # phai tuoi" - match_cast lay thang cum do, bo qua tuoi/gioi tinh.
                specs.append((None, int(g.group(1)), g.group(2) or ""))
            else:
                hong.append(p.strip())
        for h in hong:
            print(f"    ! bo qua muc khong doc duoc trong {mau.pattern[4:8]}: '{h}'"
                  f" (dung 'nu 30 Ten', '#4 Ten' hoac '@1:23:45 Ten')")
        if specs:
            return {"specs": specs}
    return None


def khong_dau(s):
    """'Lam Tri Ha' / 'Lâm Tri Hạ' -> 'LamTriHa'. Ten file khong dau cho de go.

    Chi viet hoa CHU DAU moi tu, giu nguyen phan con lai - .title() se ha
    'LamTriHa' viet lien thanh 'Lamtriha'.
    """
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    tu = [w[:1].upper() + w[1:] for w in s.split()]
    return "".join(c for c in "".join(tu) if c.isalnum())


def match_cast(clusters, order, specs, tol=AGE_TOL, da_dung=None, order_goc=None):
    """Tim cum khop 'nam ~26' / 'nu ~27'. order da sap theo diem giam dan.

    da_dung: cac cum da bi vai chinh chiem - dung khi khop tiep dong PHU de vai
    phu khong vo trung vao nguoi da co anh rieng.

    Trong so cac cum hop le, so khoang cach tuoi theo BAC 5 NAM truoc, cung bac
    thi lay cum diem cao hon (tuc tan suat/do ro mat). Hai lan chinh deu tung sai
    theo mot huong:
      - Lay diem cao nhat trong +-10 tuoi: hd041108 co be gai 5 tuoi bi doan 16,
        |16-24| = 8 < 10 nen dong 'nu 24' vo phai dua be du cum F~24 that dung
        ke duoi. Tuoi lech 8 nam la tin hieu that, phai nghe.
      - Lay gan tuoi nhat tuyet doi: ct2a041108 dong 'nam 24' bi mot cum phu doan
        dung 24 tuoi cuop cho cua tuong quan (M~22, 92 lan). Lech 2-3 nam chi la
        nhieu do cua InsightFace, khong duoc phep thang tan suat.
    Bac 5 nam tach dung hai truong hop: 8 nam -> bac 1, thua bac 0; 2-3 nam ->
    cung bac 0, roi tan suat quyet dinh.

    Tra ve (picked, miss, ten, vt): ten[i] la ten nguoi cua picked[i]. Phai build
    ten SONG SONG voi picked chu khong lay lai tu specs goc - mot spec miss hoan
    toan lam picked ngan hon specs, khi do moi vai sau cho miss se mang ten cua
    nguoi dung truoc no.

    vt[i] la CHI SO CUA SPEC da sinh ra picked[i]. Can no de chen vai ghim tay
    ('@moc thoi gian') vao dung cho: vai ghim khong di qua ham nay, nen sau do
    phai biet vai khop bang mat nam o o thu may trong dong CAST goc moi danh so
    file dung thu tu nguoi viet.
    """
    # Ghim '#N' phai tra ve bang GOC da in ra man hinh. Sau khi khop CAST xong,
    # reorder_cast day vai chinh len dau nen order da xao tron - luc khop PHU ma
    # tra vao order moi thi '#2' tro thanh nam chinh chu khong phai nguoi #2 trong bang.
    bang = order_goc if order_goc is not None else order
    used, picked, miss, ten, vt = set(da_dung or ()), [], [], [], []
    for i_spec, (sex, age, *rest) in enumerate(specs):
        if sex is None:                  # '#4 Ten' - ghim so thu tu trong bang
            hit = bang[age - 1] if 1 <= age <= len(bang) else None
            if hit is None:
                miss.append(f"#{age} -> bang chi co {len(bang)} nguoi")
            elif hit in used:
                miss.append(f"#{age} -> da bi vai truoc chiem")
                hit = None
            if hit is not None:
                used.add(hit)
                picked.append(hit)
                ten.append(rest[0] if rest else "")
                vt.append(i_spec)
            continue
        hop_le = [cid for cid in order if cid not in used
                  and clusters[cid]["sex"] == sex
                  and abs(clusters[cid]["age"] - age) <= tol]
        hit = min(hop_le, key=lambda c: (abs(clusters[c]["age"] - age) // 5,
                                         order.index(c))) if hop_le else None
        if hit is None:      # khong ai dung tuoi -> lay nguoi cung gioi tinh
            hit = next((cid for cid in order if cid not in used
                        and clusters[cid]["sex"] == sex), None)
            if hit is not None:
                miss.append(f"{sex}~{age} -> lay {clusters[hit]['sex']}"
                            f"~{clusters[hit]['age']} (lech tuoi)")
        if hit is not None:
            used.add(hit)
            picked.append(hit)
            ten.append(rest[0] if rest else "")
            vt.append(i_spec)
        else:
            miss.append(f"{sex}~{age} -> KHONG tim thay ai")
    return picked, miss, ten, vt


def bang_mat(clusters, order, records, it_nhat=8):
    """In bang nguoi xuat hien nhieu nhat: so thu tu, so lan, gioi tinh~tuoi, doan phim.

    Bang nay la thu duy nhat cho biet phai ghim '#4' hay '#2' khi hai dien vien
    cung lua tuoi khong tach duoc bang 'nam 30'. Khoang thoi gian cung to cao:
    mot cum trai deu tu 0m den het phim ma so lan lon bat thuong thuong la HAI
    nguoi bi tron lam mot -> nang SIM len.
    """
    sec_of = {r["path"]: r["sec"] for r in records}
    hang = [(i, clusters[cid]) for i, cid in enumerate(order, 1)
            if len(clusters[cid]["members"]) >= it_nhat]
    if not hang:
        return
    print(f"  bang mat (ghim bang '#so Ten Nguoi' o dong CAST/PHU neu khop sai):")
    for i, c in hang[:10]:
        ss = sorted({int(sec_of.get(f["path"], 0)) for f in c["members"]})
        print(f"    #{i:<3} {len(c['members']):>4} lan  {c['sex']}~{c['age']:<3}"
              f"  {ss[0] // 60}m -> {ss[-1] // 60}m")


def reorder_cast(clusters, order, pick):
    """Dua cac cum duoc chi dinh len dau bang xep hang, giu nguyen phan con lai."""
    rest = [c for c in order if c not in pick]
    new_order = pick + rest
    return new_order, {cid: pos for pos, cid in enumerate(new_order)}, len(pick)


def read_moments(path):
    """Doc file khoanh khac: moi dong 'MM:SS  nhan' hoac 'HH:MM:SS  nhan'.

    File nay do nguoi (hoac Claude) viet ra SAU khi doc phu de, la cach duy nhat
    de buoc chon khung bam theo cot truyen thay vi chi bam vao mat to mat net.
    """
    out, hong = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = TIME_LABEL.match(line)
        if not m:
            # CAST/PHU/SIM/THE_LOAI/THEM do ham khac doc, khong phai dong hong.
            if not any(r.match(line)
                       for r in (CAST_LINE, PHU_LINE, SIM_LINE, GENRE_LINE,
                                 THEM_LINE)):
                hong.append(line.strip())
            continue
        h, mm, ss, label = m.groups()
        out.append({"sec": int(h or 0) * 3600 + int(mm) * 60 + int(ss),
                    "label": label})
    # Nuot lang dong hong la kieu loi te nhat: moc bien mat khoi ket qua ma
    # khong bao gi. Vi du '1:44:35:20' hay '104:42' deu truot regex.
    for h in hong:
        print(f"    ! moc khong doc duoc, DA BO QUA: '{h[:70]}'"
              f"  (dung 'MM:SS  nhan' hoac 'HH:MM:SS  nhan')")
    return out


def pick_by_moments(ranked, moments, n, gap, cast_size):
    """Moi khoanh khac chinh lay MOT khung dep nhat quanh moc thoi gian cua no.

    Con thieu bao nhieu thi bu bang bang xep hang chung. Nho vay 10 anh phu
    dung cac tinh huong cua phim, khong con la 10 anh dep ngau nhien.
    """
    picked, missed = [], []
    for mo in moments[:n]:
        near = [r for r in ranked if abs(r["sec"] - mo["sec"]) <= MOMENT_WINDOW]
        if not near:
            missed.append(mo["label"])
            continue

        # Anh co ca hai vai chinh dang gia hon han lam anh tham chieu, nen cho
        # no di truoc mien la khong te hon khung don qua nhieu.
        duo = [r for r in near if len(r["cast"]) >= 2]
        if duo and duo[0]["score"] >= DUO_PREFER * near[0]["score"]:
            near = duo + [r for r in near if r not in duo]

        for rec in near:                       # ranked da sap theo diem giam dan
            if _fits(rec, picked, min(gap, MOMENT_GAP)):
                rec["moment"] = mo["label"]
                picked.append(rec)
                break
        else:
            missed.append(mo["label"])

    for rec in ranked:                          # bu cho du n anh
        if len(picked) >= n:
            break
        if rec not in picked and _fits(rec, picked, gap):
            picked.append(rec)

    return sorted(picked, key=lambda r: -r["score"]), missed


# ---------- chon khung ----------

def score_frames(records, rank, cast_size, n_vai=None):
    """Khung co nhieu nhan vat chinh cung luc duoc cong manh.

    n_vai: tong so vai CAN CO ANH RIENG (chinh + phu). rec['cast'] liet ke theo
    con so nay de vai phu con duoc nhac den; con diem cong khung dong nguoi thi
    van chi tinh vai CHINH, khong thi mot khung day vai phu lai deo diem cao hon
    khung co du hai nhan vat chinh.
    """
    n_vai = cast_size if n_vai is None else n_vai
    for rec in records:
        best, cast = 0.0, set()
        for f in rec["faces"]:
            is_main = rank.get(f.get("cid"), 99) < n_vai
            crisp = min(f.get("sharp", 0.0) / SHARP_FULL, 1.0)     # 0..1
            best = max(best, f["det"] * (0.5 + f["w"]) * (0.4 + crisp)
                       * (MAIN_BONUS if is_main else 1.0))
            if is_main:
                cast.add(rank[f["cid"]])
        for f in rec["extra"]:
            if "cid" in f and rank.get(f["cid"], 99) < n_vai:
                cast.add(rank[f["cid"]])
        rec["cast"] = sorted(cast)
        rec["n_face"] = len(rec["faces"]) + sum(1 for f in rec["extra"] if "cid" in f)
        n_chinh = sum(1 for w in cast if w < cast_size)
        rec["score"] = best * (1 + DUO_BONUS * max(0, n_chinh - 1))
    return sorted(records, key=lambda r: r["score"], reverse=True)


def _fits(rec, picked, gap):
    if any(abs(rec["sec"] - p["sec"]) < gap for p in picked):
        return False
    return all(int(np.count_nonzero(rec["hash"] != p["hash"])) >= MIN_HAMMING
               for p in picked)


def _solo_quality(rec, rank, who):
    """Do do 'dep' cua mot khuon mat rieng le trong khung: to, net, gan chinh dien."""
    best = 0.0
    for f in rec["faces"]:
        if rank.get(f.get("cid"), 99) != who:
            continue
        crisp = min(f.get("sharp", 0.0) / SHARP_FULL, 1.0)
        straight = max(0.0, 1.0 - f["yaw"] / MAX_YAW)          # 1 = chinh dien
        # Mieng khep hon thi hon. Chi la UU TIEN, khong phai bo loc: thuoc do
        # nay nhieu, nhung so sanh giua cac khung cua CUNG mot nguoi thi dung.
        khep = max(0.0, 1.0 - max(0.0, f.get("mouth", 0.5) - 0.40) / 0.40)
        best = max(best, f["det"] * (0.5 + f["w"]) * (0.4 + crisp)
                   * (0.6 + straight) * (0.55 + 0.45 * khep))
    return best


def group_quality(rec, rank, cast_size):
    """Diem cua khung CHUNG - cham theo mat YEU NHAT, khong theo mat dep nhat.

    Anh chung phai thay ro MOI nhan vat chinh. Neu cham theo mat dep nhat thi
    mot khuon mat cham to se keo ca khung len, con nguoi kia quay nghieng 58 do
    van duoc chon - AI phai bia nua mat con lai va mat giong ngay.
    Chi tinh mat trong bucket CHUAN (yaw<=30, net>=100), khong tinh noi long.
    """
    per = {}
    for f in rec["faces"]:
        who = rank.get(f.get("cid"), 99)
        if who >= cast_size:
            continue
        # Kich thuoc mat la yeu to nang nhat. Moi khung o day deu da qua nguong
        # yaw<=30 roi, nen chenh vai do goc khong dang ke bang chuyen mat to gap
        # ruoi - anh nen bi thu nho ve 16:9 nen mat nho la hong.
        size = min(f["w"] / GOOD_FACE_W, 1.0)
        crisp = min(f.get("sharp", 0.0) / SHARP_FULL, 1.0)
        straight = max(0.0, 1.0 - f["yaw"] / MAX_YAW)
        q = f["det"] * size * (0.4 + crisp) * (0.6 + 0.4 * straight)
        per[who] = max(per.get(who, 0.0), q)
    # 'per and' de khoi min() tren dict rong: cast_size=0 (moi spec CAST deu
    # miss, hoac CAST so toan id ngoai pham vi) lam 0 == 0 thanh True.
    return min(per.values()) if per and len(per) == cast_size else 0.0


def pick_for_thumbnail(ranked, rank, cast_size, moments=None, n_phu=0):
    """Bo anh toi thieu du de gen thumbnail: 1 anh CHUNG + moi nguoi 1 anh RIENG.

    Anh chung de AI biet ho dung canh nhau the nao, anh rieng de AI khoa dung
    khuon mat tung nguoi. Nhieu hon the chi lam AI nhan tin hieu mau thuan.
    """
    # Anh chung: uu tien khung ma MOI vai chinh deu ro mat. Loai nay rat hiem
    # (phim ngan hay quay cang canh mot nguoi) nen phai ha dan yeu cau.
    strict = sorted(((group_quality(r, rank, cast_size), r) for r in ranked),
                    key=lambda x: -x[0])
    strict = [r for q, r in strict if q > 0]

    groups, note = [], ""
    if strict and moments:
        near = [r for r in strict if abs(r["sec"] - moments[0]["sec"]) <= MOMENT_WINDOW]
        if near:
            groups = near[:N_GROUP]
        else:
            groups = strict[:N_GROUP]
            note = ("(khong co khung nao ro mat het vai chinh quanh moc dau tien "
                    "-> lay khung ro mat nhat toan phim.\n"
                    "   Khong sao: anh chung chi de tham khao trang phuc va chieu "
                    "cao, canh thumbnail do muc 3 cua prompt dung ra)")
    elif strict:
        groups = strict[:N_GROUP]
    else:
        # Chi dem vai CHINH - rec['cast'] chua ca vi tri vai phu (score_frames
        # chay voi n_vai = chinh + phu), khong loc thi khung 1 chinh + 2 phu
        # thang khung co du 2 vai chinh.
        n_chinh = lambda r: sum(1 for c in r["cast"] if c < cast_size)
        max_chinh = max(map(n_chinh, ranked))
        full = [r for r in ranked if n_chinh(r) == max_chinh]
        groups = (full or ranked)[:1]
        note = "! khong khung nao ro mat het vai chinh, danh lay khung tot nhat co the."

    # Loai khung trung canh, roi danh so. Xuat vai phuong an vi 'tu the dep hay
    # xau' la thu bo do mat khong cham duoc - phai mat nguoi nhin moi biet.
    keep = []
    for r in groups:
        if _fits(r, keep, 0):
            keep.append(r)
    groups = keep
    for i, r in enumerate(groups, 1):
        r["role"] = "chung"
        r["idx"] = i
    group = groups[0]
    group["note"] = note

    # Anh rieng: voi moi vai chinh, lay khung mat to/net/chinh dien nhat, uu tien
    # khung chi co mot minh ho.
    #
    # Quan trong: phim ngan Trung Quoc dung mat bang AI nen cung mot nhan vat ma
    # moi canh mat mot kieu. Neu anh rieng la kieu mat khac anh chung thi AI ve
    # thumbnail nhan hai tin hieu danh tinh nguoc nhau. Nen phai uu tien khung
    # nao co khuon mat GIONG khuon mat trong anh chung nhat.
    anchor = {}
    for f in group["faces"] + group["extra"]:
        w = rank.get(f.get("cid"), 99)
        if w < cast_size and (w not in anchor or f["w"] > anchor[w]["w"]):
            anchor[w] = f

    solos = []
    for who in range(cast_size + n_phu):
        cand = []
        for r in ranked:
            if who not in r["cast"] or r is group:
                continue
            q = _solo_quality(r, rank, who)
            if q <= 0:
                continue
            q *= 1.35 if r["n_face"] == 1 else 1.0          # thich khung mot minh
            if who in anchor:
                sim = max((float(np.dot(f["emb"], anchor[who]["emb"]))
                           for f in r["faces"] if rank.get(f.get("cid")) == who),
                          default=0.0)
                q *= 0.4 + max(0.0, sim)                     # giong anh chung thi cong
            cand.append((q, r))
        cand.sort(key=lambda x: -x[0])
        for _, rec in cand:
            if _fits(rec, groups + solos, 0):      # chi can khac canh, khong can cach xa
                rec["role"] = f"rieng #{who + 1}"
                rec["who"] = who
                solos.append(rec)
                break
    return groups + solos


def pick_frames(ranked, n, gap, cast_size):
    """Lay n khung diem cao nhat, dam bao MOI nhan vat chinh deu co mat."""
    picked = []
    for rec in ranked:
        if len(picked) >= n:
            break
        if _fits(rec, picked, gap):
            picked.append(rec)

    covered = {c for p in picked for c in p["cast"]}
    for want in range(cast_size):
        if want in covered:
            continue
        for rec in ranked:
            if want not in rec["cast"] or rec in picked:
                continue
            spare = [p for p in picked if not (set(p["cast"]) - covered - {want})]
            drop = spare[-1] if spare else (picked[-1] if picked else None)
            rest = [p for p in picked if p is not drop]
            if _fits(rec, rest, gap):
                picked = rest + [rec]
                covered = {c for p in picked for c in p["cast"]}
                break
    return sorted(picked, key=lambda r: -r["score"])


def hms(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def process(video, cast_size, rebuild, rescan, out_dir=None,
            moments=None, cast_ids=None, sim=None, phu=None, them=None,
            luong=None):
    print("=" * 58)
    print(f"{video.name}")

    # Cache duoc dat ten theo STEM. Hai phim khac nhau trung ten (dot phim moi
    # dat trung ten dot cu) se dung chung cache: anh cat ra la cua phim CU trong
    # khi thoai lai la cua phim MOI. Nhin ket qua khong the biet, vi no van chay
    # tron tru. Kich thuoc file lam dau van tay: hai phim khac nhau trung ca ten
    # lan tung byte do dai la chuyen khong xay ra.
    moc = CACHE_DIR / f"{video.stem}.nguon"
    vt = str(video.stat().st_size)
    if moc.is_file() and moc.read_text().strip() != vt:
        print(f"  ! file nguon doi roi (cung ten '{video.stem}', khac noi dung)"
              f" -> bo cache cu, quet lai tu dau")
        rebuild = rescan = True

    fps = video_fps(video)
    frames = extract_keyframes(video, CACHE_DIR / video.stem, rebuild)
    if not frames:
        print("  Khong trich duoc keyframe nao.")
        return
    print(f"  keyframe: {len(frames)}  (fps={fps:g})")

    cache_file = CACHE_DIR / f"{video.stem}_faces.pkl"
    if not cache_file.is_file() or rescan:
        print("  quet mat bang InsightFace (lan dau se lau vai phut)...")
    records, seen, tally = scan(frames, fps, cache_file, rescan, luong=luong)
    # Ghi dau van tay SAU khi quet xong. Ghi truoc ma dut giua chung thi lan sau
    # dau van tay khop voi mot cache do dang, va cai do dang do duoc dung that.
    moc.write_text(vt)
    print(f"  mat: {seen} tim thay | {tally['chuan']} dat chuan | "
          f"{tally['noi_long']} noi long | {tally['loai']} loai "
          f"(trong do {tally['mo']} do mo)")
    print(f"  khung dung duoc: {len(records)}")
    if not records:
        print("  Khong co khung nao dat. Thu ha MIN_FACE_W / MIN_DET.")
        return

    clusters, order, rank = cluster_faces(records, sim or SAME_PERSON)
    if sim:
        print(f"  nguong nhan dang: {sim}  (mac dinh {SAME_PERSON})")
    # Truyen dung nguong cua phim: cluster o 0.48 ma gan mat noi long o 0.38 la
    # tu mo lai dung cai khe 0.38-0.48 khien hai nguoi bi tron lam mot.
    n_assigned = assign_extras(records, clusters, sim or SAME_PERSON)
    print(f"  nhan dang: {len(clusters)} nguoi ({n_assigned} mat phu duoc gan)")
    bang_mat(clusters, order, records)
    order_goc = list(order)      # bang da in ra - moi ghim '#N' deu tra vao day

    # Vai ghim tay ('@2:18:57 Ten') khong di qua nhan dang mat. Tach ra truoc khi
    # khop bang mat, nho lai o goc cua no de con danh so file dung thu tu.
    specs = (cast_ids or {}).get("specs") or []
    ghim = [(i, s[1], s[2] if len(s) > 2 else "")
            for i, s in enumerate(specs) if s[0] == "@"]
    mat_specs = [s for s in specs if s[0] != "@"]
    mat_vt = [i for i, s in enumerate(specs) if s[0] != "@"]

    # ten[i] = ten nguoi cua order[i]; build song song voi pick de khong lech
    # index khi mot spec miss hoac khi CAST la dang so (khong mang ten).
    ten, vt_chinh, chi_ghim = [], [], False
    if mat_specs:                           # CAST: nam 26, nu 27
        pick, miss, ten, vt = match_cast(clusters, order, mat_specs)
        for m in miss:
            print(f"    ! {m}")
        order, rank, cast_size = reorder_cast(clusters, order, pick)
        vt_chinh = [mat_vt[i] for i in vt]
        note = " (theo mo ta trong file khoanh khac)"
    elif ghim:
        # Moi vai chinh deu ghim tay -> khong con cum nao de cham diem khung.
        # Bo han buoc xep hang: anh chung va anh rieng deu do dong CAST/THEM chi
        # dinh. Day la duong duy nhat dung cho phim ma nhan vat chinh la con vat.
        cast_size, chi_ghim = 0, True
        note = " (ghim tay theo moc thoi gian)"
    elif cast_ids and cast_ids.get("ids"):  # CAST: 1,4  (dang cu)
        pick = [order[i - 1] for i in cast_ids["ids"] if 1 <= i <= len(order)]
        order, rank, cast_size = reorder_cast(clusters, order, pick)
        ten = [""] * cast_size
        note = f" (chi dinh so thu tu: {','.join(map(str, cast_ids['ids']))})"
    elif not cast_size:                     # --cast 0 hoac bo trong -> tu doan
        cast_size = auto_cast_size(clusters, order)
        note = " (tu dong)"
    else:
        note = ""
    if not cast_size and not chi_ghim:
        # Moi spec CAST deu miss (hoac CAST so toan id ngoai pham vi) -> dung
        # tiep voi cast_size=0 la crash o group_quality. Roi ve tu doan.
        cast_size = auto_cast_size(clusters, order)
        ten = []
        vt_chinh = []
        note = " (CAST khong khop ai -> tu doan)"
        print("    ! khong khop duoc vai nao tu CAST, roi ve tu doan")
    ten += [""] * (cast_size - len(ten))
    # Vai phu: khop SAU vai chinh va khong duoc trung. Chung duoc xep ngay sau
    # vai chinh trong 'order' de co anh rieng, nhung KHONG tinh vao cast_size -
    # anh chung chi phai ro mat vai chinh thoi.
    n_phu, vt_phu = 0, []
    ghim_phu = [(i, s[1], s[2] if len(s) > 2 else "")
                for i, s in enumerate(phu or []) if s[0] == "@"]
    phu_mat = [s for s in (phu or []) if s[0] != "@"]
    phu_vt = [i for i, s in enumerate(phu or []) if s[0] != "@"]
    if phu_mat and chi_ghim:
        print("    ! bo qua vai phu khop bang mat: moi vai chinh deu ghim tay nen"
              " khong con buoc cham diem khung. Ghim vai phu bang '@moc' luon.")
    elif phu_mat:
        chinh = order[:cast_size]
        pick_phu, miss, ten_phu, vt = match_cast(clusters, order, phu_mat,
                                                 da_dung=chinh, order_goc=order_goc)
        for m in miss:
            print(f"    ! vai phu: {m}")
        if pick_phu:
            con_lai = [c for c in order if c not in chinh and c not in pick_phu]
            order = chinh + pick_phu + con_lai
            rank = {cid: pos for pos, cid in enumerate(order)}
            n_phu = len(pick_phu)
            ten += ten_phu
            vt_phu = [phu_vt[i] for i in vt]

    print(f"  {cast_size} nhan vat chinh{note}"
          f"{f' + {n_phu} nhan vat phu' if n_phu else ''}"
          f"{f' + {len(ghim) + len(ghim_phu)} vai ghim tay' if ghim or ghim_phu else ''}:")
    for pos, cid in enumerate(order[:cast_size + n_phu]):
        c = clusters[cid]
        vai = "chinh" if pos < cast_size else "phu  "
        nhan = ten[pos] if pos < len(ten) and ten[pos] else ""
        print(f"    {vai} #{pos+1}: {c['n']:4d} lan | {c['sex']} ~{c['age']} tuoi | "
              f"mat TB {c['w_avg']:.2f}  {nhan}")

    them = them or []
    them_chung = [t for t in them if t["vai"] == "chung"]
    picked = []
    if not chi_ghim:
        ranked = score_frames(records, rank, cast_size, cast_size + n_phu)
        picked = pick_for_thumbnail(ranked, rank, cast_size, moments, n_phu)
        n_ro = sum(1 for r in ranked if group_quality(r, rank, cast_size) > 0)
        print(f"  khung ro mat ca {cast_size} vai chinh: {n_ro}/{len(ranked)}")
        if them_chung:
            # Da chon tay canh chung thi bo canh chung tu dong: hai nguon mau
            # thuan nhau con te hon khong co.
            picked = [r for r in picked if r["role"] != "chung"]
            print(f"  anh chung tu dong: BO (da co {len(them_chung)} dong THEM: chung)")
        elif picked and picked[0].get("note"):
            print(f"  {picked[0]['note']}")
        elif moments:
            print(f"  anh chung lay quanh moc: {moments[0]['label']}")

    # CHI xoa anh do chinh tool nay sinh ra. Truoc day xoa ca thu muc bang rmtree,
    # nhung nguoi dung bo file cua ho vao day - title.txt lam mau tieu de, thumb.png
    # la thumbnail dat - nen rmtree la xoa mat cong nguoi ta.
    out_dir = Path(out_dir) if out_dir else REFS_DIR / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for cu in out_dir.glob("*.jpg"):
        if cu.name.startswith(("canh_chung_", "chinh_", "phu_", "vat_",
                               "chung_", "rieng_")):
            cu.unlink()

    # So thu tu file lay theo THU TU NGUOI VIET trong dong CAST/PHU, khong theo
    # thu tu tool khop duoc. Nho vay '@... Trieu Hoanh' dat dau dong CAST van ra
    # chinh_1 du anh ta khong he co cum mat nao.
    def _danh_so(vt_khop, vt_ghim):
        return {v: i + 1 for i, v in enumerate(sorted(set(vt_khop) | set(vt_ghim)))}

    so_chinh = _danh_so(vt_chinh, [g[0] for g in ghim])
    so_phu = _danh_so(vt_phu, [g[0] for g in ghim_phu])

    # Ten file noi ro VAI, vi day la thu nguoi doc thumbnail can biet ngay: ai la
    # nhan vat chinh, ai la vai phu. Ten cu 'rieng_01_M26_f216' bat phai tra lai
    # log moi biet #01 la ai. Kem moc thoi gian de doi chieu nhanh trong video.
    base_name = None
    for rec in picked:
        moc = f"{int(rec['sec']) // 60}m{int(rec['sec']) % 60:02d}"
        if rec["role"] == "chung":
            name = f"canh_chung_{rec['idx']}_{moc}.jpg"
            base_name = base_name or name
        else:
            who = rec["who"]
            c = clusters[order[who]]
            nhan = khong_dau(ten[who]) if who < len(ten) and ten[who] else ""
            nhan = nhan or f"{c['sex']}{c['age']}"
            if who < cast_size:
                so = so_chinh.get(vt_chinh[who], who + 1) if who < len(vt_chinh) \
                    else who + 1
                name = f"chinh_{so}_{nhan}_{moc}.jpg"
            else:
                j = who - cast_size
                so = so_phu.get(vt_phu[j], j + 1) if j < len(vt_phu) else j + 1
                name = f"phu_{so}_{nhan}_{moc}.jpg"
        shutil.copy(rec["path"], out_dir / name)
        ai = ",".join(f"#{c + 1}" for c in rec["cast"]) or "-"
        sh = max((f.get("sharp", 0.0) for f in rec["faces"]), default=0.0)
        print(f"  {name:<34s} {hms(rec['sec']):>8s}  {rec['n_face']} mat  "
              f"nhan vat {ai:<9s} net {sh:5.0f}")

    # --- anh ghim tay: cat thang keyframe gan moc nhat, khong qua nhan dang ---
    def _cat_ghim(sec, ten_nv, mau):
        p, that, lech = ghim_khung(frames, fps, sec)
        if p is None:
            print(f"    ! ghim {hms(sec)}: khong co keyframe nao")
            return None
        moc = f"{int(that) // 60}m{int(that) % 60:02d}"
        name = mau.format(nhan=khong_dau(ten_nv) or "ghim", moc=moc)
        shutil.copy(p, out_dir / name)
        # Keyframe cach nhau vai giay, lech 1-2s la binh thuong. Lech nhieu la
        # doan do khong co keyframe nao - khung lay ve gan nhu chac chan sai canh.
        canh = f"   ! LECH {lech:.0f}s so voi moc yeu cau {hms(sec)}" if lech > 3 else ""
        print(f"  {name:<34s} {hms(that):>8s}  ghim tay{canh}")
        return name

    n_ghim = 0
    for vi, sec, ten_nv in ghim:
        if _cat_ghim(sec, ten_nv, f"chinh_{so_chinh[vi]}_{{nhan}}_{{moc}}.jpg"):
            n_ghim += 1
    for vi, sec, ten_nv in ghim_phu:
        if _cat_ghim(sec, ten_nv, f"phu_{so_phu[vi]}_{{nhan}}_{{moc}}.jpg"):
            n_ghim += 1

    n_vat, i_chung = 0, sum(1 for r in picked if r["role"] == "chung")
    for t in them:
        if t["vai"] == "vat":
            n_vat += 1
            _cat_ghim(t["sec"], t["ten"], f"vat_{n_vat}_{{nhan}}_{{moc}}.jpg")
        else:
            i_chung += 1
            _cat_ghim(t["sec"], t["ten"], f"canh_chung_{i_chung}_{{moc}}.jpg")
            base_name = base_name or f"canh_chung_{i_chung}"

    n_anh = sum(1 for p in out_dir.glob("*.jpg")
                if p.name.startswith(("canh_chung_", "chinh_", "phu_", "vat_")))
    print(f"  -> {out_dir}/   ({n_anh} anh: {i_chung} canh chung + "
          f"{cast_size + n_phu} khop bang mat + {n_ghim} ghim tay + {n_vat} vat)")
    return base_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cast", type=int, default=CAST_SIZE, help="so nhan vat chinh")
    ap.add_argument("--rebuild", action="store_true", help="trich lai keyframe")
    ap.add_argument("--rescan", action="store_true", help="quet lai mat")
    ap.add_argument("--luong", type=int,
                    help="so khung quet cung luc (bo trong = tu chon theo may)")
    a = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("Chua co ffmpeg. Cai bang: brew install ffmpeg")

    videos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS) \
        if INPUT_DIR.is_dir() else []
    if not videos:
        sys.exit(f"Khong tim thay video nao trong {INPUT_DIR}/")

    CACHE_DIR.mkdir(exist_ok=True)
    for v in videos:
        process(v, a.cast, a.rebuild, a.rescan, luong=a.luong)
    print("=" * 58)


if __name__ == "__main__":
    main()
