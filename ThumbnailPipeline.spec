# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
from pathlib import Path
import os

block_cipher = None

# Thu thap toan bo package data, binaries va hidden imports cho InsightFace va ONNXRuntime
datas = [
    # KHONG dong 'models' vao day. Ca hai duong build (workflow GitHub va
    # build_windows.bat) deu da copy thu muc models ra CANH main.exe, ma
    # get_model_path()/build_app() cung tra cho do truoc tien. De o day nua thi
    # bo model (~420MB: buffalo_l + hai ban realesrgan) bi dong goi HAI LAN,
    # gap doi dung luong ban tai ve ma khong duoc gi.
    ('prompt_template.md', '.'),
    ('kenh_mau.tsv', '.'),
]

# Dong goi objects (chua meanshape_68.pkl) vao ca 'objects/' va 'insightface/data/objects/'
# de InsightFace get_object luon tim thay du chay frozen o che do nao
if os.path.isfile(os.path.join('objects', 'meanshape_68.pkl')):
    datas.append(('objects', 'objects'))
    datas.append(('objects', 'insightface/data/objects'))

binaries = []
# Neu co san ffmpeg.exe va ffprobe.exe trong thu muc goc thi tu dong dong goi kem
for exe in ("ffmpeg.exe", "ffprobe.exe"):
    if os.path.isfile(exe):
        binaries.append((exe, '.'))

insight_datas, insight_binaries, insight_hidden = collect_all('insightface')
onnx_datas, onnx_binaries, onnx_hidden = collect_all('onnxruntime')

datas += insight_datas + onnx_datas

# Dam bao copy meanshape_68 tu insight_datas vao objects neu co
for src, dst in insight_datas:
    if 'meanshape_68' in src:
        datas.append((src, 'objects'))

binaries += insight_binaries + onnx_binaries

hiddenimports = [
    'gdown',
    'gdown.exceptions',
    'requests',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'cv2',
    'numpy',
    'queue',
    'threading',
    'build_prompt',
    'extract_char',
    'make',
    'net',
    'noi_shape',
    'tai',
    'title',
] + insight_hidden + onnx_hidden

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX nen DLL lai, va no lam HONG DLL cua ONNX Runtime / OpenCV - kieu loi
    # chi hien khi nguoi dung chay, khong hien luc build. Runner hien khong co
    # UPX nen dong nay dang la vo hieu, tuc mot cai bay cho ngay UPX xuat hien.
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,          # xem ghi chu o EXE() phia tren
    upx_exclude=[],
    name='ThumbnailPipeline',
)
