# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files
from pathlib import Path
import os

block_cipher = None

# Thu thap toan bo package data, binaries va hidden imports cho InsightFace va ONNXRuntime
datas = [
    ('models', 'models'),
    ('prompt_template.md', '.'),
    ('kenh_mau.tsv', '.'),
]

binaries = []
# Neu co san ffmpeg.exe va ffprobe.exe trong thu muc goc thi tu dong dong goi kem
for exe in ("ffmpeg.exe", "ffprobe.exe"):
    if os.path.isfile(exe):
        binaries.append((exe, '.'))

insight_datas, insight_binaries, insight_hidden = collect_all('insightface')
onnx_datas, onnx_binaries, onnx_hidden = collect_all('onnxruntime')

datas += insight_datas + onnx_datas
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='ThumbnailPipeline',
)
