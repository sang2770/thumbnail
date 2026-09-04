#!/usr/bin/env python3
"""Launcher chinh cua Thumbnail Pipeline.

Chay truc tiep hoac dong goi thanh ThumbnailPipeline.exe cho Windows.
Ho tro ca 2 che do:
  1. Click dup (khong truyen tham so) -> Hien menu tuong tac truc quan de chon
  2. Dong lenh (CLI) -> Chay truc tiep sub-command:
       ThumbnailPipeline.exe tai [link]
       ThumbnailPipeline.exe make [ten_video] [--cast N] [--rescan]
       ThumbnailPipeline.exe net [--rong 1280] [--jpg]
       ThumbnailPipeline.exe title <anh> <tieu_de> [--goc ...]
       ThumbnailPipeline.exe gpu   (kiem tra thong tin GPU & bo tang toc)
"""

import os
import sys
from pathlib import Path

# Dam bao thu muc goc duoc them vao sys.path va PATH khi chay tu PyInstaller bundle
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Tu dong them BASE_DIR vao PATH de ffmpeg.exe / ffprobe.exe duoc nhan dien ngay lap tuc
base_dir_str = str(BASE_DIR)
current_path = os.environ.get("PATH", "")
if base_dir_str not in current_path:
    os.environ["PATH"] = base_dir_str + os.pathsep + current_path


def check_gpu():
    """Kiem tra va in thong tin phan cung GPU, ONNX Runtime Provider (CUDA/DirectML/CoreML)."""
    print("\n" + "=" * 55)
    print("       KIEM TRA PHAN CUNG & BO TANG TOC GPU")
    print("=" * 55)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[*] ONNX Runtime Version: {ort.__version__}")
        print(f"[*] Cac Provider tim thay: {', '.join(providers)}")
        
        has_gpu = False
        if "CUDAExecutionProvider" in providers:
            print("  -> [CUDA]: Co san! (Toi uu cho NVIDIA RTX 20/30/40/50 series)")
            has_gpu = True
        if "DmlExecutionProvider" in providers:
            print("  -> [DirectML]: Co san! (Toi uu cho moi card do hoa Windows, cuc chuan cho RTX 50-series)")
            has_gpu = True
        if "CoreMLExecutionProvider" in providers:
            print("  -> [CoreML]: Co san! (Toi uu cho Apple Silicon Mac)")
            has_gpu = True
        if not has_gpu:
            print("  -> [CPU]: Dang chay tren CPU (chua cai onnxruntime-gpu hoac onnxruntime-directml)")
    except Exception as e:
        print(f"[!] Khong the kiem tra ONNX Runtime: {e}")

    try:
        import cv2
        print(f"[*] OpenCV Version: {cv2.__version__}")
    except Exception as e:
        print(f"[!] Khong the kiem tra OpenCV: {e}")

    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[*] FFmpeg: Da tim thay tai {ffmpeg_path}")
    else:
        print("[!] FFmpeg: CHUA TIM THAY! Hay cai FFmpeg hoac copy ffmpeg.exe vao cung thu muc tool.")
    print("=" * 55 + "\n")


def chay_tai():
    import tai
    tai.main()


def chay_make():
    import make
    make.main()


def chay_net():
    import net
    net.main()


def chay_title():
    import title
    title.main()


def interactive_menu():
    while True:
        print("\n" + "=" * 50)
        print("          THUMBNAIL PIPELINE - MENU CHINH")
        print("=" * 50)
        print("  1. Tai video + sub tu Google Drive (tai.py)")
        print("  2. Trich xuat anh & Tao Prompt Thumbnail (make.py)")
        print("  3. Lam net anh AI trong thu muc net/ (net.py)")
        print("  4. Ghep chu tieu de vao thumbnail (title.py)")
        print("  5. Kiem tra card GPU & bo tang toc AI (CUDA/DirectML)")
        print("  0. Thoat")
        print("-" * 50)
        
        try:
            choice = input("Nhap lua chon cua ban (0-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTam biet!")
            break

        if choice == "1":
            link = input("Nhap link Google Drive (hoac de trong de doc tu clipboard): ").strip()
            sys.argv = ["tai.py"] + ([link] if link else [])
            try:
                chay_tai()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[!] Loi: {e}")

        elif choice == "2":
            ten = input("Nhap ten video (de trong de xu ly tat ca trong input/): ").strip()
            args = ["make.py"]
            if ten:
                args.append(ten)
            extra = input("Tham so phu (vd: --cast 3, --rescan) hoac Enter de bo qua: ").strip()
            if extra:
                args.extend(extra.split())
            sys.argv = args
            try:
                chay_make()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[!] Loi: {e}")

        elif choice == "3":
            print("Chon che do lam net:")
            print("  1. Chuan 1280x720 JPG (Khuyen dung cho thumbnail YouTube)")
            print("  2. Phong to 2x (PNG)")
            print("  3. Tu nhap tham so")
            opt = input("Nhap (1/2/3, Enter mac dinh la 1): ").strip()
            if opt == "2":
                sys.argv = ["net.py", "--scale", "2"]
            elif opt == "3":
                custom = input("Nhap tham so (vd: --rong 1920 --jpg): ").strip()
                sys.argv = ["net.py"] + (custom.split() if custom else [])
            else:
                sys.argv = ["net.py", "--rong", "1280", "--jpg"]
            try:
                chay_net()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[!] Loi: {e}")

        elif choice == "4":
            anh = input("Nhap duong dan file anh (vd: output/phim1/thumb.png): ").strip()
            if not anh:
                print("Chua nhap file anh!")
                continue
            td = input("Nhap tieu de tieng Viet: ").strip()
            goc = input("Goc dat chu (tren-trai, tren-phai, duoi-trai, duoi-phai - Enter mac dinh duoi-phai): ").strip()
            args = ["title.py", anh, td]
            if goc:
                args.extend(["--goc", goc])
            sys.argv = args
            try:
                chay_title()
            except SystemExit:
                pass
            except Exception as e:
                print(f"[!] Loi: {e}")

        elif choice == "5":
            check_gpu()

        elif choice == "0":
            print("Tam biet!")
            break
        else:
            print("[!] Lua chon khong hop le, vui long chon lai.")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        # Chay sub-command truc tiep qua CLI
        if cmd == "tai":
            sys.argv = ["tai.py"] + sys.argv[2:]
            chay_tai()
        elif cmd == "make":
            sys.argv = ["make.py"] + sys.argv[2:]
            chay_make()
        elif cmd == "net":
            sys.argv = ["net.py"] + sys.argv[2:]
            chay_net()
        elif cmd == "title":
            sys.argv = ["title.py"] + sys.argv[2:]
            chay_title()
        elif cmd in ("gpu", "check", "--check"):
            check_gpu()
        else:
            print(f"Lenh khong hop le: '{cmd}'. Cac lenh ho tro: tai, make, net, title, gpu")
            print("Hoac chay khong truyen tham so de mo menu tuong tac.")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
