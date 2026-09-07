#!/usr/bin/env python3
"""Noi shape co dinh cua realesrgan_x4.onnx thanh dong: [1,3,128,128] -> [N,3,H,W].

    python3 noi_shape.py                      # models/realesrgan_x4.onnx -> ..._dong.onnx
    python3 noi_shape.py vao.onnx ra.onnx

VI SAO SUA DUOC. Graph nay THUAN convolution: Conv / LeakyRelu / Concat / Add /
Mul / Clip, cong hai node Resize dung HE SO TI LE (1,1,2,2) chu khong dung kich
thuoc dich. Khong co node Reshape nao ghim cung batch. Nen chi can sua KHAI BAO
shape o dau vao/dau ra, khong dung toi mot node nao - da kiem chung: dau ra
giong het den tung bit.

VI SAO CAN. File goc ghim batch=1 va o 128x128, nen:
  - phep thu batch trong net.py LUON nem loi -> ca nhanh batch la code chet;
  - moi diem anh bi tinh (128/96)^2 = 1.78 lan vi vien chong lan.
Noi ra thi o to hon chay duoc (do duoc: 111.8us -> 69.3us moi pixel huu ich) va
batch chay that tren GPU.

DE RIENG MOT FILE. Buoc build tren GitHub goi dung ham nay. Neu de trong net.py
thi phai 'import net', ma net.py nap cv2 + numpy + onnxruntime ngay tu dau -
mot buoc chi viet lai metadata cua file .onnx khong co ly gi phai can OpenCV.
Da bi dinh dung loi do: build do vi ModuleNotFoundError: No module named 'cv2'.
"""

import sys
from pathlib import Path

MAC_DINH_VAO = Path("models/realesrgan_x4.onnx")


def noi_shape_dong(src, dest):
    """Sua khai bao shape thanh dong. Tra ve True neu ghi duoc file dest.

    Khong nem loi: thieu thu vien onnx hoac thu muc chi doc thi tra ve False de
    ben goi con duong roi ve (chay o 128 nhu cu).
    """
    src, dest = Path(src), Path(dest)
    try:
        import onnx
    except ImportError:
        return False
    try:
        m = onnx.load(str(src))
        for ten, vals in ((m.graph.input, ("N", "H", "W")),
                          (m.graph.output, ("N", "H4", "W4"))):
            for v in ten:
                d = v.type.tensor_type.shape.dim
                if len(d) != 4:
                    return False
                for k, param in zip((0, 2, 3), vals):
                    d[k].ClearField("dim_value")
                    d[k].dim_param = param
        # Bo het shape cua cac tensor TRUNG GIAN. Ban export goc ghi cung ca
        # 1093 cai theo co 128x128; de nguyen thi ONNX Runtime suy nguoc ra dau
        # ra van phai la 512x512 va canh bao om om moi o mot dong. Day chi la
        # metadata goi y, xoa di thi Runtime tu suy lai theo shape thuc.
        del m.graph.value_info[:]
        # Ghi ra file tam roi doi ten: dut giua chung thi khong de lai mot file
        # .onnx do dang ma lan sau lai tuong la dung.
        tam = dest.with_suffix(".onnx.tam")
        onnx.save(m, str(tam))
        tam.replace(dest)
        return True
    except Exception as e:
        print(f"  ! khong noi duoc shape thanh dong ({e})")
        return False


def main():
    vao = Path(sys.argv[1]) if len(sys.argv) > 1 else MAC_DINH_VAO
    ra = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        vao.with_name(vao.stem + "_dong.onnx")
    if not vao.is_file():
        sys.exit(f"Khong thay {vao}")
    if not noi_shape_dong(vao, ra):
        sys.exit(f"That bai: khong tao duoc {ra}")
    print(f"da tao {ra} ({ra.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    main()
