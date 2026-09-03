@echo off
chcp 65001 >nul
title BUILD THUMBNAIL PIPELINE - WINDOWS EXE
cls

echo =====================================================================
echo          HE THONG TU DONG DONG GOI THUMBNAIL PIPELINE SANG EXE
echo    Ho tro toi uu GPU DirectML / CUDA (Tuong thich ca NVIDIA RTX 50x)
echo =====================================================================
echo.

:: 1. Kiem tra Python
echo [*] Buoc 1: Kiem tra Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] LOI: Khong tim thay Python!
    echo     Vui long cai dat Python 3.10 hoac 3.11 tu https://www.python.org/
    echo     Luu y nho tich chon "Add python.exe to PATH" khi cai.
    goto :LOI
)
python --version

:: 2. Tao hoac kich hoat moi truong ao .venv
echo.
echo [*] Buoc 2: Kiem tra moi truong ao (.venv)...
if not exist ".venv\Scripts\activate.bat" (
    echo     -> Dang tao moi truong ao .venv...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Khong the tao .venv. Kiem tra lai quyen ghi thu muc.
        goto :LOI
    )
)

echo     -> Dang kich hoat .venv...
call .venv\Scripts\activate.bat

:: 3. Nang cap pip va cai thu vien
echo.
echo [*] Buoc 3: Cai dat cac thu vien can thiet...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pyinstaller

:: Cai dat goi tang toc GPU cho Windows (DirectML tuong thich 100% RTX 50-series / 40-series / AMD / Intel)
echo.
echo [*] Buoc 4: Cai dat bo tang toc GPU DirectML (cho RTX 50x va moi GPU)...
pip install onnxruntime-directml

:: 4. Kiem tra FFmpeg
echo.
echo [*] Buoc 5: Kiem tra cong cu FFmpeg...
if not exist "ffmpeg.exe" (
    where ffmpeg >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo     -> Tim thay ffmpeg trong PATH he thong. Dang copy vao thu muc du an...
        for /f "delims=" %%i in ('where ffmpeg') do (
            copy /y "%%i" ".\ffmpeg.exe" >nul
            goto :COPY_FFMPEG_XONG
        )
    ) else (
        echo [!] CANH BAO: Khong tim thay ffmpeg.exe trong thu muc du an hoac PATH!
        echo     De thumbnail hoat dong on dinh, hay tai ffmpeg.exe va ffprobe.exe
        echo     tu https://www.gyan.dev/ffmpeg/builds/ va dat vao thu muc nay.
    )
) else (
    echo     -> Da co ffmpeg.exe trong thu muc du an.
)
:COPY_FFMPEG_XONG

if not exist "ffprobe.exe" (
    where ffprobe >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        for /f "delims=" %%i in ('where ffprobe') do (
            copy /y "%%i" ".\ffprobe.exe" >nul
            goto :COPY_FFPROBE_XONG
        )
    )
) else (
    echo     -> Da co ffprobe.exe trong thu muc du an.
)
:COPY_FFPROBE_XONG

:: 5. Kiem tra file model ONNX
echo.
echo [*] Buoc 6: Kiem tra file model...
if not exist "models\realesrgan_x4.onnx" (
    echo [!] CANH BAO: Khong tim thay models\realesrgan_x4.onnx!
    echo     Chuc nang lam net se can file model nay.
) else (
    echo     -> Da co models\realesrgan_x4.onnx.
)

:: 6. Chay PyInstaller
echo.
echo [*] Buoc 7: Bat dau dong goi bang PyInstaller (se mat 1-3 phut)...
echo =====================================================================
pyinstaller --clean ThumbnailPipeline.spec
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] LOI: PyInstaller dong goi that bai! Hay kiem tra thong bao loi o tren.
    goto :LOI
)

:: 7. Tao thu muc input, output, net san cho ban dist
echo.
echo [*] Buoc 8: Hoan thien thu muc phan phoi (dist\ThumbnailPipeline)...
if not exist "dist\ThumbnailPipeline\input" mkdir "dist\ThumbnailPipeline\input"
if not exist "dist\ThumbnailPipeline\output" mkdir "dist\ThumbnailPipeline\output"
if not exist "dist\ThumbnailPipeline\net" mkdir "dist\ThumbnailPipeline\net"

:: Copy ffmpeg vao dist neu co
if exist "ffmpeg.exe" copy /y "ffmpeg.exe" "dist\ThumbnailPipeline\" >nul
if exist "ffprobe.exe" copy /y "ffprobe.exe" "dist\ThumbnailPipeline\" >nul

echo.
echo =====================================================================
echo                      BUILD THANH CONG!
echo =====================================================================
echo  Thu muc chuong trinh hoan chinh da san sang tai:
echo      dist\ThumbnailPipeline\
echo.
echo  File chay chinh:
echo      dist\ThumbnailPipeline\ThumbnailPipeline.exe
echo.
echo  Ban chi can nen (zip) thu muc dist\ThumbnailPipeline la co the gui
echo  cho nguoi khac su dung ngay lap tuc ma khong can cai dat Python.
echo =====================================================================
echo.
pause
exit /b 0

:LOI
echo.
echo =====================================================================
echo                  QUA TRINH BUILD GAP SU CO!
echo =====================================================================
echo Vui long xem lai cac dong bao loi phia tren de khac phuc.
echo.
pause
exit /b 1
