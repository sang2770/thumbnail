#!/usr/bin/env python3
"""Keo video + srt tu Google Drive (link share cong khai) thang vao input/.

    .venv/bin/python tai.py <link>...   # link FILE truc tiep (.mp4/.srt), may cai cung duoc
    .venv/bin/python tai.py             # khong co link thi doc clipboard - copy ca cum
                                        # link (moi dong mot cai) roi chay chay la duoc

Dung chinh voi link FILE truc tiep; link folder van chay nhung folder nhieu video
thua thi dung gui ca folder. Xong thi bao stem nao du cap video+srt de nho Claude
xu ly (viet moments.txt truoc roi moi chay make.py - dung tu chay tat).

Can file duoc share o che do "anyone with the link". File tren 100MB Drive
hay chan quet virus, gdown tu vuot bang confirm token nen khong sao.

Dut mang giua file 1GB la chuyen thuong: Drive hay tha socket chet ma khong dong.
Tool tu thu lai 6 lan, moi lan tai TIEP tu cho da dung (Range:), file do dang
nam trong input/.dangtai/ nen tat giua chung roi chay lai cung khong mat gi.
"""

import html
import inspect
import re
import sys
import time
from pathlib import Path

import requests

import gdown
from gdown.exceptions import FileURLRetrievalError

import build_prompt          # dung chung ham doc clipboard da lo he dieu hanh

INPUT_DIR = Path(__file__).parent / "input"
# Thu muc dem: file .part nam lai day giua cac lan chay de con tai tiep duoc.
# Dung TemporaryDirectory nhu truoc thi dut mang la mat sach, video 1GB phai
# tai lai tu dau.
TAM_DIR = INPUT_DIR / ".dangtai"
DUOI = {".mp4", ".mkv", ".mov", ".srt", ".ass", ".vtt"}
# gdown <6 can co fuzzy=True moi doc duoc link dang /file/d/<id>/view; tu 6.0 no
# bo tham so nay vi da tu doc moi kieu link. Truyen thua -> TypeError, thieu ->
# link view khong tai duoc. Hoi chu ky ham thay vi doan theo so phien ban.
FUZZY = "fuzzy" in inspect.signature(gdown.download).parameters
LAN_THU = 6               # so lan thu lai moi file
# gdown mac dinh gui User-Agent Chrome 39 (nam 2014). Google tra ve trang xac
# nhan kieu khac cho UA doi do, gdown parse khong ra va nem FileURLRetrievalError
# voi thong bao "may need to change the permission / had many accesses" - nghe
# nhu bi chan nhung khong phai: cung link do curl voi UA moi tai duoc ngay.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
CHO_TOI_DA = 45           # giay khong nhan duoc byte nao thi coi nhu dut


def _ep_timeout():
    """Bat moi request cua gdown phai co timeout doc.

    Drive hay tha socket chet giua chung: tien trinh con song, byte thi dung han
    - do file .part dung yen 37 phut ma khong ai bao gi. gdown khong dat timeout
    nen no doi vinh vien.

    socket.setdefaulttimeout() KHONG chua duoc: requests truyen thang timeout=None
    xuong urllib3, urllib3 goi sock.settimeout(None) -> ghi de mac dinh toan cuc,
    socket tro lai che do cho vo han. Phai chen timeout o tang Session.
    """
    goc = requests.Session.request

    def co_timeout(self, method, url, **kw):
        kw.setdefault("timeout", CHO_TOI_DA)
        return goc(self, method, url, **kw)

    requests.Session.request = co_timeout


_ep_timeout()


def lay_link_clipboard():
    """Nhat MOI dong co drive.google.com trong clipboard - dan mot cum nhieu
    link (video + srt cua vai phim) roi chay mot lenh la keo het."""
    return [d.strip() for d in build_prompt.from_clipboard().splitlines()
            if "drive.google.com" in d]


def la_folder(link):
    return "/folders/" in link or "folderview" in link


def tai_folder(link):
    """Liet ke folder truoc (khong tai), roi chi tai nhung file con thieu.

    gdown.download_folder tai ca folder mot cuc - voi video 1GB thi chay lai
    lan hai la tai lai het tu dau. skip_download=True tra ve danh sach (id,
    duong dan) de minh tu quyet tung file.
    """
    print(f"Doc danh sach folder...")
    ds = gdown.download_folder(url=link, skip_download=True, quiet=True,
                               user_agent=UA)
    if not ds:
        print("  ! Khong doc duoc folder - kiem tra link co phai 'anyone with link' khong.")
        return []
    tai_ve = []
    for f in ds:
        ten = Path(f.path).name          # bo cay thu muc con, do phang vao input/
        if Path(ten).suffix.lower() not in DUOI:
            continue
        dich = INPUT_DIR / ten
        if dich.exists() and dich.stat().st_size > 0:
            print(f"  bo qua (da co)  {ten}")
            continue
        tai_ve.append((f.id, ten))
    print(f"  {len(tai_ve)} file can tai / {len(ds)} file trong folder")
    return tai_ve


def _mot_lan(id_hoac_link, ten, la_link):
    """Goi gdown dung mot lan, tai vao TAM_DIR. Ra duong dan file da xong.

    LUON dat ten dich san khi biet ten. De trong (output='<thumuc>/') thi gdown
    tu doan ten - va voi file lon no doan tu URL chu khong tu Content-Disposition,
    ra 'download?id=...&confirm=t&uuid=...'. Duoi file luc do khong phai .mp4 nen
    o duoi vut luon ban tai xong. Ten dat san con lam moc .part on dinh, tai tiep
    moi dung mieng cu.
    """
    dau = {"resume": True, "user_agent": UA}
    if ten:
        dau["output"] = str(TAM_DIR / ten)
    else:
        dau["output"] = f"{TAM_DIR}/"
    if la_link:
        if FUZZY:
            dau["fuzzy"] = True
        return gdown.download(url=id_hoac_link, **dau)
    return gdown.download(id=id_hoac_link, **dau)


def tai_file(id_hoac_link, ten=None, la_link=False):
    """Tai vao TAM_DIR roi moi chuyen sang input/ - dut giua chung thi input/
    khong dinh file cut, chay lai la tai tiep tu dung cho do.

    Dut mang giua file 1GB la chuyen thuong. Moi lan thu lai gdown gui Range:
    bytes=<da co>- nen khong tai lai phan cu."""
    TAM_DIR.mkdir(parents=True, exist_ok=True)
    ra = None
    for lan in range(1, LAN_THU + 1):
        try:
            ra = _mot_lan(id_hoac_link, ten, la_link)
            if ra:
                break
        except FileURLRetrievalError as e:
            # KHONG phai dut mang: Drive tu choi tra link tai. Hoac file chua
            # mo 'anyone with the link', hoac dang bi chan vi truy cap qua day.
            # Dap lai lien tuc chi lam bi chan nang them - dung han o day.
            print(f"\n  ! Drive tu choi tra link tai.")
            print(f"    - file da mo che do 'Anyone with the link' chua?")
            print(f"    - neu roi thi dang bi chan tam vi truy cap qua nhieu:"
                  f" doi 15-30 phut roi chay lai.")
            return None
        except Exception as e:
            # socket.timeout, ConnectionError, IncompleteRead... deu la mang
            # tach nua chung, thu lai la tai tiep duoc.
            cho = min(5 * 2 ** (lan - 1), 60)
            print(f"\n  ! lan {lan}/{LAN_THU} dut: {type(e).__name__}: {e}")
            if lan < LAN_THU:
                print(f"  doi {cho}s roi tai tiep tu cho da dung...")
                time.sleep(cho)
    if not ra:
        return None
    ra = Path(ra)
    if ra.suffix.lower() not in DUOI:
        print(f"  ! bo qua {ra.name} (khong phai video/phu de)")
        return None
    dich = INPUT_DIR / ra.name
    if dich.exists() and dich.stat().st_size == ra.stat().st_size:
        print(f"  da co san, giu ban cu  {dich.name}")
        ra.unlink()
        return dich
    ra.replace(dich)
    return dich


PHU_DE = {".srt", ".ass", ".vtt"}
TIEU_DE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


def hoi_ten(link):
    """Hoi ten that cua file tren Drive ma KHONG tai gi ca. "" neu khong doc duoc.

    Khong dung duoc gdown cho viec nay: voi file lon Google tra
    'Content-Disposition: inline' (khong kem ten), gdown thay co header do la
    dung tim, roi roi ve lay ten tu URL - ra 'download?id=...&confirm=t&uuid=...'.
    Ten do khong co duoi .mp4 nen ban tai xong bi chinh tool vut di. File nho
    (.srt) thi header lai co ten day du, nen loi chi lo ra o video.

    Trang /file/d/<id>/view co <title> dang 'ten-that.mp4 - Google Drive' cho ca
    hai co - nhe hon nhieu so voi man xac nhan tai cua gdown.
    """
    m = re.search(r"/file/d/([\w-]+)|[?&]id=([\w-]+)", link)
    if not m:
        return ""
    fid = m.group(1) or m.group(2)
    try:
        r = requests.get(f"https://drive.google.com/file/d/{fid}/view",
                         headers={"User-Agent": UA}, timeout=30)
        t = TIEU_DE.search(r.text)
        if not t:
            return ""
        ten = html.unescape(t.group(1)).strip()
    except Exception:
        return ""
    hau = " - Google Drive"
    if not ten.endswith(hau):
        return ""
    ten = ten[:-len(hau)].strip()
    # Trang loi/dang nhap cung co <title> hop le; chi nhan khi ra dung kieu file.
    return ten if Path(ten).suffix.lower() in DUOI else ""


def soan_hang(links):
    """Hoi truoc ten file cua tung link (KHONG tai), roi sap lai hang cho hop ly.

    Hai viec, deu chi lam duoc khi biet ten file tu truoc:

    1. Bo link nao da co san trong input/. Truoc day phai tai het roi moi so
       kich thuoc de phat hien trung - chay lai lenh cu la keo lai ca 3.6GB.
    2. Phu de len truoc, video sau. Phu de vai tram KB, ve trong tich tac; video
       1GB mat ca chuc phut. Co phu de la doc duoc cot truyen va viet moments.txt
       NGAY trong luc video con dang tai. Duong truyen ~40Mbps thi day la cho
       duy nhat con tiet kiem duoc thoi gian - tai nhanh hon thi khong.

    Ton them mot request moi link. Hoi hong (folder, hoac Drive dang chan) thi
    de nguyen link o vi tri cu va cu tai binh thuong.
    """
    hang, bo_qua = [], []
    for i, link in enumerate(links):
        ten = "" if la_folder(link) else hoi_ten(link)
        if ten and (INPUT_DIR / ten).is_file():
            bo_qua.append(ten)
            continue
        hang.append((0 if Path(ten).suffix.lower() in PHU_DE else 1, i, link, ten))
    hang.sort()
    if bo_qua:
        print(f"Da co san, bo qua {len(bo_qua)} file: {', '.join(sorted(bo_qua))}")
    n_sub = sum(1 for uu, *_ in hang if uu == 0)
    if n_sub and n_sub < len(hang):
        print(f"Keo {n_sub} file phu de truoc, {len(hang) - n_sub} video sau.")
    print()
    return [(l, t) for _, _, l, t in hang]


def bao_cao():
    """Sau khi tai: stem nao du cap video+srt ma chua co output thi nhac."""
    videos = {p.stem for p in INPUT_DIR.iterdir()
              if p.suffix.lower() in (".mp4", ".mkv", ".mov")}
    subs = {p.stem for p in INPUT_DIR.iterdir() if p.suffix.lower() == ".srt"}
    out = Path(__file__).parent / "output"
    moi = sorted(s for s in videos & subs if not (out / s).is_dir())
    thieu_sub = sorted(videos - subs)
    if moi:
        print(f"\nStem moi du cap video+srt: {', '.join(moi)}")
        print("-> nho Claude: 'xu ly <ten>' (viet moments.txt truoc roi chay tool)")
    if thieu_sub:
        print(f"Video CHUA co .srt: {', '.join(thieu_sub)}")


def main():
    links = sys.argv[1:] or lay_link_clipboard()
    if not links:
        sys.exit("Khong co link. Dan link vao lenh, hoac copy link roi chay lai.")
    INPUT_DIR.mkdir(exist_ok=True)
    links = soan_hang(links)
    if not links:
        bao_cao()
        return

    for link, ten in links:
        print(f"== {ten or link[:70]}")
        if la_folder(link):
            for fid, t in tai_folder(link):
                print(f"  tai  {t}")
                if tai_file(fid, ten=t) is None:
                    print(f"  ! tai hong {t} - thu lai sau")
        else:
            ra = tai_file(link, ten=ten or None, la_link=True)
            print(f"  -> {ra.name}" if ra else "  ! tai hong - link co public khong?")

    # Con file .part la con do dang: giu lai de lan sau tai tiep, chi don khi sach.
    if TAM_DIR.is_dir():
        con = list(TAM_DIR.iterdir())
        if con:
            print(f"\nCon {len(con)} file do dang trong {TAM_DIR.name}/ - chay lai "
                  f"lenh nay se tai tiep, khong tai lai tu dau.")
        else:
            TAM_DIR.rmdir()

    bao_cao()


if __name__ == "__main__":
    main()
