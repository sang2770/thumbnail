#!/usr/bin/env python3
"""Doc file .srt trong input/ -> ghep vao prompt_template.md -> prompt san de dan.

    python3 build_prompt.py           # tao prompt, chep luon vao clipboard
    python3 build_prompt.py --no-copy # khong dung clipboard

Ket qua: prompts/<ten>_buoc1.txt
Sua noi dung prompt thi sua file prompt_template.md, khong can sua script nay.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path("input")
OUT_DIR = Path("prompts")
def get_template_path():
    possible_dirs = []
    if getattr(sys, "frozen", False):
        possible_dirs.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            possible_dirs.append(Path(sys._MEIPASS))
    possible_dirs.append(Path(__file__).resolve().parent)
    possible_dirs.append(Path.cwd().resolve())

    for d in possible_dirs:
        cand = d / "prompt_template.md"
        if cand.is_file():
            return cand
    return Path("prompt_template.md")

TEMPLATE = get_template_path()


# Gio la tuy chon (MM:SS,mmm van gap), milli-giay toi 4 chu so.
TIMECODE = re.compile(r"^(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{1,4}\s*-->")
INDEX = re.compile(r"^\d+$")
TAGS = re.compile(r"<[^>]+>|\{[^}]+\}")          # <i>, {\an8}...


def parse_srt(path):
    """Bo so thu tu, moc thoi gian, the dinh dang -> chi con thoai.

    Phai parse theo KHOI (cac dong tach nhau boi dong trong) chu khong loc tung
    dong: dong so thu tu chi la so thu tu khi no dung DAU khoi va ngay truoc
    timecode. Loc moi dong thuan so thi cau thoai '50000' (gia tien - phim nao
    cung co) bi nuot mat.

    utf-8-sig: file co BOM thi '\\ufeff1' khong khop INDEX va lot vao thoai.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    out, prev = [], None
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [TAGS.sub("", l).strip() for l in block.splitlines()]
        lines = [l for l in lines if l]
        if lines and INDEX.match(lines[0]) and len(lines) > 1 and TIMECODE.match(lines[1]):
            lines = lines[1:]
        if lines and TIMECODE.match(lines[0]):
            lines = lines[1:]
        for s in lines:
            if s == prev:                         # bo dong lap lien tiep
                continue
            out.append(s)
            prev = s
    return out


ATTACH_NOTE = """Toàn bộ lời thoại nằm trong FILE ĐÍNH KÈM tên `{FILE}`.

Hãy mở file đó và đọc HẾT từ đầu đến cuối trước khi làm bất kỳ mục nào ở trên.
Phim có thể chia nhiều phần hoặc nhiều mốc thời gian — đọc thiếu phần cuối là
tóm tắt sai.

Nếu bạn không thấy file đính kèm, hãy nói ngay. Tuyệt đối không tự bịa nội dung."""


TONG_TAI = """## THỂ LOẠI PHIM NÀY: **TỔNG TÀI — hiện đại, hào môn, thương trường**

Toàn bộ mục 3 và mục 4 phải khớp với thể loại này:

- **Bối cảnh:** đại sảnh biệt thự, sảnh khách sạn năm sao, phòng họp kính tầng cao, tiệc cưới, sảnh toà án, bệnh viện tư. Chọn theo tuyến chính của phim.
- **Trang phục:** vest tối màu cắt may, sơ mi lụa, váy dạ hội hoặc váy lụa ôm, đồng hồ, khuy măng sét, giày da bóng. Không quần áo lao động, không đồ cổ trang.
- **Vật của cải (nhóm A):** cọc tiền mặt bó dây, thẻ ngân hàng đen, chìa khoá xe sang, hộp trang sức mở nắp, sổ đỏ, séc, cặp da đầy tiền.
- **Chữ trên đồ vật:** vật in sẵn hoặc màn hình thì TIẾNG ANH IN HOA hoặc chữ số; giấy viết tay thì chữ Hán.
- **Quyền lực trong phim này đo bằng TIỀN và ĐỊA VỊ** — ai ký được, ai mua đứt được, ai bị đuổi khỏi công ty.
- **Đám đông chứng kiến:** khách dự tiệc mặc lễ phục, vệ sĩ vest đen đeo tai nghe, thư ký cầm cặp tài liệu, phóng viên.

**Bố cục đã được duyệt cho thể loại này** — dựng theo đúng bộ khung sau, chỉ thay nội dung:

Nhân vật chính đứng giữa khung, tay đưa vật phẳng cầm tay (giấy tờ, thẻ) **chếch về phía kẻ thua, mặt vật xoay ra ống kính**, ở tiền cảnh chiếm gần một phần tư khung. Đèn chùm pha lê treo ngay trên đầu. Vali tiền mặt mở nắp và hộp trang sức rải trong hậu cảnh, mỗi thứ một góc thấp. Hai kẻ thua quỳ hoặc chống tay xuống sàn đá bóng — chính là cặp `sát mép trái khung` / `sát mép phải khung` duy nhất của prompt. Hai vệ sĩ vest đen đứng sâu phía sau, cách xa nhau. Đám đông lễ phục che miệng kinh ngạc ở hậu cảnh.

Biến thể ăn khách ở kênh top khi phim có **đám cưới**: cô dâu đứng giữa làm tâm, mọi phe vây quanh thành nhóm đối đầu, châu báu/hộp quà đổ vương vãi trên bàn tiền cảnh, có thể thêm một cảnh sát đang còng tay kẻ gian ở mép khung — vụ bắt giữ ngay trong tiệc cưới là cú sốc thị giác mạnh.

**Công thức tiêu đề riêng thể loại này** — giữ nguyên nhịp bản lề đã đo: `<hoàn cảnh>, <Nào Ngờ/Ai Ngờ> <bước ngoặt> Khiến <hệ quả>`. Nhóm view cao của thể loại tổng tài vẫn dùng đúng công thức này."""


CO_TRANG = """## THỂ LOẠI PHIM NÀY: **CỔ TRANG — cung đình, phủ đệ, thời phong kiến Trung Hoa**

Toàn bộ mục 3 và mục 4 phải khớp với thể loại này:

- **Bối cảnh:** kim loan điện, chính sảnh phủ đệ, sân đá lát, hành lang gỗ chạm, yến tiệc bày mâm đồng, hiệu thuốc kê tủ thuốc bắc, ngự hoa viên.
- **Trang phục:** áo bào tay rộng nhiều lớp, đai ngọc, mũ miện, trâm cài tóc, bội ngọc, hài thêu. **TUYỆT ĐỐI không vest, không váy hiện đại, không đồng hồ, không kính.** Riêng nữ chính vẫn phải theo luật nữ chính ở mục 3: áo trong bó sát, khoác ngoài lụa mỏng buông hờ, thắt lưng siết rõ eo — nhiều lớp không có nghĩa là giấu hết dáng người.
- **Vật của cải (nhóm A):** nén bạc, thoi vàng xếp trong hộp gỗ chạm, ngọc bội, chuỗi trân châu, cuộn gấm vóc, khế ước nhà đất, mâm đồng chất lễ vật.
- **Chữ trên đồ vật:** **CHỮ HÁN PHỒN THỂ viết dọc từ trên xuống, 2-4 chữ, kèm dấu son đỏ.** Cấm tuyệt đối chữ Latin — một tờ giấy in tiếng Anh giữa đại sảnh gỗ chạm làm hỏng cả tấm ảnh. Nhớ nối cụm chặn chữ Latin vào cuối dòng Negative.
- **Quyền lực trong phim này đo bằng THỨ BẬC** — ai phải quỳ trước ai, ai được ngồi, ai bị lôi ra, ai đọc chiếu chỉ.
- **Đám đông chứng kiến:** thị vệ đội mũ giáp, cung nữ bưng khay, thái giám, quan viên đội mũ ô sa, gia nhân cúi đầu.

**Bố cục đã được duyệt cho thể loại này** — dựng theo đúng bộ khung sau, chỉ thay nội dung:

Nhân vật chính đứng giữa khung trong áo bào đỏ thêu kim tuyến, tay đưa **một CUỘN GIẤY có trục gỗ bịt đồng hai đầu** chếch về phía kẻ thua, mặt cuộn xoay ra ống kính, ở tiền cảnh chiếm gần một phần tư khung — không phải tờ giấy phẳng, cuộn mới đúng đồ cổ. Trên cuộn: hai chữ Hán lớn viết dọc, bên cạnh vài cột chữ nhỏ, dấu son đỏ vuông đóng ở đáy. Rèm lụa đỏ và cột sơn son ở hậu cảnh. Kẻ thua quỳ và người khóc chính là cặp `sát mép trái khung` / `sát mép phải khung` duy nhất của prompt. Thị vệ đội mũ giáp đứng sâu phía sau, cách xa nhau. Quan viên đội mũ ô sa che miệng phía sau. Nén vàng, ngọc tỷ, hộp gỗ chạm rải trong hậu cảnh.

Hai chi tiết kênh top hay dùng, thêm được thì thêm: **băng rôn vải đỏ treo dọc mang 4-6 chữ Hán lớn** ở hậu cảnh (chữ dọc trên vải đỏ vừa đúng thời đại vừa là mảng đỏ hút mắt), và cánh hoa đào/lá rơi lơ lửng làm chi tiết chuyển động.

**Công thức tiêu đề riêng thể loại này** — nhóm view cao mở bằng biến cố + trả thù: `Trọng Sinh`/`Trùng Sinh` (hai cách viết đều đang chạy trong ngách, chọn một dùng nhất quán trong cả 5 câu) / `Đêm Tân Hôn` / `Đích Nữ` ngay đầu câu, kết bằng `Khiến <cả phủ/hầu gia/kinh thành> <mất tất cả/chấn động>`. Ví dụ nhịp (cấm chép): `Trùng Sinh Ngày Mẹ Bị Hãm Hại, Đích Nữ Bày Mưu Báo Thù Khiến Hầu Gia Mất Tất Cả Chấn Động Kinh Thành`."""


SAN_BAT = """## THỂ LOẠI PHIM NÀY: **SĂN BẮT — đi rừng, đi biển, hệ thống, dị năng**

Toàn bộ mục 3 và mục 4 phải khớp với thể loại này. Đây là thể loại DỄ VIẾT SAI NHẤT vì phản xạ mặc định luôn kéo về đại sảnh biệt thự — ở đây thì sai hoàn toàn:

- **Bối cảnh:** bãi biển lúc triều rút, ghềnh đá, làng chài, bến cá, chợ hải sản, bìa rừng, sườn núi tuyết, sân phơi trước nhà tranh, lều bạt. Nhà cửa phải nghèo và thật — vách gỗ, mái tôn, sân đất.
- **Trang phục:** áo lao động bạc màu, áo bông vá, ủng cao su, áo mưa, quần xắn ống, khăn trùm đầu, găng tay vải. **Không vest, không váy dạ hội, không trang sức.** Ba dòng cấm này KHÔNG áp cho dáng người của nữ chính: theo luật nữ chính ở mục 3, cô vẫn phải mặc áo trong vừa khít, cúc trên mở, tay xắn cao, áo ngoài mở phanh, có thắt lưng siết eo. Vải cứ thô cứ bạc màu, nhưng không được là khối áo bông vuông vức.
- **Vật của cải (nhóm A) — đây là điểm khác biệt lớn nhất:** thứ hút mắt KHÔNG phải cọc tiền mà là **CHIẾN LỢI PHẨM CỠ LỚN lấy từ thiên nhiên** — con hổ vắt trên vai, con tôm hùm vua vằn vàng, con cua khổng lồ, sọt cá đầy ắp, củ nhân sâm nhiều nhánh, tảng trư sa. Bắt buộc có **đúng MỘT con vật hoặc vật phẩm to bất thường** ở tiền cảnh, to đến mức người xem phải dừng lại nhìn. Cọc tiền mặt và giấy nợ chỉ là lớp phụ trợ đứng sau nó.
- **Chữ trên đồ vật:** giấy nợ viết tay, biển hiệu chợ quê, bao tải ghi chữ → **chữ Hán** 2-4 chữ kèm chữ số. Giấy in sẵn hoặc màn hình → tiếng Anh in hoa hoặc chữ số.
- **Nếu phim có "hệ thống" trong đầu nhân vật: TUYỆT ĐỐI không vẽ giao diện HUD, không bảng chữ nổi, không khung thông báo phát sáng.** Model tạo ảnh in ra chữ méo và ảnh lập tức trông như ảnh chế. Sức mạnh của hệ thống thể hiện bằng KẾT QUẢ — sọt đầy, con vật to — chứ không bằng giao diện.
- **Quyền lực trong phim này đo bằng TAY NGHỀ và cái bắt được** — ai vác được con thú về, ai gọi đúng chỗ có hàng, ai được lái buôn tranh nhau mua.
- **Đám đông chứng kiến:** dân làng áo vá, dân chài, lái buôn xách cân, người vác sọt, trẻ con — mặt kinh ngạc, không phải mặt khinh miệt kiểu quý tộc.

**Bố cục đã được duyệt cho thể loại này** — khác hẳn hai thể loại kia, đọc kỹ. Mẫu này đối chiếu với 8 thumbnail 300k-843k view của các kênh cùng ngách, chúng thống nhất đến kỳ lạ:

Khung là **bức tranh RỘNG đầy ắp**, không phải chân dung cận mặt: nhân vật chính cao khoảng **một phần ba khung** (thấp hơn chuẩn 40-45% của hai thể loại kia — đây là ghi đè có chủ ý), một tay **giơ cao chiến lợi phẩm toả hào quang vàng** — củ nhân sâm, con tôm hùm vua, cây cung — phần khung còn lại nhường cho sự SUNG TÚC.

1. **Của cải TRÀN tiền cảnh, nhiều chủng loại.** Kênh top không bày một con vật mà bày cả chợ: sọt cá đầy, thịt treo, lồng thỏ gà, bao tải tiền, hộp đồ. Sự nhiều mới là thông điệp — "người này làm ra tất cả những thứ đó".
2. **Người thân đứng SÁT CẠNH nhân vật chính** (vợ/chồng bế con, mẹ già, em nhỏ) — thể loại này bán giấc mơ đổi đời cho cả nhà, không bán màn xử kẻ thù. Người thân **ngước nhìn chiến lợi phẩm** kinh ngạc, chỉ nhân vật chính nhìn ống kính.
3. **Thú rừng vây quanh làm nền:** lợn rừng, hươu, sói nằm phục, đại bàng trên trời. Dân làng há miệng kinh ngạc ở một mép — tay để yên, không chỉ trỏ (luật đám đông vẫn giữ).
4. **Trời sáng rực ban ngày, núi xanh** — không đèn chùm, không nội thất.
5. Hào quang vàng/lam được phép trên **vật phẩm và vũ khí** (mũi tên phát sáng, dòng sáng quanh tay) — cách duy nhất thể hiện "hệ thống". Giao diện chữ HUD vẫn cấm; kênh top thỉnh thoảng vẽ một tấm bảng sáng mờ nhưng đó là canh bạc chữ méo, không bắt chước.
6. **Tông mặt: quyết tâm hoặc ấm áp** — thể loại này KHÔNG cần mặt lạnh báo thù. Môi vẫn khép (nụ cười mím), nhưng ánh mắt sáng, không nheo lạnh.

Không có giấy tờ nào trong khung — vật mang chữ thuộc hai thể loại kia. Thông tin ở đây nằm ở KÍCH THƯỚC và SỐ LƯỢNG.

**Công thức tiêu đề riêng thể loại này** — đo từ nhóm video view cao nhất (20/60 câu top có "Săn", 14/60 "Đổi Đời", chỉ 8/60 dùng bản lề "Nào Ngờ"):

```
<Trọng Sinh/Xuyên Không + hoàn cảnh khốn cùng>, Dùng <lợi thế bất công> <hành động săn/hái> <Đổi Đời / Thành Tỷ Phú / Cứu Cả Nhà>
```

Ví dụ nhịp (cấm chép): `Trọng Sinh Về Trước Ngày Tan Cửa Nát Nhà, Chàng Trai Dùng Ký Ức Lên Núi Săn Thú Khủng Lấy Vợ Đẹp`. Mở đầu bằng chính từ khoá thể loại `Trọng Sinh`/`Xuyên Không` (36/60 câu top mở kiểu này) — với thể loại này nó là cái móc, không cần giấu. Lợi thế phải gọi tên cụ thể: `Ký Ức Kiếp Trước`, `Kiến Thức Hiện Đại`, `Không Gian`, `Hiểu Tiếng Động Vật`."""


MAC_DINH = """## THỂ LOẠI PHIM NÀY: **chưa xác định từ tên file**

Tên file không mang ký hiệu thể loại quen thuộc. Hãy tự đọc thoại để xác định phim
thuộc loại nào — tổng tài hiện đại, cổ trang, hay săn bắt/hệ thống — rồi giữ bối
cảnh, trang phục và vật của cải nhất quán với loại đó suốt mục 3 và mục 4. Đừng
mặc định đại sảnh biệt thự nếu phim không diễn ra ở đó."""


# Ky hieu dat ten file -> the loai. Nguoi dung dat ten theo quy uoc nay.
THE_LOAI = {
    "tt": TONG_TAI,     # tong tai
    "ct": CO_TRANG,     # co trang
    "hd": SAN_BAT,      # san ban / danh bat / he thong
    "tn": SAN_BAT,
}


def genre_block(stem, ep=None):
    """Doc ky hieu dau ten file -> khoi mo ta the loai chen vao dau prompt.

    Moi phim chi thay khoi cua RIENG no, khong thay hai khoi kia. Neu nhet ca ba
    vao template thi prompt phinh them ~40 dong va model phai tu chon - ma no
    hay chon nham sang doan sang trong nhat, tuc la doan tong tai.

    ep: ky hieu ghi thang trong file khoanh khac ('THE_LOAI: hd'), thang tien to
    ten file. Ten file khong phai luc nao cung dung the loai - mot lo phim co the
    dat ten theo ngay thay vi theo noi dung, luc do 'tt' co the la phim tham hoa
    tren bien. Doan sai the loai la prompt bat AI ve dai sanh biet thu cho mot
    phim ca map, sai tu goc.
    """
    if ep:
        kh = str(ep).strip().lower()
        if kh in THE_LOAI:
            return THE_LOAI[kh]
        print(f"    ! THE_LOAI: '{ep}' khong co trong danh sach"
              f" ({', '.join(sorted(THE_LOAI))}) -> doc theo ten file")
    s = Path(stem).stem.lower()
    for ky_hieu in sorted(THE_LOAI, key=len, reverse=True):
        if s.startswith(ky_hieu):
            return THE_LOAI[ky_hieu]
    return MAC_DINH


def tieu_de_mau(out_dir=Path("output"), bo_qua=None):
    """Gom cac tieu de NGUOI DUNG da duyet tu output/*/title.txt lam mau few-shot.

    Khac han vi du cua kenh tham chieu: day la cau chinh nguoi dung da chon dung
    cho kenh cua ho, nen no mang ca giong van lan gu chon goc nhin. File trong
    thi bo qua - co nghia phim do chua chot duoc tieu de nao dang hoc.

    Bo qua chinh phim dang lam, khong thi model chep lai y nguyen cau cu thay vi
    nghi cau moi.
    """
    mau = []
    for f in sorted(Path(out_dir).glob("*/title.txt")):
        if bo_qua and f.parent.name == bo_qua:
            continue
        for dong in f.read_text(encoding="utf-8").splitlines():
            dong = dong.strip()
            if dong and not dong.startswith("#"):   # cho phep ghi chu trong title.txt
                mau.append((f.parent.name, dong))
    if not mau:
        return ""
    dong = "\n".join(f"- `{t}`  ({ten})" for ten, t in mau)
    return f"""**Tiêu đề đã dùng thật cho các phim trước của kênh này** — đây là
gu đã được duyệt, bám sát nhịp và độ dài của chúng:

{dong}

Đọc để lấy giọng văn, KHÔNG chép lại. Phim đang làm khác hẳn những phim trên."""


ANH_MO_DAU = """**Ảnh tham chiếu đính kèm — ai là ai:**

Phim này có ĐÚNG bằng này người, không hơn một ai. Mở từng ảnh ra nhìn kỹ mặt
trước khi viết, đừng đọc lướt tên file.
"""

ANH_KET = """
**Luật dùng ảnh — đây là chỗ hỏng nhiều nhất, đọc kỹ từng dòng:**

**1. Một ảnh = một người = xuất hiện đúng một lần.** Mỗi ảnh ở trên chỉ được
dùng cho ĐÚNG một người trong khung, và mỗi người chỉ có mặt MỘT lần. Lỗi nặng
nhất từng gặp: hai người trong tranh cùng đeo một khuôn mặt (mặt của một vai
phụ), còn nhân vật chính thì biến mất khỏi khung mà không ai nhận ra.

**2. `chinh_1` BẮT BUỘC phải có trong khung.** Đó là người cả câu chuyện xoay
quanh. Thiếu người khác thì còn chữa được, thiếu `chinh_1` là bức ảnh sai phim.

**3. Nhìn từng ảnh rồi tả thành chữ — không được chỉ ghi tên file.** Mỗi lần
nhắc tới một người trong đoạn prompt bạn viết ra, phải kèm đủ ba thứ:

> **vị trí trong khung** + **tên file ảnh** + **3-4 đặc điểm bạn NHÌN THẤY trong
> chính ảnh đó** (kiểu tóc, chân mày, khuôn mặt trẻ hay già, màu và kiểu áo)

Ví dụ đúng:

> đứng chính giữa khung, người đàn ông trong ảnh `chinh_1_...jpg` — mặt trẻ
> ngoài hai mươi, tóc búi gọn trên đỉnh đầu, chân mày mảnh, áo vải tối màu có
> mảnh vá ở vai; giữ nguyên khuôn mặt này, không pha với bất kỳ ảnh nào khác.

Ví dụ SAI, tuyệt đối không viết kiểu này: `người đàn ông trong ảnh chinh_1`.
AI tạo ảnh nhận cả chùm ảnh cùng lúc; không có phần mô tả thì nó chọn khuôn mặt
nào nổi bật nhất trong chùm rồi gán cho tất cả mọi người trong tranh.

**4. Chỉ ghi thứ bạn NHÌN THẤY. Cấm suy diễn.** Không được suy đặc điểm ra từ
vai diễn hay nghề nghiệp: "thợ săn đi rừng thì chắc có râu", "bà chủ thì chắc
sang trọng", "kẻ phản bội thì chắc mặt dữ". Râu, ria, kính, sẹo, hình xăm, tóc
bạc, nốt ruồi, mũ nón — chỉ được nhắc khi bạn thực sự nhìn thấy nó **trong ảnh
của đúng người đó**.

Râu là thứ bị bịa nhiều nhất. Đã có lần một nhân vật chính bị tả là "mặt râu
phong trần" trong khi ảnh của anh ta mặt nhẵn nhụi, còn bộ râu kia là của một
vai phụ nằm ngay bên dưới trong cùng danh sách. Bức thumbnail ra một người
không có trong phim.

**Không mở được ảnh thì nói thẳng.** Nếu vì lý do gì bạn không xem được một ảnh
trong danh sách, hãy ghi rõ "không xem được `<tên file>`" rồi DỪNG lại hỏi, đừng
đoán mặt. Đoán một lần là hỏng cả bức ảnh mà không ai biết.

**5. Cặp dễ lẫn phải tách hẳn ra.** Trước khi viết, soi từng đôi trong danh sách
ảnh. Hai người cùng giới tính, xấp xỉ tuổi, cùng kiểu tóc, cùng tông áo là cặp
nguy hiểm — phim ngắn Trung Quốc dựng mặt bằng AI nên các vai nam cùng lứa
giống nhau đến mức khó tin. Với cặp đó:

- phần mô tả phải nêu thẳng ĐIỂM KHÁC NHAU, đặt cạnh nhau cho rõ ("tóc búi cao
  gọn" đối lại "tóc dài xoã sau vai"; "mày mảnh" đối lại "mày rậm thẳng");
- đặt họ xa nhau trong khung, khác tư thế, khác hướng nhìn, khác tầng sáng tối;
- vẫn thấy không tách nổi thì **bỏ hẳn người phụ ra khỏi khung**. Thà ít người
  còn hơn hai người cùng một mặt.

**6. KHÔNG thêm bất kỳ khuôn mặt nào ngoài danh sách trên.** Không tự nghĩ ra
người thân, người hầu, bạn bè, đám đông có mặt rõ. Cần đông người thì để họ quay
lưng, khuất mặt, hoặc mờ hẳn ở hậu cảnh.

**7. Thiếu ảnh thì đổi tình huống, đừng bịa mặt.** Nếu tình huống cần một người
KHÔNG có trong danh sách ảnh, chuyển sang tình huống khác chỉ gồm những người có
ảnh. Thà đơn giản còn hơn sai mặt.

**8. Luật trên chỉ nói về NGƯỜI.** Phim nào có một con vật là nhân vật trung tâm
(hổ, cá mập, gấu) thì nó BẮT BUỘC phải có trong khung, và nó thường chính là vật
hút mắt lớn nhất tiền cảnh. Có ảnh `vat_*` ở trên thì tả con vật ĐÚNG theo ảnh
đó — màu lông, cỡ, dáng — không được vẽ theo con vật mặc định trong đầu bạn.
Không có ảnh của nó thì cứ tả theo lời thoại.

**BẢNG NHẬN MẶT — in ra bảng này TRƯỚC đoạn prompt, mỗi ảnh người một dòng:**

| ảnh | nhìn thấy gì trong ảnh | đứng đâu trong thumbnail |
|---|---|---|
| `chinh_1_...` | nữ, ngoài 20, tóc tết ra sau, mày mảnh, áo bông xám cổ cao | chính giữa, quay thẳng ống kính |
| `chinh_2_...` | nam, khoảng 30, **mặt nhẵn không râu**, tóc ngắn rối, áo khoác xanh rêu | bên phải, nghiêng 3/4 |

Cột giữa chỉ ghi thứ nhìn thấy trong ảnh, không ghi tên nhân vật, không ghi vai
trò trong phim. Ảnh `vat_*` và `canh_chung_*` không cần đưa vào bảng.

Bảng này tồn tại để người dùng liếc một cái là biết bạn có mở ảnh ra nhìn thật
hay chỉ đọc tên file rồi tưởng tượng. Sai một đặc điểm trong bảng là cả bức
thumbnail ra sai người — mà đọc đoạn prompt dài thì không ai phát hiện kịp.

**Tự kiểm trước khi trả lời** (không cần in phần này ra):

1. Mỗi tên file ảnh có xuất hiện đúng một lần trong đoạn prompt không?
2. `chinh_1` có nằm trong khung không?
3. Có hai người nào bị tả bằng cùng một kiểu tóc + cùng kiểu khuôn mặt không?
4. Mô tả trong đoạn prompt có khớp từng chữ với bảng nhận mặt ở trên không?
5. Có đặc điểm nào (râu, kính, sẹo) bạn ghi ra mà không thực sự thấy trong ảnh?

Câu nào sai thì sửa rồi mới trả lời.
"""


def anh_block(files):
    """Liet ke anh tham chieu theo vai, sinh tu chinh ten file da cat ra.

    Truoc day prompt chi noi chung chung 'kem anh tham chieu chup tu phim' ma
    khong noi anh nao la ai. AI tao anh nhan mot chum anh khong nhan -> no tron
    cac guong mat lai, hoac tu them nguoi thu ba khong co trong phim.
    """
    if not files:
        return ""
    chinh, phu, vat, chung = [], [], [], []
    for f in sorted(files):
        ten = Path(f).name
        bo = ten.split("_")
        if ten.startswith("canh_chung_"):
            chung.append(ten)
        elif ten.startswith("chinh_") and len(bo) >= 3:
            chinh.append((bo[1], bo[2], ten))
        elif ten.startswith("phu_") and len(bo) >= 3:
            phu.append((bo[1], bo[2], ten))
        elif ten.startswith("vat_") and len(bo) >= 3:
            vat.append((bo[1], bo[2], ten))
    dong = [ANH_MO_DAU]
    for so, ten_nv, ten in chinh:
        # Vai chinh so 1 luon la nhan vat cot truyen xoay quanh (thu tu do nguoi
        # viet dat trong dong CAST). Danh dau han ra: lan hong nang nhat la AI
        # dap mat mot vai phu len nguoi dung giua khung roi bo quen nhan vat chinh.
        dong.append(f"- `{ten}` — NHÂN VẬT CHÍNH {so}"
                    + (f", tên trong phim: {ten_nv}" if ten_nv else "")
                    + ("  **<- cả phim xoay quanh người này, bắt buộc có trong khung**"
                       if so == "1" else ""))
    for so, ten_nv, ten in phu:
        dong.append(f"- `{ten}` — nhân vật phụ {so}"
                    + (f", tên trong phim: {ten_nv}" if ten_nv else ""))
    # Con vat / quai vat trung tam. Khong co mat nguoi nen khong lot qua duoc
    # buoc nhan dang, phai ghim tay - nhung voi thumbnail no lai la vat hut mat
    # lon nhat, va tha ra khong ta ky thi AI ve mot con hoan toan khac.
    for so, ten_nv, ten in vat:
        dong.append(f"- `{ten}` — con vật / vật thể trung tâm {so}"
                    + (f": {ten_nv}" if ten_nv else "")
                    + ". Giữ đúng loài, màu, cỡ như trong ảnh")
    for ten in chung:
        dong.append(f"- `{ten}` — ảnh chung, chỉ để tham khảo cách họ đứng cạnh nhau"
                    " và tỉ lệ to nhỏ giữa các nhân vật")
    return "\n".join(dong) + "\n" + ANH_KET


TU_CHON = """Chọn từ mục 2 một tình huống gây tò mò mạnh nhất mà diễn đạt trọn vẹn
được bằng MỘT khung hình tĩnh."""

CHI_DINH = """**Cảnh làm thumbnail đã được chỉ định sẵn:**

> {CANH}

Đây KHÔNG phải gợi ý. Người dựng video đã xem hết phim rồi mới chọn cảnh này, nên
đừng thay bằng cảnh bạn thấy hay hơn.

Việc của bạn: tìm trong mục 2 tình huống ứng với cảnh trên, ghi số thứ tự của nó
vào dòng `Tình huống:`, rồi dựng thumbnail đúng cảnh đó. Nếu mục 2 chưa có tình
huống nào ứng với nó, hãy thêm vào mục 2 rồi mới làm tiếp.
{MACH}"""

MACH_MO = """
**Khung hình phải gánh được cả mạch truyện, không chỉ một khoảnh khắc lẻ.**

Đây là những nút thắt lớn khác của phim, xếp theo độ quan trọng:

{DS}

Cảnh chỉ định ở trên là SÂN KHẤU — người, tư thế, nơi chốn đều bám theo nó. Nhưng
trong cùng khung hình đó phải nhìn thấy được thứ đang đe doạ hoặc thứ đang được
đánh đổi trong mạch truyện trên, dù chỉ ở hậu cảnh hay tiền cảnh.

Ví dụ cụ thể để bạn hiểu đúng mức độ: cảnh chỉ định là "hai người cướp xuồng cứu
sinh chạy khỏi du thuyền". Nếu chỉ vẽ hai người ngồi xuồng giữa biển thì người
lướt không hiểu vì sao phải chạy, và bức ảnh chết. Đúng phải là: hai người trên
xuồng ở tiền cảnh, VÀ vây lưng con quái vật khổng lồ rẽ nước ngay sau lưng họ,
VÀ con tàu đang nghiêng ở xa. Ba thứ trong một khung, người xem hiểu ngay toàn
bộ câu chuyện mà chưa cần đọc tiêu đề.

Quy tắc: **một tình huống làm sân khấu + ít nhất một dấu hiệu của nút thắt lớn
nhất phim.** Không kể lể nhiều cảnh chồng lên nhau — chỉ thêm đúng thứ khiến
người xem hiểu được mức độ nguy hiểm hoặc mức độ lật ngược của cả phim."""


def scene_block(label=None, mach=None):
    """Khoi chu chen vao muc 3, noi ro canh nao duoc chon lam thumbnail.

    Truoc day moc dau tien trong moments.txt chi dung noi bo de cat anh, khong
    gui cho AI - nen AI tu chon mot canh khac han, va thumbnail lech voi anh
    tham chieu. Gio truyen thang canh do vao prompt.

    mach: cac moc con lai trong file khoanh khac. Chi dua moc dau thi AI ve dung
    mot khoanh khac le - vd hai nguoi ngoi xuong cuu sinh giua bien, khong ai
    hieu vi sao phai chay. Co ca mach thi no biet phai nhet con ca map vao khung.
    """
    if not label:
        return TU_CHON
    khoi = ""
    if mach:
        ds = "\n".join(f"- {m}" for m in mach)
        khoi = MACH_MO.replace("{DS}", ds)
    return CHI_DINH.replace("{CANH}", label).replace("{MACH}", khoi)


def make_prompt(srt_path, template=TEMPLATE, attach_name=None, scene=None,
                stem=None, the_loai=None, mach=None, anh=None):
    """Doc .srt + template -> (prompt, so cau thoai, chuoi thoai).

    Prompt CHI chua phan luat, khong nhet thoai vao trong. Thoai duoc ghi ra
    mot file .txt rieng de nguoi dung tu dinh kem. Lam vay thi chi con mot ban
    prompt duy nhat cho ca ChatGPT lan Gemini - truoc day phai giu hai ban chi
    vi ChatGPT gioi han do dai mot tin nhan.

    stem: ten dinh danh phim (thuong la ten VIDEO). Phai truyen rieng vi ky hieu
    the loai va bo_qua few-shot deu doc tu ten nay - lay tu duong dan phu de thi
    phu de dat ten lech (vd sub.txt dung chung) la sai the loai luon.
    """
    tpl = Path(template).read_text(encoding="utf-8")
    dialogue = parse_srt(Path(srt_path))
    body = "\n".join(dialogue)
    stem = stem or Path(srt_path).stem
    name = attach_name or f"{stem}.txt"
    out = (tpl.replace("{TRANSCRIPT}", ATTACH_NOTE.replace("{FILE}", name))
              .replace("{CANH_THUMB}", scene_block(scene, mach))
              .replace("{ANH_THAM_CHIEU}", anh_block(anh))
              .replace("{THE_LOAI}", genre_block(stem, the_loai))
              .replace("{TIEU_DE_MAU}", tieu_de_mau(bo_qua=stem)))
    return out, len(dialogue), body


def _lenh_clipboard(ghi):
    """Lenh clipboard theo he dieu hanh -> (argv, ma_hoa) hoac None.

    Windows dung UTF-16LE: clip.exe / powershell Get-Clipboard doc va tra ve
    UTF-16, dua UTF-8 vao la ra chu Han lem nhem.
    """
    if sys.platform == "darwin":
        return (["pbcopy"] if ghi else ["pbpaste"]), "utf-8"
    if sys.platform.startswith("win"):
        return (["clip"] if ghi else
                ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]), "utf-16-le"
    # Linux: Wayland truoc, roi X11. xclip/wl-clipboard co the chua cai.
    for lenh in (["wl-copy"] if ghi else ["wl-paste", "--no-newline"],
                 ["xclip", "-selection", "clipboard"] if ghi
                 else ["xclip", "-selection", "clipboard", "-o"],
                 ["xsel", "--clipboard", "--input"] if ghi
                 else ["xsel", "--clipboard", "--output"]):
        if shutil.which(lenh[0]):
            return lenh, "utf-8"
    return None


def to_clipboard(text):
    """Chep vao clipboard, chay duoc tren macOS / Windows / Linux.

    Tren macOS phai ep LC_CTYPE=UTF-8: pbcopy doc bien moi truong de biet byte
    dau vao thuoc bang ma nao. Chay tu terminal thi thuong co san LANG=...UTF-8,
    nhung chay tu cron / editor / IDE thi bien nay rong, luc do pbcopy hieu nham
    la Mac OS Roman va "Tinh huong" thanh "T√¨nh hu·ªëng". Kiem bang pbpaste
    khong phat hien duoc vi no doc nham y het luc ghi.
    """
    lenh = _lenh_clipboard(ghi=True)
    if lenh is None:
        return False
    argv, ma_hoa = lenh
    env = dict(os.environ, LC_CTYPE="UTF-8")
    try:
        subprocess.run(argv, input=text.encode(ma_hoa), check=True, env=env)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def from_clipboard():
    """Doc clipboard ra chuoi. Tra ve '' neu he thong khong co cong cu nao."""
    lenh = _lenh_clipboard(ghi=False)
    if lenh is None:
        return ""
    argv, ma_hoa = lenh
    try:
        ra = subprocess.run(argv, capture_output=True, check=True).stdout
        return ra.decode(ma_hoa, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-copy", action="store_true", help="khong chep vao clipboard")
    a = ap.parse_args()

    if not TEMPLATE.is_file():
        sys.exit(f"Thieu {TEMPLATE}")
    if not INPUT_DIR.is_dir():
        sys.exit(f"Thieu thu muc {INPUT_DIR}/")

    srts = sorted(INPUT_DIR.glob("*.srt")) + sorted(INPUT_DIR.glob("*.txt"))
    if not srts:
        sys.exit(f"Khong tim thay file .srt hoac .txt nao trong {INPUT_DIR}/")

    OUT_DIR.mkdir(exist_ok=True)

    for srt in srts:
        prompt, n_line, body = make_prompt(srt)

        dest = OUT_DIR / f"{srt.stem}_buoc1.txt"
        dest.write_text(prompt, encoding="utf-8")
        thoai = OUT_DIR / f"{srt.stem}.txt"
        thoai.write_text(body, encoding="utf-8")

        print("=" * 54)
        print(f"{srt.name}")
        print(f"  cau thoai: {n_line}")
        print(f"  -> {dest}    {len(prompt):,} ky tu  (dan vao o chat)")
        print(f"  -> {thoai}    {len(body):,} ky tu  (dinh kem file nay)")

        if not a.no_copy and len(srts) == 1 and to_clipboard(prompt):
            print("  da chep prompt vao clipboard")

    print("=" * 54)


if __name__ == "__main__":
    main()
