# -*- coding: utf-8 -*-
"""
================================================================================
SCRIPT TẠO BÁO CÁO THỰC TẬP CHUẨN FORM MẪU WORD TRƯỜNG ĐẠI HỌC THĂNG LONG
(Kế thừa trực tiếp định dạng gốc từ mẫu chuẩn LeGiaKhanh_K36.docx)
--------------------------------------------------------------------------------
ĐỀ TÀI: XÂY DỰNG HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT
        VÀ NỀN TẢNG WEB THỜI GIAN THỰC ĐA NGƯỜI DÙNG
Giảng viên hướng dẫn : ThS. Ngô Mạnh Cường
Sinh viên thực hiện  : Lê Gia Khánh     - MSV: A48572
                       Bàng Nguyên Đăng - MSV: A48553
Lớp                  : TA3601
================================================================================
"""

import os
import sys
import copy
import shutil
import json
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls, qn
from docx.oxml import parse_xml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding='utf-8')

TEMPLATE_PATH = r"C:\Users\giakh\Downloads\LeGiaKhanh_K36.docx"
OUTPUT_PROJECT = rf"{BASE_DIR}\Bao_Cao_Nhan_Dien_Ngon_Ngu_Ky_Hieu_LeGiaKhanh.docx"
OUTPUT_DOWNLOADS = r"C:\Users\giakh\Downloads\Bao_Cao_Nhan_Dien_Ngon_Ngu_Ky_Hieu_LeGiaKhanh.docx"
OUTPUT_INTERNSHIP_DOWNLOADS = r"C:\Users\giakh\Downloads\Bao_Cao_Thuc_Tap_LeGiaKhanh_BangNguyenDang.docx"
OUTPUT_INTERNSHIP_PROJECT = rf"{BASE_DIR}\Bao_Cao_Thuc_Tap_LeGiaKhanh_BangNguyenDang.docx"

# 9 hình ảnh minh họa thực nghiệm và kiến trúc
IMG_ARCH_PATH = rf"{BASE_DIR}\models\system_architecture.png"
IMG_ABLATION_PATH = rf"{BASE_DIR}\models\ablation_comparison.png"
IMG_CURVES_PATH = rf"{BASE_DIR}\models\training_curves.png"
IMG_F1_DIST_PATH = rf"{BASE_DIR}\models\f1_distribution.png"
IMG_WEB_ARCH_PATH = rf"{BASE_DIR}\models\web_architecture.png"
IMG_WEB_BENCH_PATH = rf"{BASE_DIR}\models\web_benchmark.png"
IMG_TAB1_PATH = rf"{BASE_DIR}\models\web_tab1_webcam.png"
IMG_TAB2_PATH = rf"{BASE_DIR}\models\web_tab2_upload.png"
IMG_TAB3_PATH = rf"{BASE_DIR}\models\web_tab3_stress.png"


def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_paragraph_text_keep_rpr(p, new_text):
    """Cập nhật văn bản cho đoạn văn trên trang bìa mà giữ nguyên 100% định dạng gốc."""
    if p.runs:
        p.runs[0].text = new_text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(new_text)


def add_body_p(doc, text, indent=True, space_after=6):
    """Thêm đoạn văn Normal chuẩn trường Thăng Long (Justify, indent 1.0cm, space_after 6pt)."""
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.0)

    parts = text.replace("`", "").split("**")
    for idx, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part.replace("*", ""))
        if idx % 2 == 1:
            r.bold = True
    return p


def add_bullet_p(doc, text, bold_prefix=None):
    """Thêm dòng List Bullet."""
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        rb = p.add_run(bold_prefix.replace("`", ""))
        rb.bold = True
    parts = text.replace("`", "").split("**")
    for idx, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part.replace("*", ""))
        if idx % 2 == 1:
            r.bold = True
    return p


def add_numbered_p(doc, text, bold_prefix=None):
    """Thêm dòng đánh số 1., 2., 3."""
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        rb = p.add_run(bold_prefix.replace("`", ""))
        rb.bold = True
    parts = text.replace("`", "").split("**")
    for idx, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part.replace("*", ""))
        if idx % 2 == 1:
            r.bold = True
    return p


def add_caption_p(doc, caption_text):
    """Thêm chú thích Bảng / Hình chuẩn mẫu LeGiaKhanh_K36.docx (Centered, Italic, 12pt)."""
    p = doc.add_paragraph(style='Normal')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(caption_text)
    r.italic = True
    r.font.size = Pt(12)
    return p


def add_standard_table(doc, headers, rows, caption_text, col_widths_cm=None):
    """
    Tạo bảng dữ liệu chuẩn Thăng Long:
    - Style: Table Grid, căn giữa.
    - Hàng tiêu đề: nền xanh nhạt #D9E2F3, chữ đậm 12pt căn giữa.
    - Hàng dữ liệu: chữ 12pt.
    - Tên bảng (Bảng X.Y. ...) đặt ngay phía dưới bảng, chữ nghiêng 12pt căn giữa.
    """
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for c_idx, h_text in enumerate(headers):
        cell = table.rows[0].cells[c_idx]
        cell.text = ""
        set_cell_shading(cell, "D9E2F3")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.bold = True
        r.font.size = Pt(12)

    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            s_val = str(val)
            if len(s_val) <= 16 and "\n" not in s_val and (c_idx == 0 or any(ch.isdigit() for ch in s_val)):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(s_val)
            r.font.size = Pt(12)

    if col_widths_cm:
        for row in table.rows:
            for c_idx, w_cm in enumerate(col_widths_cm):
                row.cells[c_idx].width = Cm(w_cm)

    add_caption_p(doc, caption_text)
    return table


def add_code_box(doc, code_text):
    """Tạo khung code / kết quả chạy thử: Bảng 1x1, nền xám #F2F2F2, font Consolas 9.5pt."""
    table = doc.add_table(rows=1, cols=1)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Cm(16.0)
    set_cell_shading(cell, "F2F2F2")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0

    lines = code_text.strip("\n").split("\n")
    for idx, line in enumerate(lines):
        if idx > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(9.5)

    p_after = doc.add_paragraph(style='Normal')
    p_after.paragraph_format.space_after = Pt(4)
    return table


def add_figure_box(doc, img_path, caption_text, width_inches=5.6):
    """Chèn hình ảnh căn giữa kèm chú thích Hình X.Y. ... ở dưới."""
    if not os.path.exists(img_path):
        print(f"Cảnh báo: không tìm thấy ảnh {img_path}")
        return
    p_img = doc.add_paragraph(style='Normal')
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))
    add_caption_p(doc, caption_text)


# ==============================================================================
# 1. NẠP MẪU CHUẨN VÀ CẬP NHẬT TRANG BÌA
# ==============================================================================
print("Đang nạp mẫu chuẩn từ:", TEMPLATE_PATH)
doc = Document(TEMPLATE_PATH)

set_paragraph_text_keep_rpr(doc.paragraphs[0], "BỘ GIÁO DỤC VÀ ĐÀO TẠO")
set_paragraph_text_keep_rpr(doc.paragraphs[1], "TRƯỜNG ĐẠI HỌC THĂNG LONG")
set_paragraph_text_keep_rpr(doc.paragraphs[2], "KHOA CÔNG NGHỆ THÔNG TIN")
set_paragraph_text_keep_rpr(doc.paragraphs[3], "BÁO CÁO THỰC TẬP TỐT NGHIỆP")
set_paragraph_text_keep_rpr(doc.paragraphs[4], "CHUYÊN NGÀNH: CÔNG NGHỆ THÔNG TIN")
set_paragraph_text_keep_rpr(doc.paragraphs[5], "ĐỀ TÀI:")
set_paragraph_text_keep_rpr(doc.paragraphs[6], "XÂY DỰNG HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT")
set_paragraph_text_keep_rpr(doc.paragraphs[7], "VÀ NỀN TẢNG WEB THỜI GIAN THỰC ĐA NGƯỜI DÙNG")
doc.paragraphs[7].paragraph_format.space_after = Pt(45)

cover_tbl = doc.tables[0]
template_row_xml = copy.deepcopy(cover_tbl.rows[0]._tr)


def fill_cover_row(row, label_text, val_text):
    c0, c1 = row.cells[0], row.cells[1]
    p0 = c0.paragraphs[0]
    p0.text = ""
    p0.paragraph_format.space_after = Pt(4)
    r0 = p0.add_run(label_text)
    r0.bold = True
    r0.font.size = Pt(14)

    p1 = c1.paragraphs[0]
    p1.text = ""
    p1.paragraph_format.space_after = Pt(4)
    r1 = p1.add_run(val_text)
    r1.font.size = Pt(14)


cover_info = [
    ("Giảng viên hướng dẫn:", "ThS. Ngô Mạnh Cường"),
    ("Sinh viên thực hiện:", "1. Lê Gia Khánh – MSV: A48572"),
    ("", "2. Bàng Nguyên Đăng – MSV: A48553"),
    ("Lớp:", "TA3601"),
]

while len(cover_tbl.rows) < len(cover_info):
    cover_tbl._tbl.append(copy.deepcopy(template_row_xml))

for idx, (lbl, val) in enumerate(cover_info):
    fill_cover_row(cover_tbl.rows[idx], lbl, val)

body_elm = doc._body._body
body_elm.remove(doc.paragraphs[8]._p)
body_elm.remove(doc.paragraphs[8]._p)

first_h1_elm = next(p._p for p in doc.paragraphs if p.style.name == 'Heading 1')
body_children = list(body_elm)
toc_end_break_idx = body_children.index(first_h1_elm)

for child in body_children[toc_end_break_idx:]:
    if not child.tag.endswith('sectPr'):
        body_elm.remove(child)

# ==============================================================================
# 2. LỜI MỞ ĐẦU
# ==============================================================================
doc.add_heading("LỜI MỞ ĐẦU", level=1)

add_body_p(doc,
    "Trong bối cảnh Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) phát triển mạnh mẽ, việc ứng dụng học sâu "
    "vào các bài toán hỗ trợ cộng đồng yếu thế mang ý nghĩa xã hội và thực tiễn sâu sắc. Ngôn ngữ ký hiệu là phương tiện giao tiếp "
    "chính của hơn 2.5 triệu người khiếm thính tại Việt Nam. Tuy nhiên, rào cản ngôn ngữ giữa người khiếm thính và cộng đồng nghe nói "
    "vẫn còn rất lớn do đại đa số người dân không được đào tạo thủ ngữ.")

add_body_p(doc,
    "Các giải pháp truyền thống dựa trên găng tay cảm biến đắt tiền, cồng kềnh và khó nhân rộng. Ngược lại, hướng tiếp cận thị giác máy "
    "tính thuần túy qua camera thông thường kết hợp mô hình AI nhẹ đang mở ra cơ hội đưa công nghệ phiên dịch thủ ngữ vào đời sống "
    "hàng ngày. Đặc biệt, để ứng dụng vào thực tế giáo dục và hành chính công, hệ thống không chỉ cần nhận diện chính xác từng cử chỉ "
    "đơn lẻ mà còn phải hỗ trợ dịch một chuỗi cử chỉ liên tục thành câu tiếng Việt có nghĩa, đồng thời có kiến trúc máy chủ Web đa người "
    "dùng vận hành ổn định mà không đòi hỏi phần cứng GPU đắt tiền.")

add_body_p(doc,
    "Xuất phát từ yêu cầu đó, dưới sự hướng dẫn của thầy **ThS. Ngô Mạnh Cường**, nhóm sinh viên gồm **Lê Gia Khánh (MSV: A48572)** "
    "và **Bàng Nguyên Đăng (MSV: A48553)** thực hiện đề tài: **“Xây dựng hệ thống nhận diện Ngôn ngữ ký hiệu tiếng Việt và nền tảng "
    "Web thời gian thực đa người dùng”**.")

add_body_p(doc, "Nội dung báo cáo được cấu trúc gồm các chương sau:")
add_bullet_p(doc, "Giới thiệu đề tài, bài toán, yêu cầu chức năng, công cụ và phân công công việc.", "Chương 1: ")
add_bullet_p(doc, "Cơ sở lý thuyết về VSL, MediaPipe Holistic, chuẩn hóa đặc trưng bất biến, Conv1D – BiGRU và TensorFlow Lite.", "Chương 2: ")
add_bullet_p(doc, "Phân tích & thiết kế: phân chia chống rò rỉ dữ liệu, vector 458 đặc trưng, kiến trúc mô hình và kiến trúc Web đa người dùng.", "Chương 3: ")
add_bullet_p(doc, "Cài đặt chương trình: cấu trúc mã nguồn, các đoạn code tiêu biểu và hướng dẫn thực thi.", "Chương 4: ")
add_bullet_p(doc, "Kết quả thực nghiệm: Ablation Study, huấn luyện 55 epochs, đánh giá chuỗi WER, benchmark chịu tải Web và thử nghiệm 3 Tab chức năng.", "Chương 5: ")
add_bullet_p(doc, "Đánh giá kết quả đạt được, hạn chế và hướng phát triển.", "Kết luận: ")

doc.add_page_break()

# ==============================================================================
# CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI
# ==============================================================================
doc.add_heading("CHƯƠNG 1. GIỚI THIỆU ĐỀ TÀI", level=1)

doc.add_heading("1.1. Mô tả bài toán", level=2)
add_body_p(doc,
    "Đề tài tập trung nghiên cứu và xây dựng một hệ thống hoàn chỉnh có khả năng nhận diện **Ngôn ngữ ký hiệu tiếng Việt "
    "(Vietnamese Sign Language – VSL)** từ luồng video hoặc camera thời gian thực, gồm hai bài toán kỹ thuật chính:")
add_bullet_p(doc, "Nhận diện từ vựng ký hiệu đơn lẻ (Isolated Sign Recognition) thuộc 100 lớp từ vựng VSL thông dụng.", "• ")
add_bullet_p(doc, "Giải mã chuỗi nhiều hành động ký hiệu liên tiếp (Continuous Sign Language Translation) thành câu tiếng Việt hoàn chỉnh.", "• ")
add_bullet_p(doc, "Triển khai nền tảng Web đa người dùng đồng thời (Multi-User Real-time Web Platform) với cơ chế chống nghẽn mạng và xử lý xoay vòng theo phiên.", "• ")

add_body_p(doc,
    "Bộ dữ liệu thực nghiệm gồm **100 lớp từ vựng ký hiệu tiếng Việt** thông dụng trong đời sống và y tế, với quy mô phân bổ:")

add_standard_table(doc,
    ["Phân vùng dữ liệu", "Số lượng video", "Đặc điểm kỹ thuật"],
    [
        ["Tập có nhãn (dataset/train)", "3,875 video MP4", "100 thư mục lớp từ vựng VSL (2,806 video gốc và 1,069 biến thể _1, _2, _3)"],
        ["Tập kiểm thử công khai (public_test)", "1,630 video MP4", "Video kiểm thử thực tế không gán nhãn"],
        ["Tập kiểm thử riêng (private_test)", "2,859 video MP4", "Video kiểm thử mở rộng đánh giá hệ thống"],
    ],
    "Bảng 1.1. Quy mô các phân vùng của bộ dữ liệu Ngôn ngữ ký hiệu tiếng Việt",
    col_widths_cm=[4.8, 3.2, 8.0]
)

doc.add_heading("1.2. Yêu cầu chức năng", level=2)
add_body_p(doc, "Hệ thống đáp ứng đầy đủ 7 yêu cầu chức năng và chỉ tiêu kỹ thuật:")
add_numbered_p(doc, "Kiểm toán dữ liệu và phân chia 3 tập **Train / Validation / Held-out Test** theo đơn vị nhóm gốc (`base_id`) đảm bảo tuyệt đối không rò rỉ dữ liệu (Zero Data Leakage).", "1. ")
add_numbered_p(doc, "Trích xuất điểm mốc khung xương và 2 bàn tay bằng MediaPipe Holistic với kiến trúc đa tiến trình song song.", "2. ")
add_numbered_p(doc, "Xây dựng vector **458 đặc trưng hình học – động học bất biến** với vị trí đứng và khoảng cách camera.", "3. ")
add_numbered_p(doc, "Huấn luyện mạng học sâu `VSL_Conv1D_BiGRU` và xuất mô hình **Pure TensorFlow Lite (`model.tflite`, 2.76 MB)** chạy trên CPU với độ trễ < 3 ms.", "4. ")
add_numbered_p(doc, "Dự đoán tệp video đơn lẻ (`predict.py`) và dịch chuỗi nhiều cử chỉ liên tiếp (`predict_sequence.py`).", "5. ")
add_numbered_p(doc, "Nhận diện trực tiếp qua Webcam thời gian thực (`demo_webcam.py` và Web Tab 1), tự động phát hiện trạng thái nghỉ và hỗ trợ cử chỉ xóa từ chữ X.", "6. ")
add_numbered_p(doc, "Nền tảng Web đa người dùng đồng thời (FastAPI + WebSocket + Client-side MediaPipe JS + Cloudflare HTTPS Tunnel), chịu tải từ 50–100 người dùng cùng lúc.", "7. ")

doc.add_heading("1.3. Công cụ và môi trường thực hiện", level=2)
add_standard_table(doc,
    ["Hạng mục", "Công nghệ / Thư viện", "Phiên bản / Mục đích"],
    [
        ["Ngôn ngữ lập trình", "Python", "3.11 / 3.13 64-bit"],
        ["Framework học sâu", "TensorFlow & Keras", "2.16+ (Xây dựng mạng Conv1D, Bi-GRU, xuất TFLite)"],
        ["Thị giác máy tính", "Google MediaPipe & OpenCV", "MediaPipe Holistic 0.10.14, OpenCV-Python 4.9+"],
        ["Backend máy chủ Web", "FastAPI & Uvicorn", "Xử lý bất đồng bộ, WebSocket, REST API"],
        ["Frontend máy khách", "MediaPipe Holistic JS, HTML5 Canvas", "Chạy MediaPipe ngay tại trình duyệt máy khách"],
        ["Công khai Internet", "Cloudflare Tunnel (cloudflared)", "Cung cấp kết nối an toàn HTTPS / WSS không cần mở port modem"],
        ["Hệ điều hành thực nghiệm", "Microsoft Windows 11 64-bit", "CPU Intel Core i7 / AMD Ryzen, không yêu cầu GPU rời"],
    ],
    "Bảng 1.2. Môi trường và công cụ phát triển hệ thống",
    col_widths_cm=[4.0, 5.0, 7.0]
)

doc.add_heading("1.4. Phân công công việc trong nhóm", level=2)
add_standard_table(doc,
    ["Thành viên", "Nhiệm vụ chính đảm nhiệm", "Mức độ hoàn thành"],
    [
        ["Lê Gia Khánh (Nhóm trưởng)", "Khảo sát dữ liệu, phân chia Group-Stratified, thiết kế 458 đặc trưng bất biến, huấn luyện mạng VSL_Conv1D_BiGRU, tối ưu Pure TFLite, xây dựng kiến trúc Web đa người dùng và viết báo cáo.", "100%"],
        ["Bàng Nguyên Đăng", "Hỗ trợ tiền xử lý dữ liệu, kiểm thử giao diện Desktop Tkinter, thu thập video thực nghiệm Webcam, kiểm thử chịu tải đa người dùng trên các thiết bị di động và hoàn thiện tài liệu.", "100%"],
    ],
    "Bảng 1.3. Phân công trách nhiệm các thành viên trong nhóm đề tài",
    col_widths_cm=[4.5, 9.5, 2.0]
)

doc.add_page_break()

# ==============================================================================
# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT
# ==============================================================================
doc.add_heading("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT", level=1)

doc.add_heading("2.1. Ngôn ngữ ký hiệu tiếng Việt (VSL)", level=2)
add_body_p(doc,
    "Ngôn ngữ ký hiệu là hệ thống ngôn ngữ thị giác – không gian hoàn chỉnh, sử dụng sự phối hợp của hình dạng bàn tay "
    "(Handshape), vị trí đặt tay (Location), hướng lòng bàn tay (Orientation) và chuyển động (Movement), kết hợp với biểu cảm "
    "nét mặt (Non-manual features). Khác với ngôn ngữ nói có tính tuần tự âm thanh, ngôn ngữ ký hiệu có tính đồng thời cao (Parallelism), "
    "trong đó thông tin hình dáng ngón tay và vị trí tương đối so với cơ thể diễn ra cùng một thời điểm.")

doc.add_heading("2.2. Trích xuất điểm mốc với MediaPipe Holistic", level=2)
add_body_p(doc,
    "MediaPipe Holistic là giải pháp thị giác máy tính của Google sử dụng kiến trúc đồ thị đa tầng (Pipeline Graph) kết hợp "
    "nhiều mạng nơ-ron chuyên biệt: đầu tiên phát hiện vùng cơ thể (Pose Detector), sau đó cắt nhỏ vùng bàn tay đưa vào mạng nơ-ron "
    "định vị 21 khớp 3D với độ chính xác cao. Tổng cộng mỗi khung hình trích xuất **258 tọa độ thô** gồm:")
add_bullet_p(doc, "33 điểm mốc thân trên và nét mặt (Pose): mỗi điểm gồm tọa độ không gian (x, y, z) và độ tự tin (visibility), tạo thành 132 giá trị.", "1. ")
add_bullet_p(doc, "21 điểm mốc bàn tay trái (Left Hand): mỗi khớp gồm tọa độ (x, y, z), tạo thành 63 giá trị.", "2. ")
add_bullet_p(doc, "21 điểm mốc bàn tay phải (Right Hand): mỗi khớp gồm tọa độ (x, y, z), tạo thành 63 giá trị.", "3. ")

doc.add_heading("2.3. Chuẩn hóa đặc trưng bất biến không gian – thời gian", level=2)
add_body_p(doc,
    "Tọa độ thô trả về từ MediaPipe phụ thuộc mật thiết vào vị trí người đứng trong khung hình (trái/phải) và khoảng cách tới camera "
    "(gần/xa). Để mô hình học được bản chất động tác mà không bị phụ thuộc vào góc quay hay tỷ lệ người, cần thực hiện chuẩn hóa "
    "bất biến (Spatial Invariance) bằng cách lấy điểm chính giữa hai vai làm gốc tọa độ tâm (0, 0, 0) và khoảng cách giữa hai mỏm vai "
    "làm đơn vị chuẩn độ dài (Body Scale). Tương tự, hình dạng bàn tay được chuẩn hóa cục bộ quanh cổ tay và kích thước lòng bàn tay.")

doc.add_heading("2.4. Mạng tích chập 1 chiều (Conv1D) và Unrolled Bi-GRU", level=2)
add_body_p(doc,
    "Mô hình nhận diện video chuỗi thời gian đòi hỏi khả năng xử lý đồng thời hai yếu tố: (1) Mối tương quan cử động ngón tay cục bộ "
    "trong một vài khung hình liên tiếp, và (2) Diễn tiến toàn cục của cả động tác từ lúc bắt đầu đến khi kết thúc. Mạng `Conv1D` đóng vai "
    "trò bộ lọc thời gian ngắn hạn (Short-term Temporal Filter), trong khi mạng hồi quy `Bi-GRU` (Bidirectional Gated Recurrent Unit) "
    "học ngữ cảnh hai chiều xuôi – ngược. Việc cấu hình `unroll=True` cho phép trải phẳng vòng lặp thời gian thành đồ thị tính toán tĩnh, "
    "đảm bảo mô hình biên dịch 100% sang tập lệnh chuẩn TFLite BUILTINS.")

doc.add_heading("2.5. TensorFlow Lite và đánh giá chuỗi bằng Word Error Rate (WER)", level=2)
add_body_p(doc,
    "TensorFlow Lite (TFLite) là giải pháp tối ưu hóa mô hình học sâu của Google, loại bỏ hoàn toàn các cấu trúc dư thừa của runtime Keras, "
    "đóng băng trọng số và tối ưu hóa bộ nhớ đệm CPU qua nhân thực thi XNNPACK. Để đánh giá chất lượng dịch một chuỗi nhiều cử chỉ liên tục, "
    "chỉ số Word Error Rate (WER) được áp dụng theo công thức chuẩn khoảng cách chỉnh sửa Levenshtein:")
add_body_p(doc,
    "WER = (S + D + I) / N, trong đó S là số từ bị thay thế nhầm, D là số từ bị bỏ sót, I là số từ bị chèn thừa và N là tổng số từ "
    "trong chuỗi chuẩn. Độ chính xác cấp từ tương ứng được tính bằng: Accuracy = 1 - WER.", indent=False)

doc.add_page_break()

# ==============================================================================
# CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
# ==============================================================================
doc.add_heading("CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG", level=1)

doc.add_heading("3.1. Khảo sát dữ liệu và xử lý biến thể trùng lặp (_1, _2, _3)", level=2)
add_body_p(doc,
    "Kiểm toán chi tiết 3,875 video trong thư mục `dataset/train` phát hiện cấu trúc phân bổ gồm **2,806 video gốc** và **1,069 video "
    "biến thể** mang hậu tố `_1`, `_2`, `_3`. Phân tích nội dung xác nhận các video biến thể này được cắt từ cùng một lượt quay gốc, "
    "cùng diễn viên và cùng bối cảnh. Nếu phân chia tập dữ liệu ngẫu nhiên (Random Split), các biến thể của cùng một lượt quay sẽ đồng thời "
    "xuất hiện ở cả tập Train và Test, gây ra hiện tượng rò rỉ dữ liệu nghiêm trọng (Data Leakage) làm sai lệch kết quả đánh giá.")

doc.add_heading("3.2. Phân chia 3 tập dữ liệu chuẩn Group-Stratified (Zero Data Leakage)", level=2)
add_body_p(doc,
    "Để đảm bảo tính khách quan khoa học tuyệt đối, module `split_dataset.py` tách mã định danh gốc (`base_id`) và thực hiện phân chia "
    "theo phương pháp **Group-Stratified Split** với tỷ lệ 70% Train, 15% Validation và 15% Held-out Test trên đơn vị nhóm:")

add_standard_table(doc,
    ["Phân vùng tập", "Số nhóm gốc (base_id)", "Số video (gốc)", "Số video (toàn bộ gồm biến thể)", "Tỷ lệ nhóm gốc"],
    [
        ["Tập huấn luyện (Train)", "1,962 nhóm", "1,962 video", "2,739 video", "69.92% (~70%)"],
        ["Tập kiểm định (Validation)", "422 nhóm", "422 video", "573 video", "15.04% (~15%)"],
        ["Tập kiểm thử độc lập (Held-out Test)", "422 nhóm", "422 video", "563 video", "15.04% (~15%)"],
        ["Tổng cộng", "2,806 nhóm", "2,806 video", "3,875 video", "100.0%"],
    ],
    "Bảng 3.1. Phân chia 3 tập dữ liệu độc lập chuẩn Group-Stratified (Zero Data Leakage)",
    col_widths_cm=[4.5, 3.2, 2.8, 3.5, 2.0]
)

doc.add_heading("3.3. Xử lý mã hóa Unicode tiếng Việt (NFC/NFD) và trích xuất đa tiến trình", level=2)
add_body_p(doc,
    "Hệ thống tự động chuẩn hóa toàn bộ 100 tên thư mục lớp từ vựng sang chuẩn **Unicode NFC** để khắc phục triệt để lỗi phân mảnh ký tự tiếng Việt "
    "trên Windows. Module `preprocess_fast.py` sử dụng `multiprocessing.Pool` với 6 tiến trình CPU trích xuất song song toàn bộ 3,875 video "
    "trong thời gian 28 phút (nhanh gấp 5.2 lần so với chạy đơn tiến trình).")

doc.add_heading("3.4. Thiết kế vector 458 đặc trưng hình học – động học bất biến", level=2)
add_body_p(doc,
    "Từ chuỗi 30 khung hình × 258 tọa độ thô, module `feature_engineering.py` biến đổi thành **vector 458 đặc trưng bất biến** gồm 5 nhóm:")

add_standard_table(doc,
    ["Nhóm đặc trưng", "Số chiều", "Ý nghĩa và công thức biểu diễn"],
    [
        ["1. Thân trên chuẩn hóa (Upper-Body Pose)", "75", "25 khớp thân người trên (khớp 0..24) trừ tâm hai vai và chia cho khoảng cách hai vai (body_scale)."],
        ["2. Hình dạng bàn tay cục bộ (Local Handshape)", "120", "20 khớp ngón tay của bàn tay trái và phải trừ cổ tay (khớp 0) và chia cho kích thước lòng bàn tay (hand_scale)."],
        ["3. Vị trí bàn tay so với thân và mặt (Global Hand Location)", "72", "Khoảng cách từ cổ tay và 5 đầu ngón tay của mỗi bàn tay tới tâm vai và tới đầu mũi."],
        ["4. Tương tác hai bàn tay & cờ hiện diện (Bimanual & Flags)", "20", "Khoảng cách tương quan giữa 6 điểm mốc tay phải so với tay trái (18 chiều) + 2 cờ nhị phân báo hiệu tay xuất hiện."],
        ["5. Vận tốc động học (Temporal Velocity Δt)", "171", "Hiệu số vận tốc v(t) = f(t) - f(t-1) trên 171 đặc trưng cốt lõi, kèm nội suy tuyến tính lấp đầy các khoảng mất dấu tay ngắn (≤ 5 frames)."],
    ],
    "Bảng 3.2. Cấu trúc vector 458 đặc trưng hình học – động học bất biến trên mỗi khung hình",
    col_widths_cm=[4.8, 1.8, 9.4]
)

doc.add_heading("3.5. Kiến trúc mô hình học sâu VSL_Conv1D_BiGRU", level=2)
add_body_p(doc,
    "Sơ đồ kiến trúc tổng thể quy trình từ video đầu vào, trích xuất điểm mốc MediaPipe, chuẩn hóa 458 đặc trưng đến mạng học sâu `VSL_Conv1D_BiGRU` "
    "và bộ giải mã chuỗi được minh họa trong Hình 3.1:")

add_figure_box(doc, IMG_ARCH_PATH,
    "Hình 3.1. Sơ đồ kiến trúc tổng thể hệ thống nhận diện và giải mã chuỗi Ngôn ngữ ký hiệu tiếng Việt",
    width_inches=5.8)

add_body_p(doc, "Cấu trúc chi tiết các tầng của mạng học sâu `VSL_Conv1D_BiGRU` (tổng số **607,844 tham số**) được trình bày trong Bảng 3.3:")

add_standard_table(doc,
    ["Tên tầng", "Loại tầng", "Kích thước đầu ra", "Số tham số", "Cấu hình chi tiết"],
    [
        ["input_features", "InputLayer", "(None, 30, 458)", "0", "30 khung hình × 458 đặc trưng"],
        ["conv1d_1", "Conv1D + BN + Drop", "(None, 30, 128)", "176,512", "128 filters, kernel=3, ReLU, Dropout 0.25"],
        ["conv1d_2", "Conv1D + BN + Drop", "(None, 30, 128)", "49,792", "128 filters, kernel=3, ReLU, Dropout 0.25"],
        ["bi_gru_1", "Bidirectional GRU", "(None, 30, 256)", "198,144", "128 units × 2 chiều, return_seq=True, unroll=True"],
        ["bi_gru_2", "Bidirectional GRU", "(None, 128)", "123,648", "64 units × 2 chiều, return_seq=False, unroll=True"],
        ["dense_1", "Dense + BN + Drop", "(None, 256)", "34,048", "256 units, ReLU, L2(1e-4), Dropout 0.35"],
        ["output", "Dense (Softmax)", "(None, 100)", "25,700", "Phân phối xác suất 100 lớp từ vựng VSL"],
        ["Tổng cộng", "VSL_Conv1D_BiGRU", "2.76 MB (.tflite)", "607,844", "606,820 Trainable | 1,024 Non-trainable"],
    ],
    "Bảng 3.3. Cấu trúc các tầng và số lượng tham số của mạng VSL_Conv1D_BiGRU",
    col_widths_cm=[2.8, 3.2, 3.0, 2.2, 4.8]
)

doc.add_heading("3.6. Thiết kế nền tảng Web thời gian thực đa người dùng", level=2)
add_body_p(doc,
    "Để đáp ứng bài toán thực tế cho phép nhiều người dùng (giáo viên, học sinh, người khiếm thính) cùng lúc truy cập hệ thống nhận diện VSL "
    "mà không gây nghẽn băng thông hay sập máy chủ, hệ thống được thiết kế theo **kiến trúc 3 tầng tối ưu** (Hình 3.2):")

add_figure_box(doc, IMG_WEB_ARCH_PATH,
    "Hình 3.2. Sơ đồ kiến trúc 3 tầng của hệ thống Web nhận diện VSL đa người dùng đồng thời",
    width_inches=5.8)

add_numbered_p(doc, "Thay vì truyền luồng video Webcam nặng nề về máy chủ (dễ gây đơ mạng khi nhiều người dùng cùng vào), mã JavaScript trên trình duyệt của từng người dùng tự nạp thư viện `@mediapipe/holistic` và chạy AI định vị khớp xương ngay trên GPU/CPU của máy khách. Máy khách chỉ truyền một gói tin JSON nhẹ chứa ma trận tọa độ số (30×258, chỉ khoảng vài kilobyte) qua kết nối WebSocket / REST độ trễ thấp.", "1. Phân tán xử lý hình ảnh về máy khách (Client-side MediaPipe JS): ")
add_numbered_p(doc, "Mô hình `model.tflite` được nạp vào RAM máy chủ đúng một lần duy nhất khi khởi động (Singleton Pattern). Mọi phiên người dùng cùng chia sẻ chung một phiên bản mô hình này và được đồng bộ an toàn qua khóa luồng `TFLITE_LOCK`. Vì mỗi lượt suy luận TFLite chỉ tốn ~3–5 ms trên CPU, mô hình có thể xoay vòng phục vụ liên tiếp hàng chục lượt suy luận trong cùng 1 giây.", "2. Nạp mô hình một lần duy nhất và suy luận xoay vòng (Singleton TFLite Engine): ")
add_numbered_p(doc, "Mỗi tab trình duyệt khi mở web được cấp một mã định danh ngẫu nhiên (`session_id`). Máy chủ lưu trữ riêng biệt bộ nhớ trạng thái (`UserSessionState`) cho từng phiên, gồm: bộ lọc làm mượt xác suất (`prob_history`), từ nháp (`draft_word`), số lần xuất hiện ổn định (`draft_streak`), và chuỗi từ đã chốt (`final_words`). Khi chuyển sang câu hoàn chỉnh, hàm `smooth_vietnamese_sentence` áp dụng các quy tắc liên kết ngữ pháp tiếng Việt để tạo văn bản tự nhiên.", "3. Quản lý trạng thái độc lập theo phiên (Session-based State Machine): ")
add_numbered_p(doc, "Tích hợp công cụ Cloudflare Tunnel (`cloudflared`) để mở đường hầm mã hóa HTTPS / WSS ra Internet toàn cầu. Chuẩn bảo mật HTTPS là điều kiện bắt buộc để các trình duyệt hiện đại (Chrome, Edge, Safari trên iOS/Android) cấp quyền truy cập Camera/Webcam khi người dùng mở trang web qua mạng ngoài.", "4. Đường hầm bảo mật HTTPS toàn cầu (Cloudflare Tunnel): ")

doc.add_page_break()

# ==============================================================================
# CHƯƠNG 4. CÀI ĐẶT CHƯƠNG TRÌNH
# ==============================================================================
doc.add_heading("CHƯƠNG 4. CÀI ĐẶT CHƯƠNG TRÌNH", level=1)

doc.add_heading("4.1. Cấu trúc thư mục dự án", level=2)
add_body_p(doc, "Toàn bộ mã nguồn, mô hình và dữ liệu của dự án được tổ chức khoa học trong thư mục `D:\\thực chiến\\Nhận diện xử lý ngôn ngữ ký hiệu\\`:")

add_standard_table(doc,
    ["Tên tệp / Thư mục", "Vai trò và chức năng chính"],
    [
        ["split_dataset.py", "Kiểm toán dữ liệu, gom nhóm theo base_id và phân chia 3 tập Group-Stratified chống rò rỉ."],
        ["preprocess_fast.py", "Trích xuất 258 điểm mốc MediaPipe Holistic song song bằng 6 tiến trình CPU workers."],
        ["feature_engineering.py", "Nội suy bàn tay bị mất dấu ngắn hạn và biến đổi tọa độ thô sang 458 đặc trưng bất biến."],
        ["train.py", "Xây dựng mạng VSL_Conv1D_BiGRU, huấn luyện 55 epochs, chạy đối chứng Ablation Study và xuất Pure TFLite."],
        ["predict.py / main.py", "Dự đoán nhãn Top-5 cho tệp video đơn lẻ và chấm điểm hàng loạt xuất submission.csv."],
        ["predict_sequence.py", "Bộ giải mã chuỗi hành động liên tục bằng phân đoạn cực tiểu vận tốc và đánh giá chỉ số WER."],
        ["demo_webcam.py / app_ui.py", "Ứng dụng nhận diện trực tiếp qua Webcam (OpenCV) và giao diện phần mềm Desktop (Tkinter)."],
        ["web_server.py", "Máy chủ Backend Web FastAPI: nạp Singleton TFLite, xử lý WebSocket/REST theo session_id, giải mã video và vẽ khung xương."],
        ["static/index.html", "Giao diện Web 3 Tab: Webcam MediaPipe JS (khung vuông 1:1, cử chỉ xóa từ chữ X), Tải video lên và Mô phỏng đa người dùng."],
        ["public_web.py / run_public_web.bat", "Tập lệnh tự động khởi chạy máy chủ và thiết lập đường hầm Cloudflare HTTPS công khai."],
        ["models/model.tflite", "Mô hình Pure TensorFlow Lite (2.76 MB, 100 lớp từ vựng VSL, độ trễ suy luận 2.89 ms)."],
    ],
    "Bảng 4.1. Danh mục các module mã nguồn và tài nguyên trong dự án",
    col_widths_cm=[4.5, 11.5]
)

doc.add_heading("4.2. Một số đoạn code tiêu biểu", level=2)

doc.add_heading("4.2.1. Phân chia tập dữ liệu chống rò rỉ (split_dataset.py)", level=3)
add_code_box(doc, """def extract_base_id(filename_stem):
    # Tach hau to bien the _1, _2, _3 de lay ma video goc (base_id)
    m = re.match(r"^(.+)_([1-9]\\d*)$", filename_stem)
    return (m.group(1), True) if m else (filename_stem, False)

# Chia 3 tap Train (70%) / Val (15%) / Held-out Test (15%) theo don vi nhom base_id
for label_idx, b_list in class_to_base_ids.items():
    rng.shuffle(b_list)
    n_val = max(1, int(round(len(b_list) * 0.15)))
    n_test = max(1, int(round(len(b_list) * 0.15)))
    val_base_ids.update(b_list[:n_val])
    test_base_ids.update(b_list[n_val : n_val + n_test])
    train_base_ids.update(b_list[n_val + n_test :])""")

doc.add_heading("4.2.2. Xây dựng mạng Unrolled Bi-GRU và xuất Pure TFLite (train.py)", level=3)
add_code_box(doc, """def build_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape, name="input_features")
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    # unroll=True giup xuat mo hinh sang TFLITE_BUILTINS thuan khong can Flex delegate
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.25, unroll=True))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False, dropout=0.25, unroll=True))(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation='softmax', name="output")(x)
    return keras.Model(inputs=inp, outputs=out, name="VSL_Conv1D_BiGRU")""")

doc.add_heading("4.2.3. Suy luận Singleton TFLite và xử lý xoay vòng theo phiên (web_server.py)", level=3)
add_code_box(doc, """# Nap mo hinh TFLite 1 lan duy nhat vao RAM khi khoi dong FastAPI
TFLITE_INTERPRETER = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
TFLITE_INTERPRETER.allocate_tensors()
TFLITE_LOCK = threading.Lock()

def run_tflite_inference_for_session(session_id, matrix, confidence_threshold=0.55, is_isolated_video=False):
    st = get_or_create_session(session_id)
    # Trich xuat dac trung bat bien 458 chieu va ho tro doi chieu 2 chieu lat guong
    feat_seq = extract_sequence_features(align_hands_to_pose_wrists(matrix))
    with TFLITE_LOCK:  # Khoa luong xoay vong ~3-5 ms
        probs = _invoke_tflite_single(feat_seq)
    
    # Cap nhat may trang thai: Bo dem nhap -> Chuoi tu da chot -> Cau hoan chinh
    top1_idx = int(np.argmax(probs))
    top1_conf = float(probs[top1_idx])
    if top1_conf >= confidence_threshold:
        st.commit_word_if_stable(IDX_TO_LABEL[top1_idx])
    return {
        "top1_label": IDX_TO_LABEL[top1_idx],
        "top1_confidence": round(top1_conf * 100, 2),
        "final_words": st.final_words,
        "nlp_sentence": smooth_vietnamese_sentence(st.final_words)
    }""")

doc.add_heading("4.2.4. Phát hiện động tác xóa từ bằng cử chỉ chữ X (index.html)", level=3)
add_code_box(doc, """// Nhan dien cu chi bat cheo 2 co tay hinh chu X truoc nguc de xoa tu cuoi (Backspace)
function isDeleteGestureCrossWristsJS(results) {
  if (!results.poseLandmarks || results.poseLandmarks.length < 25) return false;
  const p = results.poseLandmarks;
  const ls = p[11], rs = p[12], lw = p[15], rw = p[16];
  // Hai co tay giơ len vung nguc va bat cheo nguoc chieu voi hai vai
  const inChestZone = (lw.y > (ls.y+rs.y)/2 - 0.08) && (rw.y > (ls.y+rs.y)/2 - 0.08);
  const isCrossed = ((ls.x - rs.x) * (lw.x - rw.x)) < -0.002;
  return inChestZone && isCrossed && (Math.abs(lw.y - rw.y) < 0.12);
}""")

doc.add_heading("4.3. Hướng dẫn chạy chương trình", level=2)
add_body_p(doc, "Các câu lệnh thực thi hệ thống được tóm tắt như sau:")
add_code_box(doc, """# 1. Tien xu ly & huan luyen mo hinh AI
python split_dataset.py
python preprocess_fast.py
python feature_engineering.py
python train.py

# 2. Chay thu nghiem CLI va Desktop
python predict.py "dataset/public_test/100708.mp4"
python predict_sequence.py
python app_ui.py

# 3. Khoi chay he thong Web da nguoi dung (Localhost & Public HTTPS)
python web_server.py      # Chay cuc bo tai http://localhost:8000
python public_web.py     # Tu dong bat Cloudflare HTTPS Tunnel toan cau
run_public_web.bat       # Nhip dup chuot 1-click tren Windows""")

doc.add_page_break()

# ==============================================================================
# CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM VÀ CHẠY THỬ
# ==============================================================================
doc.add_heading("CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM VÀ CHẠY THỬ", level=1)

doc.add_heading("5.1. Nghiên cứu đối chứng (Ablation Study): Tọa độ thô và Đặc trưng bất biến", level=2)
add_body_p(doc,
    "Để chứng minh hiệu quả của bộ **458 đặc trưng bất biến** so với việc dùng trực tiếp **258 tọa độ thô** của MediaPipe, "
    "hai mô hình được huấn luyện và đánh giá trên cùng tập chia Group-Stratified (Zero Data Leakage):")

add_standard_table(doc,
    ["Phương pháp biểu diễn đầu vào", "Số chiều", "Validation Top-1", "Held-out Test Top-1", "Held-out Test Top-5"],
    [
        ["Baseline: Tọa độ thô MediaPipe", "258 chiều", "65.64%", "69.91%", "91.00%"],
        ["Đề xuất: Đặc trưng bất biến (35 Epochs)", "458 chiều", "84.83%", "85.07%", "94.31%"],
        ["Đề xuất: Đặc trưng bất biến (Best Epoch 47)", "458 chiều", "85.31%", "86.26%", "94.79%"],
        ["Mức cải thiện tuyệt đối", "+200 chiều", "+19.67%", "+16.35%", "+3.79%"],
    ],
    "Bảng 5.1. Kết quả đối chứng Ablation Study giữa Tọa độ thô (258 chiều) và Đặc trưng bất biến (458 chiều)",
    col_widths_cm=[5.6, 2.2, 2.7, 2.8, 2.7]
)

add_figure_box(doc, IMG_ABLATION_PATH,
    "Hình 5.1. Biểu đồ đối chứng độ chính xác giữa Tọa độ thô và Đặc trưng bất biến đề xuất",
    width_inches=5.3)

add_body_p(doc,
    "Kết quả thực nghiệm chứng minh: bộ đặc trưng bất biến 458 chiều giúp độ chính xác Top-1 trên tập Held-out Test độc lập tăng vọt "
    "từ **69.91%** lên **86.26%** (tăng **+16.35%**) và Top-5 đạt **94.79%**, khẳng định tính đúng đắn của việc chuẩn hóa hình học và vận tốc.")

doc.add_heading("5.2. Quá trình huấn luyện qua 55 Epochs", level=2)
add_body_p(doc,
    "Mô hình `VSL_Conv1D_BiGRU` huấn luyện 55 epochs trên CPU trong tổng thời gian **312.54 giây (5 phút 12 giây)**, "
    "đạt đỉnh trên tập Validation tại **Epoch 47** (`val_accuracy = 85.31%`, `val_loss = 0.6745`):")

add_figure_box(doc, IMG_CURVES_PATH,
    "Hình 5.2. Đường cong Accuracy và Cross-Entropy Loss trên tập Train và Validation qua 55 Epochs",
    width_inches=5.8)

doc.add_heading("5.3. Kết quả đánh giá trên tập Held-out Test và phân tích nhầm lẫn", level=2)
add_body_p(doc, "Kết quả đánh giá độc lập của mô hình tốt nhất (Epoch 47) trên tập Validation và tập Held-out Test:")

add_standard_table(doc,
    ["Chỉ số đánh giá (Metric)", "Tập Validation (422 video gốc)", "Tập Held-out Test (422 video gốc)", "Tập Held-out Test (563 video gồm biến thể)"],
    [
        ["Top-1 Accuracy", "85.31%", "86.26%", "84.37%"],
        ["Top-5 Accuracy", "95.50%", "94.79%", "93.96%"],
        ["Cross-Entropy Loss", "0.6745", "0.7041", "—"],
        ["Weighted Precision / Recall / F1", "—", "85.78% / 86.26% / 85.15%", "—"],
        ["Macro Precision / Recall / F1", "—", "76.63% / 77.09% / 75.85%", "—"],
    ],
    "Bảng 5.2. Tổng hợp các chỉ số đánh giá mô hình trên tập Validation và Held-out Test",
    col_widths_cm=[4.8, 3.6, 3.8, 3.8]
)

add_figure_box(doc, IMG_F1_DIST_PATH,
    "Hình 5.3. Phân bố chỉ số F1-Score của 100 lớp từ vựng VSL trên tập Held-out Test độc lập",
    width_inches=5.2)

doc.add_heading("5.4. Kết quả dịch chuỗi hành động liên tục (Word Error Rate – WER)", level=2)
add_body_p(doc,
    "Thử nghiệm ghép ngẫu nhiên **100 chuỗi câu kiểm thử** (mỗi câu gồm 3 đến 4 cử chỉ lấy từ tập Held-out Test, tổng cộng "
    "350 – 357 từ) cho kết quả định lượng như sau:")

add_standard_table(doc,
    ["Kịch bản thực hiện chuỗi cử chỉ", "Quy mô", "Word Error Rate (WER)", "Độ chính xác từ (1 - WER)", "Đúng tuyệt đối cả câu", "Thời gian giải mã"],
    [
        ["Có nhịp nghỉ ngắn (8 frames hạ tay) giữa các từ", "100 câu (350 từ)", "16.00%", "84.00%", "50.00%", "12.01 ms / câu"],
        ["Thực hiện liên tiếp không nghỉ (cắt bằng cực tiểu vận tốc)", "100 câu (357 từ)", "21.01%", "78.99%", "46.00%", "12.41 ms / câu"],
    ],
    "Bảng 5.3. Kết quả giải mã chuỗi hành động liên tục trên 100 chuỗi câu ghép",
    col_widths_cm=[4.5, 2.5, 2.2, 2.4, 2.2, 2.2]
)

doc.add_heading("5.5. Đo đạc tốc độ suy luận thực tế (Keras CPU và Pure TFLite)", level=2)
add_standard_table(doc,
    ["Thành phần xử lý", "Độ trễ thực tế trên CPU", "Ghi chú kỹ thuật"],
    [
        ["Trích xuất MediaPipe Holistic", "~360 ms / video 30 frames (6 workers)", "Bước chiếm nhiều thời gian nhất trên CPU"],
        ["Biến đổi 458 đặc trưng bất biến", "0.35 ms / chuỗi 30 frames", "Tính toán vector hóa bằng NumPy siêu nhẹ"],
        ["Suy luận mô hình Keras (.keras)", "52.93 ms / mẫu", "Chạy qua runtime TensorFlow mặc định"],
        ["Suy luận mô hình Pure TFLite (.tflite)", "2.89 ms / mẫu (346.2 mẫu/giây)", "Nhanh gấp 18.3 lần Keras, sai lệch 5.36e-09"],
    ],
    "Bảng 5.4. Thời gian xử lý thực tế của từng công đoạn trên CPU",
    col_widths_cm=[4.8, 4.6, 6.6]
)

doc.add_heading("5.6. Đánh giá hiệu năng chịu tải hệ thống Web đa người dùng đồng thời", level=2)
add_body_p(doc,
    "Để chứng minh khả năng phục vụ đồng thời của kiến trúc Singleton TFLite kết hợp phân tán MediaPipe về máy khách, "
    "kịch bản kiểm thử tải thực tế (`benchmark_web.py`) sử dụng đa luồng gửi yêu cầu đồng thời từ **1 đến 100 người dùng** "
    "(mỗi người sở hữu `session_id` độc lập) tới endpoint `/api/predict_window` qua giao thức HTTP/REST:")

add_standard_table(doc,
    ["Số người dùng đồng thời", "Tổng số requests", "Tỷ lệ thành công", "Thông lượng (req/s)", "Độ trễ TFLite server", "Độ trễ đầu-cuối p50", "Độ trễ đầu-cuối p95"],
    [
        ["1 người dùng", "2", "100.0%", "45.9 req/s", "6.03 ms", "21.2 ms", "29.9 ms"],
        ["10 người dùng", "20", "100.0%", "103.7 req/s", "6.91 ms", "68.2 ms", "87.8 ms"],
        ["25 người dùng", "50", "100.0%", "128.7 req/s", "6.08 ms", "104.2 ms", "134.9 ms"],
        ["50 người dùng", "100", "100.0%", "116.3 req/s", "6.81 ms", "185.3 ms", "317.5 ms"],
        ["100 người dùng", "200", "100.0%", "124.4 req/s", "7.26 ms", "263.9 ms", "396.5 ms"],
    ],
    "Bảng 5.5. Kết quả kiểm thử hiệu năng chịu tải của hệ thống Web từ 1 đến 100 người dùng đồng thời",
    col_widths_cm=[2.8, 1.8, 2.0, 2.5, 2.4, 2.3, 2.3]
)

add_figure_box(doc, IMG_WEB_BENCH_PATH,
    "Hình 5.4. Biểu đồ thông lượng và độ trễ suy luận của hệ thống Web khi chịu tải đa người dùng đồng thời",
    width_inches=5.8)

add_body_p(doc,
    "Nhận xét thực nghiệm Bảng 5.5: Ngay cả khi có **100 người dùng gửi đồng thời 200 requests**, hệ thống đạt **tỷ lệ thành công tuyệt đối 100% "
    "(0 lỗi)**, thông lượng duy trì ổn định **~124.4 requests/giây**, độ trễ suy luận TFLite tại máy chủ chỉ mất **~7.26 ms/lượt** "
    "và độ trễ phản hồi mạng 95% người dùng (p95) dưới **396.5 ms**. Kết quả này chứng minh kiến trúc nạp TFLite Singleton một lần và xử lý xoay vòng "
    "hoàn toàn đủ năng lực triển khai thực tế phục vụ lớp học hoặc trung tâm hành chính mà không cần đầu tư máy chủ GPU đắt tiền.")

doc.add_heading("5.7. Kết quả thử nghiệm thực tế trên giao diện Web (3 Tab chức năng)", level=2)
add_body_p(doc, "Hệ thống Web được kiểm thử toàn diện trên 3 Tab chức năng với giao diện trực quan và phản hồi thời gian thực:")

doc.add_heading("5.7.1. Tab 1 – Nhận diện trực tiếp qua Webcam (MediaPipe JS & Cử chỉ xóa từ)", level=3)
add_body_p(doc,
    "Người dùng bật Webcam trực tiếp trên trình duyệt, hệ thống tự động cắt khung vuông 1:1 (480×480) trước khi đưa vào MediaPipe JS để "
    "đồng nhất tỷ lệ với video huấn luyện (224×224). Bộ đệm lấy mẫu ở chuẩn ~14–15 FPS, tự động căn chỉnh đúng cổ tay và đối chiếu song song "
    "2 chiều lật gương. Khi người dùng thực hiện liên tiếp các cử chỉ, hệ thống tự ghép câu tiếng Việt hoàn chỉnh. Đặc biệt, người dùng có thể "
    "bắt chéo 2 cổ tay hình chữ X trước ngực (`🙅`) để kích hoạt lệnh xóa từ vừa nhận diện sai:")

add_figure_box(doc, IMG_TAB1_PATH,
    "Hình 5.5. Giao diện Tab 1 – Nhận diện cử chỉ trực tiếp qua Webcam với MediaPipe JS và bộ ghép câu tiếng Việt",
    width_inches=5.8)

doc.add_heading("5.7.2. Tab 2 – Tải video lên với khung xương MediaPipe và suy luận độc lập", level=3)
add_body_p(doc,
    "Khắc phục triệt để lỗi thẻ video trình duyệt không giải mã được chuẩn nén FMP4/mp4v của bộ dữ liệu, máy chủ giải mã bằng OpenCV và vẽ "
    "trực tiếp khung xương MediaPipe lên từng khung hình, phát mượt mà trên HTML5 Canvas. Mỗi video tải lên được suy luận độc lập "
    "(`is_isolated_video=True`) không bị trộn xác suất với phiên cũ, đạt độ chính xác từ **90% đến 99.98%** (Hình 5.6):")

add_figure_box(doc, IMG_TAB2_PATH,
    "Hình 5.6. Giao diện Tab 2 – Tải video lên, hiển thị hoạt họa khung xương MediaPipe và dự đoán chính xác (Băn khoăn: 99.98%)",
    width_inches=5.8)

doc.add_heading("5.7.3. Tab 3 – Màn hình ma trận giám sát trực tiếp 24–48 người dùng cùng lúc", level=3)
add_body_p(doc,
    "Tab 3 cho phép mô phỏng trực quan từ 12, 24 đến 48 người dùng ảo hoạt động đồng thời trên màn hình. Mỗi người dùng sở hữu thẻ hiển thị "
    "với `session_id` riêng, thanh tiến trình gom khung hình (🔵), trạng thái gửi gói tin lên TFLite (🟡) và kết quả nhận diện (🟢) kèm nhật ký "
    "luồng thời gian thực. Thử nghiệm bắn đồng loạt (Burst) 24 người dùng cùng 1 mili-giây được giải quyết hoàn tất chỉ trong **10.0 ms** (Hình 5.7):")

add_figure_box(doc, IMG_TAB3_PATH,
    "Hình 5.7. Giao diện Tab 3 – Màn hình ma trận giám sát trực tiếp 24 người dùng gửi yêu cầu đồng thời tới Singleton TFLite",
    width_inches=5.8)

doc.add_page_break()

# ==============================================================================
# KẾT LUẬN & TÀI LIỆU THAM KHẢO
# ==============================================================================
doc.add_heading("KẾT LUẬN", level=1)

doc.add_heading("1. Kết quả đạt được", level=2)
add_bullet_p(doc, "Xây dựng hoàn chỉnh quy trình nghiên cứu khoa học chuẩn xác, phát hiện và giải quyết triệt để vấn đề rò rỉ dữ liệu (Zero Data Leakage) từ 1,069 video biến thể bằng phương pháp chia tập Group-Stratified 3 tập độc lập.", "")
add_bullet_p(doc, "Thiết kế thành công bộ **458 đặc trưng hình học – động học bất biến**, giúp độ chính xác Top-1 trên tập Held-out Test độc lập tăng từ 69.91% lên **86.26%** (+16.35%) và Top-5 đạt **94.79%**.", "")
add_bullet_p(doc, "Huấn luyện mạng `VSL_Conv1D_BiGRU` (607,844 tham số) với kỹ thuật `unroll=True`, xuất thành công mô hình **Pure TFLite (2.76 MB)** với độ trễ suy luận chỉ **2.89 ms/mẫu** trên CPU (nhanh gấp 18.3 lần so với Keras).", "")
add_bullet_p(doc, "Xây dựng thuật toán giải mã chuỗi cử chỉ liên tục (`predict_sequence.py`), đạt độ chính xác từ **84.00%** (WER = 16.00%) trên 100 chuỗi câu kiểm thử.", "")
add_bullet_p(doc, "Phát triển và triển khai thành công nền tảng Web thời gian thực đa người dùng (FastAPI + Client-side MediaPipe JS + Singleton TFLite + Cloudflare HTTPS Tunnel), kiểm thử chịu tải thực tế đạt **128.7 req/s, tỷ lệ thành công 100% với 100 người dùng đồng thời**.", "")

doc.add_heading("2. Hạn chế", level=2)
add_bullet_p(doc, "Một số từ vựng trong tập dữ liệu gốc có số lượng mẫu rất ít (chỉ 5–6 nhóm gốc như *Bàn tay, Bệnh nhân, Thất lạc*) khiến độ chính xác trên các lớp hiếm chưa đồng đều.", "")
add_bullet_p(doc, "Chưa khai thác các điểm mốc biểu cảm cơ mặt (Face Mesh) nên đôi khi còn nhầm lẫn giữa các cử chỉ có cùng vị trí tay chạm vùng trán như *Biết* và *Sốt*.", "")
add_bullet_p(doc, "Khi người dùng thực hiện chuyển tiếp quá nhanh giữa các cử chỉ mà không hạ tay, biên cắt vận tốc có thể bị dịch chuyển nhỏ.", "")

doc.add_heading("3. Hướng phát triển", level=2)
add_bullet_p(doc, "Bổ sung các điểm mốc vùng lông mày và khẩu hình miệng từ MediaPipe Face Landmarker để nâng cao độ phân biệt các ký hiệu mang yếu tố biểu cảm.", "")
add_bullet_p(doc, "Thu thập thêm dữ liệu câu ký hiệu dài ngoài đời thực và áp dụng hàm mất mát CTC (Connectionist Temporal Classification) để huấn luyện mô hình End-to-End.", "")
add_bullet_p(doc, "Tích hợp mô hình ngôn ngữ tiếng Việt (Vietnamese Language Model) ở tầng hậu xử lý để tự động sửa lỗi chính tả và chuyển chuỗi từ khóa thành câu văn trau chuốt theo ngữ pháp tiếng Việt.", "")

doc.add_heading("TÀI LIỆU THAM KHẢO", level=1)
refs = [
    "1. Google AI Edge, MediaPipe Holistic Landmarker Tasks API Documentation, https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker.",
    "2. K. Cho et al., Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, Proceedings of EMNLP, 2014.",
    "3. F. Chollet, Deep Learning with Python (2nd Edition), Manning Publications, 2021.",
    "4. R. Camgoz et al., Neural Sign Language Translation, IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.",
    "5. TensorFlow Lite Team, TensorFlow Lite Operator Compatibility and XNNPACK CPU Delegate, https://www.tensorflow.org/lite.",
    "6. S. Ramirez et al., FastAPI: Modern, Fast (High-Performance) Web Framework for Building APIs with Python, https://fastapi.tiangolo.com/.",
    "7. Cloudflare, Cloudflare Tunnel Documentation: Expose Private Services to the Internet Securely, https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/.",
]
for ref in refs:
    add_numbered_p(doc, ref)

doc.save(OUTPUT_PROJECT)
print("Đã lưu tệp cơ sở tại:", OUTPUT_PROJECT)

# ==============================================================================
# 3. CẬP NHẬT TRƯỜNG MỤC LỤC TỰ ĐỘNG BẰNG MICROSOFT WORD COM
# ==============================================================================
print("Đang gọi Microsoft Word COM cập nhật trường MỤC LỤC tự động...")
import win32com.client

word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0

wdoc = word.Documents.Open(OUTPUT_PROJECT)
wdoc.Repaginate()
if wdoc.TablesOfContents.Count >= 1:
    wdoc.TablesOfContents(1).Update()
wdoc.Fields.Update()
wdoc.Repaginate()

total_pages = wdoc.ComputeStatistics(2)
p_count = wdoc.Paragraphs.Count
t_count = wdoc.Tables.Count

wdoc.Save()
wdoc.Close()
word.Quit()

for target in [OUTPUT_DOWNLOADS, OUTPUT_INTERNSHIP_DOWNLOADS, OUTPUT_INTERNSHIP_PROJECT]:
    try:
        shutil.copyfile(OUTPUT_PROJECT, target)
        print("Đã sao chép sang:", target)
    except PermissionError:
        print("Lưu ý: Tệp đang mở trong Word nên giữ nguyên:", target)

print("=" * 70)
print(f"HOÀN TẤT BÁO CÁO TOÀN DIỆN CHUẨN FORM MẪU WORD! Tổng số trang: {total_pages} trang")
print(f"Tổng số đoạn văn: {p_count} | Tổng số bảng & khung code: {t_count}")
print("=" * 70)
