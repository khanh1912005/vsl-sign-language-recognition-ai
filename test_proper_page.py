import win32com.client
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def add_proper_page_number(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

d = Document()
p = d.sections[0].footer.paragraphs[0]
p.add_run("Trang ")
run_page = p.add_run()
add_proper_page_number(run_page)
path = rf"{BASE_DIR}\test_proper_page.docx"
d.save(path)

word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
ok = False
try:
    doc = word.Documents.Open(path)
    print("PROPER PAGE NUMBER IN WORD COM: PASSED! Page count:", doc.ComputeStatistics(2))
    doc.Close()
    ok = True
except Exception as e:
    print("PROPER PAGE NUMBER FAILED:", e)
word.Quit()
