import win32com.client
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_component(name, build_fn):
    d = Document()
    build_fn(d)
    path = f'{BASE_DIR}\\test_{name}.docx'
    d.save(path)
    
    word = win32com.client.Dispatch('Word.Application')
    word.Visible = False
    word.DisplayAlerts = 0
    ok = False
    try:
        doc = word.Documents.Open(path)
        ok = True
        doc.Close()
    except Exception as e:
        pass
    word.Quit()
    res = 'PASSED' if ok else 'FAILED'
    print(f'Component [{name}]: {res}')

# Test 1: Image
test_component('image', lambda d: d.add_picture(rf'{BASE_DIR}\models\training_curves.png'))

# Test 2: Footer Page numbering hack
def add_footer_hack(d):
    s = d.sections[0]
    p = s.footer.paragraphs[0]
    p.add_run('Page ')
    fld1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    p.runs[-1]._r.addnext(fld1)
    instr = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
    fld1.addnext(instr)
    fld2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    instr.addnext(fld2)
test_component('footer_hack', add_footer_hack)

# Test 3: Table with tcMar
def add_table_tcmar(d):
    t = d.add_table(rows=2, cols=2)
    c = t.rows[0].cells[0]
    tcPr = c._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:left w:w="150" w:type="dxa"/><w:right w:w="150" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)
test_component('table_tcmar', add_table_tcmar)

# Test 4: Table with shading
def add_table_shading(d):
    t = d.add_table(rows=2, cols=2)
    c = t.rows[0].cells[0]
    s = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F4E79"/>')
    c._tc.get_or_add_tcPr().append(s)
test_component('table_shading', add_table_shading)
