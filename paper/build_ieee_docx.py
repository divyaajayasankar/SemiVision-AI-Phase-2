from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re

ROOT=Path(__file__).resolve().parents[1]
md=(ROOT/'paper'/'IEEE_Paper.md').read_text(encoding='utf-8')
doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.62); sec.left_margin=Inches(.65); sec.right_margin=Inches(.65)
sec.page_width=Inches(8.5); sec.page_height=Inches(11)

styles=doc.styles
styles['Normal'].font.name='Times New Roman'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman'); styles['Normal'].font.size=Pt(10)
styles['Normal'].paragraph_format.space_after=Pt(2); styles['Normal'].paragraph_format.line_spacing=1.0
for name in ['Title','Heading 1','Heading 2','Heading 3']:
    styles[name].font.name='Times New Roman'; styles[name]._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman')
styles['Title'].font.size=Pt(18); styles['Title'].font.bold=True
styles['Heading 1'].font.size=Pt(10); styles['Heading 1'].font.bold=False
styles['Heading 2'].font.size=Pt(9.5); styles['Heading 2'].font.bold=False; styles['Heading 2'].font.italic=True

# helper
def set_cell_text(cell, text, size=7.5, bold=False):
    cell.text=''
    p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(0)
    r=p.add_run(text); r.font.name='Times New Roman'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman'); r.font.size=Pt(size); r.font.bold=bold
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def set_cols(section, n=2, space_twips=360):
    sectPr=section._sectPr
    cols=sectPr.xpath('./w:cols')
    if cols: cols_el=cols[0]
    else:
        cols_el=OxmlElement('w:cols'); sectPr.append(cols_el)
    cols_el.set(qn('w:num'),str(n)); cols_el.set(qn('w:space'),str(space_twips))

def add_body_p(text):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.first_line_indent=Inches(.15); p.paragraph_format.space_after=Pt(2)
    r=p.add_run(text); r.font.name='Times New Roman'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman'); r.font.size=Pt(10)
    return p

lines=md.splitlines()
# Title / author / abstract extraction
idx=0
assert lines[0].startswith('# ')
title=lines[0][2:].strip(); p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(title)
idx=2
author=lines[idx].replace('**','').strip(); p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(author); r.font.name='Times New Roman'; r.font.size=Pt(10); idx+=1
# skip blank
while idx<len(lines) and not lines[idx].strip(): idx+=1
# abstract
if lines[idx].startswith('## Abstract'):
    idx+=1
    while idx<len(lines) and not lines[idx].strip(): idx+=1
    abstract=lines[idx].strip(); idx+=1
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.left_indent=Inches(.25); p.paragraph_format.right_indent=Inches(.25)
    r=p.add_run('Abstract—'); r.bold=True; r.font.name='Times New Roman'; r.font.size=Pt(9)
    r=p.add_run(abstract); r.font.name='Times New Roman'; r.font.size=Pt(9)
while idx<len(lines) and not lines[idx].strip(): idx+=1
if idx<len(lines) and lines[idx].startswith('**Index Terms'):
    t=lines[idx].replace('**','').strip(); idx+=1
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.left_indent=Inches(.25); p.paragraph_format.right_indent=Inches(.25)
    r=p.add_run(t); r.font.name='Times New Roman'; r.font.size=Pt(9)
# start two-column continuous section
sec2=doc.add_section(WD_SECTION.CONTINUOUS); sec2.top_margin=Inches(.62); sec2.bottom_margin=Inches(.62); sec2.left_margin=Inches(.65); sec2.right_margin=Inches(.65); set_cols(sec2,2,360)

# custom tables inserted before normal text when section headings encountered
image_table_done=loc_table_done=ablation_done=drift_done=False

while idx < len(lines):
    line=lines[idx].rstrip(); idx+=1
    if not line.strip(): continue
    if line.startswith('## REFERENCES'):
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run('REFERENCES'); r.font.name='Times New Roman'; r.font.size=Pt(10); r.font.bold=True
        continue
    if line.startswith('## '):
        txt=line[3:].strip(); p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(6); p.paragraph_format.space_after=Pt(3); r=p.add_run(txt); r.font.name='Times New Roman'; r.font.size=Pt(10); r.font.bold=False
        continue
    if line.startswith('### '):
        txt=line[4:].strip(); p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(4); p.paragraph_format.space_after=Pt(2); r=p.add_run(txt); r.font.name='Times New Roman'; r.font.size=Pt(9.5); r.font.italic=True
        continue
    if line.startswith('**Algorithm 1:'):
        p=doc.add_paragraph(); r=p.add_run(line.replace('**','')); r.bold=True; r.font.name='Times New Roman'; r.font.size=Pt(9)
        # following code block
        continue
    if line.strip()=='```':
        code=[]
        while idx<len(lines) and lines[idx].strip()!='```': code.append(lines[idx]); idx+=1
        if idx<len(lines): idx+=1
        p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.1); p.paragraph_format.right_indent=Inches(.05); p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(4)
        r=p.add_run('\n'.join(code)); r.font.name='Courier New'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'Courier New'); r.font.size=Pt(7.2)
        continue
    if line.startswith('[1]') or re.match(r'^\[\d+\]',line):
        p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.18); p.paragraph_format.first_line_indent=Inches(-.18); p.paragraph_format.space_after=Pt(1); r=p.add_run(line); r.font.name='Times New Roman'; r.font.size=Pt(7.8)
        continue
    # Figure insertion after architecture paragraph trigger
    if 'The output of the system is therefore a set of screening decisions' in line:
        add_body_p(line)
        fig=ROOT/'diagrams'/'system_architecture.png'
        if fig.exists():
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(fig),width=Inches(3.3))
            cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run('Fig. 1. Two-phase SEM anomaly-screening architecture.'); rr.font.name='Times New Roman'; rr.font.size=Pt(7.5)
        continue
    # Insert result tables after relevant discussion paragraphs
    if 'The Phase-2 RCC-ConvAE framework retained essentially the same image-level performance' in line:
        add_body_p(line)
        fig=ROOT/'results'/'figures'/'table_image_metrics.png'
        if fig.exists():
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(fig),width=Inches(3.0))
        cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run('TABLE I. IMAGE-LEVEL REFERENCE-RUN PERFORMANCE'); rr.font.name='Times New Roman'; rr.font.size=Pt(7.2); rr.font.bold=True
        continue
    if 'The largest controlled improvement was observed in localization.' in line:
        add_body_p(line)
        figtab=ROOT/'results'/'figures'/'table_localization.png'
        if figtab.exists():
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(figtab),width=Inches(3.0))
        cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run('TABLE II. LOCALIZATION PERFORMANCE'); rr.font.name='Times New Roman'; rr.font.size=Pt(7.2); rr.font.bold=True
        fig=ROOT/'results'/'figures'/'reference_localization_comparison.png'
        if fig.exists():
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(fig),width=Inches(3.25))
            cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run('Fig. 2. Historical verified localization metrics.'); rr.font.name='Times New Roman'; rr.font.size=Pt(7.5)
        continue
    if 'The archived drift table showed the same aggregate drift values' in line:
        add_body_p(line)
        fig=ROOT/'results'/'figures'/'reference_drift_balanced_accuracy.png'
        if fig.exists():
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(fig),width=Inches(3.25))
            cp=doc.add_paragraph(); cp.alignment=WD_ALIGN_PARAGRAPH.CENTER; rr=cp.add_run('Fig. 3. Frozen versus recalibrated performance under controlled drift.'); rr.font.name='Times New Roman'; rr.font.size=Pt(7.5)
        continue
    add_body_p(line)

out=ROOT/'paper'/'IEEE_Paper_Risk_Calibrated_Semiconductor.docx'; doc.save(out); print(out)
