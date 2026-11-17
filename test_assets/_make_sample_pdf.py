"""Throwaway helper: render sample_syllabus.txt into sample_syllabus.pdf."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

source = Path("sample_syllabus.txt")
target = Path("sample_syllabus.pdf")
lines = source.read_text(encoding="utf-8").splitlines()

page_width, page_height = A4
pdf = canvas.Canvas(str(target), pagesize=A4)
pdf.setTitle("Class 10 Science Syllabus")
x, y = 50, page_height - 60
pdf.setFont("Helvetica", 10)

for line in lines:
    if y < 60:
        pdf.showPage()
        pdf.setFont("Helvetica", 10)
        y = page_height - 60
    pdf.drawString(x, y, line)
    y -= 15

pdf.save()
print(f"wrote {target} from {len(lines)} lines")
