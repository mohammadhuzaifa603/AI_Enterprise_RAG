"""
Creates a synthetic *scanned-style* invoice PDF: the invoice text is
rendered onto an image and embedded into a PDF page as an image only
(no selectable text layer), so the ingestion pipeline's OCR fallback
path is genuinely exercised.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "sample_documents"
OUT.mkdir(exist_ok=True)

lines = [
    "INVOICE",
    "",
    "Vendor: Hazelwood IT Services Ltd.",
    "Invoice Number: INV-2026-77820",
    "Invoice Date: 2026-08-01",
    "",
    "Bill To: Northwind Analytics",
    "",
    "Item                    Qty   Unit Price   Line Total",
    "Laptop Deployment         3      $85.00        $255.00",
    "Network Setup             1     $600.00        $600.00",
    "",
    "Subtotal: $855.00",
    "Tax: $70.54",
    "Total: $925.54",
    "Currency: USD",
]

img = Image.new("RGB", (1275, 1650), "white")
draw = ImageDraw.Image if False else ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
except Exception:
    font = ImageFont.load_default()

y = 100
for line in lines:
    draw.text((100, y), line, fill="black", font=font)
    y += 45

img_path = OUT / "_scanned_invoice_source.png"
img.save(img_path)

pdf_path = OUT / "Synthetic_Scanned_Invoice.pdf"
c = canvas.Canvas(str(pdf_path), pagesize=letter)
width, height = letter
c.drawImage(str(img_path), 0, 0, width=width, height=height)
c.showPage()
c.save()
img_path.unlink()

print(f"Wrote scanned-style invoice to {pdf_path}")
