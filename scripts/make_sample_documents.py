"""
Generates realistic *synthetic* sample PDFs into sample_documents/.
No real company data is used - all content is invented for demo purposes.
Run once: python scripts/make_sample_documents.py
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

OUT = Path(__file__).resolve().parent.parent / "sample_documents"
OUT.mkdir(exist_ok=True)


def write_text_pdf(path, pages):
    """pages: list[list[str]] - each inner list is the lines for one page."""
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    for lines in pages:
        y = height - 1 * inch
        for line in lines:
            if line.startswith("# "):
                c.setFont("Helvetica-Bold", 15)
                c.drawString(1 * inch, y, line[2:])
                y -= 0.32 * inch
                c.setFont("Helvetica", 11)
            else:
                c.setFont("Helvetica", 11)
                c.drawString(1 * inch, y, line)
                y -= 0.22 * inch
        c.showPage()
    c.save()


# ---------------------------------------------------------------------
# 1. Employee Handbook
# ---------------------------------------------------------------------
handbook = [
    [
        "# Northwind Analytics — Employee Handbook",
        "",
        "Section 1: Company Overview",
        "Northwind Analytics is a data consulting firm founded in 2016,",
        "headquartered in Austin, Texas, with a distributed team across",
        "North America and Europe. We help mid-market companies build",
        "data platforms and analytics practices.",
        "",
        "Section 2: Working Hours",
        "Core working hours are 10:00 AM to 4:00 PM in the employee's local",
        "time zone. Employees are expected to work 40 hours per week,",
        "arranged flexibly around the core hours.",
    ],
    [
        "# Section 3: Remote Work Policy",
        "",
        "The company allows eligible employees to work remotely up to",
        "three days per week. Employees who wish to work remotely five",
        "days per week must submit a request to their manager and to HR",
        "for approval, renewed annually. Fully remote roles are explicitly",
        "marked as such in the job posting.",
        "",
        "Employees working remotely are expected to maintain the same",
        "core working hours and to attend all scheduled meetings with",
        "video enabled unless otherwise agreed with their manager.",
    ],
    [
        "# Section 4: Paid Time Off",
        "",
        "Full-time employees accrue 20 days of paid time off (PTO) per",
        "calendar year, accrued monthly. Unused PTO of up to 5 days may be",
        "carried over into the following year. PTO requests should be",
        "submitted at least 5 business days in advance through the HR",
        "portal, except in emergencies.",
        "",
        "Section 5: Sick Leave",
        "Employees receive 10 paid sick days per year, separate from PTO.",
        "A doctor's note is required for sick leave lasting more than 3",
        "consecutive days.",
    ],
    [
        "# Section 6: Code of Conduct",
        "",
        "All employees are expected to treat colleagues, clients, and",
        "partners with respect. Harassment, discrimination, and retaliation",
        "of any kind are strictly prohibited and will result in disciplinary",
        "action up to and including termination.",
        "",
        "Section 7: Expense Reimbursement",
        "Business expenses over $25 require a receipt. Reimbursement",
        "requests must be submitted within 30 days of the expense through",
        "the finance portal and are typically processed within 10 business",
        "days.",
    ],
]
write_text_pdf(OUT / "Employee_Handbook.pdf", handbook)

# ---------------------------------------------------------------------
# 2. Remote Work Policy (standalone, overlapping content on purpose to
#    exercise multi-document retrieval / citation grounding)
# ---------------------------------------------------------------------
remote_policy = [
    [
        "# Northwind Analytics — Remote Work Policy (Detailed)",
        "",
        "1. Eligibility",
        "All employees who have completed their 90-day probationary period",
        "are eligible to request remote work arrangements. Eligibility is",
        "reviewed annually by HR in partnership with the employee's manager.",
        "",
        "2. Frequency",
        "Standard remote work allowance is up to three days per week.",
        "Employees based more than 50 miles from a Northwind office may",
        "request a fully remote arrangement of five days per week.",
    ],
    [
        "# 3. Equipment",
        "Northwind provides a laptop, monitor, and a one-time $300 home",
        "office stipend to employees approved for remote work. Additional",
        "equipment requests are reviewed on a case-by-case basis by IT.",
        "",
        "4. Security Requirements",
        "Remote employees must use the company VPN when accessing internal",
        "systems and must enable full-disk encryption on all company",
        "devices. Public Wi-Fi may only be used with the VPN active.",
        "",
        "5. Review",
        "This policy is reviewed annually every January by the People",
        "Operations team and may be updated based on business needs.",
    ],
]
write_text_pdf(OUT / "Remote_Work_Policy.pdf", remote_policy)

# ---------------------------------------------------------------------
# 3. Sales Report (contains numeric data useful for multi-hop questions)
# ---------------------------------------------------------------------
sales_report = [
    [
        "# Northwind Analytics — Q2 2026 Sales Report",
        "",
        "Executive Summary",
        "Q2 2026 revenue reached $4.82 million, an increase of 14 percent",
        "compared to Q1 2026 ($4.23 million). Growth was driven primarily",
        "by the Financial Services and Healthcare verticals.",
        "",
        "Revenue by Region",
        "North America: $2.9 million",
        "Europe: $1.4 million",
        "Rest of World: $0.52 million",
    ],
    [
        "# Revenue by Vertical",
        "",
        "Financial Services: $1.8 million (up 22% quarter over quarter)",
        "Healthcare: $1.3 million (up 19% quarter over quarter)",
        "Retail: $0.95 million (up 4% quarter over quarter)",
        "Manufacturing: $0.77 million (down 3% quarter over quarter)",
        "",
        "Sales Pipeline",
        "The sales pipeline entering Q3 2026 stands at $9.1 million in",
        "qualified opportunities, with an average deal size of $185,000.",
    ],
    [
        "# Q3 2026 Outlook",
        "",
        "Leadership expects Q3 2026 revenue to grow between 8 and 12",
        "percent quarter over quarter, driven by continued momentum in",
        "Financial Services and the launch of the new Northwind Insights",
        "product line in September 2026.",
        "",
        "Risks include lengthening sales cycles in the Manufacturing",
        "vertical and increased competition in the Healthcare analytics",
        "space.",
    ],
]
write_text_pdf(OUT / "Q2_2026_Sales_Report.pdf", sales_report)

# ---------------------------------------------------------------------
# 4. Product Guide
# ---------------------------------------------------------------------
product_guide = [
    [
        "# Northwind Insights — Product Guide",
        "",
        "Overview",
        "Northwind Insights is a self-serve analytics product that lets",
        "business users build dashboards on top of their existing data",
        "warehouse without writing SQL.",
        "",
        "Key Features",
        "- Drag-and-drop dashboard builder",
        "- Natural language query interface",
        "- Scheduled email and Slack reports",
        "- Row-level security integrated with existing SSO roles",
    ],
    [
        "# Supported Data Warehouses",
        "Northwind Insights connects to Snowflake, BigQuery, Redshift, and",
        "Postgres. Connection credentials are stored encrypted at rest",
        "using AES-256.",
        "",
        "Pricing",
        "Northwind Insights is priced per active user per month, starting",
        "at $39/user/month for the Starter tier and $89/user/month for the",
        "Professional tier, which adds row-level security and API access.",
    ],
]
write_text_pdf(OUT / "Northwind_Insights_Product_Guide.pdf", product_guide)

# ---------------------------------------------------------------------
# 5. Synthetic Invoice (clean, native text - for Module 2)
# ---------------------------------------------------------------------
invoice_lines = [
    [
        "# INVOICE",
        "",
        "Vendor: BrightPath Office Supplies Inc.",
        "Invoice Number: INV-2026-04471",
        "Invoice Date: 2026-06-14",
        "",
        "Bill To: Northwind Analytics, 500 Congress Ave, Austin, TX",
        "",
        "Item                         Qty      Unit Price     Line Total",
        "Standing Desks                4         $410.00       $1,640.00",
        "Ergonomic Chairs               6         $265.00       $1,590.00",
        "Monitor Arms                  8          $45.00         $360.00",
        "",
        "Subtotal: $3,590.00",
        "Tax: $296.18",
        "Total: $3,886.18",
        "Currency: USD",
    ]
]
write_text_pdf(OUT / "Synthetic_Invoice_Clean.pdf", invoice_lines)

# ---------------------------------------------------------------------
# 6. Synthetic invoice with an inconsistent total (to exercise validation)
# ---------------------------------------------------------------------
invoice_bad_lines = [
    [
        "# INVOICE",
        "",
        "Vendor: Lockhart Print & Signage LLC",
        "Invoice Number: INV-2026-00931",
        "Invoice Date: 07/02/2026",
        "",
        "Bill To: Northwind Analytics",
        "",
        "Item                         Qty      Unit Price     Line Total",
        "Banner Printing                2         $180.00        $360.00",
        "Trade Show Booth Signage       1       $1,250.00      $1,250.00",
        "",
        "Subtotal: $1,610.00",
        "Tax: $132.83",
        "Total: $1,900.00",
        "Currency: USD",
    ]
]
write_text_pdf(OUT / "Synthetic_Invoice_Inconsistent_Total.pdf", invoice_bad_lines)

print(f"Wrote {len(list(OUT.glob('*.pdf')))} sample PDFs to {OUT}")
