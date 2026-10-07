"""Deterministic invoice validation using Decimal arithmetic."""
from __future__ import annotations
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS = ["vendor_name", "invoice_number", "total"]
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%d-%m-%Y"]
VALID_CURRENCIES = {"USD", "EUR", "GBP", "PKR", "CAD", "AUD"}
TOLERANCE = Decimal("0.02")

def _parse_date(value: str):
    for fmt in DATE_FORMATS:
        try: return datetime.strptime(value, fmt)
        except ValueError: pass
    return None

def _d(value):
    try: return Decimal(str(value)) if value is not None else None
    except Exception: return None

def validate_invoice_fields(fields: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors, warnings = [], []
    for field in REQUIRED_FIELDS:
        if not fields.get(field): errors.append(f"Missing required field: {field}")

    amounts = {}
    for field in ("subtotal", "discount", "tax", "shipping", "fees", "total"):
        value = _d(fields.get(field))
        amounts[field] = value
        if value is not None and value < 0:
            errors.append(f"Field '{field}' cannot be negative: {value}")

    date_value = fields.get("invoice_date")
    if date_value and _parse_date(str(date_value)) is None:
        warnings.append(f"invoice_date '{date_value}' is not in a recognized format")
    elif not date_value:
        warnings.append("invoice_date is missing")

    currency = fields.get("currency")
    if currency and str(currency).upper() not in VALID_CURRENCIES:
        warnings.append(f"Unrecognized currency code: {currency}")
    elif not currency:
        warnings.append("currency is missing")

    subtotal, discount, tax, shipping, fees, total = (amounts[x] for x in ("subtotal", "discount", "tax", "shipping", "fees", "total"))
    if subtotal is not None and total is not None:
        expected = subtotal - (discount or Decimal("0")) + (tax or Decimal("0")) + (shipping or Decimal("0")) + (fees or Decimal("0"))
        if abs(expected - total) > max(TOLERANCE, abs(total) * TOLERANCE):
            errors.append(f"subtotal ({subtotal}) + adjustments = {expected:.2f}, which does not match total ({total})")
    else:
        warnings.append("Could not verify invoice arithmetic because subtotal or total is missing.")

    inv_num = fields.get("invoice_number")
    if inv_num and not re.match(r"^[A-Za-z0-9\-/ #]{2,40}$", str(inv_num)):
        warnings.append(f"invoice_number has an unusual format: {inv_num!r}")

    valid = not errors
    return valid, errors, warnings
