"""
check_extractor_vision.py
=========================
Extracts structured fields from scanned check images using the
Anthropic Claude Vision API — the same vision model used to read
the check in the conversation.

This approach has major advantages over PaddleOCR:
  - No local model downloads or dependencies
  - Understands MICR font natively, including E-13B special characters
  - Knows banking domain context (routing number format, ABA checksum, etc.)
  - Handles any check layout, font, handwriting, or image orientation
  - Returns structured JSON directly — no spatial zone parsing needed

Requirements:
  pip install anthropic Pillow

Usage:
  python check_extractor_vision.py --image path/to/check.jpg
  python check_extractor_vision.py --image path/to/check.jpg --output result.json
  python check_extractor_vision.py --image path/to/check.jpg --debug

Environment variable:
  ANTHROPIC_API_KEY  — your Anthropic API key (required)
  Get one at: https://console.anthropic.com/
"""

import os
import re
import json
import base64
import argparse
from pathlib import Path

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if not _env_path.exists():
        _env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # dotenv not installed — rely on environment variables only

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert check processing system. When given an image
of a bank check, you extract specific fields and return them as valid JSON.

You have deep knowledge of:
- US check layouts and standard field positions
- MICR E-13B font used on the bottom of checks for routing/account numbers
- ABA routing number format (always exactly 9 digits, passes ABA checksum)
- How to distinguish routing numbers, account numbers, and check numbers
  from the MICR line, even when the special delimiter symbols are unclear

Always return ONLY a valid JSON object with no preamble, markdown, or explanation.
"""

EXTRACTION_PROMPT = """Extract all fields from this check image and return a JSON
object with exactly these keys:

{
  "owner_name":       "full name or company name of the account holder (top-left header)",
  "address_line1":    "street address from the header",
  "address_line2":    "city, state, zip from the header",
  "phone":            "phone number from the header if present, else empty string",
  "date":             "date written on the check (MM/DD/YYYY or as written)",
  "payee":            "name on the Pay To The Order Of line",
  "amount_numeric":   "dollar amount from the box (e.g. $294.00)",
  "amount_written":   "written-out dollar amount (e.g. TWO HUNDRED NINETY FOUR AND 00/100)",
  "memo":             "memo/for field content if present, else empty string",
  "routing_number":   "9-digit ABA routing number from the MICR line at the bottom",
  "account_number":   "bank account number from the MICR line",
  "check_number":     "check number (appears top-right and/or in MICR line)",
  "bank_name":        "name of the bank printed on the check",
  "micr_raw":         "the complete raw text of the MICR line at the bottom as you read it"
}

Important rules:
- routing_number must be exactly 9 digits. Use your knowledge of ABA routing
  numbers and their checksum to correct any OCR errors in the MICR line.
- If a field is not visible or not present on the check, use an empty string "".
- Do not include any text outside the JSON object.
- amount_numeric should include the $ sign and decimal point.
- For the MICR line, the layout is: [check#] [routing#] [account#]
  The routing number is surrounded by transit symbols (⑆) and the account
  number is followed by an on-us symbol (⑈).
"""


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image_base64(image_path: str) -> tuple[str, str]:
    """
    Load an image and return (base64_data, media_type).
    Converts non-JPEG/PNG formats to JPEG automatically.
    Resizes very large images to stay within API limits.
    """
    from PIL import Image
    import io

    img = Image.open(image_path)

    # Auto-rotate based on EXIF orientation
    try:
        from PIL.ExifTags import TAGS
        exif = img._getexif()
        if exif:
            for tag, value in exif.items():
                if TAGS.get(tag) == 'Orientation':
                    rotations = {3: 180, 6: 270, 8: 90}
                    if value in rotations:
                        img = img.rotate(rotations[value], expand=True)
                    break
    except Exception:
        pass

    # Resize if too large (Claude Vision handles up to ~5MB / 8000px)
    max_dim = 3000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"[vision] Resized image to {new_size[0]}x{new_size[1]}")

    # Determine output format
    suffix = Path(image_path).suffix.lower()
    if suffix in ('.jpg', '.jpeg'):
        fmt, media_type = 'JPEG', 'image/jpeg'
    elif suffix == '.png':
        fmt, media_type = 'PNG', 'image/png'
    else:
        fmt, media_type = 'JPEG', 'image/jpeg'

    # Convert to RGB if needed (PNG with alpha, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    b64 = base64.standard_b64encode(buf.getvalue()).decode('utf-8')
    return b64, media_type


# ── API call ──────────────────────────────────────────────────────────────────

def extract_with_claude(image_path: str, debug: bool = False) -> dict:
    """
    Send the check image to Claude Vision and return structured extracted fields.
    """
    import anthropic

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Get your key at: https://console.anthropic.com/\n"
            "Then set it with:\n"
            "  Windows PowerShell: $env:ANTHROPIC_API_KEY='your-key-here'\n"
            "  Windows CMD:        set ANTHROPIC_API_KEY=your-key-here\n"
            "  Linux/macOS:        export ANTHROPIC_API_KEY='your-key-here'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Load and encode image
    print(f"[vision] Loading image: {image_path}")
    b64_data, media_type = load_image_base64(image_path)
    print(f"[vision] Sending to Claude Vision API...")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    }
                ],
            }
        ],
    )

    raw_text = response.content[0].text.strip()

    if debug:
        print(f"\n── Raw API response ────────────────────────────────")
        print(raw_text)
        print(f"────────────────────────────────────────────────────\n")
        print(f"[vision] Input tokens:  {response.usage.input_tokens}")
        print(f"[vision] Output tokens: {response.usage.output_tokens}\n")

    # Parse JSON response
    # Strip markdown code fences if model added them despite instructions
    clean = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
    clean = re.sub(r'```\s*$', '', clean, flags=re.MULTILINE).strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude returned invalid JSON: {e}\nRaw response:\n{raw_text}"
        )

    # Normalise keys — ensure all expected fields are present
    expected_keys = [
        "owner_name", "address_line1", "address_line2", "phone",
        "date", "payee", "amount_numeric", "amount_written",
        "memo", "routing_number", "account_number", "check_number",
        "bank_name", "micr_raw"
    ]
    for key in expected_keys:
        if key not in result:
            result[key] = ""

    return result


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_routing(routing: str) -> tuple[bool, str]:
    """Validate routing number using ABA checksum."""
    if not re.match(r'^\d{9}$', routing):
        return False, f"Not 9 digits (got {len(routing)})"
    d = [int(c) for c in routing]
    total = 3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])
    if total % 10 != 0:
        return False, f"ABA checksum failed (sum={total}, mod10={total%10})"
    return True, "Valid"


def validate_result(result: dict) -> list[str]:
    """Run validation checks and return list of warnings."""
    warnings = []

    routing = result.get("routing_number", "")
    if routing:
        valid, msg = validate_routing(routing)
        if not valid:
            warnings.append(f"Routing number '{routing}': {msg}")
    else:
        warnings.append("Routing number not found")

    if not result.get("amount_numeric"):
        warnings.append("Numeric amount not found")

    if not result.get("payee"):
        warnings.append("Payee not found")

    return warnings


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract check fields using Claude Vision API."
    )
    parser.add_argument("--image",  required=True,
                        help="Path to check image (JPEG/PNG/TIFF/BMP)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save JSON output")
    parser.add_argument("--debug",  action="store_true",
                        help="Print raw API response and token usage")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}")
        return

    print(f"\n📄 Processing: {args.image}")
    print("─" * 55)

    result = extract_with_claude(args.image, debug=args.debug)

    # Validate
    warnings = validate_result(result)

    print("\n✅ Extracted Fields:")
    print(f"  Owner Name    : {result['owner_name']}")
    print(f"  Address       : {result['address_line1']}")
    print(f"                  {result['address_line2']}")
    print(f"  Phone         : {result['phone']}")
    print(f"  Bank          : {result['bank_name']}")
    print(f"  Date          : {result['date']}")
    print(f"  Payee         : {result['payee']}")
    print(f"  Amount ($)    : {result['amount_numeric']}")
    print(f"  Amount (text) : {result['amount_written']}")
    print(f"  Memo          : {result['memo']}")
    print(f"  Routing #     : {result['routing_number']}")
    print(f"  Account #     : {result['account_number']}")
    print(f"  Check #       : {result['check_number']}")
    print(f"  MICR raw      : {result['micr_raw']}")

    # Validation warnings
    if warnings:
        print("\n⚠️  Validation warnings:")
        for w in warnings:
            print(f"   • {w}")
    else:
        print("\n✓  All validation checks passed")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
