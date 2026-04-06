"""
check_extractor_gemini.py
=========================
Extracts structured fields from scanned check images using the
Google Gemini Vision API — completely free via Google AI Studio.

Advantages over PaddleOCR:
  - No local model downloads or version conflicts
  - Strong document understanding including MICR font
  - Handles any check layout without coordinate tuning
  - Free tier is generous enough for testing hundreds of checks

Requirements:
  pip install google-generativeai Pillow

Setup:
  1. Go to https://aistudio.google.com/
  2. Sign in with your Google account (free)
  3. Click "Get API Key" → "Create API Key"
  4. Set the environment variable:
       PowerShell:  $env:GEMINI_API_KEY = "your-key-here"
       CMD:         set GEMINI_API_KEY=your-key-here
       Linux/macOS: export GEMINI_API_KEY="your-key-here"

Usage:
  python check_extractor_gemini.py --image path/to/check.jpg
  python check_extractor_gemini.py --image path/to/check.jpg --output result.json
  python check_extractor_gemini.py --image path/to/check.jpg --debug
"""

import os
import re
import json
import argparse
from pathlib import Path

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    # Walk up from current file location to find .env
    _env_path = Path(__file__).parent / ".env"
    if not _env_path.exists():
        _env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # dotenv not installed — rely on environment variables only

# ── Prompt ────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert check processing system.

Extract all fields from this bank check image and return ONLY a valid JSON
object with no preamble, markdown code fences, or explanation.

Return exactly these keys:

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
  numbers and their checksum to correct any single-digit errors in the MICR line.
- The MICR line at the bottom uses this layout:
    [check#] [transit symbol] [routing 9 digits] [transit symbol] [account] [on-us symbol]
- If a field is not visible or not present, use an empty string "".
- Do NOT include any text outside the JSON object.
- amount_numeric should always include the $ sign and two decimal places.
"""


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(image_path: str):
    """Load image for Gemini API. Returns a PIL Image."""
    from PIL import Image
    img = Image.open(image_path)

    # Auto-rotate based on EXIF
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

    # Resize if too large
    max_dim = 3000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        print(f"[vision] Resized to {new_size[0]}x{new_size[1]}")

    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    return img


# ── API call ──────────────────────────────────────────────────────────────────

def extract_with_gemini(image_path: str, debug: bool = False) -> dict:
    """
    Send the check image to Gemini Vision and return structured fields.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai not installed.\n"
            "Run: pip install google-genai"
        )

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get a free key at: https://aistudio.google.com/\n"
            "Then set it:\n"
            "  PowerShell: $env:GEMINI_API_KEY='your-key-here'\n"
            "  CMD:        set GEMINI_API_KEY=your-key-here\n"
            "  Linux/mac:  export GEMINI_API_KEY='your-key-here'"
        )

    client = genai.Client(api_key=api_key)

    # Model selection — try in order until one works on your free tier
    # These are the exact model names from your account (run list_models.py to refresh)
    MODELS_TO_TRY = [
        "models/gemini-2.5-flash-lite",    # lightest quota usage
        "models/gemini-2.0-flash-lite",    # fast, free-tier friendly
        "models/gemini-2.0-flash",         # good balance
        "models/gemini-2.5-flash",         # very capable
        "models/gemini-2.5-pro",           # most powerful — use as last resort
    ]

    print(f"[vision] Loading image: {image_path}")
    img = load_image(image_path)

    # Convert PIL image to bytes for the new API
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    raw_text = None
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            print(f"[vision] Trying model: {model_name} ...")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=EXTRACTION_PROMPT),
                ]
            )
            # Handle empty or blocked response
            if response is None or not response.text:
                print(f"[vision] {model_name} returned empty response, trying next model...")
                continue
            raw_text = response.text.strip()
            if not raw_text:
                print(f"[vision] {model_name} returned blank text, trying next model...")
                continue
            print(f"[vision] Success with {model_name}")
            break
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "quota" in err_str.lower() or "429" in err_str or "exhausted" in err_str.lower() or "resource" in err_str.lower():
                print(f"[vision] {model_name} quota exceeded, trying next model...")
                import time
                time.sleep(2)
                continue
            raise  # non-quota error — re-raise immediately

    if raw_text is None:
        raise RuntimeError(
            f"All models exhausted their quota.\n"
            f"Last error: {last_error}\n"
            f"Try again in a few minutes, or visit https://aistudio.google.com/ "
            f"to check your quota usage."
        )

    if debug:
        print(f"\n── Raw API response ────────────────────────────────")
        print(raw_text)
        print(f"────────────────────────────────────────────────────\n")

    # Strip markdown fences if model added them
    clean = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
    clean = re.sub(r'```\s*$', '', clean, flags=re.MULTILINE).strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON: {e}\nRaw response:\n{raw_text}"
        )

    # Ensure all expected keys are present
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


# ── Validation ────────────────────────────────────────────────────────────────

def validate_routing(routing: str) -> tuple:
    """Validate routing number using ABA checksum."""
    if not re.match(r'^\d{9}$', routing):
        return False, f"Not 9 digits (got {len(routing)})"
    d = [int(c) for c in routing]
    total = 3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])
    if total % 10 != 0:
        return False, f"ABA checksum failed"
    return True, "Valid"


def validate_result(result: dict) -> list:
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
        description="Extract check fields using Google Gemini Vision API (free)."
    )
    parser.add_argument("--image",  required=True,
                        help="Path to check image (JPEG/PNG/TIFF/BMP)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save JSON output")
    parser.add_argument("--debug",  action="store_true",
                        help="Print raw API response")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found: {args.image}")
        return

    print(f"\n📄 Processing: {args.image}")
    print("─" * 55)

    result = extract_with_gemini(args.image, debug=args.debug)
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
