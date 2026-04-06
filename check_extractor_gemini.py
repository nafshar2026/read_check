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

# ── Prompts ───────────────────────────────────────────────────────────────────

# Shorter fallback prompt — used when main prompt triggers safety filters
EXTRACTION_PROMPT_SHORT = (
    "Please read this check image and return a JSON object with these fields: "
    "owner_name (name printed top-left), "
    "address_line1 (street address top-left), "
    "address_line2 (city state zip top-left), "
    "phone (phone number top-left), "
    "bank_name (bank name on check), "
    "date (date written on check), "
    "payee (name on Pay To The Order Of line), "
    "amount_numeric (dollar amount like $294.00 in the box right of Pay To line), "
    "amount_written (amount spelled out in words), "
    "memo (text on memo line), "
    "routing_number (9-digit number from bottom MICR line between transit symbols), "
    "account_number (account number from bottom MICR line), "
    "check_number (number printed top-right), "
    "micr_raw (copy the entire bottom line exactly as printed). "
    "Use empty string for any field not found. Return JSON only."
)

EXTRACTION_PROMPT = """Please read this bank check image carefully and extract the information into a JSON object.

Return ONLY the JSON object below with no other text, markdown, or explanation.

{
  "owner_name":       "Name or company printed in the top-left header of the check",
  "address_line1":    "Street address from the top-left header",
  "address_line2":    "City, state, and zip code from the top-left header",
  "phone":            "Phone number from the top-left header, or empty string",
  "bank_name":        "Name of the bank printed on the check",
  "date":             "Date written on the check, e.g. 8/11/93",
  "payee":            "Name written on the Pay To The Order Of line",
  "amount_numeric":   "Dollar amount from the small box on the right side of the Pay To line, e.g. $294.00",
  "amount_written":   "The amount written out in words, e.g. TWO HUNDRED NINETY FOUR AND 00/100",
  "memo":             "Text written on the memo or for line, or empty string",
  "routing_number":   "The 9-digit routing number on the bottom line of the check. It is between the two transit symbols that look like this: ⑆122000661⑆. Copy all 9 digits carefully.",
  "account_number":   "The account number on the bottom line, after the routing number",
  "check_number":     "The check number printed in the top-right corner",
  "micr_raw":         "Copy the entire bottom line of the check exactly as it appears"
}

Use empty string for any field you cannot find. Return the JSON object only."""


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

    # Model preference order — more accurate models first.
    # gemini-2.5-flash gives better MICR accuracy than flash-lite.
    # flash-lite is faster but tends to merge or drop MICR digits.
    MODELS_TO_TRY = [
        "models/gemini-2.5-flash",       # best accuracy for document extraction
        "models/gemini-2.5-flash-lite",  # faster but less accurate on MICR
        "models/gemini-2.5-pro",         # most powerful fallback
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
            # Use simplified prompt first — it's less likely to trigger safety filters
            # and is faster. Fall back to full prompt only if it fails.
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=EXTRACTION_PROMPT_SHORT),
                ]
            )
            if response is None or not response.text:
                print(f"[vision] {model_name} simplified prompt empty, trying full prompt...")
                import time; time.sleep(1)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=EXTRACTION_PROMPT),
                    ]
                )
                if response is None or not response.text:
                    print(f"[vision] {model_name} still empty, trying next model...")
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
            if any(x in err_str.lower() for x in ["quota", "429", "exhausted", "resource", "not found", "404", "no longer available"]):
                print(f"[vision] {model_name} unavailable ({type(e).__name__}), trying next model...")
                import time
                time.sleep(1)
                continue
            raise  # unexpected error — re-raise immediately

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

    # Apply post-processing fixes
    result = postprocess_result(result)

    return result


# ── Validation & post-processing ─────────────────────────────────────────────

def _aba_valid(routing: str) -> bool:
    if not re.match(r'^\d{9}$', routing):
        return False
    d = [int(c) for c in routing]
    return (3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])) % 10 == 0


def _fix_routing(routing: str) -> str:
    """
    Attempt to correct a routing number that fails the ABA checksum.
    Handles:
      - Single digit substitution errors  (e.g. 021000021 -> 021000089)
      - Truncated 8-digit numbers         (e.g. 02100002  -> 021000021)
      - Leading/trailing digit dropped    (e.g. 21000021  -> 021000021)
    Returns the corrected routing or the original if no fix found.
    """
    if _aba_valid(routing):
        return routing

    digits = re.sub(r'[^\d]', '', routing)

    CONFUSION = [('1','7'),('7','1'),('1','2'),('2','1'),('0','6'),('6','0'),
                 ('3','8'),('8','3'),('5','6'),('6','5'),('4','9'),('9','4')]

    # Case 1: exactly 9 digits — collect ALL single-digit corrections, pick best
    if len(digits) == 9:
        candidates = []
        # Try confusion pairs first
        for pos in range(9):
            for orig, replacement in CONFUSION:
                if digits[pos] == orig:
                    candidate = digits[:pos] + replacement + digits[pos+1:]
                    if _aba_valid(candidate) and candidate[0] in '0123':
                        prefix_int = int(candidate[:2])
                        # Score: prefer valid Federal Reserve district prefix
                        score = (5 if 1 <= prefix_int <= 12 else
                                 2 if 21 <= prefix_int <= 32 else 0)
                        candidates.append((score, candidate))
        # Try all single digit replacements
        for pos in range(9):
            for d in '0123456789':
                candidate = digits[:pos] + d + digits[pos+1:]
                if _aba_valid(candidate) and candidate[0] in '0123':
                    prefix_int = int(candidate[:2])
                    score = (5 if 1 <= prefix_int <= 12 else
                             2 if 21 <= prefix_int <= 32 else 0)
                    candidates.append((score, candidate))
        if candidates:
            # Pick highest scoring candidate
            best = max(candidates, key=lambda x: x[0])[1]
            print(f"[validation] Corrected routing {routing} → {best}")
            return best

    # Case 2: 8 digits — a digit was dropped
    # Try appending first (last digit most commonly dropped by OCR),
    # then prepending, then inserting at each internal position
    elif len(digits) == 8:
        # Append (most likely — OCR cuts off last digit)
        for d in '0123456789':
            candidate = digits + d
            if _aba_valid(candidate) and candidate[0] in '0123':
                print(f"[validation] Corrected truncated routing {routing} → {candidate}")
                return candidate
        # Internal insertion
        for pos in range(1, 8):
            for d in '0123456789':
                candidate = digits[:pos] + d + digits[pos:]
                if _aba_valid(candidate) and candidate[0] in '0123':
                    print(f"[validation] Corrected truncated routing {routing} → {candidate}")
                    return candidate
        # Prepend (least likely)
        for d in '0123456789':
            candidate = d + digits
            if _aba_valid(candidate) and candidate[0] in '0123':
                print(f"[validation] Corrected truncated routing {routing} → {candidate}")
                return candidate

    # Case 3: 10 digits — an extra digit was added, collect all valid removals.
    # Score by: Federal Reserve prefix (01-12 best) + position bonus.
    # OCR most commonly adds an extra digit at the END, so removing the last
    # digit gets a tiebreak bonus.
    elif len(digits) == 10:
        candidates = []
        for pos in range(10):
            candidate = digits[:pos] + digits[pos+1:]
            if _aba_valid(candidate) and candidate[0] in '0123':
                pi = int(candidate[:2])
                prefix_score = 5 if 1 <= pi <= 12 else (2 if 21 <= pi <= 32 else 1)
                # Tiebreak: prefer removing last digit (pos 9) then first (pos 0)
                position_bonus = 2 if pos == 9 else (1 if pos == 0 else 0)
                candidates.append((prefix_score + position_bonus, candidate))
        if candidates:
            best = max(candidates, key=lambda x: x[0])[1]
            print(f"[validation] Corrected over-read routing {routing} → {best}")
            return best

    return routing


def _fix_amount(amount: str) -> str:
    """Normalise amount string to .xx format."""
    if not amount:
        return amount
    # Extract digits and decimals
    m = re.search(r'(\d{1,6})[.,](\d{2})', amount)
    if m:
        return f"${m.group(1)}.{m.group(2)}"
    # Zero-padded MICR amount e.g. 0000150000 = $1500.00
    m = re.match(r'0*(\d+)(\d{2})$', re.sub(r'[^\d]', '', amount))
    if m and len(re.sub(r'[^\d]', '', amount)) >= 6:
        return f"${m.group(1)}.{m.group(2)}"
    return amount


def _extract_routing_from_micr(micr_raw: str) -> str:
    """
    Extract the ABA routing number from the MICR text string returned by the model.

    Core principle (your suggestion): treat each SOLID run of consecutive digits
    as a single field. Any digit group interrupted by non-digit characters is
    noise and is ignored. This mirrors exactly how the MICR line is structured —
    each field (check#, routing, account) is a clean unbroken digit sequence.

    US MICR layout (ANSI X9.27):
        [check# solid run] [separator(s)] [routing solid run] [separator(s)] [account solid run]

    Strategy:
      1. Extract all solid digit runs (re.findall r'\d+')
      2. From runs of 8-10 digits, find one that passes ABA checksum
         — prefer runs that start with a valid Federal Reserve prefix (01-12)
      3. If no exact match, try single-digit correction on 8-10 digit runs
      4. Fallback: position-based scan of the full cleaned digit string
         (handles the case where the model returns no separators at all)
    """
    if not micr_raw:
        return ""

    def aba_score(routing):
        """Score a valid 9-digit routing by Federal Reserve prefix quality."""
        if not (_aba_valid(routing) and routing[0] in '0123'):
            return -1
        pi = int(routing[:2])
        return 5 if 1 <= pi <= 12 else (2 if 21 <= pi <= 32 else 1)

    # ── Strategy 1: solid consecutive digit runs ──────────────────────────
    # Each field in the MICR line is an unbroken digit sequence.
    # We only consider runs of 8-10 digits as routing candidates.
    # Runs shorter than 8 are check# fragments or account sub-fields.
    # Runs longer than 10 are merged fields — handled by strategy 2.
    solid_runs = re.findall(r'\d+', micr_raw)
    routing_candidates = [r for r in solid_runs if 8 <= len(r) <= 10]

    # Pass A: exact ABA match
    scored = []
    for run in routing_candidates:
        s = aba_score(run)
        if s > 0:
            scored.append((s, run))
    if scored:
        return max(scored, key=lambda x: x[0])[1]

    # Pass B+C: run both corrections together and pick the highest scoring result.
    # Pass B: single-digit substitution correction
    # Pass C: prepend a digit (handles stolen leading digit case)
    #         e.g. OCR reads "220006611" but real routing is "122000661"
    #         because the leading "1" was absorbed into the check# chunk
    all_corrections = []

    for run in routing_candidates:
        # Pass B: single-digit fix
        fixed = _fix_routing(run)
        s = aba_score(fixed)
        if s > 0:
            all_corrections.append((s, fixed))

        # Pass C: prepend digit + trim
        if len(run) == 9:
            for d in '0123456789':
                candidate = (d + run)[:9]
                s = aba_score(candidate)
                if s > 0:
                    all_corrections.append((s, candidate))

    if all_corrections:
        return max(all_corrections, key=lambda x: x[0])[1]

    # ── Strategy 2: position-based scan with correction ───────────────────
    # Used when the model returns one long digit string with no separators,
    # or when fields are merged (e.g. "12200066123624").
    # MICR layout: routing starts ~6 digits from the left of the cleaned string.
    all_digits = re.sub(r'[^\d]', '', micr_raw)
    if len(all_digits) >= 15:
        best = None
        best_score = -1
        for start in range(len(all_digits) - 8):
            window = all_digits[start:start+9]
            # Try exact, then single-digit fix
            for candidate in (window, _fix_routing(window)):
                s = aba_score(candidate)
                if s < 0:
                    continue
                proximity     = max(0, 10 - abs(start - 6))
                early_penalty = 3 if start < 4 else 0
                score = s + proximity - early_penalty
                if score > best_score:
                    best_score = score
                    best = candidate
        if best:
            return best

    return ""


def postprocess_result(result: dict) -> dict:
    """Apply fixes and normalisation to the raw model output."""
    # If routing is missing or invalid, try extracting from MICR raw
    routing = result.get("routing_number", "")
    if not routing or not _aba_valid(re.sub(r'[^\d]', '', routing)):
        micr_routing = _extract_routing_from_micr(result.get("micr_raw", ""))
        if micr_routing:
            result["routing_number"] = micr_routing
            routing = micr_routing

    # Fix routing number via ABA checksum correction
    if result.get("routing_number"):
        result["routing_number"] = _fix_routing(result["routing_number"])

    # Fix amount format
    if result.get("amount_numeric"):
        result["amount_numeric"] = _fix_amount(result["amount_numeric"])

    # Normalise all fields — convert None/null to empty string, strip quotes
    for key in list(result.keys()):
        val = result[key]
        if val is None or str(val).lower() in ('none', 'null', 'n/a', 'na'):
            result[key] = ""
        elif isinstance(val, str):
            result[key] = val.strip().strip('"').strip("'").strip()

    return result


def validate_routing(routing: str) -> tuple:
    """Validate routing number using ABA checksum."""
    if not re.match(r'^\d{9}$', routing):
        return False, f"Not 9 digits (got '{routing}'  len={len(routing)})"
    if not _aba_valid(routing):
        return False, "ABA checksum failed"
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
