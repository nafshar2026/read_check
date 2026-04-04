# read_check
Check Reader

A Python application that extracts structured information from an image of a handwritten check using OCR (Optical Character Recognition).

## Extracted Fields

| Field | Description |
|-------|-------------|
| **Name & Address** | Account-holder name and mailing address (top-left of check) |
| **Amount** | Dollar amount written in the numeric box |
| **Routing Number** | 9-digit ABA routing number from the MICR line |
| **Account Number** | Bank account number from the MICR line |

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your `PATH`
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: download the installer from the Tesseract GitHub releases page

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Human-readable output
python main.py check.jpg

# JSON output
python main.py check.png --json

# Specify Tesseract path explicitly
python main.py check.tiff --tesseract-cmd /usr/local/bin/tesseract
```

**Example output:**

```
==================================================
CHECK INFORMATION
==================================================
Name / Address   :
John Doe
123 Main Street
Springfield, IL 62701
--------------------------------------------------
Amount           : $1,234.56
Routing Number   : 021000021
Account Number   : 1234567890
==================================================
```

### Python API

```python
from check_reader import CheckReader

reader = CheckReader("check.jpg")
data = reader.read()

print(data.name_address)    # "John Doe\n123 Main Street\n..."
print(data.amount)          # "$1,234.56"
print(data.routing_number)  # "021000021"
print(data.account_number)  # "1234567890"

# As a dictionary
print(data.to_dict())
```

## How It Works

1. **Image Loading** – The check image is loaded via Pillow (supports JPEG, PNG, TIFF, BMP, and more).
2. **Preprocessing** – Each region is upscaled, converted to greyscale, and binarised with adaptive thresholding to improve OCR accuracy.
3. **Region OCR** – Tesseract is applied to specific regions of the check:
   - Top-left → name and address
   - Upper-right numeric box → dollar amount
   - Bottom strip → MICR line (routing and account numbers)
4. **MICR Parsing** – The routing number is validated using the standard ABA mod-10 checksum. The account number is extracted from the digit group that follows the routing number.
5. **Fallback** – If a field cannot be found in its expected region, the parser falls back to a full-page OCR scan.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
read_check/
├── check_reader.py        # Core CheckReader class and helpers
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── tests/
│   └── test_check_reader.py
└── README.md
```
