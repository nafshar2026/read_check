# Check Extractor — PaddleOCR

Extract structured fields from scanned US check images using PaddleOCR.

---

## Extracted Fields

| Field | Description |
|---|---|
| `owner.name` | Account holder name (printed header) |
| `owner.address_line1` | Street address |
| `owner.address_line2` | City / State / ZIP |
| `owner.phone` | Phone number |
| `date` | Check date |
| `payee` | Pay To The Order Of |
| `amount_numeric` | Dollar amount from the $ box (e.g. `$294.00`) |
| `amount_written` | Written-out amount line |
| `memo` | Memo / For field |
| `routing_number` | 9-digit ABA routing number (from MICR line) |
| `account_number` | Bank account number (from MICR line) |
| `check_number` | Check number |
| `micr_raw` | Raw MICR bottom line (for debugging) |

---

## Project Structure

```
check_extractor/
├── check_extractor.py   # Core extraction logic
├── batch_process.py     # Batch folder processing
├── download_models.py   # Pre-download model weights for offline/Azure use
├── requirements.txt     # Python dependencies
├── conda.yaml           # Azure ML environment definition
└── README.md
```

---

## Local Installation (Windows / macOS)

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** PaddleOCR downloads ~300 MB of model weights on first run.
> GPU users: replace `paddlepaddle` with `paddlepaddle-gpu` in requirements.txt.

---

## Usage

### Single check image

```bash
python check_extractor.py --image path/to/check.jpg
```

Save results to JSON:

```bash
python check_extractor.py --image check.jpg --output result.json
```

Debug mode (prints all raw OCR detections for troubleshooting):

```bash
python check_extractor.py --image check.jpg --debug
```

### Using the launcher scripts (Windows — recommended for PaddleOCR v3)

If you are on Windows with PaddleOCR v3 and see the `ConvertPirAttribute2RuntimeAttribute`
error, use the launcher scripts instead of calling Python directly. They set the
required environment variables before the Python process starts:

```powershell
# PowerShell
.un.ps1 --image .\images\check01.jpeg
.un.ps1 --image .\images\check01.jpeg --output result.json
.un.ps1 --image .\images\check01.jpeg --debug
```

```cmd
:: Command Prompt
run.bat --image .\images\check01.jpeg
run.bat --image .\images\check01.jpeg --output result.json
```

### Batch processing (folder of checks)

```bash
# Export to CSV
python batch_process.py --folder ./images --output results.csv

# Export to JSON
python batch_process.py --folder ./images --output results.json
```

---

## Example Output

```
📄 Processing: check01.jpeg
───────────────────────────────────────────────────────

✅ Extracted Fields:
  Owner Name    : CURT C. FLOOD ENTERPRISES
  Address       : 4139 CLOVERDALE AVE.
                  LOS ANGELES, CA 90008
  Phone         : PH. 213-290-3264
  Date          : 8/11/93
  Payee         : DMV
  Amount ($)    : $294.00
  Amount (text) : TWO HUNDRED NINETY FOUR AND NO/100
  Memo          : Merz
  Routing #     : 122000661
  Account #     : 236243058
  Check #       : 3092
```

---

## How It Works

```
Input Image
     │
     ▼
┌──────────────┐
│ Pre-process  │  Deskew, denoise, adaptive threshold
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  PaddleOCR   │  PP-OCRv5 detection + recognition
│  (PP-OCRv5)  │  Returns bounding boxes + text + confidence
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│        Spatial Zone Extraction       │
│                                      │
│  Top-left   → Owner name & address   │
│  Top-right  → Date, Check #          │
│  Mid-left   → Payee name             │
│  Mid-right  → Numeric amount         │
│  Middle     → Written amount         │
│  Lower-left → Memo                   │
│  Bottom     → MICR line              │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│  MICR Parser │  Routing # / Account # / Check #
└──────┬───────┘
       │
       ▼
  Structured JSON output
```

---

## Deploying to Azure AI Foundry

Azure AI Foundry compute runs on **Ubuntu Linux**. Follow the steps below to
migrate from your Windows development environment.

### Step 1 — Pre-download model weights (do this on Windows before deploying)

PaddleOCR downloads ~300 MB of model weights on first run. Azure AI Foundry
compute may have restricted outbound internet access, so download the models
locally first and bundle them with your code.

```bash
# Run once on your Windows machine while you still have internet access
python download_models.py --dir ./models

# Verify what was downloaded
python download_models.py --list
```

The `./models/` folder will be created alongside your code. Commit it to your
repository or upload it to your Azure ML datastore before deploying.

---

### Step 2 — Update requirements.txt for Linux

On Linux servers there is no display driver, so `opencv-python` will fail to
import. Replace it with the headless build. Also switch to the GPU-enabled
PaddlePaddle if your compute SKU has a GPU.

```text
# requirements.txt for Azure / Linux

paddlepaddle-gpu    # GPU compute (NC/ND series VMs) — recommended for speed
# paddlepaddle      # CPU-only compute (DS/D series VMs)

paddleocr>=2.7.3
opencv-python-headless   # <-- headless build required on Linux servers
Pillow>=10.0.0
numpy>=1.24.0
```

To check your Azure compute SKU:
- NC / ND / NV series  → GPU available, use `paddlepaddle-gpu`
- DS / D / F series    → CPU only, use `paddlepaddle`

---

### Step 3 — Use the conda.yaml for Azure ML environments

Azure ML environments are best defined with a `conda.yaml` file. Create this
file in your project root:

```yaml
# conda.yaml
name: check-extractor
channels:
  - defaults
dependencies:
  - python=3.11
  - pip:
    - paddlepaddle-gpu       # or paddlepaddle for CPU
    - paddleocr>=2.7.3
    - opencv-python-headless
    - Pillow>=10.0.0
    - numpy>=1.24.0
```

Register it in Azure ML:

```bash
az ml environment create \
  --name check-extractor-env \
  --conda-file conda.yaml \
  --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
```

---

### Step 4 — Point PaddleOCR at your local model cache

After uploading your `./models/` folder to Azure, tell PaddleOCR to use it
instead of trying to download from the internet. Set the environment variable
before running:

```bash
# Linux / Azure ML job
export PADDLEX_HOME="./models"
python check_extractor.py --image ./images/check01.jpeg
```

Or set it permanently in your Azure ML job YAML:

```yaml
# job.yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: python check_extractor.py --image ${{inputs.image_folder}}
environment_variables:
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: "True"
  FLAGS_use_mkldnn: "0"        # keep off unless you confirm oneDNN works on your SKU
  PADDLEX_HOME: "./models"
inputs:
  image_folder:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/check_images/
```

---

### Step 5 — oneDNN / MKL flag on Azure

The `FLAGS_use_mkldnn=0` flag in the code was added to fix a Windows-specific
crash. On Azure Linux with a GPU you may actually want oneDNN enabled for
better performance. Test with it on and off:

```bash
# Test with oneDNN disabled (safe default, matches Windows behaviour)
FLAGS_use_mkldnn=0 python check_extractor.py --image ./images/check01.jpeg

# Test with oneDNN enabled (potentially faster on Linux + GPU)
FLAGS_use_mkldnn=1 python check_extractor.py --image ./images/check01.jpeg
```

If you get no errors with oneDNN enabled, remove the `FLAGS_use_mkldnn` and
`PADDLE_DISABLE_ONEDNN` lines from the top of `check_extractor.py`.

---

### Step 6 — File paths

No code changes needed. The codebase uses `pathlib.Path` throughout, which
handles Linux forward-slash paths automatically. Just make sure to use Linux
paths when calling the script:

```bash
# Linux paths — forward slashes, no drive letters
python check_extractor.py --image ./images/check01.jpeg
python batch_process.py --folder ./images --output ./output/results.csv
```

---

### Step 7 — Write permissions for processed images

The preprocessor saves a `_processed` copy of each image in the same folder
as the source. Make sure your Azure compute has write access to the image
folder, or use `--no-preprocess` to skip this step entirely if you want
read-only access to your image store:

```bash
python check_extractor.py --image ./images/check01.jpeg --no-preprocess
```

---

### Azure Deployment Checklist

| # | Task | Done |
|---|---|---|
| 1 | Run `python download_models.py` on Windows to pre-fetch weights | ☐ |
| 2 | Commit `./models/` folder or upload to Azure ML datastore | ☐ |
| 3 | Replace `opencv-python` with `opencv-python-headless` in requirements.txt | ☐ |
| 4 | Switch to `paddlepaddle-gpu` if compute SKU has a GPU | ☐ |
| 5 | Create `conda.yaml` and register Azure ML environment | ☐ |
| 6 | Set `PADDLEX_HOME` env var to point at local model cache | ☐ |
| 7 | Test oneDNN flag — disable if errors, enable for GPU performance | ☐ |
| 8 | Confirm write permissions on image folder (or use `--no-preprocess`) | ☐ |

---

## Tips for Best Accuracy

- **Resolution**: Scan at 300 DPI or higher
- **Lighting**: Even lighting, no shadows or glare
- **Orientation**: Keep checks upright (the preprocessor deskews minor tilts)
- **MICR line**: If routing/account extraction is poor, try `--no-preprocess` —
  sometimes the raw image gives a cleaner MICR read than the processed one

---

## Known Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `Unknown argument: show_log` | PaddleOCR v3 removed that param | Already fixed — code auto-detects v2 vs v3 |
| `ConvertPirAttribute2RuntimeAttribute` | PaddlePaddle C++ runtime loads oneDNN before Python env vars take effect | **Option A (recommended):** downgrade: `pip install "numpy<2.0" paddlepaddle paddleocr==2.7.3`<br>**Option B:** use `run.ps1` / `run.bat` launchers which set env vars before Python starts<br>**Option C:** set permanently in PowerShell: `$env:FLAGS_use_mkldnn="0"` |
| `can't open/read file: _processed.\images\...` | Windows path bug | Fixed — uses `pathlib.Path` for all paths |
| `numpy.core.multiarray failed to import` | NumPy 2.x vs PaddleOCR 2.7.x ABI mismatch | Run `pip install "numpy<2.0"` |
| `No module named cv2` on Linux server | `opencv-python` needs a display | Replace with `opencv-python-headless` |

---

## Claude Vision API Backend (Recommended)

The `check_extractor_vision.py` script uses the Claude Vision API instead of
PaddleOCR. This is the recommended approach because:

- **No local model downloads** — no paddlepaddle, no opencv, no numpy conflicts
- **Reads MICR natively** — understands E-13B font, delimiter symbols, and ABA
  routing number format including checksum validation
- **Any check layout** — no coordinate zones, no geometry assumptions
- **Handles handwriting** — payee, memo, amount written fields are read reliably
- **Self-correcting** — if a digit looks ambiguous, Claude uses banking domain
  knowledge to pick the right value

### Setup

```bash
# 1. Install the Anthropic SDK (only dependency besides Pillow)
pip install anthropic Pillow

# 2. Set your API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # PowerShell
set ANTHROPIC_API_KEY=sk-ant-...          # CMD
export ANTHROPIC_API_KEY="sk-ant-..."     # Linux/macOS

# Get your key at: https://console.anthropic.com/
```

### Usage

```bash
# Single check
python check_extractor_vision.py --image .\images\check01.jpeg

# Save to JSON
python check_extractor_vision.py --image .\images\check01.jpeg --output result.json

# Debug — shows raw API response and token usage
python check_extractor_vision.py --image .\images\check01.jpeg --debug

# Batch — whole folder
python batch_process_vision.py --folder .\images --output results.csv
```

### Cost

Each check image costs approximately **$0.01–$0.03** using claude-opus-4-5,
depending on image size. For high-volume processing, claude-sonnet-4-5 is
significantly cheaper with similar accuracy.

To switch models, change the `model=` line in `check_extractor_vision.py`:
```python
model="claude-sonnet-4-5",   # cheaper, still very accurate
model="claude-opus-4-5",     # most accurate, higher cost
```

### Comparison: PaddleOCR vs Claude Vision

| | PaddleOCR | Claude Vision |
|---|---|---|
| Setup | Complex (model downloads, version conflicts) | `pip install anthropic` |
| MICR accuracy | Poor (no MICR training) | Excellent (domain knowledge) |
| Handwriting | Moderate | Good |
| Any check layout | Needs tuning | Yes |
| Works offline | Yes | No (API call) |
| Cost per check | Free | ~$0.01–0.03 |
| Internet required | For model download only | Always |

---

## Google Gemini Vision Backend (Free — Recommended for Testing)

The `check_extractor_gemini.py` script uses Google Gemini Vision API which has
a **completely free tier** — no credit card required.

**Free tier limits:**
- 15 requests per minute
- 1,500 requests per day
- More than enough for testing all your checks

### Setup

```bash
# 1. Get a free API key
#    Go to: https://aistudio.google.com/
#    Sign in with Google → click "Get API Key" → "Create API Key"

# 2. Install the SDK
pip install google-generativeai Pillow

# 3. Set your key
$env:GEMINI_API_KEY = "your-key-here"    # PowerShell
set GEMINI_API_KEY=your-key-here         # CMD
export GEMINI_API_KEY="your-key-here"    # Linux/macOS
```

### Usage

```bash
# Single check
python check_extractor_gemini.py --image .\images\check01.jpeg

# Save to JSON
python check_extractor_gemini.py --image .\images\check01.jpeg --output result.json

# Debug — shows raw API response
python check_extractor_gemini.py --image .\images\check01.jpeg --debug

# Batch — whole folder (4 second delay between requests to respect rate limit)
python batch_process_gemini.py --folder .\images --output results.csv
```

### Comparison of all three backends

| | PaddleOCR | Claude Vision | Gemini Vision |
|---|---|---|---|
| Cost | Free | ~$0.01–0.03/check | Free |
| Credit card needed | No | Yes | No |
| Works offline | Yes | No | No |
| MICR accuracy | Poor | Excellent | Very good |
| Any check layout | Needs tuning | Yes | Yes |
| Setup complexity | High | Low | Low |
| Best for | Offline/air-gapped | Production | Testing & development |
