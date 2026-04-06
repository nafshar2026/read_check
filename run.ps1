# run.ps1
# Windows PowerShell launcher for check_extractor.
# Sets environment variables BEFORE Python starts so PaddlePaddle's
# C++ runtime picks them up at load time.
#
# Usage:
#   .\run.ps1 --image .\images\check01.jpeg
#   .\run.ps1 --image .\images\check01.jpeg --output result.json
#   .\run.ps1 --image .\images\check01.jpeg --debug

$env:FLAGS_use_mkldnn = "0"
$env:PADDLE_DISABLE_ONEDNN = "1"
$env:FLAGS_onednn_cpu_disabled = "1"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"

python check_extractor.py @args
