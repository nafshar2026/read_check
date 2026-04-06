@echo off
REM run.bat — Windows CMD launcher for check_extractor.
SET FLAGS_use_mkldnn=0
SET PADDLE_DISABLE_ONEDNN=1
SET FLAGS_onednn_cpu_disabled=1
SET PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

python check_extractor.py %*
