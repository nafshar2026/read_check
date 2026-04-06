"""
list_models.py
==============
Lists all Gemini models available on your account.
Run this to find out which models you can use.

Usage:
  python list_models.py
"""
import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from google import genai

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("Error: GEMINI_API_KEY not set in .env or environment")
    exit(1)

client = genai.Client(api_key=api_key)

print("Available models that support generateContent:\n")
for model in client.models.list():
    if 'generateContent' in (model.supported_actions or []):
        print(f"  {model.name}")
