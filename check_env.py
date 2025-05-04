import os
import sys
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

print("Checking environment variables...")

# Check for GEMINI_API_KEY
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
    print("Summarization will use local fallback methods.")
else:
    print("✓ GEMINI_API_KEY is set")

# Check for PORT
port = os.environ.get("PORT")
if port:
    print(f"✓ PORT is set to {port}")
else:
    print("PORT is not set. Will default to 5000 in local development.")

print("\nEnvironment check complete.") 