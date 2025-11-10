import os
import hashlib
from llama_cpp import Llama

# === CONFIGURATION ===
# Exact path to the failing model
MODEL_PATH = r"C:\Users\dimbe\Documents\plugin brain v3\plug-in brain v5\models\Phi-3-mini-4k-instruct-q4.gguf"
# =====================

print(f"\n🔬 STARTING DIAGNOSTIC PROBE")
print(f"📍 Target: {MODEL_PATH}")

# 1. FILE INTEGRITY CHECK
if not os.path.exists(MODEL_PATH):
    print("❌ FATAL: File not found at specified path.")
    exit()

size_bytes = os.path.getsize(MODEL_PATH)
size_gb = size_bytes / (1024**3)
print(f"📁 File Size: {size_gb:.2f} GB ({size_bytes:,} bytes)")

if size_gb < 0.1:
    print("❌ CRITICAL: File is too small. It's likely an incomplete download or an HTML error page saved as a .gguf.")
    exit()

print("⏳ Calculating SHA-256 Hash (this might take a minute)...")
# Quick partial hash to save time, usually enough to catch corruption
with open(MODEL_PATH, "rb") as f:
    # Read first 10MB and last 10MB for a quick integrity check
    start_chunk = f.read(1024 * 1024 * 10)
    try:
        f.seek(-1024 * 1024 * 10, 2)
        end_chunk = f.read()
    except OSError: # Handle files smaller than 20MB just in case
        f.seek(0)
        end_chunk = f.read()
    quick_hash = hashlib.sha256(start_chunk + end_chunk).hexdigest()
print(f"🔑 Quick Hash signature: {quick_hash[:16]}...")

# 2. LOAD ATTEMPT (VERBOSE MODE)
print("\n🧠 ATTEMPTING VERBOSE LOAD...")
print("--- BEGIN LLAMA.CPP LOGS ---")
try:
    # We use n_gpu_layers=0 first to rule out VRAM issues.
    # verbose=True will show us the exact C++ error.
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=0, 
        n_ctx=512, 
        verbose=True
    )
    print("--- END LLAMA.CPP LOGS ---")
    print("\n🎉 DIAGNOSIS: SUCCESS! The model file is healthy and loadable.")
    print("The issue might be related to GPU drivers or VRAM when loaded with other models.")

except Exception as e:
    print("--- END LLAMA.CPP LOGS ---")
    print(f"\n❌ DIAGNOSIS: LOAD FAILED.")
    print(f"Error details: {e}")
    print("\nINTERPRETATION:")
    err_str = str(e).lower()
    if "magic" in err_str or "invalid" in err_str:
        print("👉 The file is corrupted. It's not a valid GGUF container.")
    elif "unsupported" in err_str or "arch" in err_str:
        print("👉 Your installed 'llama-cpp-python' is too old for this newer Phi-3 model file.")
    else:
        print("👉 Likely incomplete download or disk read error.")