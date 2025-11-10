# ============================================
# FILE: check_mistral_rs.py
# PURPOSE: A simple script to verify that the mistral.rs engine
#          and the Python bindings can successfully load and run a model.
# ============================================
import time
from pathlib import Path
import traceback

print("--- Mistral.rs Engine Verification Script ---")

try:
    import mistral_rs
    print("[SUCCESS] `mistral_rs` library is installed.")
except ImportError:
    print("[FATAL ERROR] `mistral_rs` library is not installed in your venv!")
    print("Please run: pip install mistral_rs")
    exit()

# --- Configuration ---
# This path points to your folder containing the .gguf model files.
MODELS_DIR = Path.home() / "Documents" / "plugin brain v3" / "models"
MODEL_TO_TEST = "gemma-2-2b-it-Q4_K_M.gguf"
MODEL_PATH = MODELS_DIR / MODEL_TO_TEST
MODEL_ID = "gemma_test" # An alias for the model in the engine

print(f"Model Directory: {MODELS_DIR}")
print(f"Model to Test:   {MODEL_PATH}")

# --- Verification Steps ---

# 1. Check if the model file actually exists.
if not MODEL_PATH.exists():
    print(f"\n[FATAL ERROR] Model file not found at the specified path!")
    print("Please make sure the model file exists and the path is correct.")
    exit()
else:
    print(f"\n[SUCCESS] Model file found.")

# 2. Initialize the mistral.rs engine.
engine = None
try:
    print("\nAttempting to initialize the mistral.rs inference engine...")
    start_time = time.time()
    engine = mistral_rs.MistralRs(which=None, gqa=1)
    end_time = time.time()
    print(f"[SUCCESS] Inference engine initialized in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"\n[FATAL ERROR] Failed to initialize the mistral.rs engine.")
    print("This could be due to a problem with the Rust backend or system dependencies.")
    traceback.print_exc()
    exit()

# 3. Load the model into the engine.
try:
    print(f"\nAttempting to load '{MODEL_TO_TEST}' into the engine...")
    start_time = time.time()
    engine.load_model_from_path(
        model_id=MODEL_ID,
        model_path=str(MODEL_PATH),
    )
    end_time = time.time()
    print(f"[SUCCESS] Model loaded successfully in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"\n[FATAL ERROR] Failed to load the model into the engine.")
    print("The GGUF file might be corrupted or incompatible.")
    traceback.print_exc()
    exit()

# 4. Run a test inference.
try:
    print("\nAttempting to run a test query...")
    prompt = "What is a VST audio plugin? Respond in one clear sentence."
    start_time = time.time()
    result = engine.chat(
        model_id=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1,
    )
    response_text = result.choices[0].message.content
    end_time = time.time()
    
    print(f"[SUCCESS] Test query completed in {end_time - start_time:.2f} seconds.")
    print("\n--- MODEL RESPONSE ---")
    print(response_text)
    print("------------------------")

except Exception as e:
    print(f"\n[FATAL ERROR] Test inference failed.")
    print("The model loaded but failed to generate a response.")
    traceback.print_exc()
    exit()

print("\n[FINAL VERDICT] Your `mistral.rs` setup is working correctly with your Python environment!")