import os
from llama_cpp import Llama

# Exact path from your logs
model_path = r"C:\Users\dimbe\Documents\plugin brain v3\plug-in brain v5\models\Phi-3-mini-4k-instruct-q4.gguf"

print(f"Testing model at: {model_path}")

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ File found. Size: {size_mb:.2f} MB")
    
    print("\n--- ATTEMPT 1: MINIMAL CPU LOAD ---")
    try:
        # Try loading with absolutely minimal parameters, no GPU
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=0, 
            verbose=True # Turn on C++ logs to see the real error
        )
        print("\n🎉 SUCCESS! Phi-3 is healthy and loaded on CPU.")
    except Exception as e:
        print(f"\n❌ FAILED CPU LOAD: {e}")
        print("Likely cause: Corrupted file or incompatible GGUF version.")
else:
    print("❌ File NOT FOUND by Python. Check permissions or path typos.")
```

### Step 2: Run the Test
Open standard PowerShell in that folder (or just use your existing one) and run:

```powershell
python test_phi.py