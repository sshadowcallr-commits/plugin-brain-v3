import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

# --- Neuro-Sensory Initialization ---
try:
    from llama_cpp import Llama
    _phi_model = None
except ImportError:
    Llama = None
    _phi_model = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# UPDATED: Now using Qwen2-1.5B for ultra-fast sensory input (optimized for 6GB VRAM)
MODEL_PATH = BASE_DIR / "models" / "qwen2-1_5b-instruct-q4_k_m.gguf"

def get_sensory_model():
    """JIT loading of the auditory sensory model (Qwen2)"""
    global _phi_model
    if _phi_model is None and Llama and MODEL_PATH.exists():
        print("   🔊 Initializing sensory cortex (Qwen2)...")
        try:
            # Low context (512) is enough for channel detection, saves VRAM
            _phi_model = Llama(model_path=str(MODEL_PATH), n_ctx=512, n_gpu_layers=-1, verbose=False)
        except Exception as e:
            print(f"   ❌ Sensory cortex failure: {e}")
    return _phi_model

CHANNEL_PATTERNS = {
    "mono": re.compile(r"\b(mono|monophonic|1ch)\b", re.IGNORECASE),
    "stereo": re.compile(r"\b(stereo|stereophonic|2ch)\b", re.IGNORECASE),
    "5.1": re.compile(r"\b(5\.1|surround|six channel)\b", re.IGNORECASE),
    "7.1": re.compile(r"\b(7\.1|eight channel)\b", re.IGNORECASE),
    "atmos": re.compile(r"\b(atmos|immersive|spatial|binaural|ambisonics|3d audio)\b", re.IGNORECASE),
    "mid-side": re.compile(r"\b(m/s|mid-side|mid side)\b", re.IGNORECASE)
}

def detect_channels(plugin_name: str, nfo_content: str = "") -> str:
    raw_input = f"{plugin_name} {nfo_content[:1000]}".lower()
    detected = {config for config, pattern in CHANNEL_PATTERNS.items() if pattern.search(raw_input)}

    if "atmos" in detected: return "atmos"
    if "7.1" in detected: return "7.1"
    if "5.1" in detected: return "5.1"
    if "mid-side" in detected: return "mid-side"
    if "stereo" in detected and len(detected) == 1: return "stereo"
    if "mono" in detected and len(detected) == 1: return "mono"

    model = get_sensory_model()
    if model:
        prompt = f"""TASK: Identify audio channels in this plugin text.
TEXT: {plugin_name} {nfo_content[:300]}
OPTIONS: [mono, stereo, 5.1, 7.1, atmos, mid-side]
RESULT (ONE WORD):"""
        try:
            output = model(prompt, max_tokens=6, temperature=0.1, stop=["\n", ",", "."])
            result = output['choices'][0]['text'].strip().lower()
            for option in CHANNEL_PATTERNS.keys():
                if option in result: return option
        except Exception as e:
            print(f"   ⚠ Sensory overload: {e}")

    return "unknown"