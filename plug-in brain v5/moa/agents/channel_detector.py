import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# =========================================
# SETUP
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

phi_model = None

def get_phi_model():
    """Lazy loader for Phi-3 model to avoid circular imports or slow startups if not needed."""
    global phi_model
    if phi_model is None:
        if not MODEL_PATH.exists():
             print(f"⚠ Channel Detector: Phi-3 model not found at {MODEL_PATH}")
             return None
        try:
            # Load specifically for quick inference tasks
            phi_model = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=2048,          # Smaller context for speed
                n_gpu_layers=-1,     # Use GPU if available
                verbose=False
            )
        except Exception as e:
             print(f"❌ Channel Detector: Failed to load Phi-3: {e}")
             return None
    return phi_model

# Regex for common channel configurations to avoid expensive LLM calls
CHANNEL_PATTERNS = {
    "mono": re.compile(r"\b(mono|monophonic|1ch)\b", re.IGNORECASE),
    "stereo": re.compile(r"\b(stereo|stereophonic|2ch)\b", re.IGNORECASE),
    "5.1": re.compile(r"\b(5\.1|surround|six channel)\b", re.IGNORECASE),
    "7.1": re.compile(r"\b(7\.1|eight channel)\b", re.IGNORECASE),
    "atmos": re.compile(r"\b(atmos|immersive|spatial|binaural|ambisonics)\b", re.IGNORECASE),
    "mid-side": re.compile(r"\b(m/s|mid-side|mid side)\b", re.IGNORECASE)
}

def detect_channels(plugin_name: str, nfo_content: str = "") -> str:
    """
    Determines the audio channel configuration of a plugin.
    Prioritizes regex matching, falls back to Phi-3 LLM if ambiguous.
    """
    text_to_scan = f"{plugin_name} {nfo_content}".lower()
    
    detected = []
    for config, pattern in CHANNEL_PATTERNS.items():
        if pattern.search(text_to_scan):
            detected.append(config)
            
    # If we found exactly one clear configuration via Regex, return it.
    if len(detected) == 1:
        return detected[0]
    # If we found multiple (e.g. "Stereo to 5.1"), might need LLM to clarify, 
    # but often standardizing on the most complex one is safe.
    elif "atmos" in detected: return "atmos"
    elif "7.1" in detected: return "7.1"
    elif "5.1" in detected: return "5.1"
    elif "stereo" in detected: return "stereo"

    # Fallback: Use LLM if no regex matched
    model = get_phi_model()
    if model:
        prompt = f"""Identify audio channels for plugin: '{plugin_name}'
Metadata: {nfo_content[:300]}
Options: [mono, stereo, 5.1, 7.1, atmos, mid-side]
Return ONLY one option word."""
        try:
            output = model(prompt, max_tokens=10, temperature=0.1)
            result = output["choices"][0]["text"].strip().lower()
            # Basic validation to ensure LLM didn't hallucinate wildly
            for option in CHANNEL_PATTERNS.keys():
                if option in result:
                    return option
        except Exception as e:
            print(f"⚠ Channel LLM failed: {e}")
            
    return "unknown"