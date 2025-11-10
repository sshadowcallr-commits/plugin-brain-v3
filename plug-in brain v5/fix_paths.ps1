# Quick Path Fix
Write-Host "Fixing model paths..." -ForegroundColor Cyan

$v5 = "C:\Users\dimbe\Documents\plugin brain v3\plug-in brain v5"

# Fix channel_detector.py
$chan = @'
from llama_cpp import Llama
import re, json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
phi = Llama(str(MODEL_PATH), n_gpu_layers=-1, verbose=False)

VARIATIONS = [
    "mono", "stereo", "m to s", "mono to stereo", "m2s", "s to m", "stereo to mono", "s2m",
    "mid-side", "m/s", "mid side", "ms", "side only", "mid only",
    "5.1", "5 point 1", "surround 5.1", "6 channel", "lfe",
    "7.1", "7 point 1", "surround 7.1", "8 channel",
    "9.1", "11.1", "atmos", "dolby atmos", "object based",
    "360", "binaural", "ambisonics", "immersive", "spatial",
    "quad", "quadraphonic", "4 channel", "hex", "true mono", "fold down"
]

PATTERN = re.compile(r"(?i)\b(" + "|".join(re.escape(v) for v in VARIATIONS) + r")\b")

def detect_channels(name: str, nfo: str) -> dict:
    text = f"{name} {nfo}".lower()
    matches = set(PATTERN.findall(text))
    if matches:
        return {"channels": "|".join(sorted(matches)), "confidence": 0.92}
    prompt = f"""Analyze VST name and NFO for channel config.
Name: {name}
NFO: {nfo[:1800]}
Possible: {', '.join(VARIATIONS)}
Output ONLY JSON:
{{"channels": "mono|stereo|m/s|5.1|unknown", "confidence": 0.0-1.0}}"""
    try:
        out = phi(prompt, max_tokens=80, temperature=0.1)["choices"][0]["text"]
        return json.loads(re.search(r"\{.*\}", out, re.S).group(0))
    except:
        return {"channels": "unknown", "confidence": 0.3}
'@

Set-Content -Path "$v5\moa\agents\channel_detector.py" -Value $chan -Encoding utf8

# Read llm_core.py and fix the model loading lines
$llmPath = "$v5\moa\llm_core.py"
$content = Get-Content $llmPath -Raw

# Replace model loading lines
$content = $content -replace 'gemma = Llama\("models/', 'BASE_DIR = Path(__file__).parent.parent; MODEL_DIR = BASE_DIR / "models"; gemma = Llama(str(MODEL_DIR / "'
$content = $content -replace 'mistral = Llama\("models/', 'mistral = Llama(str(MODEL_DIR / "'
$content = $content -replace 'phi = Llama\("models/', 'phi = Llama(str(MODEL_DIR / "'
$content = $content -replace '\.gguf"', '.gguf")'

Set-Content -Path $llmPath -Value $content -Encoding utf8

Write-Host "Done! Now run: .\start_backend.ps1" -ForegroundColor Green