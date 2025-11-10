# PLUGIN BRAIN v5 - FIXED INSTALLATION SCRIPT
# Uses existing v3 setup, fixes all syntax errors
# Run as Administrator

# Check admin rights
$user = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]$user
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "`nERROR: Requires Administrator privileges" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

# Set error handling
$ErrorActionPreference = "Stop"

# Paths
$base = "C:\Users\dimbe\Documents\plugin brain v3"
$v5 = "$base\plug-in brain v5"

Write-Host "=== Plugin Brain v5 Installation ===" -ForegroundColor Cyan
Write-Host "Base directory: $base" -ForegroundColor White

# Verify base directory exists
if (-not (Test-Path $base)) {
    Write-Host "ERROR: Base directory not found: $base" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# 1. Create v5 structure
Write-Host "`n[1/9] Creating v5 directory structure..." -ForegroundColor Green
$dirs = @(
    "$v5",
    "$v5\models",
    "$v5\data",
    "$v5\moa",
    "$v5\moa\agents",
    "$v5\api",
    "$v5\database",
    "$v5\logs",
    "$v5\plugin-brain-ui",
    "$v5\plugin-brain-ui\src",
    "$v5\plugin-brain-ui\src\components",
    "$v5\plugin-brain-ui\src\api"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

# Create __init__.py files
New-Item -ItemType File -Force -Path "$v5\moa\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "$v5\moa\agents\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "$v5\api\__init__.py" | Out-Null

Write-Host "✓ Directory structure created" -ForegroundColor Green

# 2. Copy models
Write-Host "`n[2/9] Copying models..." -ForegroundColor Green
if (Test-Path "$base\models\*.gguf") {
    Copy-Item -Path "$base\models\*.gguf" -Destination "$v5\models\" -Force
    Write-Host "✓ Models copied" -ForegroundColor Green
} else {
    Write-Host "WARNING: No .gguf models found in $base\models" -ForegroundColor Yellow
}

# 3. Copy data
Write-Host "`n[3/9] Copying data files..." -ForegroundColor Green
if (Test-Path "$base\data\native_plugins.json") {
    Copy-Item -Path "$base\data\native_plugins.json" -Destination "$v5\data\" -Force
    Write-Host "✓ native_plugins.json copied" -ForegroundColor Green
} else {
    Write-Host "WARNING: native_plugins.json not found" -ForegroundColor Yellow
    # Create empty file
    '{"natives": []}' | Out-File "$v5\data\native_plugins.json" -Encoding utf8
}

# 4. Copy UI files
Write-Host "`n[4/9] Copying UI files..." -ForegroundColor Green
if (Test-Path "$base\plugin-brain-ui") {
    Copy-Item -Path "$base\plugin-brain-ui\*" -Destination "$v5\plugin-brain-ui\" -Recurse -Force
    Write-Host "✓ UI files copied" -ForegroundColor Green
} else {
    Write-Host "WARNING: plugin-brain-ui folder not found" -ForegroundColor Yellow
}

# 5. Install Python dependencies
Write-Host "`n[5/9] Installing Python dependencies..." -ForegroundColor Green
$venvPath = "$base\venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    Write-Host "Using existing venv from v3..." -ForegroundColor Cyan
    & $venvPath
    pip install --upgrade pip --quiet
    pip install fastapi uvicorn langgraph sentence-transformers llama-cpp-python httpx pydantic sqlalchemy --quiet
    Write-Host "✓ Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "WARNING: venv not found, creating new one..." -ForegroundColor Yellow
    python -m venv "$v5\.venv"
    & "$v5\.venv\Scripts\Activate.ps1"
    pip install --upgrade pip --quiet
    pip install fastapi uvicorn langgraph sentence-transformers llama-cpp-python httpx pydantic sqlalchemy --quiet
    Write-Host "✓ New venv created and dependencies installed" -ForegroundColor Green
}

# 6. Write channel_detector.py
Write-Host "`n[6/9] Creating channel_detector.py..." -ForegroundColor Green
$chanDetector = @'
from llama_cpp import Llama
import re, json

phi = Llama("models/Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers=-1, verbose=False)

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

Set-Content -Path "$v5\moa\agents\channel_detector.py" -Value $chanDetector -Encoding utf8
Write-Host "✓ channel_detector.py created" -ForegroundColor Green

# 7. Write llm_core.py (shortened for script size, but complete)
Write-Host "`n[7/9] Creating llm_core.py..." -ForegroundColor Green
$llmCore = @'
import os, json, re, shutil, sqlite3, time, random
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import httpx

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from .agents.channel_detector import detect_channels

# Initialize models
gemma = Llama("models/gemma-2-2b-it-Q4_K_M.gguf", n_gpu_layers=-1, verbose=False)
mistral = Llama("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_gpu_layers=-1, verbose=False)
phi = Llama("models/Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers=-1, verbose=False)
sim_model = SentenceTransformer("intfloat/e5-small-v2")

# Database setup
KB_PATH = Path("database/knowledge_base.db")
KB_PATH.parent.mkdir(exist_ok=True)
conn = sqlite3.connect(str(KB_PATH), check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS plugins (
    name TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT,
    vendor TEXT,
    path_schema TEXT,
    channels TEXT,
    description TEXT,
    confidence REAL,
    source TEXT
)""")
conn.commit()

def load_natives() -> set:
    f = Path("data/native_plugins.json")
    return {p.lower() for p in json.load(f.open()).get("natives", [])} if f.exists() else set()

NATIVES = load_natives()

class AgentState(BaseModel):
    plugin_path: str
    plugin_name: str
    nfo_content: str = ""
    classification: Optional[Dict] = None
    confidence: float = 0.0
    channels: str = "unknown"
    kb_entry: Optional[Dict] = None
    logs: List[str] = Field(default_factory=list)
    approved: bool = False

def log(state: AgentState, msg: str):
    state.logs.append(msg)
    log_path = Path("logs/scan.log")
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "plugin": state.plugin_name,
            "msg": msg,
            "conf": state.confidence,
            "channels": state.channels,
            "ts": time.time()
        }) + "\n")

def similarity_check(state: AgentState) -> AgentState:
    if state.approved or state.plugin_name.lower() in NATIVES:
        log(state, "Skipped (native/approved)")
        return state
    
    emb = sim_model.encode(state.plugin_name, normalize_embeddings=True)
    rows = conn.execute("SELECT name, path_schema, confidence, channels FROM plugins").fetchall()
    if not rows:
        return state
    
    names = [r[0] for r in rows]
    embs = sim_model.encode(names, normalize_embeddings=True)
    sims = emb @ embs.T
    best = sims.argmax()
    
    if sims[best] > 0.86:
        row = dict(zip(["name", "path_schema", "confidence", "channels"], rows[best]))
        state.kb_entry = row
        state.confidence = float(sims[best])
        state.channels = row["channels"]
        log(state, f"KB match: {names[best]} ({sims[best]:.3f})")
    
    return state

def parse_context(state: AgentState) -> AgentState:
    nfo_path = Path(state.plugin_path).with_suffix(".nfo")
    if nfo_path.exists():
        state.nfo_content = nfo_path.read_text(encoding="utf-8", errors="ignore")[:4000]
    
    chan = detect_channels(state.plugin_name, state.nfo_content)
    state.channels = chan["channels"]
    state.confidence = max(state.confidence, chan["confidence"])
    
    prompt = f"""Classify VST plugin.
Name: {state.plugin_name}
Channels: {state.channels}
NFO: {state.nfo_content[:500]}

Return JSON:
{{"category": "Effects|Generators", "subcategory": "Mastering/Delay", "vendor": "iZotope", "description": "AI mastering suite", "path_schema": "Effects/Mastering/Dynamic-EQ/iZotope/Ozone 10.fst"}}"""
    
    try:
        out = mistral(prompt, max_tokens=200, temperature=0.1)["choices"][0]["text"]
        data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        state.classification = data
        state.confidence = 0.78
        log(state, f"Parsed: {data['category']}/{data['subcategory']} [{state.channels}]")
    except:
        state.confidence = 0.3
        log(state, "Parse failed")
    
    return state

def web_enrich(state: AgentState) -> AgentState:
    if state.confidence >= 0.82:
        return state
    
    rss_url = "https://www.kvraudio.com/rss/products.xml"
    time.sleep(random.uniform(3, 6))
    
    try:
        r = httpx.get(rss_url, timeout=15)
        r.raise_for_status()
        titles = re.findall(r'<title>([^<]+)</title>', r.text)
        matches = [t for t in titles if state.plugin_name.lower() in t.lower()][:3]
        
        if matches and state.classification:
            prompt = f"""Internet correction for: {state.plugin_name}
RSS matches: {matches}
Current: {state.classification}

Infer correct category, subcategory, vendor, channels.

JSON:
{{"category": "...", "subcategory": "...", "vendor": "...", "channels": "...", "confidence_adjust": 0.0-0.3}}"""
            
            out = phi(prompt, max_tokens=180, temperature=0.1)["choices"][0]["text"]
            update = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            state.classification.update(update)
            state.confidence += update.get("confidence_adjust", 0)
            state.channels = update.get("channels", state.channels)
            log(state, f"Web enriched: {matches[0][:50]}...")
    except Exception as e:
        log(state, f"Web failed: {e}")
    
    if state.classification:
        conn.execute("""INSERT OR REPLACE INTO plugins
            (name, category, subcategory, vendor, path_schema, channels, description, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (state.plugin_name, state.classification.get("category"),
             state.classification.get("subcategory"), state.classification.get("vendor"),
             state.classification.get("path_schema"), state.channels,
             state.classification.get("description"), state.confidence, "web"))
        conn.commit()
    
    return state

def apply_known_rule(state: AgentState) -> AgentState:
    if not state.kb_entry:
        return state
    
    kb = state.kb_entry
    state.classification = {
        "category": kb.get("category", "Effects"),
        "subcategory": kb.get("subcategory", "Uncategorized"),
        "vendor": kb.get("vendor", "Unknown"),
        "description": "",
        "path_schema": kb["path_schema"]
    }
    state.channels = kb["channels"]
    state.confidence = float(kb["confidence"])
    log(state, f"Applied KB rule (conf={state.confidence:.2f})")
    return state

def route_after_similarity(state: AgentState) -> str:
    if state.approved:
        return END
    if state.kb_entry:
        return "apply_known_rule"
    if state.confidence < 0.60:
        return "web_enrich"
    return "parse_context"

# Build workflow
workflow = StateGraph(AgentState)
workflow.set_entry_point("similarity_check")
workflow.add_node("similarity_check", similarity_check)
workflow.add_node("parse_context", parse_context)
workflow.add_node("web_enrich", web_enrich)
workflow.add_node("apply_known_rule", apply_known_rule)

workflow.add_conditional_edges(
    "similarity_check",
    route_after_similarity,
    {"apply_known_rule": "apply_known_rule", "parse_context": "parse_context", "web_enrich": "web_enrich", END: END}
)
workflow.add_edge("apply_known_rule", END)
workflow.add_edge("parse_context", END)
workflow.add_edge("web_enrich", END)

checkpointer = SqliteSaver.from_conn_string(str(KB_PATH))
app = workflow.compile(checkpointer=checkpointer)

def smart_scan(installed_path: str, db_path: str, dry_run: bool = True) -> Dict[str, Any]:
    installed = Path(installed_path)
    db = Path(db_path)
    effects_root = db / "Effects"
    gens_root = db / "Generators"
    effects_root.mkdir(parents=True, exist_ok=True)
    gens_root.mkdir(parents=True, exist_ok=True)
    
    placed = {p.stem for p in effects_root.rglob("*.fst")} | {p.stem for p in gens_root.rglob("*.fst")}
    new_fst = [p for p in installed.glob("*.fst") if p.stem not in placed]
    
    summary = {
        "processed": 0,
        "copied": 0,
        "low_conf": [],
        "errors": [],
        "skipped_natives": 0,
        "kb_hits": 0,
        "web_enriched": 0
    }
    
    for fst in new_fst:
        nfo = fst.with_suffix(".nfo")
        if not nfo.exists():
            summary["errors"].append(f"Missing .nfo for {fst.name}")
            continue
        
        state = AgentState(plugin_path=str(fst), plugin_name=fst.stem)
        
        try:
            final: AgentState = app.invoke(state)
        except Exception as e:
            summary["errors"].append(f"{fst.name} -> {e}")
            continue
        
        summary["processed"] += 1
        
        if final.approved or final.plugin_name.lower() in NATIVES:
            summary["skipped_natives"] += 1
            continue
        
        if final.kb_entry:
            summary["kb_hits"] += 1
        if "web" in str(final.logs):
            summary["web_enriched"] += 1
        
        if not final.classification:
            summary["errors"].append(f"{fst.name} -> no classification")
            continue
        
        cat = final.classification.get("category", "Effects")
        schema = final.classification.get("path_schema", "")
        
        if not schema:
            sub = final.classification.get("subcategory", "Uncategorized")
            ven = final.classification.get("vendor", "Unknown")
            schema = f"{cat}/{sub}/{ven}/{fst.name}"
        
        parts = Path(schema).parts[:-1]
        root = effects_root if cat == "Effects" else gens_root
        target_dir = root.joinpath(*parts)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_fst = target_dir / fst.name
        target_nfo = target_dir / nfo.name
        
        if not dry_run:
            try:
                shutil.copy2(str(fst), str(target_fst))
                shutil.copy2(str(nfo), str(target_nfo))
                summary["copied"] += 1
            except Exception as e:
                summary["errors"].append(f"Copy failed {fst.name}: {e}")
        
        if final.confidence < 0.80:
            summary["low_conf"].append({
                "name": final.plugin_name,
                "confidence": round(final.confidence, 3),
                "channels": final.channels,
                "suggested_path": str(target_fst),
                "reason": "low_confidence"
            })
        
        conn.execute("""INSERT OR REPLACE INTO plugins
            (name, category, subcategory, vendor, path_schema, channels, description, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (final.plugin_name, final.classification.get("category"),
             final.classification.get("subcategory"), final.classification.get("vendor"),
             schema, final.channels, final.classification.get("description", ""), final.confidence, "scan"))
        conn.commit()
    
    return summary

def editor_move(plugin_name: str, old_fst_path: str, new_fst_path: str) -> bool:
    old = Path(old_fst_path)
    new = Path(new_fst_path)
    if not old.exists():
        return False
    
    new.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(old), str(new))
        old_nfo = old.with_suffix(".nfo")
        if old_nfo.exists():
            shutil.move(str(old_nfo), new.with_suffix(".nfo"))
        
        conn.execute("INSERT OR REPLACE INTO plugins (name, confidence, source) VALUES (?, ?, ?)",
                    (plugin_name, 1.0, "user"))
        conn.commit()
        return True
    except Exception as e:
        print(f"Move failed: {e}")
        return False
'@

Set-Content -Path "$v5\moa\llm_core.py" -Value $llmCore -Encoding utf8
Write-Host "✓ llm_core.py created" -ForegroundColor Green

# 8. Write api_server.py
Write-Host "`n[8/9] Creating api_server.py..." -ForegroundColor Green
$apiServer = @'
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from moa.llm_core import smart_scan, editor_move, conn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ScanReq(BaseModel):
    installed_path: str
    db_path: str
    dry_run: bool = True

class MoveReq(BaseModel):
    plugin_name: str
    old_fst_path: str
    new_fst_path: str

@app.post("/scan/start")
async def scan(req: ScanReq, bg: BackgroundTasks):
    bg.add_task(smart_scan, req.installed_path, req.db_path, req.dry_run)
    return {"status": "started"}

@app.get("/plugins")
async def plugins(low_conf_only: bool = False):
    sql = "SELECT name, category, confidence, path_schema, channels FROM plugins"
    if low_conf_only:
        sql += " WHERE confidence < 0.8"
    rows = conn.execute(sql).fetchall()
    return [{"name": r[0], "cat": r[1], "conf": r[2], "path": r[3], "channels": r[4]} for r in rows]

@app.post("/bulk-approve")
async def approve(names: List[str]):
    for n in names:
        conn.execute("UPDATE plugins SET confidence=1.0, source='user' WHERE name=?", (n,))
    conn.commit()
    return {"approved": len(names)}

@app.post("/editor-move")
async def move(req: MoveReq):
    return {"success": editor_move(req.plugin_name, req.old_fst_path, req.new_fst_path)}

@app.get("/")
async def root():
    return {"message": "Plugin Brain v5 API", "status": "online"}
'@

Set-Content -Path "$v5\api\api_server.py" -Value $apiServer -Encoding utf8
Write-Host "✓ api_server.py created" -ForegroundColor Green

# 9. Create launch scripts
Write-Host "`n[9/9] Creating launch scripts..." -ForegroundColor Green

# Backend launcher
$backendLauncher = @"
# Backend Launcher for Plugin Brain v5
`$base = "C:\Users\dimbe\Documents\plugin brain v3"
`$v5 = "`$base\plug-in brain v5"

Write-Host "Starting Plugin Brain v5 Backend..." -ForegroundColor Cyan
Set-Location `$v5

# Activate venv
if (Test-Path "`$base\venv\Scripts\Activate.ps1") {
    & "`$base\venv\Scripts\Activate.ps1"
} else {
    & "`$v5\.venv\Scripts\Activate.ps1"
}

# Start server
Write-Host "Backend running at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
uvicorn api.api_server:app --reload --host 127.0.0.1 --port 8000
"@

Set-Content -Path "$v5\start_backend.ps1" -Value $backendLauncher -Encoding utf8

# Frontend launcher
$frontendLauncher = @"
# Frontend Launcher for Plugin Brain v5
`$v5 = "C:\Users\dimbe\Documents\plugin brain v3\plug-in brain v5"

Write-Host "Starting Plugin Brain v5 Frontend..." -ForegroundColor Cyan
Set-Location "`$v5\plugin-brain-ui"

if (Test-Path "package.json") {
    npm run electron
} else {
    Write-Host "ERROR: package.json not found. Run 'npm install' first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
"@

Set-Content -Path "$v5\start_frontend.ps1" -Value $frontendLauncher -Encoding utf8

Write-Host "✓ Launch scripts created" -ForegroundColor Green

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nProject location:" -ForegroundColor White
Write-Host "  $v5" -ForegroundColor Yellow

Write-Host "`nTo start the system:" -ForegroundColor White
Write-Host "  1. Backend:  .\start_backend.ps1" -ForegroundColor Yellow
Write-Host "  2. Frontend: .\start_frontend.ps1 (in new window)" -ForegroundColor Yellow

Write-Host "`nOr manually:" -ForegroundColor White
Write-Host "  Backend:  cd '$v5'; uvicorn api.api_server:app --reload" -ForegroundColor Cyan
Write-Host "  Frontend: cd '$v5\plugin-brain-ui'; npm run electron" -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  - Install UI deps: cd plugin-brain-ui && npm install" -ForegroundColor Cyan
Write-Host "  - Test backend: http://127.0.0.1:8000/docs" -ForegroundColor Cyan

Write-Host "`nPress Enter to exit..." -ForegroundColor Gray
Read-Host