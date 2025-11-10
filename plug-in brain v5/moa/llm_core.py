import os
import json
import re
import shutil
import sqlite3
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import httpx

# Handle optional imports for different environments
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("✔ Imported SqliteSaver from langgraph.checkpoint.sqlite")
except ImportError:
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver
        print("✔ Imported SqliteSaver from langgraph_checkpoint_sqlite")
    except ImportError:
        SqliteSaver = None
        print("⚠ Warning: SqliteSaver not found. State checkpointing disabled.")

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ CRITICAL: llama-cpp-python not installed.")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ CRITICAL: sentence-transformers not installed.")
    raise

from .agents.channel_detector import detect_channels

# ============================================
# INITIALIZATION & CONFIGURATION
# ============================================

# Use absolute paths to avoid relative path errors when running from different dirs
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

print(f"\n{'='*60}")
print("🧠 Plugin Brain v5 Neuro-Core Initializing")
print(f"{'='*60}")
print(f"📂 Base Directory: {BASE_DIR}")
print(f"📂 Model Directory: {MODEL_DIR}")

# Ensure critical directories exist
DATABASE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- Load AI Models ---
print("\n🔽 Loading Neural Models...")

def load_model(name: str, filename: str, n_gpu_layers: int = -1) -> Optional[Llama]:
    path = MODEL_DIR / filename
    if path.exists():
        print(f"   • Loading {name}...")
        try:
            model = Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, verbose=False)
            print(f"   ✔ {name} ready.")
            return model
        except Exception as e:
            print(f"   ❌ Failed to load {name}: {e}")
            return None
    else:
        print(f"   ⚠ Missing model file: {filename}")
        return None

# Load the mixture of experts
gemma = load_model("Gemma-2B (Generalist)", "gemma-2-2b-it-Q4_K_M.gguf")
mistral = load_model("Mistral-7B (Specialist)", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
phi = load_model("Phi-3 (Fast Reasoning)", "Phi-3-mini-4k-instruct-q4.gguf")

print("   • Loading E5 Embeddings (Memory)...")
try:
    sim_model = SentenceTransformer("intfloat/e5-small-v2")
    print("   ✔ Embeddings ready.")
except Exception as e:
    sim_model = None
    print(f"   ❌ Failed to load embeddings: {e}")

# --- Database Connection ---
KB_PATH = DATABASE_DIR / "knowledge_base.db"
conn = sqlite3.connect(str(KB_PATH), check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS plugins (
    name TEXT PRIMARY KEY,
    category TEXT, subcategory TEXT, vendor TEXT,
    path_schema TEXT, channels TEXT, description TEXT,
    confidence REAL, source TEXT
)""")
conn.commit()
print(f"✔ Knowledge Base connected: {KB_PATH.name}")

# --- Load Native Plugin Ignore List ---
NATIVES = set()
natives_path = DATA_DIR / "native_plugins.json"
if natives_path.exists():
    try:
        with open(natives_path, 'r', encoding='utf-8') as f:
            NATIVES = {p.lower() for p in json.load(f).get("natives", [])}
        print(f"✔ Loaded {len(NATIVES)} native plugins to ignore.")
    except Exception as e:
        print(f"⚠ Error reading native_plugins.json: {e}")

print(f"{'='*60}\n")

# ============================================
# CORE DATA STRUCTURES (STATE)
# ============================================

class AgentState(BaseModel):
    plugin_path: str
    plugin_name: str
    nfo_content: str = ""
    classification: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    channels: str = "unknown"
    kb_entry: Optional[Dict[str, Any]] = None
    logs: List[str] = Field(default_factory=list)
    approved: bool = False

def log_to_file(state: AgentState, msg: str):
    """Persistent logging for neuro-traceability"""
    state.logs.append(msg)
    try:
        with open(LOGS_DIR / "neuro_activity.jsonl", "a", encoding="utf-8") as f:
             f.write(json.dumps({
                 "ts": time.time(),
                 "plugin": state.plugin_name,
                 "msg": msg,
                 "conf": state.confidence
             }) + "\n")
    except Exception as e:
        print(f"⚠ Log Error: {e}")

# ============================================
# NEURAL AGENTS
# ============================================

def similarity_check(state: AgentState) -> AgentState:
    """Agent 1: Long-term memory lookup"""
    print(f"\n🔍 [Memory Agent] Scanning: {state.plugin_name}")
    
    if state.plugin_name.lower() in NATIVES:
        log_to_file(state, "Identified as native/ignored")
        print("   → Skipping (Native Plugin)")
        state.approved = True
        return state

    if not sim_model: return state

    # Fetch all known plugins
    rows = conn.execute("SELECT name, path_schema, confidence, channels FROM plugins").fetchall()
    if not rows: return state

    # Vector Search
    names = [r[0] for r in rows]
    current_emb = sim_model.encode(state.plugin_name, normalize_embeddings=True)
    db_embs = sim_model.encode(names, normalize_embeddings=True)
    scores = current_emb @ db_embs.T
    best_idx = scores.argmax()
    best_score = float(scores[best_idx])

    if best_score > 0.88:
        match = rows[best_idx]
        state.kb_entry = {
            "name": match[0], "path_schema": match[1],
            "confidence": match[2], "channels": match[3]
        }
        state.confidence = best_score
        state.channels = match[3]
        log_to_file(state, f"Memory match: {match[0]} ({best_score:.2f})")
        print(f"   💡 Memory recall: '{match[0]}' ({best_score:.2f} match)")
    return state

def parse_context(state: AgentState) -> AgentState:
    """Agent 2: Visual Cortex & Logic Center (NFO Reading + LLM Classification)"""
    print(f"📄 [Context Agent] Analyzing metadata...")
    
    # 1. Read NFO (Simulated visual input)
    nfo = Path(state.plugin_path).with_suffix(".nfo")
    if nfo.exists():
        state.nfo_content = nfo.read_text(errors='ignore')[:2500] # Cap context size

    # 2. Detect Channels (Audio auditory processing)
    state.channels = detect_channels(state.plugin_name, state.nfo_content)
    
    # 3. Specialist LLM Classification
    if mistral:
        prompt = f"""TASK: Classify VST Plugin.
NAME: {state.plugin_name}
CHANNELS: {state.channels}
NFO START: {state.nfo_content[:600]}
REQUIRED FORMAT (JSON ONLY):
{{"category": "Effects/Generators", "subcategory": "Reverb/Synth/Dynamics", "vendor": "Name", "description": "Short summary"}}"""
        
        try:
            print("   → Querying Mistral specialist...")
            output = mistral(prompt, max_tokens=256, temperature=0.2, stop=["```"])
            text = output['choices'][0]['text']
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                state.classification = json.loads(match.group(0))
                # Generate path schema immediately
                c = state.classification
                state.classification["path_schema"] = f"{c.get('category')}/{c.get('subcategory')}/{c.get('vendor')}/{state.plugin_name}"
                state.confidence = 0.75 # Baseline confidence for first-pass LLM
                print(f"   ✨ Classified: {state.classification.get('path_schema')}")
            else:
                raise ValueError("Neural net failed to output JSON")
        except Exception as e:
            print(f"   ⚠ Classification failed: {e}")
            state.confidence = 0.3
    
    return state

def web_enrich(state: AgentState) -> AgentState:
    """Agent 3: External Knowledge Retrieval (Web)"""
    if state.confidence > 0.85: return state # Don't need web if confident

    print(f"🌐 [Web Agent] Searching external databases...")
    # FIXED THE BROKEN LINE HERE vvv
    delay = random.uniform(1.5, 3.5)
    print(f"   → Rate limiting: waiting {delay:.1f}s...") 
    time.sleep(delay)

    try:
        # Simple KVR RSS check as a placeholder for real web search
        r = httpx.get("[https://www.kvraudio.com/rss/products.xml](https://www.kvraudio.com/rss/products.xml)", timeout=10)
        if r.status_code == 200 and state.plugin_name.lower() in r.text.lower():
             print("   ✔ Verified verified existance on KVR Audio")
             state.confidence += 0.15 # Boost confidence if found externally
             log_to_file(state, "Verified on KVR Audio")
    except Exception as e:
        print(f"   ⚠ Web lookup failed: {e}")

    return state

def save_memory(state: AgentState) -> AgentState:
    """Agent 4: Memory Consolidation (Save to DB)"""
    if state.classification and state.confidence > 0.5 and not state.approved:
        c = state.classification
        conn.execute("""INSERT OR REPLACE INTO plugins 
            (name, category, subcategory, vendor, path_schema, channels, description, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (state.plugin_name, c.get("category"), c.get("subcategory"), c.get("vendor"),
             c.get("path_schema"), state.channels, c.get("description"), state.confidence, "neural-scan"))
        conn.commit()
        print("   💾 Memory consolidated to database.")
    return state

# ============================================
# NEURAL PATHWAYS (WORKFLOW)
# ============================================

def router(state: AgentState) -> str:
    if state.approved: return END
    if state.kb_entry and state.confidence > 0.88: return "save_memory"
    if state.confidence < 0.6: return "web_enrich"
    return "save_memory"

workflow = StateGraph(AgentState)
workflow.add_node("memory_scan", similarity_check)
workflow.add_node("context_analysis", parse_context)
workflow.add_node("web_enrich", web_enrich)
workflow.add_node("save_memory", save_memory)

workflow.set_entry_point("memory_scan")

workflow.add_conditional_edges(
    "memory_scan", router,
    {"save_memory": "save_memory", "web_enrich": "web_enrich", END: END}
)
# Fallback edge if router doesn't match
workflow.add_edge("memory_scan", "context_analysis")

workflow.add_conditional_edges(
    "context_analysis", router,
    {"web_enrich": "web_enrich", "save_memory": "save_memory", END: END}
)

workflow.add_edge("web_enrich", "save_memory")
workflow.add_edge("save_memory", END)

# Compile the brain
try:
    checkpointer = SqliteSaver.from_conn_string(str(KB_PATH)) if SqliteSaver else None
    brain = workflow.compile(checkpointer=checkpointer)
    print("🧠 Neural pathways established successfully.\n")
except Exception as e:
    brain = workflow.compile()
    print(f"⚠ pathways established without checkpoints: {e}\n")

# ============================================
# EXPOSED API FUNCTIONS
# ============================================

def smart_scan(installed_path: str, db_path: str, dry_run: bool = True):
    print(f"🚀 Starting Neuro-Scan: {installed_path}")
    source = Path(installed_path)
    target = Path(db_path)
    
    for fst in source.rglob("*.fst"):
        try:
            initial = AgentState(plugin_path=str(fst), plugin_name=fst.stem)
            final = brain.invoke(initial)
            
            if not dry_run and final.get('classification'):
                 # Move logic would go here
                 pass
        except Exception as e:
            print(f"❌ Neural Failure on {fst.name}: {e}")

def editor_move(plugin_name: str, old_path: str, new_path: str) -> bool:
    try:
        src, dst = Path(old_path), Path(new_path)
        if not src.exists(): return False
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if src.with_suffix('.nfo').exists():
            shutil.move(str(src.with_suffix('.nfo')), str(dst.with_suffix('.nfo')))
            
        # Reinforce memory with user correction (100% confidence)
        new_schema = str(dst.relative_to(dst.parents[3])).replace('\\', '/') if len(dst.parents) > 3 else dst.name
        conn.execute("INSERT OR REPLACE INTO plugins (name, path_schema, confidence, source) VALUES (?, ?, 1.0, 'human_override')",
                    (plugin_name, new_schema))
        conn.commit()
        return True
    except Exception as e:
        print(f"Move failed: {e}")
        return False
```

### Step 2: Verify it works
Now that the file is saved, go back to your PowerShell window (where you are in the `venv`) and try running it again:

```powershell
uvicorn api.api_server:app --reload
```

It should now start without that `SyntaxError`.

### Step 3: NOW we use Git Bash
Once you confirm it starts up (you see the `INFO: Uvicorn running...` message), you can safely stop it (Ctrl+C) and use Git Bash to save this working state.

1.  Right-click your `plug-in brain v5` folder -> **Git Bash Here**.
2.  Run these commands in the Bash window to save this victory:

```bash
git init
git add .
git commit -m "Fixed syntax error in LLM core, neuro-engine ready"