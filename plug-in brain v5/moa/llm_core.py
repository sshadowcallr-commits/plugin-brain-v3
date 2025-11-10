import os
import json
import re
import shutil
import sqlite3
import time
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- Advanced Imports with Fallbacks ---
try:
    from langgraph_checkpoint_sqlite import SqliteSaver
    print("✔ NEURO-CORE: Active Memory (SqliteSaver) engaged.")
except ImportError:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        print("✔ NEURO-CORE: Active Memory (SqliteSaver) engaged (legacy mode).")
    except ImportError:
        SqliteSaver = None
        print("⚠ NEURO-CORE: Active Memory disabled. Operating in transient state.")

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ CRITICAL FAILURE: Neural pathway 'llama-cpp-python' severed.")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ CRITICAL FAILURE: Semantic processor 'sentence-transformers' missing.")
    raise

from .agents.channel_detector import detect_channels

# ============================================
# NEURO-CORE INITIALIZATION
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

print(f"\n{'='*60}")
print("🧠 PLUGIN BRAIN v5: NEURO-RESILIENT CORE (6GB OPTIMIZED) ONLINE")
print(f"{'='*60}")

# Ensure neural pathways exist
for d in [DATABASE_DIR, LOGS_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

# --- Load Neural Models (Experts) ---
print("\n🔽 Synchronizing Neuro-Slim Experts...")

def load_expert(name: str, filename: str, n_gpu_layers: int = -1) -> Optional[Llama]:
    """Universal model loader with GPU fallback and safety logging."""
    path = MODEL_DIR / filename
    if not path.exists():
        print(f"   ⚠ Expert missing: {filename}")
        return None

    print(f"   • Syncing {name}...")
    try:
        # Try GPU load first (unless n_gpu_layers=0 forces CPU)
        model = Llama(model_path=str(path), n_gpu_layers=n_gpu_layers, verbose=False, n_ctx=2048)
        device = "GPU" if n_gpu_layers != 0 else "CPU"
        print(f"   ✔ {name} online ({device} Mode).")
        return model
    except Exception:
        print(f"   ⚠ GPU sync failed for {name}. Rerouting to CPU...")
        try:
            model = Llama(model_path=str(path), n_gpu_layers=0, verbose=False, n_ctx=2048)
            print(f"   ✔ {name} online (CPU Mode - Fallback).")
            return model
        except Exception as e:
            print(f"   ❌ Failed to sync {name}: {e}")
            return None


# ============================================================
# 🧠 PLUGIN BRAIN v5 — NEURO-CORE INITIALIZATION (6GB MODE)
# ============================================================

# Generalist: Gemma-2-2B (Main reasoning core)
gemma = load_expert(
    "Gemma-2-2B (Generalist)",
    "gemma-2-2b-it-Q4_K_M.gguf",
    n_gpu_layers=-1  # Use GPU acceleration
)

# Specialist: reuse Gemma instance (saves VRAM)
mistral = gemma  # VRAM-efficient reuse of Gemma as Specialist

# Sensory: Qwen2-1.5B (fast perception & text parsing)
phi = load_expert(
    "Qwen2-1.5B (Sensory)",
    "qwen2-1_5b-instruct-q4_k_m.gguf",
    n_gpu_layers=-1  # Use GPU acceleration
)

# Semantic Memory: Sentence Transformer (runs on CPU)
print("   • Syncing Semantic Memory (E5)...")
try:
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer("intfloat/e5-small-v2")
    print("   ✔ Semantic Memory online.")
except Exception as e:
    sim_model = None
    print(f"   ❌ Semantic Memory failure: {e}")

# --- Knowledge Base Connection ---
KB_PATH = DATABASE_DIR / "knowledge_base.db"
conn = sqlite3.connect(str(KB_PATH), check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS plugins (
    name TEXT PRIMARY KEY,
    category TEXT, subcategory TEXT, vendor TEXT,
    path_schema TEXT, channels TEXT, description TEXT,
    confidence REAL, source TEXT
)""")
conn.commit()
print(f"✔ Knowledge Base linked: {KB_PATH.name}")

# --- Load Inhibitory patterns (Native Plugins) ---
NATIVES = set()
natives_path = DATA_DIR / "native_plugins.json"
if natives_path.exists():
    try:
        with open(natives_path, 'r', encoding='utf-8') as f:
            NATIVES = {p.lower() for p in json.load(f).get("natives", [])}
        print(f"✔ Loaded {len(NATIVES)} inhibitory patterns.")
    except Exception:
        pass 

print(f"{'='*60}\n")

# ============================================
# COGNITIVE STATE
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

def neuro_log(state: AgentState, msg: str, level: str = "INFO"):
    """Persistent, structured logging of cognitive processes."""
    entry = {
        "ts": time.time(),
        "level": level,
        "plugin": state.plugin_name,
        "msg": msg,
        "conf": state.confidence
    }
    state.logs.append(f"[{level}] {msg}")
    try:
        with open(LOGS_DIR / "neuro_activity.jsonl", "a", encoding="utf-8") as f:
             f.write(json.dumps(entry) + "\n")
    except Exception:
        pass 

# ============================================
# COGNITIVE AGENTS (NEURO-ENHANCED)
# ============================================

def memory_recall(state: AgentState) -> AgentState:
    """Agent 1: Long-term Memory Lookup (High speed, low energy)"""
    print(f"\n🔍 [Memory] Scanning: {state.plugin_name}")
    
    if state.plugin_name.lower() in NATIVES:
        neuro_log(state, "Matched inhibitory pattern (Native)", "INFO")
        print("   → Inhibited (Native Plugin)")
        state.approved = True
        return state

    if not sim_model: return state

    try:
        rows = conn.execute("SELECT name, path_schema, confidence, channels FROM plugins").fetchall()
        if not rows: return state

        names = [r[0] for r in rows]
        current_emb = sim_model.encode(state.plugin_name, normalize_embeddings=True)
        db_embs = sim_model.encode(names, normalize_embeddings=True)
        scores = current_emb @ db_embs.T
        best_idx = scores.argmax()
        best_score = float(scores[best_idx])

        # Adaptive threshold: Higher confidence required for automatic action
        if best_score > 0.89:
            match = rows[best_idx]
            state.kb_entry = {
                "name": match[0], "path_schema": match[1],
                "confidence": match[2], "channels": match[3]
            }
            state.confidence = best_score
            state.channels = match[3]
            neuro_log(state, f"Recall success: {match[0]} ({best_score:.2f})", "SUCCESS")
            print(f"   💡 Recall: '{match[0]}' ({best_score:.2f})")
    except Exception as e:
        neuro_log(state, f"Memory failure: {e}", "ERROR")
        print(f"   ⚠ Memory glitch: {e}")

    return state

def cognitive_analysis(state: AgentState) -> AgentState:
    """Agent 2: Deep Analysis (Visual + Auditory + Logic)"""
    print(f"🧠 [Cortex] Analyzing...")
    
    # 1. Visual Processing (NFO)
    nfo = Path(state.plugin_path).with_suffix(".nfo")
    if nfo.exists():
        try:
            state.nfo_content = nfo.read_text(errors='ignore')[:3000]
        except Exception:
             pass

    # 2. Auditory Processing (Channels)
    state.channels = detect_channels(state.plugin_name, state.nfo_content)
    
    # 3. Logical Processing (LLM)
    # Use specialist (Phi-3) if available, else generalist (Gemma-2)
    expert = mistral if mistral else gemma
    if expert:
        prompt = f"""TASK: Classify Audio Plugin accurately.
PLUGIN: {state.plugin_name}
CHANNELS: {state.channels}
METADATA: {state.nfo_content[:800]}
OUTPUT JSON ONLY:
{{"category": "Effects" or "Generators", "subcategory": "Specific Type", "vendor": "Manufacturer", "description": "Brief summary"}}"""
        
        try:
            # Reduced temperature for higher precision
            output = expert(prompt, max_tokens=300, temperature=0.1, stop=["```", "\n\n"])
            text = output['choices'][0]['text'].strip()
            # Robust JSON extraction
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                c = json.loads(match.group(0))
                # Sanitize inputs
                cat = c.get("category", "Effects").strip()
                sub = c.get("subcategory", "Uncategorized").strip().replace("/", "-")
                ven = c.get("vendor", "Unknown").strip().replace("/", "-")
                
                state.classification = c
                state.classification["category"] = cat
                state.classification["subcategory"] = sub
                state.classification["vendor"] = ven
                state.classification["path_schema"] = f"{cat}/{sub}/{ven}/{state.plugin_name}"
                
                state.confidence = 0.78 # High baseline for specialist
                print(f"   ✨ Cognition: {state.classification['path_schema']}")
            else:
                raise ValueError("Expert failed to structure output")
        except Exception as e:
            neuro_log(state, f"Cognitive failure: {e}", "ERROR")
            print(f"   ⚠ Cognitive slip: {e}")
            state.confidence = 0.2
    
    return state

# --- GROK-ENHANCED RESILIENCE ---
# Uses 'tenacity' to retry transient network errors with exponential backoff.
# It waits 1s, then 2s, then 4s, up to 10s, before giving up.
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.HTTPError)
)
def _secure_web_lookup(plugin_name: str) -> bool:
    with httpx.Client(timeout=5.0) as client:
        r = client.get("[https://www.kvraudio.com/rss/products.xml](https://www.kvraudio.com/rss/products.xml)")
        r.raise_for_status()
        return plugin_name.lower() in r.text.lower()

def external_validation(state: AgentState) -> AgentState:
    """Agent 3: External Reality Check (Web) - Neuro-Resilient"""
    if state.confidence > 0.85: return state

    print(f"🌐 [External] Validating reality...")
    # Neuro-mimetic rate limiting: Random pause simulates "thinking" time and prevents overloading
    # Jitter: Add randomness to avoid robotic patterns
    time.sleep(random.uniform(1.0, 3.0)) 

    try:
        if _secure_web_lookup(state.plugin_name):
             print("   ✔ External validation confirmed.")
             state.confidence += 0.15
             neuro_log(state, "Validated via external feed", "SUCCESS")
    except Exception as e:
        print(f"   ⚠ External link severed after retries: {e}")

    return state

def memory_consolidation(state: AgentState) -> AgentState:
    """Agent 4: Long-term Potentiation (Save to DB)"""
    if state.classification and state.confidence > 0.5 and not state.approved:
        try:
            c = state.classification
            conn.execute("""INSERT OR REPLACE INTO plugins 
                (name, category, subcategory, vendor, path_schema, channels, description, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (state.plugin_name, c.get("category"), c.get("subcategory"), c.get("vendor"),
                 c.get("path_schema"), state.channels, c.get("description"), state.confidence, "neuro-scan"))
            conn.commit()
            print("   💾 Memory consolidated.")
        except Exception as e:
             neuro_log(state, f"Consolidation failed: {e}", "CRITICAL")
             print(f"   ❌ Memory write error: {e}")
    return state

# ============================================
# NEURAL PATHWAYS (WORKFLOW)
# ============================================
def router(state: AgentState) -> str:
    """Dynamic routing based on confidence levels"""
    if state.approved: return END
    # High confidence recall -> skip expensive analysis
    if state.kb_entry and state.confidence > 0.92: return "memory_consolidation"
    # Low confidence -> seek external validation
    if state.confidence < 0.65: return "external_validation"
    return "memory_consolidation"

workflow = StateGraph(AgentState)
workflow.add_node("memory_recall", memory_recall)
workflow.add_node("cognitive_analysis", cognitive_analysis)
workflow.add_node("external_validation", external_validation)
workflow.add_node("memory_consolidation", memory_consolidation)

workflow.set_entry_point("memory_recall")

workflow.add_conditional_edges(
    "memory_recall", router,
    {"memory_consolidation": "memory_consolidation", "external_validation": "external_validation", END: END}
)
workflow.add_edge("memory_recall", "cognitive_analysis") # Default fallthrough

workflow.add_conditional_edges(
    "cognitive_analysis", router,
    {"external_validation": "external_validation", "memory_consolidation": "memory_consolidation", END: END}
)

workflow.add_edge("external_validation", "memory_consolidation")
workflow.add_edge("memory_consolidation", END)

try:
    checkpointer = SqliteSaver.from_conn_string(str(KB_PATH)) if SqliteSaver else None
    brain = workflow.compile(checkpointer=checkpointer)
    print("🧠 Neural pathways established. Neuro-Resilience engaged.\n")
except Exception as e:
    brain = workflow.compile()
    print(f"⚠ Pathways established in fallback mode: {e}\n")

# ============================================
# CORTEX API (EXPOSED FUNCTIONS)
# ============================================
async def smart_scan_async(installed_path: str, db_path: str, dry_run: bool = True):
    """Async wrapper for the blocking scan to keep API responsive"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _smart_scan_sync, installed_path, db_path, dry_run)

def _smart_scan_sync(installed_path: str, db_path: str, dry_run: bool = True):
    print(f"🚀 NEURO-SCAN INITIATED: {installed_path}")
    source = Path(installed_path)
    # target = Path(db_path) # Target used by move logic if implemented
    
    count = 0
    for fst in source.rglob("*.fst"):
        try:
            initial = AgentState(plugin_path=str(fst), plugin_name=fst.stem)
            brain.invoke(initial, {"recursion_limit": 10})
            count += 1
        except Exception as e:
             print(f"❌ Neural Failure on {fst.name}: {e}")
    print(f"🏁 Neuro-scan complete. Processed {count} entities.")

# Backward compatibility alias for API server
smart_scan = smart_scan_async

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