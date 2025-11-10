from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from moa.llm_core import smart_scan, editor_move, conn
# We need to import the model loaders to reload them
from moa.llm_core import load_expert, MODEL_DIR

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ScanReq(BaseModel): installed_path: str; db_path: str; dry_run: bool = True
class MoveReq(BaseModel): plugin_name: str; old_fst_path: str; new_fst_path: str

@app.post("/set-mode")
async def set_mode(mode: str):
    # Simple placeholder re-load logic. 
    # In a full prod app, we'd update the global model variables in llm_core.
    print(f" NEURO-CORE: Switching inference mode to {mode}")
    return {"status": f"Mode switched to {mode} (Simulated for now - restart backend to fully apply)"}

@app.post("/scan/start")
async def scan(req: ScanReq, bg: BackgroundTasks):
    bg.add_task(smart_scan, req.installed_path, req.db_path, req.dry_run)
    return {"status": "started"}

@app.get("/plugins")
async def plugins(low_conf_only: bool = False):
    sql = "SELECT name, category, confidence, path_schema, channels FROM plugins"
    if low_conf_only: sql += " WHERE confidence < 0.8"
    rows = conn.execute(sql).fetchall()
    return [{"name":r[0],"cat":r[1],"conf":r[2],"path":r[3],"channels":r[4]} for r in rows]

@app.post("/bulk-approve")
async def approve(names: List[str]):
    for n in names:
        conn.execute("UPDATE plugins SET confidence=1.0, source='user' WHERE name=?", (n,))
    conn.commit()
    return {"approved": len(names)}

@app.post("/editor-move")
async def move(req: MoveReq):
    return {"success": editor_move(req.plugin_name, req.old_fst_path, req.new_fst_path)}
