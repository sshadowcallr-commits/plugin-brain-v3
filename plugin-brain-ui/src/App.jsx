// src/App.jsx
// This is the complete, fixed file with proxy support.

import React, { useState, useEffect, useRef } from "react";
import {
  Brain,
  Settings,
  Activity,
  Zap,
  FileText,
  ChevronRight,
  Play,
  Pause,
  Database,
  TrendingUp,
  Search,
} from "lucide-react";

// --- COMPONENTS ---
const Toggle = ({ label, active, onToggle }) => (
  <div className="flex items-center gap-3">
    <span className="text-sm text-gray-400 uppercase tracking-wide">{label}</span>
    <button
      onClick={onToggle}
      className={`w-14 h-7 rounded-full transition-all relative ${
        active ? "bg-cyan-500" : "bg-gray-700"
      }`}
    >
      <div
        className={`w-5 h-5 rounded-full bg-white absolute top-1 transition-all ${
          active ? "left-8" : "left-1"
        }`}
      ></div>
    </button>
  </div>
);

const AgentStatusBar = ({ agents }) => (
  <div className="container mx-auto px-6">
    <div className="flex items-center gap-8 mt-6 text-xs flex-wrap pb-4">
      {agents.map((a, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              a.active ? "bg-cyan-400 animate-pulse" : "bg-gray-600"
            }`}
          ></div>
          <span className="text-gray-400">{a.name}:</span>
          <span className={a.active ? "text-cyan-400 font-medium" : "text-gray-500"}>
            {a.status}
          </span>
        </div>
      ))}
    </div>
  </div>
);

const PluginCard = ({ plugin, selected, onSelect }) => (
  <div
    onClick={() => onSelect(plugin)}
    className={`bg-gray-900/50 backdrop-blur-xl rounded-xl border p-5 cursor-pointer transition-all hover:scale-[1.02] ${
      selected?.path === plugin.path
        ? "border-cyan-500 shadow-lg shadow-cyan-500/20"
        : "border-cyan-500/20 hover:border-cyan-500/50"
    }`}
  >
    <div className="flex items-start justify-between mb-3">
      <span className="text-3xl">{plugin.format === 'FST' ? '🎹' : '🎛️'}</span>
      <div
        className={`w-2 h-2 rounded-full ${
           plugin.version !== "Unknown" ? "bg-cyan-400 animate-pulse" : "bg-gray-500"
        }`}
      ></div>
    </div>
    <h3 className="text-base font-semibold text-white mb-2 truncate" title={plugin.name}>{plugin.name}</h3>
    <p className="text-xs text-gray-400 mb-1 truncate">{plugin.vendor}</p>
    <p className="text-xs text-cyan-400 font-medium">{plugin.version !== "Unknown" ? plugin.version : "Standard"}</p>
    <p className="text-xs text-gray-500 mt-1 truncate" title={plugin.path}>{plugin.path}</p>
  </div>
);

const StatsCard = ({ title, value, icon: Icon, color }) => (
  <div className={`bg-gradient-to-br ${color} rounded-xl p-5 border border-white/10`}>
    <div className="flex items-center justify-between mb-3">
      <Icon className="w-6 h-6 text-white/80" />
      <div className="w-3 h-3 rounded-full bg-white/30 animate-pulse"></div>
    </div>
    <p className="text-3xl font-bold text-white mb-1">{value}</p>
    <p className="text-sm text-white/70">{title}</p>
  </div>
);

// Placeholder activity chart
const ActivityChart = () => (
  <div className="bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-cyan-500/20 p-6">
    <h3 className="text-sm text-gray-400 mb-4 uppercase tracking-wide">System Activity Timeline</h3>
    <div className="h-48 relative flex items-center justify-center border border-dashed border-gray-700 rounded-lg">
       <p className="text-gray-500 text-sm">Activity visualization ready for live data</p>
    </div>
  </div>
);

// --- MAIN APP ---
export default function PluginBrain() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [aiActive] = useState(true);
  const [smartMode, setSmartMode] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [classifying, setClassifying] = useState(false);
  const [selectedPlugin, setSelectedPlugin] = useState(null);

  // Data from Python
  const [plugins, setPlugins] = useState([]);
  const [consoleLog, setConsoleLog] = useState([{ type: 'info', text: 'Ready to connect to Python Brain.'}]);
  const [config, setConfig] = useState({ installed_path: '', target_root: '' });
  const logEndRef = useRef(null);

  // --- API ACTIONS ---
  const log = (text, type="info") => {
      setConsoleLog(prev => [...prev.slice(-100), { type, text }]); // Keep last 100 logs
  }

  // Scroll log to bottom
  useEffect(() => {
      logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [consoleLog]);

  // Initial connection check
  useEffect(() => {
      fetch(`/api/config`) // Use proxy
          .then(res => {
              if (!res.ok) throw new Error("Backend not responding");
              return res.json();
          })
          .then(data => {
              setConfig(data);
              log(`Connected to backend. Installed path: ${data.installed_path}`, "success");
          })
          .catch(err => log(`Failed to connect to backend: ${err.message}`, "error"));
  }, []);

  const handleScan = async () => {
      setScanning(true);
      log("Starting scan...", "info");
      setSelectedPlugin(null);
      setPlugins([]);
      try {
          const res = await fetch(`/api/scan`); // Use proxy
          if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
          const data = await res.json();
          setPlugins(data);
          log(`Scan complete. Found ${data.length} plugins.`, "success");
      } catch (e) {
          log(`Scan error: ${e.message}`, "error");
      } finally {
          setScanning(false);
      }
  };

  const handleClassify = async () => {
      setClassifying(true);
      log("Starting classification (this may take a while)...", "info");
      try {
          const res = await fetch(`/api/classify-all`, { method: 'POST' }); // Use proxy
           if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
          const data = await res.json();
          log(data.message, "success");
          if (data.errors > 0) log(`Finished with ${data.errors} errors. Check Python console.`, "warning");
          // Auto-refresh the list after classifying
          handleScan();
      } catch (e) {
           log(`Classification error: ${e.message}`, "error");
      } finally {
          setClassifying(false);
      }
  };
  
  const handleEditorMove = async () => {
      const newPath = document.getElementById('newPathInput')?.value;
      if (!selectedPlugin || !newPath) {
          log("Select a plugin and enter a new path to move.", "warning");
          return;
      }

      log(`Attempting editor move to: ${newPath}...`, "info");
      try {
           const res = await fetch(`/api/editor-move`, { // Use proxy
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ plugin_path: selectedPlugin.path, new_relative_path: newPath })
           });
           if(!res.ok) throw new Error((await res.json()).detail || res.statusText);
           const data = await res.json();
           log(data.message, "success");
           // remove from local list
           setPlugins(prev => prev.filter(p => p.path !== selectedPlugin.path));
           setSelectedPlugin(null);
      } catch (e) {
          log(`Move failed: ${e.message}`, "error");
      }
  }

  const agents = [
    { name: "Gemma Brain", status: "Online", active: true },
    { name: "Mistral Scanner", status: "Ready", active: true },
    { name: "Phi-3 Enricher", status: "Idle", active: true },
    { name: "File Agent", status: "Standing By", active: true },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-slate-900 to-gray-950 text-white font-sans flex flex-col">
      {/* HEADER */}
      <header className="border-b border-cyan-500/20 bg-gray-900/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-30 rounded-full"></div>
                <Brain className="w-10 h-10 text-cyan-400 relative" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  PLUGIN BRAIN v3
                </h1>
                <p className="text-xs text-gray-400 mt-0.5">AI-Driven Neural Organizer</p>
              </div>
            </div>
             <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${aiActive ? "bg-cyan-400 animate-pulse" : "bg-gray-600"}`}></div>
                <span className="text-gray-300">AI Brain {aiActive ? "ONLINE" : "OFFLINE"}</span>
              </div>
        </div>
        <AgentStatusBar agents={agents} />
      </header>

      {/* MAIN LAYOUT */}
      <div className="flex-1 container mx-auto px-6 py-8 grid grid-cols-12 gap-6 overflow-hidden">
          
          {/* SIDEBAR */}
          <aside className="col-span-2 space-y-2">
            {[
              { id: "dashboard", icon: Activity, label: "Dashboard" },
              { id: "classify", icon: Zap, label: "Classify" },
              { id: "editor", icon: FileText, label: "Editor" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full px-4 py-3 rounded-lg flex items-center gap-3 transition-all ${
                  activeTab === tab.id
                    ? "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400"
                    : "bg-gray-800/30 text-gray-400 hover:bg-gray-800/50"
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span className="text-sm font-medium">{tab.label}</span>
              </button>
            ))}
          </aside>

          {/* MAIN PANEL */}
          <main className="col-span-7 space-y-6 flex flex-col">
            {activeTab === "dashboard" && (
              <div className="space-y-6 animate-in fade-in duration-500">
                 <div className="grid grid-cols-3 gap-4">
                  <StatsCard title="Total Plugins Found" value={plugins.length} icon={Database} color="from-blue-600/80 to-blue-900/80" />
                  <StatsCard title="Classification Rate" value={"N/A"} icon={TrendingUp} color="from-cyan-600/80 to-cyan-900/80" />
                  <StatsCard title="System Status" value={scanning ? "SCANNING" : classifying ? "CLASSIFYING" : "READY"} icon={Activity} color="from-purple-600/80 to-purple-900/80" />
                </div>
                <ActivityChart />
              </div>
            )}

            {activeTab === "classify" && (
               <div className="space-y-4 animate-in fade-in duration-500 flex flex-col flex-1">
                   <div className="flex justify-between items-center">
                     <h2 className="text-xl font-semibold text-cyan-400">Plugin Grid ({plugins.length})</h2>
                      <div className="relative w-64">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                        <input type="text" placeholder="Filter plugins..." className="w-full pl-10 pr-4 py-2 bg-gray-800/50 rounded-lg border border-cyan-500/30 text-sm focus:outline-none focus:border-cyan-500/70 transition-colors" />
                      </div>
                   </div>

                   {/* PLUGIN GRID SCROLL AREA */}
                   <div className="flex-1 grid grid-cols-2 gap-4 overflow-y-auto pr-2" style={{maxHeight: 'calc(100vh - 300px)'}}>
                        {plugins.length === 0 && !scanning && (
                             <div className="col-span-2 h-40 flex items-center justify-center border border-dashed border-gray-700 rounded-xl text-gray-500">
                                 No plugins loaded. Click "Start Scan" on the right.
                             </div>
                        )}
                        {plugins.map((p) => (
                            <PluginCard key={p.path} plugin={p} selected={selectedPlugin} onSelect={setSelectedPlugin} />
                        ))}
                   </div>
                   {scanning && <div className="h-1 w-full bg-gray-800 overflow-hidden rounded-full"><div className="h-full bg-cyan-500 animate-pulse w-1/2 mx-auto rounded-full"></div></div>}
               </div>
            )}

            {activeTab === "editor" && (
                <div className="bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-cyan-500/20 p-8 animate-in fade-in duration-500">
                    <h2 className="text-xl font-semibold text-cyan-400 mb-6">Manual Editor</h2>
                    {selectedPlugin ? (
                        <div className="space-y-6">
                             <div>
                                <label className="text-sm text-gray-400">Selected Plugin</label>
                                <input type="text" value={`${selectedPlugin.name} (${selectedPlugin.version})`} readOnly className="w-full mt-1 px-4 py-2 bg-black/50 border border-gray-700 rounded-lg text-gray-300" />
                             </div>
                             <div>
                                 <label className="text-sm text-gray-400">Current Location</label>
                                 <p className="text-xs text-gray-500 font-mono break-all mt-1">{selectedPlugin.path}</p>
                             </div>
                             <div>
                                <label className="text-sm text-cyan-400">Move to New Relative Path</label>
                                <input type="text" id="newPathInput" placeholder="e.g., Effects/EQ/MyFolder" className="w-full mt-1 px-4 py-3 bg-gray-900 border border-cyan-500/50 rounded-lg focus:outline-none focus:border-cyan-400 transition-colors" />
                                <p className="text-xs text-gray-500 mt-2">This will physically move the .fst file to: <br/><span className="font-mono text-gray-400">{config.target_root}\&lt;your path&gt;</span></p>
                             </div>
                             <button 
                                onClick={handleEditorMove}
                                className="w-full py-3 bg-red-600 hover:bg-red-500 rounded-lg font-bold transition-colors"
                             >
                                 EXECUTE MOVE
                             </button>
                        </div>
                    ) : (
                        <div className="text-center text-gray-500 py-20">
                            Select a plugin from the 'Classify' tab to edit it here.
                        </div>
                    )}
                </div>
            )}

          </main>

          {/* RIGHT CONTROL PANEL */}
          <aside className="col-span-3 space-y-6 flex flex-col h-full">
              {/* BRAIN CONSOLE */}
              <div className="bg-black/40 backdrop-blur-xl rounded-2xl border border-cyan-500/30 p-5 flex-1 flex flex-col">
                  <h3 className="text-sm text-cyan-400 mb-3 uppercase tracking-widest font-bold">Brain Console</h3>
                  
                  {/* Log Output Area */}
                  <div className="flex-1 bg-black/50 rounded-lg p-3 mb-4 overflow-y-auto font-mono text-xs space-y-1" style={{minHeight: '200px'}}>
                      {consoleLog.map((log, i) => (
                          <div key={i} className={`${log.type === 'error' ? 'text-red-400' : log.type === 'success' ? 'text-green-400' : log.type === 'warning' ? 'text-yellow-400' : 'text-cyan-300/70'}`}>
                              <span className="opacity-50 mr-2">&gt;</span>{log.text}
                          </div>
                      ))}
                      <div ref={logEndRef} />
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-3">
                      <button
                        onClick={handleScan}
                        disabled={scanning || classifying}
                        className={`w-full py-3 rounded-lg border font-bold flex items-center justify-center gap-2 transition-all ${scanning ? 'bg-gray-800 border-gray-700 text-gray-500' : 'bg-cyan-950/50 border-cyan-500/50 hover:bg-cyan-900/50 text-cyan-400'} disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                          {scanning ? <><Pause className="w-4 h-4 animate-pulse"/><span>SCANNING...</span></> : <><Play className="w-4 h-4"/><span>START SCAN</span></>}
                      </button>

                       <button
                        onClick={handleClassify}
                        disabled={scanning || classifying || plugins.length === 0}
                        className={`w-full py-3 rounded-lg border font-bold flex items-center justify-center gap-2 transition-all ${classifying ? 'bg-gray-800 border-gray-700 text-gray-500' : 'bg-blue-950/50 border-blue-500/50 hover:bg-blue-900/50 text-blue-400'} disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                          {classifying ? <><Activity className="w-4 h-4 animate-pulse"/><span>CLASSIFYING...</span></> : <><Zap className="w-4 h-4"/><span>CLASSIFY ALL</span></>}
                      </button>
                  </div>
              </div>

               {/* Config Info */}
               <div className="bg-gray-900/50 backdrop-blur-xl rounded-xl border border-white/5 p-4 text-xs">
                  <h4 className="text-gray-400 mb-2 uppercase">Active Configuration</h4>
                   <div className="space-y-2">
                       <div>
                           <p className="text-gray-600">Source (Installed)</p>
                           <p className="text-gray-400 truncate" title={config.installed_path}>{config.installed_path || "Loading..."}</p>
                       </div>
                       <div>
                           <p className="text-gray-600">Target (Organized)</p>
                           <p className="text-gray-400 truncate" title={config.target_root}>{config.target_root || "Loading..."}</p>
                       </div>
                   </div>
               </div>
          </aside>
      </div>
    </div>
  );
}