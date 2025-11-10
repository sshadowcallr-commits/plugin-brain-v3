import { useEffect, useState, useRef } from 'react';
import { Terminal } from 'lucide-react';

export function ActivityStream() {
  const [logs, setLogs] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Simulated logs for instant gratification before backend connects
    setLogs([
        '[SYSTEM] Neuro-Core initialized.',
        '[GPU] CUDA detected: NVIDIA GeForce GTX 1060 6GB',
        '[MOA] Agents ready: Gemma-2, Phi-3, Qwen2',
        '[WAITING] Awaiting scan command...'
    ]);

    const interval = setInterval(() => {
      fetch('http://127.0.0.1:8000/logs/tail') // We will add this endpoint
        .then(res => res.json())
        .then(data => { if(data.logs) setLogs(prev => [...prev.slice(-50), ...data.logs]); })
        .catch(() => {}); // Silent fail if backend offline
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="h-48 bg-black/60 border-t border-white/5 backdrop-blur-md flex flex-col">
        <div className="px-4 py-2 bg-black/40 flex items-center gap-2 text-xs font-mono text-muted-foreground border-b border-white/5">
            <Terminal size={12} className="text-neon" /> NEURO-ACTIVITY STREAM
        </div>
        <div className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1">
            {logs.map((log, i) => (
                <div key={i} className="text-green-500/80 border-l-2 border-transparent hover:border-neon pl-2 transition-all">
                    <span className="opacity-50 mr-2">{new Date().toLocaleTimeString()}</span>
                    {log}
                </div>
            ))}
            <div ref={endRef} />
        </div>
    </div>
  );
}
