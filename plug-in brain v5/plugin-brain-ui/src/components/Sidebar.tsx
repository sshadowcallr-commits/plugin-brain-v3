import { LayoutDashboard, BrainCircuit, Library, Settings, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

const items = [
    { icon: LayoutDashboard, label: 'Dashboard', active: true },
    { icon: BrainCircuit, label: 'Classify' },
    { icon: Library, label: 'Memory Bank' },
    { icon: Activity, label: 'Live Logs' },
    { icon: Settings, label: 'Settings' },
];

export function Sidebar() {
    return (
        <div className="w-20 bg-black/40 backdrop-blur-xl border-r border-white/5 flex flex-col items-center py-6 gap-4">
            {items.map((item, i) => (
                <button 
                    key={i} 
                    className={cn(
                        "w-12 h-12 flex items-center justify-center rounded-xl transition-all duration-300 group relative",
                        item.active ? "bg-neon/10 text-neon" : "text-muted-foreground hover:bg-white/5 hover:text-white"
                    )}
                >
                    <item.icon className={cn("h-6 w-6 transition-transform group-hover:scale-110", item.active && "drop-shadow-[0_0_8px_rgba(0,240,255,0.5)]")} />
                    {item.active && <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-neon rounded-l-full" />}
                </button>
            ))}
        </div>
    );
}
