import { Cpu, Search } from 'lucide-react';
import { Input } from './ui/input';

export function Header() {
  return (
    <header className="h-16 border-b border-white/5 px-6 flex items-center justify-between bg-black/20 backdrop-blur-lg">
        <div className="flex items-center gap-3">
            <div className="h-8 w-8 bg-neon/10 rounded flex items-center justify-center border border-neon/50 shadow-neon">
                <Cpu className="text-neon h-5 w-5" />
            </div>
            <h1 className="text-xl font-black tracking-wider text-white">
                PLUGIN <span className="text-neon neon-text">BRAIN</span> <span className="text-xs bg-neon/20 text-neon px-1.5 py-0.5 rounded">v5</span>
            </h1>
        </div>
        <div className="w-96 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search knowledge base..." className="pl-10 bg-black/40 border-white/5 focus-visible:bg-black/60" />
        </div>
    </header>
  );
}
