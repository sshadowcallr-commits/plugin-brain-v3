import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { ActivityStream } from './components/ActivityStream';

function App() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />
        <main className="flex-1 overflow-hidden relative">
            <Dashboard />
        </main>
        <ActivityStream />
      </div>
    </div>
  );
}

export default App;
