import { useState } from 'react';
import axios from 'axios';

export function GPUToggle() {
  const [mode, setMode] = useState("GPU");
  const changeMode = async (newMode) => {
    try {
      await axios.post('http://127.0.0.1:8000/set-gpu-mode', { mode: newMode });
      setMode(newMode);
      alert('Mode changed. Please restart the backend server to apply.');
    } catch (error) {
      alert('Failed to set GPU mode.');
    }
  };
  const btnClass = (m) => px-2 py-1 rounded text-xs ;
  return (
    <div className="flex space-x-2 items-center p-2">
      <span className="text-xs text-gray-400">Compute:</span>
      <button onClick={() => changeMode("CPU")} className={btnClass("CPU")}>CPU</button>
      <button onClick={() => changeMode("GPU")} className={btnClass("GPU")}>GPU</button>
    </div>
  );
}
