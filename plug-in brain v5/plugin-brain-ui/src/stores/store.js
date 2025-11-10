import { create } from 'zustand';

export const usePluginStore = create((set) => ({
  plugins: [],
  loadPlugins: async () => {
    const res = await axios.get('http://localhost:8000/api/plugins');
    set({ plugins: res.data });
  },
  classifyPlugin: async (pluginData) => {
    const res = await axios.post('http://localhost:8000/api/classify', pluginData);
    set((state) => ({ plugins: [...state.plugins, res.data] }));
  },
}));