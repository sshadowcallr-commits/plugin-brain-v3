import { usePluginStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const Dashboard = () => {
  const { plugins, loadPlugins } = usePluginStore();

  useQuery(['plugins'], loadPlugins, { enabled: plugins.length === 0 });

  return (
    <div className="p-4 bg-gray-100 dark:bg-gray-900">
      <h1 className="text-2xl font-bold">Plugin Dashboard</h1>
      <ul className="mt-4">
        {plugins.map((plugin) => (
          <li key={plugin.id} className="p-2 border-b">
            {plugin.name} - Classified as: {plugin.category}
          </li>
        ))}
      </ul>
      <button onClick={loadPlugins} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
        Refresh Plugins
      </button>
    </div>
  );
};

export default Dashboard;