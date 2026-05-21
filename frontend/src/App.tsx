import React, { useEffect } from 'react';
import { useAppStore } from './lib/store';
import { Home } from './pages/Home';

export const App: React.FC = () => {
  const { theme } = useAppStore();

  useEffect(() => {
    // Initialise document class matching active store state on boot
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return <Home />;
};

export default App;
