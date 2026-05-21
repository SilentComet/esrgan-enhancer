import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useAppStore } from '../lib/store';

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useAppStore();

  return (
    <button
      onClick={toggleTheme}
      className="relative p-2.5 rounded-xl border border-slate-200/60 dark:border-slate-800/80 bg-white/60 dark:bg-slate-900/60 backdrop-blur-md hover:bg-slate-100/80 dark:hover:bg-slate-800/80 hover:scale-105 active:scale-95 transition-all duration-300 group shadow-sm hover:shadow-glow-primary/20"
      aria-label="Toggle Theme"
    >
      <div className="relative w-5 h-5 overflow-hidden">
        <div
          className={`absolute inset-0 transform transition-transform duration-500 ease-spring ${
            theme === 'dark' ? 'translate-y-0 rotate-0' : '-translate-y-10 -rotate-90'
          }`}
        >
          <Sun className="w-5 h-5 text-amber-400 fill-amber-400/20 group-hover:animate-spin-slow" />
        </div>
        <div
          className={`absolute inset-0 transform transition-transform duration-500 ease-spring ${
            theme === 'light' ? 'translate-y-0 rotate-0' : 'translate-y-10 rotate-90'
          }`}
        >
          <Moon className="w-5 h-5 text-indigo-400 fill-indigo-400/20 group-hover:rotate-12 transition-transform duration-300" />
        </div>
      </div>
    </button>
  );
};
