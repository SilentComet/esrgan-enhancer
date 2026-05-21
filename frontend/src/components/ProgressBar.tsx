import React from 'react';

interface ProgressBarProps {
  progress: number;
  status: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ progress, status }) => {
  // Ensure progress is bounded [0, 100]
  const percentage = Math.min(Math.max(0, progress), 100);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1.5 text-xs font-semibold tracking-wider uppercase text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></span>
          {status}
        </span>
        <span className="font-mono text-indigo-600 dark:text-indigo-400">{percentage}%</span>
      </div>
      
      <div className="w-full h-2.5 bg-slate-100 dark:bg-slate-800/80 rounded-full overflow-hidden border border-slate-200/20 shadow-inner">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
          style={{ width: `${percentage}%` }}
        >
          {/* Subtle moving shine effect */}
          <div className="absolute inset-0 bg-[linear-gradient(45deg,rgba(255,255,255,0.15)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.15)_50%,rgba(255,255,255,0.15)_75%,transparent_75%,transparent)] bg-[length:1rem_1rem] animate-[bar-stripes_1s_linear_infinite]" />
        </div>
      </div>
    </div>
  );
};
