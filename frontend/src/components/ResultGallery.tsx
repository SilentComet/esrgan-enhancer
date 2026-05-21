import React, { useState } from 'react';
import { Download, Trash2, Maximize2, AlertCircle, RefreshCw, X } from 'lucide-react';
import { useAppStore } from '../lib/store';
import { EnhancementTask } from '../types';
import { ProgressBar } from './ProgressBar';
import { ComparisonSlider } from './ComparisonSlider';

export const ResultGallery: React.FC = () => {
  const { tasks, removeTask } = useAppStore();
  const [selectedTask, setSelectedTask] = useState<EnhancementTask | null>(null);

  const downloadImage = (task: EnhancementTask) => {
    if (!task.outputUrl) return;
    const link = document.createElement('a');
    link.href = task.outputUrl;
    // Format download name cleanly
    const dotIndex = task.inputFile.name.lastIndexOf('.');
    const baseName = dotIndex !== -1 ? task.inputFile.name.substring(0, dotIndex) : task.inputFile.name;
    link.download = `${baseName}_enhanced_x4.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'queued':
        return <RefreshCw className="w-5 h-5 text-indigo-500 animate-pulse" />;
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-indigo-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-rose-500" />;
      default:
        return null;
    }
  };

  if (tasks.length === 0) return null;

  return (
    <div className="w-full space-y-6">
      <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800/80 pb-4">
        <h3 className="text-xl font-bold tracking-tight text-slate-800 dark:text-slate-200">
          Enhancement Queue ({tasks.length})
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {tasks.map((task) => {
          const isPending = task.status === 'queued' || task.status === 'processing';
          const isDone = task.status === 'completed';
          const isFailed = task.status === 'failed';

          return (
            <div
              key={task.taskId}
              className="group relative flex flex-col justify-between overflow-hidden rounded-2xl bg-white/40 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/80 backdrop-blur-md p-5 hover:border-slate-300 dark:hover:border-slate-700/80 hover:shadow-md transition-all duration-300"
            >
              {/* Header and details */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex gap-3 items-center min-w-0">
                  <div className="flex-shrink-0">
                    {getStatusIcon(task.status)}
                    {isDone && (
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-glow-green" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 truncate" title={task.inputFile.name}>
                      {task.inputFile.name}
                    </h4>
                    <p className="text-xs text-slate-400 dark:text-slate-500 font-mono mt-0.5">
                      Task ID: {task.taskId.substring(0, 15)}...
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-1.5 flex-shrink-0">
                  {isDone && task.outputUrl && (
                    <>
                      <button
                        onClick={() => setSelectedTask(task)}
                        className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 hover:scale-105 transition-all duration-200"
                        title="Interactive Slider Compare"
                      >
                        <Maximize2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => downloadImage(task)}
                        className="p-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm hover:scale-105 transition-all duration-200"
                        title="Download Enhanced Image"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                    </>
                  )}
                  <button
                    onClick={() => removeTask(task.taskId)}
                    className="p-1.5 rounded-lg bg-slate-100 hover:bg-rose-50 dark:bg-slate-800 dark:hover:bg-rose-950/20 text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 hover:scale-105 transition-all duration-200"
                    title="Remove from List"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Central Area: Previews or Progress */}
              <div className="my-5 flex-grow">
                {isPending && (
                  <div className="py-4">
                    <ProgressBar
                      progress={task.progress}
                      status={task.status === 'queued' ? 'Queued in scheduler...' : 'Enhancing and denoising...'}
                    />
                  </div>
                )}

                {isFailed && (
                  <div className="flex items-start gap-2.5 p-3.5 bg-rose-50/50 dark:bg-rose-950/10 border border-rose-200/20 rounded-xl text-rose-600 dark:text-rose-400">
                    <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <div className="text-xs">
                      <p className="font-bold">Enhancement Failed</p>
                      <p className="mt-0.5 leading-relaxed opacity-90">{task.error}</p>
                    </div>
                  </div>
                )}

                {isDone && task.outputUrl && (
                  <div className="relative aspect-video rounded-xl overflow-hidden border border-slate-200/50 dark:border-slate-800/60 bg-slate-100 dark:bg-slate-900 group/image cursor-pointer" onClick={() => setSelectedTask(task)}>
                    <img
                      src={task.outputUrl}
                      alt="Upscaled Preview"
                      className="w-full h-full object-cover transition-transform duration-500 group-hover/image:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover/image:opacity-100 flex items-center justify-center transition-all duration-300">
                      <span className="px-3.5 py-1.5 bg-white/95 dark:bg-slate-900/95 text-slate-800 dark:text-slate-200 rounded-lg text-xs font-bold flex items-center gap-1.5 border border-slate-200/10 shadow-lg">
                        <Maximize2 className="w-3.5 h-3.5" /> Compare Before / After
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Footer details */}
              <div className="flex items-center justify-between text-xs text-slate-400 dark:text-slate-500 mt-2 font-medium">
                <span>{(task.inputFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                {task.estimatedTime && isPending && (
                  <span>Est: {task.estimatedTime}s</span>
                )}
                {isDone && (
                  <span className="text-emerald-500 font-semibold bg-emerald-50 dark:bg-emerald-950/20 px-2 py-0.5 rounded-full">
                    Completed
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison Modal Overlay */}
      {selectedTask && selectedTask.outputUrl && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-md animate-fade-in">
          <div className="relative w-full max-w-4xl bg-white dark:bg-slate-900 rounded-3xl overflow-hidden border border-slate-200/80 dark:border-slate-800/80 shadow-2xl flex flex-col max-h-[90vh]">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800/80">
              <div>
                <h3 className="font-bold text-slate-800 dark:text-slate-200">
                  Interactive Super-Resolution Inspector
                </h3>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5 truncate max-w-lg">
                  Comparing {selectedTask.inputFile.name}
                </p>
              </div>
              <button
                onClick={() => setSelectedTask(null)}
                className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body with Slider */}
            <div className="p-6 overflow-y-auto flex-grow flex items-center justify-center bg-slate-50 dark:bg-slate-950">
              <div className="w-full max-w-3xl">
                <ComparisonSlider
                  original={selectedTask.inputUrl}
                  enhanced={selectedTask.outputUrl}
                />
              </div>
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 dark:border-slate-800/80 bg-slate-50 dark:bg-slate-900/50">
              <span className="text-xs font-semibold text-slate-400 dark:text-slate-500">
                Drag the divider handle left/right to inspect enhancement details.
              </span>
              <button
                onClick={() => downloadImage(selectedTask)}
                className="px-5 py-2 text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 rounded-xl flex items-center gap-1.5 shadow-md shadow-glow-primary/20 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
              >
                <Download className="w-4 h-4" /> Download Upscaled Image
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
