import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Sparkles, AlertCircle } from 'lucide-react';
import { useAppStore } from '../lib/store';
import { enhanceImage, getTaskStatus, getResultUrl } from '../lib/api';
import { EnhancementTask } from '../types';

export const ImageUploader: React.FC = () => {
  const { scaleFactor, setScaleFactor, addTask, updateTask } = useAppStore();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    for (const file of acceptedFiles) {
      // Validate file size (10MB limit)
      const maxSize = 10 * 1024 * 1024;
      if (file.size > maxSize) {
        alert(`File ${file.name} exceeds the maximum size limit of 10MB.`);
        continue;
      }

      // Generate a temporary local task ID and input object URL
      const tempTaskId = `task_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      const inputUrl = URL.createObjectURL(file);

      const newTask: EnhancementTask = {
        taskId: tempTaskId,
        inputFile: file,
        inputUrl,
        status: 'queued',
        progress: 0,
      };

      addTask(newTask);

      try {
        // Trigger server upload and enhancement
        const data = await enhanceImage(file, scaleFactor);
        const serverTaskId = data.task_id;
        const estimatedTime = data.estimated_time_seconds;

        // Re-map the task to its server side taskId and initial status
        updateTask(tempTaskId, {
          taskId: serverTaskId,
          status: 'processing',
          estimatedTime,
        });

        // Initialize background status polling
        pollStatus(serverTaskId);
      } catch (err: any) {
        console.error('Enhancement upload failure:', err);
        const errMsg = err.response?.data?.detail || err.message || 'Server upload failed';
        updateTask(tempTaskId, {
          status: 'failed',
          error: errMsg,
        });
      }
    }
  }, [scaleFactor, addTask, updateTask]);

  const pollStatus = (taskId: string) => {
    const interval = 2000;
    const maxRetries = 150; // 5 minutes
    let retries = 0;

    const check = async () => {
      try {
        const response = await getTaskStatus(taskId);
        const { status, progress, error } = response;

        updateTask(taskId, {
          status,
          progress: progress || 0,
        });

        if (status === 'completed') {
          const outputUrl = getResultUrl(taskId);
          updateTask(taskId, {
            outputUrl,
            progress: 100,
          });
        } else if (status === 'failed') {
          updateTask(taskId, {
            error: error || 'Enhancement processing failed on worker.',
          });
        } else if (retries < maxRetries) {
          retries++;
          setTimeout(check, interval);
        } else {
          updateTask(taskId, {
            status: 'failed',
            error: 'Enhancement operation timed out.',
          });
        }
      } catch (err) {
        console.error('Polling failure for task:', taskId, err);
        if (retries < maxRetries) {
          retries++;
          setTimeout(check, interval);
        }
      }
    };

    setTimeout(check, interval);
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
    },
    maxFiles: 5,
  });

  return (
    <div className="w-full space-y-6">
      {/* Drop Zone Box */}
      <div
        {...getRootProps()}
        className={`relative overflow-hidden group border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 backdrop-blur-md ${
          isDragActive
            ? 'border-indigo-500 bg-indigo-500/5 dark:bg-indigo-500/10 shadow-glow-primary'
            : 'border-slate-300 dark:border-slate-700/80 bg-white/40 dark:bg-slate-900/40 hover:border-indigo-400 dark:hover:border-indigo-500 hover:bg-white/60 dark:hover:bg-slate-900/60'
        }`}
      >
        <input {...getInputProps()} />
        
        {/* Subtle mesh background grid details */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:14px_24px] pointer-events-none" />

        <div className="relative z-10 flex flex-col items-center">
          <div className="p-4 bg-indigo-50 dark:bg-indigo-950/50 rounded-2xl text-indigo-500 dark:text-indigo-400 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300 shadow-sm border border-slate-100 dark:border-slate-800">
            <Upload className="w-8 h-8" />
          </div>
          
          <h3 className="mt-4 text-lg font-bold tracking-tight text-slate-800 dark:text-slate-200">
            {isDragActive ? 'Release to drop your images' : 'Upload images for AI enhancement'}
          </h3>
          
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400 max-w-sm">
            Drag and drop your images here, or click to browse. Supports JPG, PNG, and WEBP.
          </p>

          <div className="mt-6 flex flex-wrap justify-center items-center gap-4 text-xs font-semibold text-slate-400 dark:text-slate-500">
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-100/60 dark:bg-slate-800/40 rounded-full border border-slate-200/10">
              Max 10MB per file
            </span>
            <span>•</span>
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-100/60 dark:bg-slate-800/40 rounded-full border border-slate-200/10">
              Up to 5 files at once
            </span>
          </div>
        </div>

        {isDragReject && (
          <div className="absolute inset-0 bg-rose-500/10 dark:bg-rose-500/20 backdrop-blur-sm flex flex-col items-center justify-center gap-2 border border-rose-500">
            <AlertCircle className="w-10 h-10 text-rose-500" />
            <span className="font-bold text-rose-500 text-sm">Unsupported file format selected.</span>
          </div>
        )}
      </div>

      {/* Control Panel (Scale settings) */}
      <div className="p-5 rounded-2xl bg-white/40 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/80 backdrop-blur-md shadow-sm">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h4 className="font-bold text-sm text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-indigo-500" />
              Upscaling Scale Factor
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Choose the resolution scaling factor powered by GAN.
            </p>
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto p-1 bg-slate-100/80 dark:bg-slate-950/60 rounded-xl border border-slate-200/20">
            {[2, 4, 8].map((scale) => (
              <button
                key={scale}
                onClick={() => setScaleFactor(scale)}
                className={`flex-1 sm:flex-initial px-5 py-2 text-xs font-bold rounded-lg transition-all duration-300 ${
                  scaleFactor === scale
                    ? 'bg-gradient-to-r from-indigo-500 to-indigo-600 text-white shadow-glow-primary/30 shadow-md'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800/30'
                }`}
              >
                {scale}×
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
