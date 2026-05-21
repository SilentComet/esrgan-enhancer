import React, { useEffect } from 'react';
import { Sparkles, Cpu, Image as ImageIcon, Zap, ShieldCheck } from 'lucide-react';
import { useAppStore } from '../lib/store';
import { getModelInfo } from '../lib/api';
import { ThemeToggle } from '../components/ThemeToggle';
import { ImageUploader } from '../components/ImageUploader';
import { ResultGallery } from '../components/ResultGallery';

export const Home: React.FC = () => {
  const { modelInfo, setModelInfo, theme, tasks } = useAppStore();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const stats = await getModelInfo();
        setModelInfo(stats);
      } catch (err) {
        console.error('Failed to pull system ML specifications:', err);
      }
    };
    fetchStats();
  }, [setModelInfo]);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-200 transition-colors duration-300 relative overflow-hidden flex flex-col justify-between">
      {/* Decorative colored glow blur effects */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-500/10 dark:bg-indigo-500/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-500/10 dark:bg-purple-500/5 blur-[120px] pointer-events-none" />

      {/* Main Container */}
      <div className="relative z-10 flex-grow">
        {/* Header */}
        <header className="sticky top-0 z-30 w-full bg-white/40 dark:bg-slate-950/40 backdrop-blur-lg border-b border-slate-200/60 dark:border-slate-900/60">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-tr from-indigo-500 to-indigo-600 rounded-xl text-white shadow-glow-primary shadow-sm hover:rotate-12 transition-transform duration-300">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-lg font-extrabold tracking-tight bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 dark:from-white dark:via-indigo-200 dark:to-white bg-clip-text text-transparent">
                  ESRGAN Enhancer
                </h1>
                <p className="text-2xs text-slate-500 dark:text-slate-400 font-semibold tracking-widest uppercase">
                  AI-Powered Super-Resolution
                </p>
              </div>
            </div>

            <ThemeToggle />
          </div>
        </header>

        {/* Dashboard Content */}
        <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
          
          {/* Welcome Banner & System Status */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Left side: Quick overview */}
            <div className="lg:col-span-2 flex flex-col justify-center space-y-3">
              <h2 className="text-3xl lg:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
                Upscale and Restore Your Images with <span className="bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 bg-clip-text text-transparent">State-of-the-Art GANs</span>
              </h2>
              <p className="text-slate-500 dark:text-slate-400 text-sm max-w-xl leading-relaxed">
                ESRGAN Enhancer uses deep Relativistic Generative Adversarial Networks to reconstruct realistic high-frequency textures, remove noise, and magnify details by up to 8× with visual fidelity.
              </p>
            </div>

            {/* Right side: Modern System Specifications Card */}
            <div className="p-5 rounded-2xl bg-white/40 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/80 backdrop-blur-md shadow-sm space-y-4">
              <h4 className="font-bold text-xs text-slate-400 dark:text-slate-500 uppercase tracking-widest flex items-center gap-1.5 border-b border-slate-100 dark:border-slate-800 pb-2">
                <Cpu className="w-3.5 h-3.5 text-indigo-500" />
                ML Backend Environment
              </h4>
              
              {modelInfo ? (
                <div className="grid grid-cols-2 gap-y-3 gap-x-4 text-xs font-medium">
                  <div>
                    <span className="text-slate-400 dark:text-slate-500 block mb-0.5">Execution Device</span>
                    <span className="text-slate-700 dark:text-slate-300 font-bold bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 dark:text-indigo-400 px-2 py-0.5 rounded-md inline-block">
                      {modelInfo.device.toUpperCase()}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 dark:text-slate-500 block mb-0.5">CUDA Support</span>
                    <span className={`font-bold px-2 py-0.5 rounded-md inline-block ${
                      modelInfo.cuda_available 
                        ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400' 
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                    }`}>
                      {modelInfo.cuda_available ? 'ONLINE' : 'UNAVAILABLE'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 dark:text-slate-500 block mb-0.5">Network Precision</span>
                    <span className="text-slate-700 dark:text-slate-300 font-mono">
                      {modelInfo.precision}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 dark:text-slate-500 block mb-0.5">Engine Version</span>
                    <span className="text-slate-700 dark:text-slate-300 font-mono">
                      v{modelInfo.version}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 py-1 animate-pulse">
                  <div className="h-3.5 bg-slate-200 dark:bg-slate-800 rounded w-2/3" />
                  <div className="h-3.5 bg-slate-200 dark:bg-slate-800 rounded w-1/2" />
                  <div className="h-3.5 bg-slate-200 dark:bg-slate-800 rounded w-3/4" />
                </div>
              )}
            </div>
          </div>

          {/* Grid Layout: Control Panel vs Results */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Uploader Section */}
            <div className="lg:col-span-5 space-y-6">
              <ImageUploader />
              
              {/* Features breakdown panel */}
              <div className="p-5 rounded-2xl bg-white/20 dark:bg-slate-900/10 border border-slate-200/40 dark:border-slate-800/40 backdrop-blur-xs space-y-3.5 text-xs text-slate-500 dark:text-slate-400">
                <div className="flex gap-3">
                  <ImageIcon className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-bold text-slate-700 dark:text-slate-300">Intelligent Denoising</p>
                    <p className="mt-0.5">Cleans JPEG artifacts and random noise while retaining structural edges.</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Zap className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-bold text-slate-700 dark:text-slate-300">Asynchronous Processing</p>
                    <p className="mt-0.5">Powered by a background Celery task queue for seamless scaling without locking.</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <ShieldCheck className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-bold text-slate-700 dark:text-slate-300">Secure Storage Sandbox</p>
                    <p className="mt-0.5">Images are validated recursively and auto-purged from the server within 24 hours.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Results Section */}
            <div className="lg:col-span-7">
              {tasks.length > 0 ? (
                <ResultGallery />
              ) : (
                <div className="h-full min-h-[350px] flex flex-col items-center justify-center text-center p-8 rounded-2xl border-2 border-dashed border-slate-200 dark:border-slate-800 bg-white/20 dark:bg-slate-900/10 backdrop-blur-xs">
                  <div className="p-4 bg-slate-100 dark:bg-slate-900 rounded-full text-slate-400 dark:text-slate-600 mb-4 border border-slate-200/20 shadow-sm animate-pulse-slow">
                    <Sparkles className="w-8 h-8" />
                  </div>
                  <h4 className="font-bold text-slate-800 dark:text-slate-200">No active enhancement tasks</h4>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 max-w-xs leading-relaxed">
                    Upload images using the drag-and-drop zone to start the super-resolution ML pipeline.
                  </p>
                </div>
              )}
            </div>

          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="w-full bg-white dark:bg-slate-950 border-t border-slate-200/60 dark:border-slate-900/60 py-6 mt-16 text-center text-xs text-slate-500 dark:text-slate-500 z-10 relative">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>© 2026 ESRGAN Image Enhancement Platform. All rights reserved.</p>
          <div className="flex items-center gap-4 font-semibold">
            <span className="hover:text-indigo-500 transition-colors cursor-help">Relativistic GAN (RaGAN)</span>
            <span>•</span>
            <span className="hover:text-indigo-500 transition-colors cursor-help">FastAPI + React</span>
            <span>•</span>
            <span className="hover:text-indigo-500 transition-colors cursor-help">Dockerized Infrastructure</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default Home;
