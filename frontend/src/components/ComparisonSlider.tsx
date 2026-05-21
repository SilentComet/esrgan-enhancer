import React, { useState, useRef, useEffect } from 'react';

interface ComparisonSliderProps {
  original: string;
  enhanced: string;
  originalLabel?: string;
  enhancedLabel?: string;
}

export const ComparisonSlider: React.FC<ComparisonSliderProps> = ({
  original,
  enhanced,
  originalLabel = 'Original',
  enhancedLabel = 'Upscaled AI',
}) => {
  const [sliderPosition, setSliderPosition] = useState<number>(50); // 0 to 100 percentage
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e: TouchEvent) => {
    if (!isDragging) return;
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove, { passive: true });
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging]);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onTouchStart = () => {
    setIsDragging(true);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-video rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 shadow-lg select-none cursor-ew-resize bg-slate-100 dark:bg-slate-900"
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
    >
      {/* Enhanced Image (Underlay / Background) */}
      <img
        src={enhanced}
        alt="Enhanced"
        className="absolute inset-0 w-full h-full object-cover"
        draggable={false}
      />
      <span className="absolute bottom-3 right-3 z-10 px-2.5 py-1 text-xs font-bold bg-slate-950/80 text-white rounded-md backdrop-blur-sm pointer-events-none uppercase tracking-widest border border-white/10">
        {enhancedLabel}
      </span>

      {/* Original Image (Overlay / Cropped) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${sliderPosition}%` }}
      >
        <img
          src={original}
          alt="Original"
          className="absolute inset-0 w-full h-full object-cover"
          style={{ width: containerRef.current?.getBoundingClientRect().width }}
          draggable={false}
        />
        <span className="absolute bottom-3 left-3 z-10 px-2.5 py-1 text-xs font-bold bg-slate-950/80 text-white rounded-md backdrop-blur-sm pointer-events-none uppercase tracking-widest border border-white/10">
          {originalLabel}
        </span>
      </div>

      {/* Interactive Divider Line & Handle */}
      <div
        className="absolute inset-y-0 w-1 bg-white shadow-lg cursor-ew-resize select-none pointer-events-none transition-shadow"
        style={{ left: `${sliderPosition}%` }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white dark:bg-slate-950 text-slate-800 dark:text-white border-2 border-indigo-500 shadow-glow-primary flex items-center justify-center font-bold text-lg select-none">
          <svg
            className="w-4 h-4 fill-current rotate-90"
            viewBox="0 0 24 24"
          >
            <path d="M12 2A10 10 0 0 0 2 12a10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2m0 2a8 8 0 0 1 8 8 8 8 0 0 1-8 8 8 8 0 0 1-8-8 8 8 0 0 1 8-8m-1 3v3H8l4 4 4-4h-3V7h-2Z" />
          </svg>
        </div>
      </div>
    </div>
  );
};
