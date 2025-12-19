/**
 * WorldControls - Control bar for world viewer
 */

import React from 'react';

interface WorldControlsProps {
  isConnected: boolean;
  isPaused: boolean;
  isFullscreen: boolean;
  latency: number | null;
  frameCount: number;
  onPauseToggle: () => void;
  onFullscreenToggle: () => void;
}

const WorldControls: React.FC<WorldControlsProps> = ({
  isConnected,
  isPaused,
  isFullscreen,
  latency,
  frameCount,
  onPauseToggle,
  onFullscreenToggle,
}) => {
  return (
    <div className="absolute top-0 left-0 right-0 p-3 flex items-center justify-between bg-gradient-to-b from-black/60 to-transparent">
      {/* Left: Status */}
      <div className="flex items-center gap-3">
        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}
          />
          <span className="text-white text-xs">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>

        {/* Latency */}
        {latency !== null && (
          <div className="text-gray-400 text-xs">
            {latency.toFixed(0)}ms
          </div>
        )}

        {/* Frame Count */}
        <div className="text-gray-400 text-xs">
          Frame: {frameCount}
        </div>
      </div>

      {/* Right: Control Buttons */}
      <div className="flex items-center gap-2">
        {/* Pause/Play Button */}
        <button
          onClick={onPauseToggle}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          title={isPaused ? 'Resume' : 'Pause'}
        >
          {isPaused ? (
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
            </svg>
          )}
        </button>

        {/* Fullscreen Button */}
        <button
          onClick={onFullscreenToggle}
          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
          title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        >
          {isFullscreen ? (
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
};

export default WorldControls;
