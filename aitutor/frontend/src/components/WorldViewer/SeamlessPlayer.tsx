/**
 * SeamlessPlayer - Video player for pre-generated worlds
 *
 * Uses dual video element switching for seamless transitions between segments.
 * Provides interactive controls and AI tutor integration.
 */

import React, { useCallback, useState, useEffect } from 'react';
import { useWorldPlayback, UseWorldPlaybackOptions } from '../../hooks/useWorldPlayback';
import { Action } from '../../services/worldPlayback';

export interface SeamlessPlayerProps {
  worldId: string;
  className?: string;
  showControls?: boolean;
  showNavigationHints?: boolean;
  autoPlay?: boolean;
  onPositionChange?: (position: string, description: string) => void;
  onReady?: () => void;
  onError?: (error: string) => void;
}

const SeamlessPlayer: React.FC<SeamlessPlayerProps> = ({
  worldId,
  className = '',
  showControls = true,
  showNavigationHints = true,
  autoPlay = true,
  onPositionChange,
  onReady,
  onError,
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPaused, setIsPaused] = useState(!autoPlay);
  const [showHelpOverlay, setShowHelpOverlay] = useState(false);

  const options: UseWorldPlaybackOptions = {
    autoPlay,
    onPositionChange,
    onReady,
    onError,
  };

  const {
    containerRef,
    isReady,
    error,
    state,
    metadata,
    currentPosition,
    positionDescription,
    handleAction,
    play,
    pause,
  } = useWorldPlayback(worldId, options);

  // Handle play/pause toggle
  const togglePause = useCallback(async () => {
    if (isPaused) {
      await play();
      setIsPaused(false);
    } else {
      pause();
      setIsPaused(true);
    }
  }, [isPaused, play, pause]);

  // Handle fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    if (!isFullscreen) {
      container.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, [isFullscreen, containerRef]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Direction button click handler
  const handleDirectionClick = useCallback(
    async (action: Action) => {
      await handleAction(action);
    },
    [handleAction]
  );

  return (
    <div
      ref={containerRef as React.RefObject<HTMLDivElement>}
      className={`seamless-player relative bg-black rounded-lg overflow-hidden ${className}`}
      style={{ position: 'relative', minHeight: '400px' }}
      tabIndex={0}
    >
      {/* Videos are dynamically inserted here by the engine */}

      {/* Loading State */}
      {!isReady && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-white border-t-transparent mx-auto mb-4" />
            <p className="text-white text-lg">Loading {metadata?.name || worldId}...</p>
            <p className="text-gray-400 text-sm mt-2">Preparing your immersive experience</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
          <div className="text-center text-white max-w-md">
            <div className="text-red-500 text-4xl mb-4">!</div>
            <p className="text-xl mb-2">Unable to load world</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Position Description Overlay */}
      {isReady && positionDescription && (
        <div className="absolute bottom-20 left-4 right-4 z-20">
          <div className="bg-black/70 backdrop-blur-sm text-white px-4 py-3 rounded-lg max-w-lg">
            <p className="text-sm text-gray-300">Current view:</p>
            <p className="text-lg">{positionDescription}</p>
          </div>
        </div>
      )}

      {/* Navigation Controls */}
      {showNavigationHints && isReady && (
        <div className="absolute bottom-4 right-4 z-20">
          <div className="bg-black/50 backdrop-blur-sm rounded-lg p-3">
            {/* Direction buttons */}
            <div className="grid grid-cols-3 gap-1">
              {/* Top row */}
              <div />
              <button
                onClick={() => handleDirectionClick('forward')}
                className="w-10 h-10 bg-white/20 hover:bg-white/40 rounded flex items-center justify-center text-white transition"
                title="Move Forward (W)"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
              </button>
              <div />

              {/* Middle row */}
              <button
                onClick={() => handleDirectionClick('left')}
                className="w-10 h-10 bg-white/20 hover:bg-white/40 rounded flex items-center justify-center text-white transition"
                title="Move Left (A)"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <button
                className="w-10 h-10 bg-white/10 rounded flex items-center justify-center text-white/50"
                title="Current Position"
              >
                <div className="w-2 h-2 bg-white/50 rounded-full" />
              </button>
              <button
                onClick={() => handleDirectionClick('right')}
                className="w-10 h-10 bg-white/20 hover:bg-white/40 rounded flex items-center justify-center text-white transition"
                title="Move Right (D)"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>

              {/* Bottom row */}
              <div />
              <button
                onClick={() => handleDirectionClick('backward')}
                className="w-10 h-10 bg-white/20 hover:bg-white/40 rounded flex items-center justify-center text-white transition"
                title="Move Backward (S)"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              <div />
            </div>

            {/* Turn buttons */}
            <div className="flex justify-center gap-2 mt-2">
              <button
                onClick={() => handleDirectionClick('turn_left')}
                className="px-3 py-1 bg-white/20 hover:bg-white/40 rounded text-white text-xs transition"
                title="Turn Left (Q)"
              >
                Q Turn
              </button>
              <button
                onClick={() => handleDirectionClick('turn_right')}
                className="px-3 py-1 bg-white/20 hover:bg-white/40 rounded text-white text-xs transition"
                title="Turn Right (E)"
              >
                Turn E
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Help Overlay */}
      {showHelpOverlay && (
        <div
          className="absolute inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => setShowHelpOverlay(false)}
        >
          <div className="bg-gray-900 rounded-lg p-6 max-w-md text-white">
            <h3 className="text-xl font-bold mb-4">Navigation Controls</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">W / Arrow Up</span>
                <p>Move Forward</p>
              </div>
              <div>
                <span className="text-gray-400">S / Arrow Down</span>
                <p>Move Backward</p>
              </div>
              <div>
                <span className="text-gray-400">A</span>
                <p>Strafe Left</p>
              </div>
              <div>
                <span className="text-gray-400">D</span>
                <p>Strafe Right</p>
              </div>
              <div>
                <span className="text-gray-400">Q / Arrow Left</span>
                <p>Turn Left</p>
              </div>
              <div>
                <span className="text-gray-400">E / Arrow Right</span>
                <p>Turn Right</p>
              </div>
            </div>
            <p className="text-gray-400 text-xs mt-4">Click anywhere to close</p>
          </div>
        </div>
      )}

      {/* Control Bar */}
      {showControls && (
        <div className="absolute top-0 left-0 right-0 p-3 flex items-center justify-between bg-gradient-to-b from-black/60 to-transparent z-20">
          {/* Left: Status */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isReady ? 'bg-green-500' : 'bg-yellow-500'
                }`}
              />
              <span className="text-white text-xs">
                {isReady ? 'Ready' : 'Loading'}
              </span>
            </div>
            <div className="text-gray-400 text-xs">
              {currentPosition}
            </div>
          </div>

          {/* Right: Control Buttons */}
          <div className="flex items-center gap-2">
            {/* Help Button */}
            <button
              onClick={() => setShowHelpOverlay(true)}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
              title="Controls Help"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </button>

            {/* Pause/Play Button */}
            <button
              onClick={togglePause}
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
              onClick={toggleFullscreen}
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
      )}

      {/* Pause Overlay */}
      {isPaused && isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-white text-2xl font-bold">PAUSED</div>
        </div>
      )}

      {/* Keyboard hint */}
      {isReady && (
        <div className="absolute bottom-4 left-4 text-white/50 text-xs z-20">
          Use WASD or arrow keys to navigate
        </div>
      )}
    </div>
  );
};

export default SeamlessPlayer;
