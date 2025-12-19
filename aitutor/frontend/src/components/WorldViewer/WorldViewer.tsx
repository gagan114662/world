/**
 * WorldViewer - Main component for displaying and interacting with 3D worlds
 *
 * Renders video frames from HY-WorldPlay and captures user input
 * for interactive world exploration.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useWorldStream } from '../../hooks/useWorldStream';
import { useWorldControls } from '../../hooks/useWorldControls';
import WorldCanvas from './WorldCanvas';
import WorldOverlay from './WorldOverlay';
import WorldControls from './WorldControls';
import WorldLoadingState from './WorldLoadingState';

export interface WorldViewerProps {
  /** Session ID for the world generation */
  sessionId: string;
  /** Subject area (physics, biology, etc.) */
  subject: string;
  /** Topic within the subject */
  topic: string;
  /** AI tutor narration text */
  narration?: string;
  /** Called when user interacts with the world */
  onInteraction?: (action: WorldAction) => void;
  /** Called when world generation ends */
  onWorldEnd?: () => void;
  /** Show controls overlay */
  showControls?: boolean;
  /** Enable keyboard/mouse controls */
  controlsEnabled?: boolean;
  /** CSS class name */
  className?: string;
}

export interface WorldAction {
  keyboard: KeyboardState;
  mouse: MouseState;
  timestamp: number;
}

export interface KeyboardState {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
  space: boolean;
  shift: boolean;
}

export interface MouseState {
  dx: number;
  dy: number;
  scroll: number;
  leftButton: boolean;
  rightButton: boolean;
}

const WorldViewer: React.FC<WorldViewerProps> = ({
  sessionId,
  subject,
  topic,
  narration,
  onInteraction,
  onWorldEnd,
  showControls = true,
  controlsEnabled = true,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // Connect to world stream
  const {
    frames,
    isConnected,
    latency,
    frameCount,
    error,
    sendAction,
    pause,
    resume,
  } = useWorldStream(sessionId);

  // Capture keyboard/mouse controls
  const { currentAction, isCapturing, startCapture, stopCapture } = useWorldControls(
    containerRef,
    controlsEnabled
  );

  // Send actions to server when controls change
  useEffect(() => {
    if (currentAction && isConnected && !isPaused) {
      sendAction(currentAction);
      onInteraction?.(currentAction);
    }
  }, [currentAction, isConnected, isPaused, sendAction, onInteraction]);

  // Render frames to canvas
  useEffect(() => {
    if (!canvasRef.current || frames.length === 0) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Get the latest frame
    const latestFrame = frames[frames.length - 1];
    if (latestFrame) {
      // Draw the frame
      ctx.drawImage(latestFrame, 0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [frames]);

  // Handle pause/resume
  const togglePause = useCallback(() => {
    if (isPaused) {
      resume();
      setIsPaused(false);
    } else {
      pause();
      setIsPaused(true);
    }
  }, [isPaused, pause, resume]);

  // Handle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Start capturing controls when component mounts and is focused
  useEffect(() => {
    if (controlsEnabled) {
      startCapture();
    }
    return () => stopCapture();
  }, [controlsEnabled, startCapture, stopCapture]);

  return (
    <div
      ref={containerRef}
      className={`world-viewer relative bg-black rounded-lg overflow-hidden ${className}`}
      tabIndex={0}
      onFocus={startCapture}
      onBlur={stopCapture}
    >
      {/* Main Canvas */}
      <WorldCanvas
        ref={canvasRef}
        width={854}
        height={480}
        className="w-full h-full object-contain"
      />

      {/* Loading State */}
      {!isConnected && !error && (
        <WorldLoadingState subject={subject} topic={topic} />
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-center text-white">
            <p className="text-red-500 text-lg mb-2">Connection Error</p>
            <p className="text-sm text-gray-400">{error}</p>
          </div>
        </div>
      )}

      {/* AI Narration Overlay */}
      {narration && <WorldOverlay narration={narration} />}

      {/* Controls Overlay */}
      {showControls && (
        <WorldControls
          isConnected={isConnected}
          isPaused={isPaused}
          isFullscreen={isFullscreen}
          latency={latency}
          frameCount={frameCount}
          onPauseToggle={togglePause}
          onFullscreenToggle={toggleFullscreen}
        />
      )}

      {/* Pause Overlay */}
      {isPaused && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="text-white text-2xl font-bold">PAUSED</div>
        </div>
      )}

      {/* Control Instructions (shown when not capturing) */}
      {!isCapturing && controlsEnabled && isConnected && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-lg text-sm">
          Click to enable controls (WASD to move, Mouse to look)
        </div>
      )}
    </div>
  );
};

export default WorldViewer;
