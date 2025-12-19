/**
 * useWorldPlayback - React hook for world playback engine
 *
 * Provides seamless integration of pre-generated world playback with React components.
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import {
  WorldPlaybackEngine,
  WorldState,
  WorldMetadata,
  Action,
  PlaybackCallbacks,
} from '../services/worldPlayback';

export interface UseWorldPlaybackOptions {
  autoPlay?: boolean;
  onPositionChange?: (position: string, description: string) => void;
  onError?: (error: string) => void;
  onReady?: () => void;
}

export interface UseWorldPlaybackReturn {
  // Container ref to attach to wrapper div
  containerRef: React.RefObject<HTMLDivElement | null>;

  // State
  state: WorldState | null;
  metadata: WorldMetadata | null;
  isReady: boolean;
  error: string | null;

  // Controls
  handleAction: (action: Action) => Promise<void>;
  play: () => Promise<void>;
  pause: () => void;

  // Position info
  currentPosition: string;
  positionDescription: string;
  positionHistory: string[];

  // AI integration
  guideTo: (action: Action) => Promise<void>;
  playGuidedSequence: (actions: Action[]) => Promise<void>;
}

export function useWorldPlayback(
  worldId: string,
  options: UseWorldPlaybackOptions = {}
): UseWorldPlaybackReturn {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<WorldPlaybackEngine | null>(null);

  // Store callbacks in refs to avoid stale closures
  const callbacksRef = useRef(options);
  callbacksRef.current = options;

  const [state, setState] = useState<WorldState | null>(null);
  const [metadata, setMetadata] = useState<WorldMetadata | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPosition, setCurrentPosition] = useState('start');
  const [positionDescription, setPositionDescription] = useState('');
  const [positionHistory, setPositionHistory] = useState<string[]>(['start']);

  // Initialize engine when component mounts and container is available
  useEffect(() => {
    if (!containerRef.current) return;

    // Reset state on worldId change
    setIsReady(false);
    setError(null);
    setCurrentPosition('start');
    setPositionDescription('');
    setPositionHistory(['start']);

    const engine = new WorldPlaybackEngine(worldId);
    engineRef.current = engine;

    const callbacks: PlaybackCallbacks = {
      onReady: () => {
        setIsReady(true);
        setMetadata(engine.getMetadata());
        setState(engine.getState());
        setPositionDescription(engine.getPositionDescription());
        // Use ref to get latest callback
        callbacksRef.current.onReady?.();
      },
      onPositionChange: (position, description) => {
        setCurrentPosition(position);
        setPositionDescription(description);
        setPositionHistory(engine.getPositionHistory());
        setState(engine.getState());
        // Use ref to get latest callback
        callbacksRef.current.onPositionChange?.(position, description);
      },
      onError: (err) => {
        setError(err);
        // Use ref to get latest callback
        callbacksRef.current.onError?.(err);
      },
    };

    engine.initialize(containerRef.current, callbacks).catch((err) => {
      console.error('Failed to initialize playback engine:', err);
      setError(err.message || 'Failed to initialize');
    });

    return () => {
      engine.dispose();
      engineRef.current = null;
    };
  }, [worldId]); // Only re-initialize when worldId changes

  // Handle keyboard input
  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      if (!isReady || !engineRef.current) return;

      // Don't capture if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const actionMap: Record<string, Action> = {
        w: 'forward',
        W: 'forward',
        s: 'backward',
        S: 'backward',
        a: 'left',
        A: 'left',
        d: 'right',
        D: 'right',
        q: 'turn_left',
        Q: 'turn_left',
        e: 'turn_right',
        E: 'turn_right',
        ArrowUp: 'forward',
        ArrowDown: 'backward',
        ArrowLeft: 'turn_left',
        ArrowRight: 'turn_right',
      };

      const action = actionMap[e.key];
      if (action) {
        e.preventDefault();
        await engineRef.current.handleAction(action);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isReady]);

  const handleAction = useCallback(async (action: Action) => {
    if (engineRef.current) {
      await engineRef.current.handleAction(action);
    }
  }, []);

  const play = useCallback(async () => {
    if (engineRef.current) {
      await engineRef.current.play();
    }
  }, []);

  const pause = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.pause();
    }
  }, []);

  const guideTo = useCallback(async (action: Action) => {
    if (engineRef.current) {
      await engineRef.current.guideTo(action);
    }
  }, []);

  const playGuidedSequence = useCallback(async (actions: Action[]) => {
    if (engineRef.current) {
      await engineRef.current.playGuidedSequence(actions);
    }
  }, []);

  return {
    containerRef,
    state,
    metadata,
    isReady,
    error,
    handleAction,
    play,
    pause,
    currentPosition,
    positionDescription,
    positionHistory,
    guideTo,
    playGuidedSequence,
  };
}

export default useWorldPlayback;
