/**
 * useWorldControls - Hook for capturing keyboard and mouse input
 *
 * Captures WASD + mouse input for world navigation.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { WorldAction, KeyboardState, MouseState } from '../components/WorldViewer/WorldViewer';

interface UseWorldControlsReturn {
  /** Current action state */
  currentAction: WorldAction | null;
  /** Whether controls are being captured */
  isCapturing: boolean;
  /** Start capturing input */
  startCapture: () => void;
  /** Stop capturing input */
  stopCapture: () => void;
}

export function useWorldControls(
  containerRef: React.RefObject<HTMLElement>,
  enabled: boolean = true
): UseWorldControlsReturn {
  const [isCapturing, setIsCapturing] = useState(false);
  const [currentAction, setCurrentAction] = useState<WorldAction | null>(null);

  // Track key states
  const keysRef = useRef<KeyboardState>({
    w: false,
    a: false,
    s: false,
    d: false,
    space: false,
    shift: false,
  });

  // Track mouse state
  const mouseRef = useRef<MouseState>({
    dx: 0,
    dy: 0,
    scroll: 0,
    leftButton: false,
    rightButton: false,
  });

  // Track if pointer is locked
  const pointerLockedRef = useRef(false);

  // Update action state
  const updateAction = useCallback(() => {
    setCurrentAction({
      keyboard: { ...keysRef.current },
      mouse: { ...mouseRef.current },
      timestamp: Date.now(),
    });

    // Reset mouse delta after capturing
    mouseRef.current.dx = 0;
    mouseRef.current.dy = 0;
    mouseRef.current.scroll = 0;
  }, []);

  // Handle key down
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled || !isCapturing) return;

      const key = event.key.toLowerCase();

      switch (key) {
        case 'w':
          keysRef.current.w = true;
          break;
        case 'a':
          keysRef.current.a = true;
          break;
        case 's':
          keysRef.current.s = true;
          break;
        case 'd':
          keysRef.current.d = true;
          break;
        case ' ':
          keysRef.current.space = true;
          event.preventDefault(); // Prevent page scroll
          break;
        case 'shift':
          keysRef.current.shift = true;
          break;
        case 'escape':
          // Exit pointer lock
          if (document.pointerLockElement) {
            document.exitPointerLock();
          }
          break;
      }

      updateAction();
    },
    [enabled, isCapturing, updateAction]
  );

  // Handle key up
  const handleKeyUp = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      const key = event.key.toLowerCase();

      switch (key) {
        case 'w':
          keysRef.current.w = false;
          break;
        case 'a':
          keysRef.current.a = false;
          break;
        case 's':
          keysRef.current.s = false;
          break;
        case 'd':
          keysRef.current.d = false;
          break;
        case ' ':
          keysRef.current.space = false;
          break;
        case 'shift':
          keysRef.current.shift = false;
          break;
      }

      updateAction();
    },
    [enabled, updateAction]
  );

  // Handle mouse move
  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!enabled || !isCapturing) return;

      // Use movementX/Y when pointer is locked for better precision
      if (pointerLockedRef.current) {
        mouseRef.current.dx += event.movementX;
        mouseRef.current.dy += event.movementY;
      }

      updateAction();
    },
    [enabled, isCapturing, updateAction]
  );

  // Handle mouse button
  const handleMouseDown = useCallback(
    (event: MouseEvent) => {
      if (!enabled || !isCapturing) return;

      if (event.button === 0) {
        mouseRef.current.leftButton = true;

        // Request pointer lock on click
        if (!document.pointerLockElement && containerRef.current) {
          containerRef.current.requestPointerLock();
        }
      } else if (event.button === 2) {
        mouseRef.current.rightButton = true;
      }

      updateAction();
    },
    [enabled, isCapturing, updateAction, containerRef]
  );

  const handleMouseUp = useCallback(
    (event: MouseEvent) => {
      if (!enabled) return;

      if (event.button === 0) {
        mouseRef.current.leftButton = false;
      } else if (event.button === 2) {
        mouseRef.current.rightButton = false;
      }

      updateAction();
    },
    [enabled, updateAction]
  );

  // Handle scroll
  const handleWheel = useCallback(
    (event: WheelEvent) => {
      if (!enabled || !isCapturing) return;

      mouseRef.current.scroll = event.deltaY;
      updateAction();
    },
    [enabled, isCapturing, updateAction]
  );

  // Handle pointer lock change
  const handlePointerLockChange = useCallback(() => {
    pointerLockedRef.current = document.pointerLockElement === containerRef.current;
  }, [containerRef]);

  // Start capturing
  const startCapture = useCallback(() => {
    if (!enabled) return;
    setIsCapturing(true);
  }, [enabled]);

  // Stop capturing
  const stopCapture = useCallback(() => {
    setIsCapturing(false);

    // Reset keys
    keysRef.current = {
      w: false,
      a: false,
      s: false,
      d: false,
      space: false,
      shift: false,
    };

    // Exit pointer lock
    if (document.pointerLockElement) {
      document.exitPointerLock();
    }

    setCurrentAction(null);
  }, []);

  // Add event listeners
  useEffect(() => {
    if (!enabled) return;

    const container = containerRef.current;
    if (!container) return;

    // Keyboard events (on window to capture when focused)
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Mouse events (on container)
    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mousedown', handleMouseDown);
    container.addEventListener('mouseup', handleMouseUp);
    container.addEventListener('wheel', handleWheel, { passive: false });

    // Pointer lock
    document.addEventListener('pointerlockchange', handlePointerLockChange);

    // Prevent context menu
    const preventContextMenu = (e: Event) => {
      if (isCapturing) e.preventDefault();
    };
    container.addEventListener('contextmenu', preventContextMenu);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mousedown', handleMouseDown);
      container.removeEventListener('mouseup', handleMouseUp);
      container.removeEventListener('wheel', handleWheel);
      document.removeEventListener('pointerlockchange', handlePointerLockChange);
      container.removeEventListener('contextmenu', preventContextMenu);
    };
  }, [
    enabled,
    isCapturing,
    containerRef,
    handleKeyDown,
    handleKeyUp,
    handleMouseMove,
    handleMouseDown,
    handleMouseUp,
    handleWheel,
    handlePointerLockChange,
  ]);

  // Send periodic updates (throttled)
  useEffect(() => {
    if (!isCapturing) return;

    const interval = setInterval(() => {
      if (
        keysRef.current.w ||
        keysRef.current.a ||
        keysRef.current.s ||
        keysRef.current.d ||
        keysRef.current.space ||
        keysRef.current.shift ||
        mouseRef.current.dx !== 0 ||
        mouseRef.current.dy !== 0
      ) {
        updateAction();
      }
    }, 50); // 20 updates per second

    return () => clearInterval(interval);
  }, [isCapturing, updateAction]);

  return {
    currentAction,
    isCapturing,
    startCapture,
    stopCapture,
  };
}

export default useWorldControls;
