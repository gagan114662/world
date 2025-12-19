/**
 * useWorldStream - Hook for managing WebSocket connection to world stream
 *
 * Handles bidirectional video streaming and action transmission.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { WorldAction } from '../components/WorldViewer/WorldViewer';

const WORLDPLAY_WS_URL = import.meta.env.VITE_WORLDPLAY_WS_URL || 'ws://localhost:8010';

interface UseWorldStreamReturn {
  /** Array of decoded video frames */
  frames: ImageBitmap[];
  /** Whether connected to stream */
  isConnected: boolean;
  /** Current latency in ms */
  latency: number | null;
  /** Total frames received */
  frameCount: number;
  /** Error message if any */
  error: string | null;
  /** Send user action to server */
  sendAction: (action: WorldAction) => void;
  /** Pause the stream */
  pause: () => void;
  /** Resume the stream */
  resume: () => void;
}

export function useWorldStream(sessionId: string): UseWorldStreamReturn {
  const [frames, setFrames] = useState<ImageBitmap[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState<number | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const frameBufferRef = useRef<ImageBitmap[]>([]);
  const lastPingRef = useRef<number>(0);

  // Max frames to keep in buffer (2 seconds at 24 FPS)
  const MAX_BUFFER_SIZE = 48;

  // Connect to WebSocket
  useEffect(() => {
    if (!sessionId) return;

    const connect = () => {
      try {
        const ws = new WebSocket(`${WORLDPLAY_WS_URL}/ws/worlds/${sessionId}/stream`);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
          console.log('WorldStream connected');
          setIsConnected(true);
          setError(null);

          // Send ping for latency measurement
          lastPingRef.current = Date.now();
          ws.send(JSON.stringify({ type: 'ping', timestamp: lastPingRef.current }));
        };

        ws.onclose = (event) => {
          console.log('WorldStream disconnected', event.code);
          setIsConnected(false);

          // Attempt reconnection on abnormal close
          if (event.code !== 1000 && event.code !== 1001) {
            setTimeout(connect, 2000);
          }
        };

        ws.onerror = (event) => {
          console.error('WorldStream error', event);
          setError('Connection error');
        };

        ws.onmessage = async (event) => {
          try {
            if (event.data instanceof ArrayBuffer) {
              // Binary data - video frame
              await handleFrameData(event.data);
            } else {
              // JSON message
              const message = JSON.parse(event.data);
              handleMessage(message);
            }
          } catch (err) {
            console.error('Error processing message', err);
          }
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Failed to connect', err);
        setError('Failed to connect');
      }
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [sessionId]);

  // Handle binary frame data
  const handleFrameData = async (data: ArrayBuffer) => {
    try {
      // Create blob from data
      const blob = new Blob([data], { type: 'image/jpeg' });

      // Decode to ImageBitmap
      const bitmap = await createImageBitmap(blob);

      // Add to buffer
      frameBufferRef.current.push(bitmap);

      // Trim buffer if too large
      while (frameBufferRef.current.length > MAX_BUFFER_SIZE) {
        const oldFrame = frameBufferRef.current.shift();
        oldFrame?.close();
      }

      // Update state
      setFrames([...frameBufferRef.current]);
      setFrameCount((prev) => prev + 1);
    } catch (err) {
      console.error('Error decoding frame', err);
    }
  };

  // Handle JSON messages
  const handleMessage = (message: any) => {
    switch (message.type) {
      case 'connected':
        console.log('Session info:', message.payload);
        break;

      case 'metadata':
        if (message.payload.latency_ms !== undefined) {
          setLatency(message.payload.latency_ms);
        }
        break;

      case 'pong':
        const rtt = Date.now() - lastPingRef.current;
        setLatency(rtt / 2);
        break;

      case 'narration':
        // Narration handled separately by parent component
        break;

      case 'error':
        setError(message.payload.message || 'Unknown error');
        break;
    }
  };

  // Send action to server
  const sendAction = useCallback((action: WorldAction) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'action',
          payload: action,
        })
      );
    }
  }, []);

  // Pause stream
  const pause = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'control',
          payload: { command: 'pause' },
        })
      );
    }
  }, []);

  // Resume stream
  const resume = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'control',
          payload: { command: 'resume' },
        })
      );
    }
  }, []);

  // Cleanup frames on unmount
  useEffect(() => {
    return () => {
      frameBufferRef.current.forEach((frame) => frame.close());
      frameBufferRef.current = [];
    };
  }, []);

  return {
    frames,
    isConnected,
    latency,
    frameCount,
    error,
    sendAction,
    pause,
    resume,
  };
}

export default useWorldStream;
