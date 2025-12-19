/**
 * WorldCanvas - Canvas component for rendering video frames
 */

import React, { forwardRef } from 'react';

interface WorldCanvasProps {
  width: number;
  height: number;
  className?: string;
}

const WorldCanvas = forwardRef<HTMLCanvasElement, WorldCanvasProps>(
  ({ width, height, className = '' }, ref) => {
    return (
      <canvas
        ref={ref}
        width={width}
        height={height}
        className={`world-canvas ${className}`}
        style={{
          imageRendering: 'auto',
        }}
      />
    );
  }
);

WorldCanvas.displayName = 'WorldCanvas';

export default WorldCanvas;
