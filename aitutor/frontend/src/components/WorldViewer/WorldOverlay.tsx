/**
 * WorldOverlay - Displays AI tutor narration over the world
 */

import React from 'react';

interface WorldOverlayProps {
  narration: string;
  highlights?: string[];
  className?: string;
}

const WorldOverlay: React.FC<WorldOverlayProps> = ({
  narration,
  highlights = [],
  className = '',
}) => {
  return (
    <div className={`world-overlay absolute bottom-16 left-4 right-4 ${className}`}>
      {/* Narration Text */}
      <div className="bg-black/80 backdrop-blur-sm rounded-lg p-4 text-white shadow-lg">
        <div className="flex items-start gap-3">
          {/* AI Avatar */}
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>

          {/* Text */}
          <div className="flex-1">
            <p className="text-sm font-medium text-blue-400 mb-1">AI Tutor</p>
            <p className="text-base leading-relaxed">{narration}</p>
          </div>
        </div>

        {/* Highlights */}
        {highlights.length > 0 && (
          <div className="mt-3 pt-3 border-t border-white/20">
            <p className="text-xs text-gray-400 mb-2">Look at:</p>
            <div className="flex flex-wrap gap-2">
              {highlights.map((highlight, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-blue-500/30 rounded text-xs text-blue-300"
                >
                  {highlight}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WorldOverlay;
