/**
 * WorldLoadingState - Loading animation while world generates
 */

import React from 'react';

interface WorldLoadingStateProps {
  subject: string;
  topic: string;
  progress?: number;
}

const WorldLoadingState: React.FC<WorldLoadingStateProps> = ({
  subject,
  topic,
  progress,
}) => {
  // Subject-specific loading messages
  const loadingMessages: Record<string, string[]> = {
    physics: [
      'Calculating physics laws...',
      'Simulating gravitational fields...',
      'Rendering particle interactions...',
    ],
    biology: [
      'Growing cellular structures...',
      'Simulating ecosystems...',
      'Rendering biological processes...',
    ],
    history: [
      'Reconstructing historical setting...',
      'Loading period-accurate details...',
      'Rendering ancient architecture...',
    ],
    chemistry: [
      'Arranging molecular structures...',
      'Simulating chemical bonds...',
      'Rendering atomic interactions...',
    ],
    geography: [
      'Mapping terrain features...',
      'Simulating geological processes...',
      'Rendering landscape...',
    ],
    math: [
      'Constructing geometric shapes...',
      'Plotting coordinate systems...',
      'Rendering mathematical space...',
    ],
  };

  const messages = loadingMessages[subject.toLowerCase()] || [
    'Generating world...',
    'Building environment...',
    'Preparing experience...',
  ];

  const [messageIndex, setMessageIndex] = React.useState(0);

  // Cycle through messages
  React.useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % messages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [messages.length]);

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900 via-black to-gray-800">
      <div className="text-center">
        {/* Animated Globe/World Icon */}
        <div className="relative w-24 h-24 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 animate-pulse" />
          <div className="absolute inset-2 rounded-full bg-black" />
          <div className="absolute inset-4 rounded-full bg-gradient-to-br from-blue-400/30 to-purple-500/30 animate-spin" style={{ animationDuration: '3s' }}>
            {/* Grid lines */}
            <div className="absolute inset-0 rounded-full border border-blue-500/30" />
            <div className="absolute top-1/2 left-0 right-0 h-px bg-blue-500/30" />
            <div className="absolute top-0 bottom-0 left-1/2 w-px bg-blue-500/30" />
          </div>

          {/* Center Icon */}
          <div className="absolute inset-0 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>

        {/* Subject Badge */}
        <div className="inline-block px-3 py-1 mb-3 rounded-full bg-blue-500/20 text-blue-400 text-xs font-medium uppercase tracking-wider">
          {subject}
        </div>

        {/* Topic */}
        <h3 className="text-white text-lg font-semibold mb-2 capitalize">
          {topic.replace(/_/g, ' ')}
        </h3>

        {/* Loading Message */}
        <p className="text-gray-400 text-sm mb-4 transition-opacity duration-300">
          {messages[messageIndex]}
        </p>

        {/* Progress Bar */}
        {progress !== undefined && (
          <div className="w-48 mx-auto h-1 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Loading Dots */}
        {progress === undefined && (
          <div className="flex justify-center gap-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default WorldLoadingState;
