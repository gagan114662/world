/**
 * WorldViewerDemo - Demo/test page for the HybridWorldViewer
 *
 * This component provides a standalone demo of the immersive world viewer
 * for testing and showcasing the feature.
 *
 * Access via: Add ?demo=world to the URL
 */

import React, { useState } from 'react';
import HybridWorldViewer from './HybridWorldViewer';
import { getAvailableWorlds, WorldMetadata } from '../../data/worldMetadata';
import { StudentProfile, WorldExplorationSession } from '../../services/educationService';
import { MathProblem, EXAMPLE_ALGEBRA_PROBLEMS, EXAMPLE_GEOMETRY_PROBLEMS } from './MathOverlay';

// Demo student profile
const DEMO_STUDENT: StudentProfile = {
  id: 'demo-student',
  name: 'Demo Student',
  gradeLevel: '7',
  age: 12,
  learningStyle: 'visual',
  pacePreference: 'medium',
  interests: ['space', 'science', 'history'],
  accommodations: [],
  language: 'en',
};

const WorldViewerDemo: React.FC = () => {
  const [selectedWorld, setSelectedWorld] = useState<string>('solar_system');
  const [mode, setMode] = useState<'explore' | 'lesson' | 'practice'>('explore');
  const [teacherMode, setTeacherMode] = useState(false);
  const [sessionData, setSessionData] = useState<WorldExplorationSession | null>(null);

  const availableWorlds = getAvailableWorlds();

  // Select problems based on mode
  const getProblems = (): MathProblem[] => {
    if (mode === 'practice') {
      return [...EXAMPLE_ALGEBRA_PROBLEMS, ...EXAMPLE_GEOMETRY_PROBLEMS];
    }
    return EXAMPLE_ALGEBRA_PROBLEMS;
  };

  const handleSessionStart = (session: WorldExplorationSession) => {
    console.log('Session started:', session);
    setSessionData(session);
  };

  const handleSessionEnd = (session: WorldExplorationSession) => {
    console.log('Session ended:', session);
    setSessionData(session);
  };

  const handleProgress = (progress: { position: string; timeSpent: number }) => {
    console.log('Progress:', progress);
  };

  const handleTeacherAction = (action: string, data?: any) => {
    console.log('Teacher action:', action, data);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-black/50 backdrop-blur-sm border-b border-white/10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">World Viewer Demo</h1>
            <p className="text-gray-400 text-sm">Immersive AI Tutor Experience</p>
          </div>

          <div className="flex items-center gap-4">
            {/* World Selector */}
            <div>
              <label className="text-xs text-gray-400 block mb-1">World</label>
              <select
                value={selectedWorld}
                onChange={(e) => setSelectedWorld(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
              >
                {availableWorlds.map((world) => (
                  <option key={world.id} value={world.id}>
                    {world.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Mode Selector */}
            <div>
              <label className="text-xs text-gray-400 block mb-1">Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as any)}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
              >
                <option value="explore">Explore</option>
                <option value="lesson">Lesson</option>
                <option value="practice">Practice</option>
              </select>
            </div>

            {/* Teacher Mode Toggle */}
            <div>
              <label className="text-xs text-gray-400 block mb-1">Teacher Mode</label>
              <button
                onClick={() => setTeacherMode(!teacherMode)}
                className={`px-3 py-1.5 rounded text-sm transition ${
                  teacherMode
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-400 border border-gray-700'
                }`}
              >
                {teacherMode ? 'On' : 'Off'}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* World Viewer - Main Area */}
          <div className="lg:col-span-3">
            <div className="bg-black rounded-xl overflow-hidden" style={{ height: '70vh' }}>
              <HybridWorldViewer
                key={`${selectedWorld}-${mode}`}
                worldId={selectedWorld}
                mode={mode}
                studentProfile={DEMO_STUDENT}
                teacherMode={teacherMode}
                onTeacherAction={handleTeacherAction}
                problems={getProblems()}
                onSessionStart={handleSessionStart}
                onSessionEnd={handleSessionEnd}
                onProgress={handleProgress}
                className="w-full h-full"
              />
            </div>

            {/* Controls Legend */}
            <div className="mt-4 bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Navigation Controls</h3>
              <div className="grid grid-cols-4 gap-4 text-xs text-gray-400">
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">W</kbd> Forward</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">S</kbd> Backward</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">A</kbd> Left</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">D</kbd> Right</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">Q</kbd> Turn Left</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">E</kbd> Turn Right</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">↑↓←→</kbd> Arrows</div>
                <div><kbd className="bg-gray-700 px-2 py-1 rounded">Space</kbd> Pause</div>
              </div>
            </div>
          </div>

          {/* Sidebar - Session Info */}
          <div className="lg:col-span-1 space-y-4">
            {/* World Info */}
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Current World</h3>
              <div className="text-gray-400 text-sm">
                <p><strong>{availableWorlds.find(w => w.id === selectedWorld)?.name}</strong></p>
                <p className="mt-1">{availableWorlds.find(w => w.id === selectedWorld)?.description}</p>
              </div>
            </div>

            {/* Session Stats */}
            {sessionData && (
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h3 className="text-sm font-semibold mb-2">Session Stats</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Positions Visited</span>
                    <span>{sessionData.positionsVisited.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Questions Asked</span>
                    <span>{sessionData.questionsAsked.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Correct Answers</span>
                    <span className="text-green-400">
                      {sessionData.comprehensionIndicators.questionsAnsweredCorrectly}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Incorrect Answers</span>
                    <span className="text-red-400">
                      {sessionData.comprehensionIndicators.questionsAnsweredIncorrectly}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Engagement Score</span>
                    <span className="text-blue-400">{sessionData.engagementScore}/100</span>
                  </div>
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Instructions</h3>
              <ul className="text-gray-400 text-xs space-y-1">
                <li>• Use WASD or arrow keys to navigate</li>
                <li>• AI tutor provides educational narration</li>
                <li>• Click on-screen controls as alternative</li>
                <li>• Enable Teacher Mode for classroom controls</li>
                <li>• Practice mode shows math problems</li>
              </ul>
            </div>

            {/* Debug Info */}
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold mb-2">Debug</h3>
              <div className="text-gray-500 text-xs font-mono">
                <p>World: {selectedWorld}</p>
                <p>Mode: {mode}</p>
                <p>Teacher: {teacherMode ? 'true' : 'false'}</p>
                <p>CDN: {import.meta.env.VITE_WORLD_CDN_URL || 'localhost:8010'}</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-8 border-t border-white/10 p-4 text-center text-gray-500 text-xs">
        <p>WorldPlay AI Tutor Demo | Pre-generated worlds with AI narration</p>
        <p className="mt-1">
          Note: Video segments must be generated using{' '}
          <code className="bg-gray-800 px-1 rounded">generate_world_library.py</code>
        </p>
      </footer>
    </div>
  );
};

export default WorldViewerDemo;
