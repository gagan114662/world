/**
 * HybridWorldViewer - Combines immersive video worlds with interactive overlays
 *
 * For visual topics (science, history): Pure video exploration
 * For abstract topics (math): Video backdrop + interactive problem overlay
 *
 * This is the main component students and teachers interact with.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import SeamlessPlayer from './SeamlessPlayer';
import { MathOverlay, MathProblem, EXAMPLE_ALGEBRA_PROBLEMS } from './MathOverlay';
import { WorldAITutor, createWorldAITutor } from '../../services/worldAI';
import {
  StudentProfile,
  findMatchingStandards,
  WorldExplorationSession,
  calculateEngagementScore,
} from '../../services/educationService';
import { getWorldMetadata, createMinimalMetadata } from '../../data/worldMetadata';

// ============================================================================
// TYPES
// ============================================================================

export interface HybridWorldViewerProps {
  worldId: string;
  mode: 'explore' | 'lesson' | 'practice';

  // Student info for personalization
  studentProfile?: StudentProfile;

  // Teacher controls
  teacherMode?: boolean;
  onTeacherAction?: (action: string, data?: any) => void;

  // Lesson content for practice mode
  problems?: MathProblem[];

  // Callbacks
  onSessionStart?: (session: WorldExplorationSession) => void;
  onSessionEnd?: (session: WorldExplorationSession) => void;
  onProgress?: (progress: { position: string; timeSpent: number }) => void;

  className?: string;
}

type OverlayMode = 'none' | 'problem' | 'quiz' | 'info';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const HybridWorldViewer: React.FC<HybridWorldViewerProps> = ({
  worldId,
  mode = 'explore',
  studentProfile,
  teacherMode = false,
  onTeacherAction,
  problems = EXAMPLE_ALGEBRA_PROBLEMS,
  onSessionStart,
  onSessionEnd,
  onProgress,
  className = '',
}) => {
  // State
  const [isReady, setIsReady] = useState(false);
  const [currentPosition, setCurrentPosition] = useState('start');
  const [overlayMode, setOverlayMode] = useState<OverlayMode>(
    mode === 'practice' ? 'problem' : 'none'
  );
  const [aiTutor, setAiTutor] = useState<WorldAITutor | null>(null);
  const [narration, setNarration] = useState<string>('');
  const [showNarration, setShowNarration] = useState(true);

  // Session tracking
  const [session, setSession] = useState<WorldExplorationSession | null>(null);
  const [sessionStartTime] = useState(new Date());

  // Use ref to access latest session in cleanup
  const sessionRef = useRef<WorldExplorationSession | null>(null);
  sessionRef.current = session;

  // Store callbacks in refs to avoid stale closures
  const onSessionEndRef = useRef(onSessionEnd);
  onSessionEndRef.current = onSessionEnd;
  const onSessionStartRef = useRef(onSessionStart);
  onSessionStartRef.current = onSessionStart;

  // Note: learningParams can be used for AI personalization in future
  // const learningParams = studentProfile ? generateLearningParameters(studentProfile) : null;

  // Initialize AI tutor and session
  useEffect(() => {
    let isMounted = true;

    const initTutor = async () => {
      const metadata = (await getWorldMetadata(worldId)) || createMinimalMetadata(worldId);

      if (!isMounted) return;

      const tutor = createWorldAITutor({
        worldId,
        metadata,
        onNarration: (text) => {
          if (!isMounted) return;
          setNarration(text);
          setShowNarration(true);
          // Auto-hide after delay
          setTimeout(() => {
            if (isMounted) setShowNarration(false);
          }, 8000);
        },
      });

      setAiTutor(tutor);

      // Create session
      const newSession: WorldExplorationSession = {
        sessionId: `session-${Date.now()}`,
        worldId,
        studentId: studentProfile?.id || 'anonymous',
        startTime: sessionStartTime,
        positionsVisited: ['start'],
        questionsAsked: [],
        conceptsExplored: [],
        standardsCovered: findMatchingStandards(metadata.subject, metadata.name).map(
          (s) => s.id
        ),
        engagementScore: 0,
        comprehensionIndicators: {
          questionsAnsweredCorrectly: 0,
          questionsAnsweredIncorrectly: 0,
          conceptsRevisited: 0,
        },
      };

      setSession(newSession);
      onSessionStartRef.current?.(newSession);

      // Greet student
      await tutor.greet();
    };

    initTutor();

    // Cleanup on unmount
    return () => {
      isMounted = false;
      // Use ref to get latest session value
      if (sessionRef.current) {
        const finalSession = {
          ...sessionRef.current,
          endTime: new Date(),
          engagementScore: calculateEngagementScore(sessionRef.current),
        };
        onSessionEndRef.current?.(finalSession);
      }
    };
  }, [worldId, studentProfile?.id, sessionStartTime]);

  // Handle position changes
  const handlePositionChange = useCallback(
    async (position: string, description: string) => {
      setCurrentPosition(position);

      // Update session
      if (session) {
        const updatedSession = {
          ...session,
          positionsVisited: [...session.positionsVisited, position],
        };
        setSession(updatedSession);
      }

      // Notify AI tutor
      if (aiTutor) {
        await aiTutor.onPositionChange(position, description);
      }

      onProgress?.({
        position,
        timeSpent: (Date.now() - sessionStartTime.getTime()) / 1000,
      });
    },
    [aiTutor, session, sessionStartTime, onProgress]
  );

  // Handle problem answers
  const handleAnswer = useCallback(
    (problemId: string, answer: string, correct: boolean) => {
      if (session) {
        const indicators = { ...session.comprehensionIndicators };
        if (correct) {
          indicators.questionsAnsweredCorrectly++;
        } else {
          indicators.questionsAnsweredIncorrectly++;
        }
        setSession({ ...session, comprehensionIndicators: indicators });
      }

      // AI tutor responds
      if (aiTutor && correct) {
        setNarration("Great job! That's correct. Let's continue exploring!");
        setShowNarration(true);
      }
    },
    [session, aiTutor]
  );

  // Handle hint requests
  const handleHintRequest = useCallback(
    (problemId: string) => {
      if (session) {
        setSession({
          ...session,
          questionsAsked: [...session.questionsAsked, `hint:${problemId}`],
        });
      }
    },
    [session]
  );

  // Handle skip
  const handleSkip = useCallback(
    (problemId: string) => {
      // Track skipped problems
    },
    []
  );

  // Teacher controls
  const handleTeacherPause = useCallback(() => {
    onTeacherAction?.('pause');
    setNarration("The teacher has paused the exploration. Let's discuss what we've seen.");
    setShowNarration(true);
  }, [onTeacherAction]);

  const handleTeacherGuide = useCallback(
    (direction: string) => {
      onTeacherAction?.('guide', { direction });
      setNarration(`Let's look ${direction}...`);
      setShowNarration(true);
    },
    [onTeacherAction]
  );

  const handleTeacherShowProblem = useCallback(() => {
    setOverlayMode('problem');
    onTeacherAction?.('showProblem');
  }, [onTeacherAction]);

  return (
    <div className={`hybrid-world-viewer relative w-full h-full ${className}`}>
      {/* Video World Layer */}
      <SeamlessPlayer
        worldId={worldId}
        className="absolute inset-0"
        showControls={!teacherMode}
        showNavigationHints={overlayMode === 'none'}
        autoPlay={true}
        onPositionChange={handlePositionChange}
        onReady={() => setIsReady(true)}
      />

      {/* AI Narration Overlay */}
      {showNarration && narration && (
        <div className="absolute bottom-24 left-4 right-4 z-20 pointer-events-none">
          <div className="bg-black/70 backdrop-blur-sm text-white px-6 py-4 rounded-xl max-w-2xl mx-auto">
            <p className="text-lg leading-relaxed">{narration}</p>
          </div>
        </div>
      )}

      {/* Math/Problem Overlay */}
      {overlayMode === 'problem' && (
        <MathOverlay
          problems={problems}
          onAnswer={handleAnswer}
          onHintRequest={handleHintRequest}
          onSkip={handleSkip}
          visible={true}
          position="center"
        />
      )}

      {/* Mode Toggle (for exploration mode) */}
      {mode === 'explore' && isReady && (
        <div className="absolute top-16 right-4 z-20">
          <div className="bg-black/50 backdrop-blur-sm rounded-lg p-2 flex gap-2">
            <button
              onClick={() => setOverlayMode('none')}
              className={`px-3 py-1 rounded text-sm transition ${
                overlayMode === 'none'
                  ? 'bg-blue-600 text-white'
                  : 'text-white/70 hover:text-white'
              }`}
            >
              Explore
            </button>
            <button
              onClick={() => setOverlayMode('problem')}
              className={`px-3 py-1 rounded text-sm transition ${
                overlayMode === 'problem'
                  ? 'bg-blue-600 text-white'
                  : 'text-white/70 hover:text-white'
              }`}
            >
              Practice
            </button>
          </div>
        </div>
      )}

      {/* Teacher Control Panel */}
      {teacherMode && isReady && (
        <div className="absolute top-4 left-4 z-30">
          <div className="bg-black/80 backdrop-blur-sm rounded-xl p-4 space-y-3">
            <h3 className="text-white font-bold text-sm uppercase tracking-wide">
              Teacher Controls
            </h3>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={handleTeacherPause}
                className="px-3 py-2 bg-yellow-600 text-white rounded-lg text-sm
                           hover:bg-yellow-700 transition"
              >
                Pause Class
              </button>
              <button
                onClick={handleTeacherShowProblem}
                className="px-3 py-2 bg-purple-600 text-white rounded-lg text-sm
                           hover:bg-purple-700 transition"
              >
                Show Problem
              </button>
            </div>

            <div className="pt-2 border-t border-white/20">
              <p className="text-white/50 text-xs mb-2">Guide class to:</p>
              <div className="grid grid-cols-4 gap-1">
                {['forward', 'left', 'right', 'back'].map((dir) => (
                  <button
                    key={dir}
                    onClick={() => handleTeacherGuide(dir)}
                    className="px-2 py-1 bg-white/10 text-white text-xs rounded
                               hover:bg-white/20 transition capitalize"
                  >
                    {dir}
                  </button>
                ))}
              </div>
            </div>

            <div className="pt-2 border-t border-white/20 text-white/50 text-xs">
              <p>Position: {currentPosition}</p>
              <p>Students exploring: 24</p>
            </div>
          </div>
        </div>
      )}

      {/* Session Info (debug) */}
      {process.env.NODE_ENV === 'development' && session && (
        <div className="absolute bottom-4 right-4 z-10 bg-black/50 text-white/50 text-xs p-2 rounded">
          <p>Session: {session.sessionId.slice(-8)}</p>
          <p>Positions: {session.positionsVisited.length}</p>
          <p>Correct: {session.comprehensionIndicators.questionsAnsweredCorrectly}</p>
        </div>
      )}
    </div>
  );
};

export default HybridWorldViewer;
