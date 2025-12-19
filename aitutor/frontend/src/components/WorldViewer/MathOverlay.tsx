/**
 * MathOverlay - Interactive overlay for abstract/math topics
 *
 * Used with video worlds to add interactive elements for topics that
 * need more than pure visual exploration (equations, problems, graphs).
 *
 * The video world provides immersive context while this overlay
 * adds interactive educational elements.
 */

import React, { useState, useCallback, useEffect } from 'react';

// ============================================================================
// TYPES
// ============================================================================

export interface MathProblem {
  id: string;
  type: 'equation' | 'multiple-choice' | 'graph' | 'drag-drop' | 'fill-blank';
  question: string;
  hint?: string;
  correctAnswer: string | string[];
  options?: string[]; // for multiple choice
  difficulty: 'easy' | 'medium' | 'hard';
  topic: string;
  visualElements?: {
    type: 'equation' | 'graph' | 'shape' | 'number-line';
    data: any;
  }[];
}

export interface MathOverlayProps {
  problems: MathProblem[];
  currentProblemIndex?: number;
  onAnswer: (problemId: string, answer: string, correct: boolean) => void;
  onHintRequest: (problemId: string) => void;
  onSkip: (problemId: string) => void;
  visible?: boolean;
  position?: 'center' | 'bottom' | 'side';
  className?: string;
}

// ============================================================================
// EQUATION RENDERER
// ============================================================================

interface EquationProps {
  equation: string;
  size?: 'small' | 'medium' | 'large';
}

const Equation: React.FC<EquationProps> = ({ equation, size = 'medium' }) => {
  const fontSize = {
    small: 'text-lg',
    medium: 'text-2xl',
    large: 'text-4xl',
  }[size];

  // Simple equation renderer - safer version without dangerouslySetInnerHTML
  // In production, use KaTeX or MathJax
  const renderEquation = (eq: string): React.ReactNode => {
    // Split equation into parts and render with proper React elements
    const parts: React.ReactNode[] = [];
    let remaining = eq;
    let key = 0;

    // Process superscripts (^2 -> ²)
    remaining = remaining
      .replace(/\^2/g, '²')
      .replace(/\^3/g, '³')
      .replace(/\^(\d+)/g, '^$1');

    // Process common math symbols
    remaining = remaining
      .replace(/\*/g, '×')
      .replace(/\//g, '÷')
      .replace(/sqrt/g, '√')
      .replace(/<=/g, '≤')
      .replace(/>=/g, '≥')
      .replace(/!=/g, '≠')
      .replace(/pi/gi, 'π');

    // Simple parse for remaining superscripts
    const superscriptRegex = /\^(\d+)/g;
    let lastIndex = 0;
    let match;

    while ((match = superscriptRegex.exec(remaining)) !== null) {
      if (match.index > lastIndex) {
        parts.push(<span key={key++}>{remaining.slice(lastIndex, match.index)}</span>);
      }
      parts.push(<sup key={key++}>{match[1]}</sup>);
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < remaining.length) {
      parts.push(<span key={key++}>{remaining.slice(lastIndex)}</span>);
    }

    return parts.length > 0 ? parts : remaining;
  };

  return (
    <div className={`font-mono ${fontSize} text-white`}>
      {renderEquation(equation)}
    </div>
  );
};

// ============================================================================
// PROBLEM CARD COMPONENT
// ============================================================================

interface ProblemCardProps {
  problem: MathProblem;
  onAnswer: (answer: string) => void;
  onHint: () => void;
  onSkip: () => void;
  showHint: boolean;
}

const ProblemCard: React.FC<ProblemCardProps> = ({
  problem,
  onAnswer,
  onHint,
  onSkip,
  showHint,
}) => {
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(null);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleSubmit = () => {
    const answer = problem.type === 'multiple-choice' ? selectedOption : userAnswer;
    if (!answer) return;

    const isCorrect = Array.isArray(problem.correctAnswer)
      ? problem.correctAnswer.includes(answer.trim().toLowerCase())
      : answer.trim().toLowerCase() === problem.correctAnswer.toLowerCase();

    setFeedback(isCorrect ? 'correct' : 'incorrect');
    onAnswer(answer);

    // Reset after delay
    setTimeout(() => {
      if (isCorrect) {
        setUserAnswer('');
        setSelectedOption(null);
        setFeedback(null);
      }
    }, 1500);
  };

  return (
    <div className="bg-black/80 backdrop-blur-md rounded-xl p-6 max-w-lg mx-auto border border-white/20">
      {/* Difficulty badge */}
      <div className="flex justify-between items-center mb-4">
        <span className="text-xs text-gray-400 uppercase tracking-wide">
          {problem.topic}
        </span>
        <span
          className={`text-xs px-2 py-1 rounded ${
            problem.difficulty === 'easy'
              ? 'bg-green-500/20 text-green-400'
              : problem.difficulty === 'medium'
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-red-500/20 text-red-400'
          }`}
        >
          {problem.difficulty}
        </span>
      </div>

      {/* Question */}
      <div className="mb-6">
        <Equation equation={problem.question} size="large" />
      </div>

      {/* Visual elements if any */}
      {problem.visualElements?.map((element, i) => (
        <div key={i} className="mb-4">
          {element.type === 'equation' && (
            <Equation equation={element.data} size="medium" />
          )}
          {element.type === 'shape' && (
            <div className="text-center text-6xl">{element.data}</div>
          )}
        </div>
      ))}

      {/* Answer input based on type */}
      {problem.type === 'multiple-choice' && problem.options && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          {problem.options.map((option) => (
            <button
              key={option}
              onClick={() => setSelectedOption(option)}
              className={`p-3 rounded-lg border transition ${
                selectedOption === option
                  ? 'border-blue-500 bg-blue-500/20'
                  : 'border-white/20 hover:border-white/40'
              }`}
            >
              <Equation equation={option} size="small" />
            </button>
          ))}
        </div>
      )}

      {problem.type === 'equation' || problem.type === 'fill-blank' ? (
        <div className="mb-4">
          <input
            type="text"
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Type your answer..."
            className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3
                       text-white text-xl font-mono focus:outline-none focus:border-blue-500"
          />
        </div>
      ) : null}

      {/* Hint */}
      {showHint && problem.hint && (
        <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
          <p className="text-yellow-400 text-sm">
            <span className="font-bold">Hint:</span> {problem.hint}
          </p>
        </div>
      )}

      {/* Feedback */}
      {feedback && (
        <div
          className={`mb-4 p-3 rounded-lg ${
            feedback === 'correct'
              ? 'bg-green-500/20 border border-green-500/50'
              : 'bg-red-500/20 border border-red-500/50'
          }`}
        >
          <p className={feedback === 'correct' ? 'text-green-400' : 'text-red-400'}>
            {feedback === 'correct' ? 'Correct! Great job!' : 'Not quite. Try again!'}
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          onClick={onHint}
          className="px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-lg
                     hover:bg-yellow-500/30 transition text-sm"
        >
          Hint
        </button>
        <button
          onClick={handleSubmit}
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg
                     hover:bg-blue-700 transition font-medium"
        >
          Check Answer
        </button>
        <button
          onClick={onSkip}
          className="px-4 py-2 bg-white/10 text-gray-400 rounded-lg
                     hover:bg-white/20 transition text-sm"
        >
          Skip
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN OVERLAY COMPONENT
// ============================================================================

export const MathOverlay: React.FC<MathOverlayProps> = ({
  problems,
  currentProblemIndex = 0,
  onAnswer,
  onHintRequest,
  onSkip,
  visible = true,
  position = 'center',
  className = '',
}) => {
  const [showHint, setShowHint] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(currentProblemIndex);

  const currentProblem = problems[currentIndex];

  useEffect(() => {
    setCurrentIndex(currentProblemIndex);
    setShowHint(false);
  }, [currentProblemIndex]);

  const handleAnswer = useCallback(
    (answer: string) => {
      if (!currentProblem) return;

      const isCorrect = Array.isArray(currentProblem.correctAnswer)
        ? currentProblem.correctAnswer.includes(answer.trim().toLowerCase())
        : answer.trim().toLowerCase() === currentProblem.correctAnswer.toLowerCase();

      onAnswer(currentProblem.id, answer, isCorrect);

      if (isCorrect && currentIndex < problems.length - 1) {
        setTimeout(() => {
          setCurrentIndex((prev) => prev + 1);
          setShowHint(false);
        }, 1500);
      }
    },
    [currentProblem, currentIndex, problems.length, onAnswer]
  );

  const handleHint = useCallback(() => {
    if (currentProblem) {
      setShowHint(true);
      onHintRequest(currentProblem.id);
    }
  }, [currentProblem, onHintRequest]);

  const handleSkip = useCallback(() => {
    if (currentProblem) {
      onSkip(currentProblem.id);
      if (currentIndex < problems.length - 1) {
        setCurrentIndex((prev) => prev + 1);
        setShowHint(false);
      }
    }
  }, [currentProblem, currentIndex, problems.length, onSkip]);

  if (!visible || !currentProblem) return null;

  const positionClasses = {
    center: 'items-center justify-center',
    bottom: 'items-end justify-center pb-24',
    side: 'items-center justify-end pr-8',
  }[position];

  return (
    <div className={`absolute inset-0 flex ${positionClasses} z-30 ${className}`}>
      {/* Progress indicator */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2">
        <div className="flex gap-2">
          {problems.map((_, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full transition ${
                i < currentIndex
                  ? 'bg-green-500'
                  : i === currentIndex
                  ? 'bg-blue-500'
                  : 'bg-white/30'
              }`}
            />
          ))}
        </div>
        <p className="text-white/50 text-xs text-center mt-2">
          Problem {currentIndex + 1} of {problems.length}
        </p>
      </div>

      {/* Problem card */}
      <ProblemCard
        problem={currentProblem}
        onAnswer={handleAnswer}
        onHint={handleHint}
        onSkip={handleSkip}
        showHint={showHint}
      />
    </div>
  );
};

// ============================================================================
// EXAMPLE PROBLEMS
// ============================================================================

export const EXAMPLE_ALGEBRA_PROBLEMS: MathProblem[] = [
  {
    id: 'alg-1',
    type: 'equation',
    question: 'x + 5 = 12',
    hint: 'Subtract 5 from both sides',
    correctAnswer: '7',
    difficulty: 'easy',
    topic: 'Basic Algebra',
  },
  {
    id: 'alg-2',
    type: 'equation',
    question: '2x = 14',
    hint: 'Divide both sides by 2',
    correctAnswer: '7',
    difficulty: 'easy',
    topic: 'Basic Algebra',
  },
  {
    id: 'alg-3',
    type: 'multiple-choice',
    question: 'Solve: 3x + 2 = 11',
    options: ['x = 2', 'x = 3', 'x = 4', 'x = 5'],
    correctAnswer: 'x = 3',
    difficulty: 'medium',
    topic: 'Linear Equations',
  },
  {
    id: 'geo-1',
    type: 'multiple-choice',
    question: 'Area of a rectangle with length 5 and width 3?',
    options: ['8', '15', '16', '12'],
    correctAnswer: '15',
    difficulty: 'easy',
    topic: 'Geometry',
    visualElements: [{ type: 'shape', data: '▭' }],
  },
];

export const EXAMPLE_GEOMETRY_PROBLEMS: MathProblem[] = [
  {
    id: 'geo-area-1',
    type: 'equation',
    question: 'Area of triangle: base = 6, height = 4',
    hint: 'Area = (1/2) × base × height',
    correctAnswer: '12',
    difficulty: 'easy',
    topic: 'Triangle Area',
    visualElements: [{ type: 'shape', data: '△' }],
  },
  {
    id: 'geo-vol-1',
    type: 'equation',
    question: 'Volume of cube with side = 3',
    hint: 'Volume = side³',
    correctAnswer: '27',
    difficulty: 'medium',
    topic: '3D Geometry',
  },
];

export default MathOverlay;
