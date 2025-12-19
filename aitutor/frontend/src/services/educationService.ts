/**
 * Education Service - Curriculum alignment, progress tracking, personalization
 *
 * Inspired by 2wai's approach:
 * - Curriculum-based responses
 * - Personalized learning parameters
 * - Teacher dashboards & analytics
 * - Age-appropriate content safety
 */

// ============================================================================
// CURRICULUM STANDARDS
// ============================================================================

export interface CurriculumStandard {
  id: string;
  subject: string;
  gradeLevel: string;
  topic: string;
  objectives: string[];
  keywords: string[];
}

// Common Core / NGSS aligned standards (examples)
export const CURRICULUM_STANDARDS: CurriculumStandard[] = [
  // Science - NGSS
  {
    id: 'NGSS-MS-ESS1-2',
    subject: 'earth_science',
    gradeLevel: '6-8',
    topic: 'Solar System',
    objectives: [
      'Develop and use a model to describe the role of gravity in the motions within galaxies and the solar system',
    ],
    keywords: ['gravity', 'orbit', 'planets', 'solar system', 'motion'],
  },
  {
    id: 'NGSS-MS-LS1-2',
    subject: 'life_science',
    gradeLevel: '6-8',
    topic: 'Cell Structure',
    objectives: [
      'Develop and use a model to describe the function of a cell as a whole',
    ],
    keywords: ['cell', 'organelle', 'nucleus', 'mitochondria', 'membrane'],
  },
  {
    id: 'NGSS-MS-ESS2-2',
    subject: 'earth_science',
    gradeLevel: '6-8',
    topic: 'Plate Tectonics',
    objectives: [
      'Construct an explanation based on evidence for how geoscience processes have changed Earth\'s surface',
    ],
    keywords: ['volcano', 'earthquake', 'tectonic', 'magma', 'lava'],
  },
  // History
  {
    id: 'NCSS-D2.His.1.6-8',
    subject: 'history',
    gradeLevel: '6-8',
    topic: 'Ancient Civilizations',
    objectives: [
      'Analyze connections among events and developments in broader historical contexts',
    ],
    keywords: ['rome', 'egypt', 'ancient', 'civilization', 'empire'],
  },
  // Math
  {
    id: 'CCSS-MATH-6.G.A.1',
    subject: 'geometry',
    gradeLevel: '6',
    topic: '3D Geometry',
    objectives: [
      'Find the area of polygons and surface area of 3D figures',
    ],
    keywords: ['area', 'volume', 'surface', 'polygon', 'prism', 'pyramid'],
  },
];

/**
 * Find curriculum standards that match a world's subject/topic
 */
export function findMatchingStandards(
  worldSubject: string,
  worldName: string,
  gradeLevel?: string
): CurriculumStandard[] {
  const searchTerms = [
    worldSubject.toLowerCase(),
    ...worldName.toLowerCase().split(/[\s_]+/),
  ];

  return CURRICULUM_STANDARDS.filter((standard) => {
    // Match by grade if specified
    if (gradeLevel && !standard.gradeLevel.includes(gradeLevel)) {
      return false;
    }

    // Match by subject or keywords
    const matchesSubject = searchTerms.some(
      (term) =>
        standard.subject.includes(term) ||
        standard.topic.toLowerCase().includes(term) ||
        standard.keywords.some((kw) => kw.includes(term) || term.includes(kw))
    );

    return matchesSubject;
  });
}

// ============================================================================
// STUDENT PROFILE & PERSONALIZATION
// ============================================================================

export interface StudentProfile {
  id: string;
  name: string;
  gradeLevel: string;
  age: number;
  learningStyle: 'visual' | 'auditory' | 'kinesthetic' | 'reading';
  pacePreference: 'slow' | 'medium' | 'fast';
  interests: string[];
  accommodations: string[];
  language: string;
}

export interface LearningParameters {
  // Content complexity
  vocabularyLevel: 'simple' | 'intermediate' | 'advanced';
  explanationDepth: 'brief' | 'moderate' | 'detailed';

  // Interaction style
  questionFrequency: 'low' | 'medium' | 'high';
  encouragementLevel: 'minimal' | 'moderate' | 'high';
  hintStyle: 'direct' | 'socratic' | 'exploratory';

  // Pacing
  pauseBetweenConcepts: boolean;
  repeatImportantPoints: boolean;
  allowSkipAhead: boolean;

  // Safety
  ageAppropriate: boolean;
  maxComplexity: number; // 1-10 scale
}

/**
 * Generate learning parameters from student profile
 */
export function generateLearningParameters(
  profile: StudentProfile
): LearningParameters {
  const isYounger = profile.age < 10;
  const isOlder = profile.age >= 14;

  return {
    vocabularyLevel: isYounger ? 'simple' : isOlder ? 'advanced' : 'intermediate',
    explanationDepth: profile.pacePreference === 'slow' ? 'detailed' : 'moderate',
    questionFrequency: profile.learningStyle === 'kinesthetic' ? 'high' : 'medium',
    encouragementLevel: isYounger ? 'high' : 'moderate',
    hintStyle:
      profile.learningStyle === 'reading'
        ? 'direct'
        : profile.learningStyle === 'auditory'
        ? 'socratic'
        : 'exploratory',
    pauseBetweenConcepts: profile.pacePreference === 'slow',
    repeatImportantPoints: profile.pacePreference !== 'fast',
    allowSkipAhead: profile.pacePreference === 'fast',
    ageAppropriate: true,
    maxComplexity: Math.min(10, Math.max(1, profile.age - 3)),
  };
}

// ============================================================================
// PROGRESS TRACKING
// ============================================================================

export interface WorldExplorationSession {
  sessionId: string;
  worldId: string;
  studentId: string;
  startTime: Date;
  endTime?: Date;
  positionsVisited: string[];
  questionsAsked: string[];
  conceptsExplored: string[];
  standardsCovered: string[];
  engagementScore: number; // 0-100
  comprehensionIndicators: {
    questionsAnsweredCorrectly: number;
    questionsAnsweredIncorrectly: number;
    conceptsRevisited: number;
  };
}

export interface StudentProgress {
  studentId: string;
  worldsExplored: string[];
  totalExplorationTime: number; // minutes
  standardsMastered: string[];
  standardsInProgress: string[];
  recentSessions: WorldExplorationSession[];
  strengths: string[];
  areasForGrowth: string[];
  recommendedNextWorlds: string[];
}

/**
 * Calculate engagement score from session data
 */
export function calculateEngagementScore(session: WorldExplorationSession): number {
  const uniquePositions = new Set(session.positionsVisited).size;
  const questionsAsked = session.questionsAsked.length;
  const sessionDuration = session.endTime
    ? (session.endTime.getTime() - session.startTime.getTime()) / 60000
    : 0;

  // Score components
  const explorationScore = Math.min(100, uniquePositions * 10);
  const curiosityScore = Math.min(100, questionsAsked * 15);
  const durationScore = Math.min(100, sessionDuration * 5);

  return Math.round((explorationScore + curiosityScore + durationScore) / 3);
}

/**
 * Generate recommendations for next worlds based on progress
 */
export function recommendNextWorlds(
  progress: StudentProgress,
  allWorldIds: string[]
): string[] {
  const unexplored = allWorldIds.filter(
    (id) => !progress.worldsExplored.includes(id)
  );

  // Prioritize worlds that cover standards in progress
  const prioritized = unexplored.filter((worldId) => {
    const standards = findMatchingStandards(worldId, worldId);
    return standards.some((s) => progress.standardsInProgress.includes(s.id));
  });

  return prioritized.length > 0 ? prioritized.slice(0, 3) : unexplored.slice(0, 3);
}

// ============================================================================
// TEACHER DASHBOARD DATA
// ============================================================================

export interface ClassroomAnalytics {
  classId: string;
  teacherId: string;
  studentCount: number;

  // Engagement metrics
  averageSessionDuration: number;
  averageWorldsExplored: number;
  mostPopularWorlds: { worldId: string; visits: number }[];

  // Learning metrics
  standardsCoverage: { standardId: string; percentStudentsMastered: number }[];
  classStrengths: string[];
  classAreasForGrowth: string[];

  // Individual tracking
  studentsNeedingSupport: string[]; // student IDs
  highPerformers: string[];
  recentActivity: {
    studentId: string;
    worldId: string;
    timestamp: Date;
    duration: number;
  }[];
}

export interface LessonPlan {
  id: string;
  title: string;
  teacherId: string;
  gradeLevel: string;
  subject: string;
  objectives: string[];
  standardsAddressed: string[];
  worldsIncluded: string[];
  estimatedDuration: number; // minutes
  activities: {
    type: 'explore' | 'discuss' | 'quiz' | 'reflect';
    worldId?: string;
    instructions: string;
    duration: number;
  }[];
  assessmentQuestions: {
    question: string;
    expectedAnswer: string;
    standardId: string;
  }[];
}

/**
 * Generate a lesson plan for a world
 */
export function generateLessonPlan(
  worldId: string,
  worldName: string,
  worldSubject: string,
  gradeLevel: string
): LessonPlan {
  const standards = findMatchingStandards(worldSubject, worldName, gradeLevel);

  return {
    id: `lesson-${worldId}-${Date.now()}`,
    title: `Exploring ${worldName}`,
    teacherId: '',
    gradeLevel,
    subject: worldSubject,
    objectives: standards.flatMap((s) => s.objectives),
    standardsAddressed: standards.map((s) => s.id),
    worldsIncluded: [worldId],
    estimatedDuration: 45,
    activities: [
      {
        type: 'explore',
        worldId,
        instructions: `Students freely explore ${worldName} using WASD controls while AI tutor narrates`,
        duration: 15,
      },
      {
        type: 'discuss',
        instructions: 'Class discussion: What did you discover? What surprised you?',
        duration: 10,
      },
      {
        type: 'explore',
        worldId,
        instructions: 'Guided exploration: Teacher directs class to specific locations',
        duration: 10,
      },
      {
        type: 'reflect',
        instructions: 'Students write or discuss 3 things they learned',
        duration: 10,
      },
    ],
    assessmentQuestions: standards.map((s) => ({
      question: `Based on your exploration, explain one thing about ${s.topic}`,
      expectedAnswer: s.objectives[0],
      standardId: s.id,
    })),
  };
}

// ============================================================================
// AI TUTOR CONTEXT GENERATION
// ============================================================================

/**
 * Generate AI tutor system prompt with personalization and curriculum alignment
 */
export function generatePersonalizedTutorPrompt(
  worldName: string,
  worldSubject: string,
  studentProfile: StudentProfile,
  learningParams: LearningParameters,
  standards: CurriculumStandard[]
): string {
  const standardsContext = standards.length > 0
    ? `\n\nCurriculum Standards to Address:\n${standards.map(s =>
        `- ${s.id}: ${s.objectives.join('; ')}`
      ).join('\n')}`
    : '';

  return `You are an engaging educational AI tutor guiding ${studentProfile.name} through "${worldName}".

Student Profile:
- Grade Level: ${studentProfile.gradeLevel}
- Age: ${studentProfile.age}
- Learning Style: ${studentProfile.learningStyle}
- Interests: ${studentProfile.interests.join(', ')}
${studentProfile.accommodations.length > 0 ? `- Accommodations: ${studentProfile.accommodations.join(', ')}` : ''}

Teaching Parameters:
- Vocabulary: Use ${learningParams.vocabularyLevel} language
- Explanations: ${learningParams.explanationDepth} depth
- Questions: Ask questions ${learningParams.questionFrequency === 'high' ? 'frequently to check understanding' : 'occasionally'}
- Encouragement: ${learningParams.encouragementLevel} level of positive reinforcement
- Hints: Use ${learningParams.hintStyle} questioning style
${learningParams.pauseBetweenConcepts ? '- Pause between major concepts to allow processing' : ''}
${learningParams.repeatImportantPoints ? '- Repeat and reinforce key points' : ''}
${standardsContext}

Your role:
1. Narrate what the student is seeing in age-appropriate language
2. Connect observations to ${worldSubject} concepts
3. Ask thought-provoking questions matching their learning style
4. Provide encouragement and celebrate discoveries
5. Guide toward curriculum objectives naturally
6. Keep responses concise (2-3 sentences) unless asked for more

Remember: This is exploration-based learning. Let curiosity drive the experience!`;
}

// ============================================================================
// CONTENT SAFETY
// ============================================================================

export interface ContentSafetyConfig {
  maxAgeRating: number;
  blockedTopics: string[];
  requireEducatorReview: boolean;
  allowCustomWorlds: boolean;
}

/**
 * Check if world content is appropriate for student
 */
export function isContentAppropriate(
  worldPrompt: string,
  worldSubject: string,
  safetyConfig: ContentSafetyConfig
): { safe: boolean; reason?: string } {
  const lowercasePrompt = worldPrompt.toLowerCase();

  // Check blocked topics
  for (const topic of safetyConfig.blockedTopics) {
    if (lowercasePrompt.includes(topic.toLowerCase())) {
      return { safe: false, reason: `Contains blocked topic: ${topic}` };
    }
  }

  // Educational subjects are generally safe
  const educationalSubjects = [
    'astronomy', 'biology', 'chemistry', 'physics', 'geology',
    'history', 'geography', 'math', 'geometry', 'ecology',
  ];

  if (educationalSubjects.some(s => worldSubject.toLowerCase().includes(s))) {
    return { safe: true };
  }

  // Custom worlds may need review
  if (!safetyConfig.allowCustomWorlds) {
    return { safe: false, reason: 'Custom worlds require educator approval' };
  }

  return { safe: true };
}
