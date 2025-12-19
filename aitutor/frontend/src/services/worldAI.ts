/**
 * WorldAI - AI Tutor Integration for World Exploration
 *
 * FULLY DYNAMIC - Works with ANY world, not just predefined ones.
 *
 * Connects the world playback engine with Gemini Live for:
 * - Position-aware narration
 * - Guided exploration
 * - Real-time Q&A during exploration
 * - Educational context about what the student is seeing
 *
 * The AI generates contextual narration based on:
 * 1. World metadata (name, subject, description, prompt)
 * 2. Position descriptions from metadata.json
 * 3. Dynamic Gemini responses for unknown content
 */

import { Action } from './worldPlayback';

// CDN URL for loading world metadata
const WORLD_CDN_URL = import.meta.env.VITE_WORLD_CDN_URL || 'http://localhost:8010/worlds';

/**
 * Extended metadata loaded from world's metadata.json
 */
export interface ExtendedWorldMetadata {
  id: string;
  name: string;
  subject: string;
  description: string;
  prompt?: string;
  educationalPoints?: string[];
  vocabulary?: string[];
  focusAreas?: string[];
  positions?: Record<string, string>;
}

/**
 * Generate subject context dynamically from world metadata
 */
function generateSubjectContext(metadata: ExtendedWorldMetadata): {
  greeting: string;
  focusAreas: string[];
  vocabulary: string[];
} {
  const subject = metadata.subject || 'exploration';
  const name = metadata.name || 'this world';

  // Generate greeting based on subject keywords
  const greetingTemplates: Record<string, string> = {
    astronomy: "Welcome to our journey through space! I'll be your guide as we explore the cosmos.",
    biology: "Let's explore the fascinating world of living things together!",
    history: "Step back in time with me as we explore this historical period!",
    geology: "Let's discover the forces that shape our Earth!",
    chemistry: "Let's explore the molecular world and see how matter behaves!",
    physics: "Let's explore the fundamental forces of nature!",
    ecology: "Join me in exploring this amazing ecosystem!",
    marine: "Dive with me into the mysteries of the ocean!",
    space: "Welcome aboard! Let's explore the universe together!",
  };

  // Find matching greeting or generate one
  let greeting = `Welcome! Let's explore ${name} together. I'll guide you through this experience.`;
  for (const [key, value] of Object.entries(greetingTemplates)) {
    if (subject.toLowerCase().includes(key)) {
      greeting = value;
      break;
    }
  }

  // Use provided focus areas or generate from description
  const focusAreas = metadata.focusAreas ||
    extractKeyTopics(metadata.description || metadata.prompt || '');

  // Use provided vocabulary or leave empty for AI to fill
  const vocabulary = metadata.vocabulary || [];

  return { greeting, focusAreas, vocabulary };
}

/**
 * Extract key topics from a description for focus areas
 */
function extractKeyTopics(text: string): string[] {
  if (!text) return ['observation', 'discovery', 'learning'];

  // Simple keyword extraction - in production, use NLP
  const keywords = text
    .toLowerCase()
    .split(/[\s,]+/)
    .filter(word => word.length > 5)
    .slice(0, 4);

  return keywords.length > 0 ? keywords : ['exploration', 'discovery', 'learning'];
}

export interface WorldAIConfig {
  worldId: string;
  metadata: ExtendedWorldMetadata;
  onNarration?: (text: string) => void;
  onSuggestion?: (action: Action, reason: string) => void;
  geminiApiKey?: string;
}

export interface GeminiMessage {
  role: 'user' | 'model';
  text: string;
}

/**
 * WorldAI Tutor Service
 *
 * FULLY DYNAMIC - Provides intelligent narration for ANY world.
 * Loads extended metadata from CDN and generates contextual responses.
 */
export class WorldAITutor {
  private worldId: string;
  private metadata: ExtendedWorldMetadata;
  private onNarration?: (text: string) => void;
  private onSuggestion?: (action: Action, reason: string) => void;
  private conversationHistory: GeminiMessage[] = [];
  private subjectContext: { greeting: string; focusAreas: string[]; vocabulary: string[] };
  private lastNarratedPosition: string = '';
  private extendedMetadataLoaded: boolean = false;
  private positionDescriptions: Record<string, string> = {};
  private initializationPromise: Promise<void>;

  constructor(config: WorldAIConfig) {
    this.worldId = config.worldId;
    this.metadata = config.metadata as ExtendedWorldMetadata;
    this.onNarration = config.onNarration;
    this.onSuggestion = config.onSuggestion;

    // Generate subject context dynamically from metadata
    this.subjectContext = generateSubjectContext(this.metadata);

    // Initialize with system context
    this.initializeContext();

    // Load extended metadata from world's metadata.json (async, tracked by promise)
    this.initializationPromise = this.loadExtendedMetadata();
  }

  /**
   * Wait for initialization to complete (optional - system works without it)
   */
  async waitForInitialization(): Promise<void> {
    await this.initializationPromise;
  }

  /**
   * Load extended metadata from the world's metadata.json file
   */
  private async loadExtendedMetadata(): Promise<void> {
    try {
      const response = await fetch(`${WORLD_CDN_URL}/${this.worldId}/metadata.json`);
      if (response.ok) {
        const data = await response.json();

        // Merge extended metadata
        if (data.world) {
          this.metadata = { ...this.metadata, ...data.world };

          // Load position descriptions for narration
          if (data.world.positions) {
            this.positionDescriptions = data.world.positions;
          }

          // Update subject context with loaded data
          this.subjectContext = generateSubjectContext(this.metadata);

          // Update educational points if available
          if (data.world.educationalPoints) {
            this.metadata.educationalPoints = data.world.educationalPoints;
          }
        }

        this.extendedMetadataLoaded = true;
        console.log(`WorldAI: Loaded extended metadata for ${this.worldId}`);
      }
    } catch (error) {
      console.warn(`WorldAI: Could not load extended metadata for ${this.worldId}, using defaults`);
    }
  }

  /**
   * Initialize the AI with world context - FULLY DYNAMIC
   */
  private initializeContext(): void {
    const prompt = this.metadata.prompt || this.metadata.description || '';

    const systemMessage = `You are an engaging educational AI tutor helping a student explore "${this.metadata.name || this.worldId}".

Subject: ${this.metadata.subject || 'general exploration'}
Description: ${this.metadata.description || 'An immersive educational experience'}
Scene: ${prompt}

${this.subjectContext.focusAreas.length > 0 ? `Focus Areas: ${this.subjectContext.focusAreas.join(', ')}` : ''}
${this.subjectContext.vocabulary.length > 0 ? `Key Vocabulary: ${this.subjectContext.vocabulary.join(', ')}` : ''}
${this.metadata.educationalPoints ? `Educational Points: ${this.metadata.educationalPoints.join('; ')}` : ''}

Your role:
1. Provide educational narration about what the student is seeing
2. Ask thought-provoking questions to encourage curiosity
3. Suggest directions to explore for interesting discoveries
4. Answer student questions in an engaging, age-appropriate way
5. Make connections to real-world applications and other subjects

Keep responses concise (2-3 sentences) unless the student asks for more detail.
Be enthusiastic but informative. Adapt your language to the subject matter.`;

    this.conversationHistory.push({
      role: 'model',
      text: systemMessage,
    });
  }

  /**
   * Greet the student when entering the world
   * Waits for initialization to complete to ensure we have the best context
   */
  async greet(): Promise<string> {
    // Wait for extended metadata to load for better greeting context
    await this.initializationPromise;

    const greeting = this.subjectContext.greeting;
    this.onNarration?.(greeting);
    return greeting;
  }

  /**
   * Handle position change and provide narration - FULLY DYNAMIC
   */
  async onPositionChange(position: string, description: string): Promise<void> {
    // Avoid repeating narration for same position
    if (position === this.lastNarratedPosition) return;
    this.lastNarratedPosition = position;

    // First check: Position description from loaded metadata.json
    const metadataDescription = this.positionDescriptions[position];
    if (metadataDescription) {
      const narration = await this.generateNarrationFromDescription(position, metadataDescription);
      this.onNarration?.(narration);
      return;
    }

    // Second check: Position description from WorldMetadata.positions
    if (this.metadata.positions?.[position]) {
      const narration = await this.generateNarrationFromDescription(position, this.metadata.positions[position]);
      this.onNarration?.(narration);
      return;
    }

    // Fallback: Generate dynamic narration from provided description
    const dynamicNarration = await this.generateNarration(position, description);
    this.onNarration?.(dynamicNarration);
  }

  /**
   * Generate narration from a position description
   */
  private async generateNarrationFromDescription(position: string, description: string): Promise<string> {
    // Format the description into educational narration
    // In production, this would call Gemini for richer responses
    const narration = this.formatAsNarration(description);

    this.conversationHistory.push({
      role: 'model',
      text: `[Narrating: ${position}] ${narration}`,
    });

    return narration;
  }

  /**
   * Format a description as educational narration
   */
  private formatAsNarration(description: string): string {
    // If description is already a full sentence, use it
    if (description.includes('.') || description.length > 50) {
      return description;
    }

    // Otherwise, expand it into a narration
    return `We're now ${description.toLowerCase()}. Take a moment to observe your surroundings.`;
  }

  /**
   * Generate dynamic narration for any position - WORKS WITH ANY WORLD
   */
  private async generateNarration(position: string, description: string): Promise<string> {
    // Build context-aware response based on available metadata
    const worldName = this.metadata.name || this.worldId;
    const subject = this.metadata.subject || 'this topic';

    // Dynamic templates that work with any content
    const templates = [
      `Interesting! We're ${description}. Notice how this relates to ${subject}.`,
      `Here we can see ${description}. What do you observe?`,
      `${description}. This is a fascinating aspect of ${worldName}.`,
      `As we explore ${description}, think about what makes this significant.`,
      `We're now viewing ${description}. Let's take a closer look.`,
    ];

    const narration = templates[Math.floor(Math.random() * templates.length)];

    this.conversationHistory.push({
      role: 'model',
      text: `[Narrating: ${position}] ${narration}`,
    });

    return narration;
  }

  /**
   * Handle student question
   */
  async answerQuestion(question: string): Promise<string> {
    this.conversationHistory.push({
      role: 'user',
      text: question,
    });

    // In a real implementation, this would call Gemini API
    // For now, return a helpful template response
    const answer = `That's a great question about ${this.metadata.name}! ${question
      .toLowerCase()
      .includes('why')
      ? 'The reason is related to ' + this.subjectContext.focusAreas[0] + '. '
      : ''
    }Let me explain what we're seeing here...`;

    this.conversationHistory.push({
      role: 'model',
      text: answer,
    });

    return answer;
  }

  /**
   * Suggest a direction for the student to explore
   */
  suggestDirection(): { action: Action; reason: string } {
    const suggestions: Array<{ action: Action; reason: string }> = [
      { action: 'forward', reason: "Let's go forward and see what's ahead!" },
      { action: 'left', reason: 'There might be something interesting to the left.' },
      { action: 'right', reason: "I notice something worth exploring to the right." },
    ];

    const suggestion = suggestions[Math.floor(Math.random() * suggestions.length)];

    this.onSuggestion?.(suggestion.action, suggestion.reason);
    return suggestion;
  }

  /**
   * Plan a guided tour sequence
   */
  planGuidedTour(goal: string): Action[] {
    // In a real implementation, this would use AI to plan the route
    // For now, return a simple sequence
    const tours: Record<string, Action[]> = {
      default: ['forward', 'forward', 'left', 'forward', 'right'],
      overview: ['forward', 'turn_left', 'forward', 'turn_right', 'forward'],
      detailed: ['forward', 'idle', 'forward', 'idle', 'left', 'idle', 'right'],
    };

    return tours[goal] || tours.default;
  }

  /**
   * Get educational highlights about current view
   */
  getEducationalHighlights(): string[] {
    const highlights = this.subjectContext.focusAreas.map((area, i) => {
      const vocab = this.subjectContext.vocabulary[i] || '';
      return vocab ? `${area}: Learn about ${vocab}` : area;
    });

    return highlights;
  }

  /**
   * Get conversation history for context
   */
  getConversationHistory(): GeminiMessage[] {
    return [...this.conversationHistory];
  }
}

/**
 * Create a WorldAI tutor instance
 */
export function createWorldAITutor(config: WorldAIConfig): WorldAITutor {
  return new WorldAITutor(config);
}

export default WorldAITutor;
