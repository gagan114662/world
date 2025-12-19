/**
 * WorldPlay API Client
 *
 * API client for communicating with the WorldPlay Gateway service.
 */

const WORLDPLAY_API_URL = import.meta.env.VITE_WORLDPLAY_API_URL || 'http://localhost:8010';

export interface GenerateWorldRequest {
  sessionId?: string;
  subject: string;
  topic: string;
  prompt: string;
  learningObjective?: string;
  referenceImageUrl?: string;
  camera?: CameraConfig;
  tutoringSessionId?: string;
}

export interface CameraConfig {
  position?: { x: number; y: number; z: number };
  rotation?: { pitch: number; yaw: number; roll: number };
  mode?: 'first_person' | 'third_person' | 'orbit';
  fov?: number;
}

export interface WorldSession {
  sessionId: string;
  userId?: string;
  tutoringSessionId?: string;
  subject: string;
  topic: string;
  prompt: string;
  learningObjective?: string;
  status: 'pending' | 'generating' | 'active' | 'paused' | 'completed' | 'error';
  frameCount: number;
  durationSeconds: number;
  createdAt: string;
  lastInteraction: string;
  errorMessage?: string;
}

export interface GenerateWorldResponse {
  sessionId: string;
  status: string;
  websocketUrl: string;
  estimatedStartTimeMs: number;
  message: string;
}

export interface Subject {
  name: string;
  topics: Topic[];
}

export interface Topic {
  name: string;
  templates: string[];
  camera: string;
}

class WorldPlayApi {
  private baseUrl: string;

  constructor(baseUrl: string = WORLDPLAY_API_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate a new world
   */
  async generateWorld(request: GenerateWorldRequest): Promise<GenerateWorldResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: request.sessionId,
        subject: request.subject,
        topic: request.topic,
        prompt: request.prompt,
        learning_objective: request.learningObjective,
        reference_image_url: request.referenceImageUrl,
        camera: request.camera,
        tutoring_session_id: request.tutoringSessionId,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to generate world');
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      status: data.status,
      websocketUrl: data.websocket_url,
      estimatedStartTimeMs: data.estimated_start_time_ms,
      message: data.message,
    };
  }

  /**
   * Get world session status
   */
  async getSession(sessionId: string): Promise<WorldSession> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/${sessionId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get session');
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      userId: data.user_id,
      tutoringSessionId: data.tutoring_session_id,
      subject: data.subject,
      topic: data.topic,
      prompt: data.prompt,
      learningObjective: data.learning_objective,
      status: data.status,
      frameCount: data.frame_count,
      durationSeconds: data.duration_seconds,
      createdAt: data.created_at,
      lastInteraction: data.last_interaction,
      errorMessage: data.error_message,
    };
  }

  /**
   * Pause world generation
   */
  async pauseWorld(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/${sessionId}/pause`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to pause world');
    }
  }

  /**
   * Resume world generation
   */
  async resumeWorld(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/${sessionId}/resume`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to resume world');
    }
  }

  /**
   * Transition to a new scene
   */
  async transitionScene(
    sessionId: string,
    newPrompt: string,
    transitionType: string = 'fade',
    durationMs: number = 1000
  ): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/${sessionId}/transition`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        new_prompt: newPrompt,
        transition_type: transitionType,
        duration_ms: durationMs,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to transition scene');
    }
  }

  /**
   * End world session
   */
  async endSession(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/v1/worlds/${sessionId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to end session');
    }
  }

  /**
   * List available subjects
   */
  async listSubjects(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/subjects`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to list subjects');
    }

    const data = await response.json();
    return data.subjects;
  }

  /**
   * List topics for a subject
   */
  async listTopics(subject: string): Promise<Topic[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/subjects/${subject}/topics`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to list topics');
    }

    const data = await response.json();
    return data.topics;
  }

  /**
   * Get world templates for a topic
   */
  async getTopicTemplates(subject: string, topic: string): Promise<{
    templates: string[];
    defaultCamera: string;
    interactiveElements: string[];
  }> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/subjects/${subject}/topics/${topic}/templates`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get templates');
    }

    const data = await response.json();
    return {
      templates: data.templates,
      defaultCamera: data.default_camera,
      interactiveElements: data.interactive_elements,
    };
  }

  /**
   * Get pre-generated worlds for a topic
   */
  async getPregeneratedWorlds(subject: string, topic: string): Promise<any[]> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/worlds/pregenerated/${subject}/${topic}`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get pre-generated worlds');
    }

    const data = await response.json();
    return data.pregenerated_worlds;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; gpuWorker: string; redis: string }> {
    const response = await fetch(`${this.baseUrl}/health`);

    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();
    return {
      status: data.status,
      gpuWorker: data.gpu_worker,
      redis: data.redis,
    };
  }
}

// Export singleton instance
export const worldPlayApi = new WorldPlayApi();

export default worldPlayApi;
