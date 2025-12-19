/**
 * WorldPlaybackEngine - Seamless playback of pre-generated worlds
 *
 * This engine handles:
 * - Loading and caching video segments from CDN/storage
 * - Seamless transitions between segments using dual video elements
 * - Position tracking for AI tutor integration
 * - Preloading of next possible segments
 *
 * NOTE: We use dual video element switching instead of MediaSource API
 * because HY-WorldPlay outputs standard MP4 files (not fragmented MP4).
 */

// CDN URL for world library (configurable)
const WORLD_CDN_URL = import.meta.env.VITE_WORLD_CDN_URL || 'http://localhost:8010/worlds';

export type Action = 'idle' | 'forward' | 'backward' | 'left' | 'right' | 'turn_left' | 'turn_right';

export interface WorldMetadata {
  id: string;
  name: string;
  subject: string;
  description: string;
  positions: Record<string, string>;
  educationalPoints?: string[];
}

export interface WorldState {
  worldId: string;
  currentSegment: string;
  position: string;
  positionHistory: string[];
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface PlaybackCallbacks {
  onPositionChange?: (position: string, description: string) => void;
  onSegmentStart?: (segmentId: string) => void;
  onSegmentEnd?: (segmentId: string) => void;
  onError?: (error: string) => void;
  onReady?: () => void;
}

/**
 * World Playback Engine
 *
 * Uses dual video elements for seamless transitions between pre-generated segments.
 * One video plays while the next one preloads, then we crossfade.
 */
export class WorldPlaybackEngine {
  private state: WorldState;
  private metadata: WorldMetadata | null = null;
  private activeVideo: HTMLVideoElement | null = null;
  private preloadVideo: HTMLVideoElement | null = null;
  private container: HTMLElement | null = null;
  private preloadedUrls: Map<string, string> = new Map();
  private loadingSegments: Set<string> = new Set();
  private callbacks: PlaybackCallbacks = {};
  private isInitialized = false;
  private transitionDuration = 300; // ms for crossfade

  constructor(worldId: string) {
    this.state = {
      worldId,
      currentSegment: 'start',
      position: 'start',
      positionHistory: ['start'],
      isPlaying: false,
      isLoading: true,
      error: null,
    };
  }

  /**
   * Initialize the playback engine
   * @param container - Container element where videos will be rendered
   * @param callbacks - Event callbacks
   */
  async initialize(
    container: HTMLElement,
    callbacks: PlaybackCallbacks = {}
  ): Promise<void> {
    this.container = container;
    this.callbacks = callbacks;

    try {
      // Create dual video elements for seamless transitions
      this.activeVideo = this.createVideoElement('active');
      this.preloadVideo = this.createVideoElement('preload');

      this.container.appendChild(this.activeVideo);
      this.container.appendChild(this.preloadVideo);

      // Load world metadata
      await this.loadMetadata();

      // Load and play starting segment
      await this.loadAndPlaySegment('start');

      // Preload possible next segments
      this.preloadNextSegments();

      this.isInitialized = true;
      this.state.isLoading = false;
      this.callbacks.onReady?.();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to initialize';
      this.state.error = errorMessage;
      this.callbacks.onError?.(errorMessage);
      throw error;
    }
  }

  /**
   * Alternative initialization with existing video element
   */
  async initializeWithVideo(
    videoElement: HTMLVideoElement,
    callbacks: PlaybackCallbacks = {}
  ): Promise<void> {
    this.activeVideo = videoElement;
    this.callbacks = callbacks;

    // Create preload video as hidden sibling
    this.preloadVideo = this.createVideoElement('preload');
    this.preloadVideo.style.display = 'none';
    videoElement.parentElement?.appendChild(this.preloadVideo);

    try {
      await this.loadMetadata();
      await this.loadAndPlaySegment('start');
      this.preloadNextSegments();

      this.isInitialized = true;
      this.state.isLoading = false;
      this.callbacks.onReady?.();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to initialize';
      this.state.error = errorMessage;
      this.callbacks.onError?.(errorMessage);
      throw error;
    }
  }

  /**
   * Create a video element with proper styling
   */
  private createVideoElement(id: string): HTMLVideoElement {
    const video = document.createElement('video');
    video.id = `world-video-${id}`;
    video.className = 'world-video';

    // CRITICAL: Preload video must start hidden
    const isPreload = id === 'preload';
    video.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      transition: opacity ${this.transitionDuration}ms ease-in-out;
      opacity: ${isPreload ? '0' : '1'};
      display: ${isPreload ? 'none' : 'block'};
    `;
    video.playsInline = true;
    video.muted = true; // Audio handled separately
    video.preload = 'auto';

    // Handle video end - loop or wait for next action
    video.addEventListener('ended', () => {
      this.callbacks.onSegmentEnd?.(this.state.currentSegment);
    });

    return video;
  }

  /**
   * Load world metadata from CDN
   */
  private async loadMetadata(): Promise<void> {
    const url = `${WORLD_CDN_URL}/${this.state.worldId}/metadata.json`;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to load metadata: ${response.status}`);
      const data = await response.json();
      this.metadata = data.world as WorldMetadata;
    } catch (error) {
      console.warn('Could not load metadata from CDN, using defaults');
      this.metadata = {
        id: this.state.worldId,
        name: this.state.worldId,
        subject: 'general',
        description: '',
        positions: {},
      };
    }
  }

  /**
   * Get URL for a video segment
   */
  private getSegmentUrl(segmentId: string): string {
    return `${WORLD_CDN_URL}/${this.state.worldId}/${segmentId}.mp4`;
  }

  /**
   * Load and play a video segment
   * Falls back to demo mode if videos aren't available
   */
  private async loadAndPlaySegment(segmentId: string): Promise<void> {
    if (!this.activeVideo) return;

    const url = this.getSegmentUrl(segmentId);

    return new Promise((resolve, reject) => {
      const video = this.activeVideo!;

      const onCanPlay = () => {
        video.removeEventListener('canplay', onCanPlay);
        video.removeEventListener('error', onError);
        video.play()
          .then(() => {
            this.state.isPlaying = true;
            resolve();
          })
          .catch(reject);
      };

      const onError = () => {
        video.removeEventListener('canplay', onCanPlay);
        video.removeEventListener('error', onError);

        // FALLBACK: If video doesn't exist, use a placeholder/demo mode
        console.warn(`Video not found: ${url}. Running in demo mode.`);

        // Create a canvas-based placeholder that shows position info
        this.createDemoPlaceholder(segmentId);
        this.state.isPlaying = true;
        resolve();
      };

      video.addEventListener('canplay', onCanPlay);
      video.addEventListener('error', onError);
      video.src = url;
      video.load();
    });
  }

  /**
   * Create a demo placeholder when videos aren't available
   */
  private createDemoPlaceholder(segmentId: string): void {
    if (!this.container || !this.activeVideo) return;

    // Hide video element
    this.activeVideo.style.display = 'none';

    // Check if placeholder already exists
    let placeholder = this.container.querySelector('.demo-placeholder') as HTMLDivElement;
    if (!placeholder) {
      placeholder = document.createElement('div');
      placeholder.className = 'demo-placeholder';
      placeholder.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: white;
        font-family: system-ui, sans-serif;
      `;
      this.container.appendChild(placeholder);
    }

    const description = this.getPositionDescription();
    const worldName = this.metadata?.name || this.state.worldId;

    placeholder.innerHTML = `
      <div style="text-align: center; max-width: 600px; padding: 20px;">
        <div style="font-size: 48px; margin-bottom: 20px;">🌌</div>
        <h2 style="font-size: 28px; margin-bottom: 10px; background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
          ${worldName}
        </h2>
        <p style="font-size: 18px; color: #a0aec0; margin-bottom: 30px;">
          ${description}
        </p>
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
          <p style="font-size: 14px; color: #718096; margin-bottom: 10px;">DEMO MODE</p>
          <p style="font-size: 16px; color: #e2e8f0;">
            Use <kbd style="background: #4a5568; padding: 4px 8px; border-radius: 4px;">W</kbd>
            <kbd style="background: #4a5568; padding: 4px 8px; border-radius: 4px;">A</kbd>
            <kbd style="background: #4a5568; padding: 4px 8px; border-radius: 4px;">S</kbd>
            <kbd style="background: #4a5568; padding: 4px 8px; border-radius: 4px;">D</kbd>
            to navigate
          </p>
        </div>
        <p style="font-size: 12px; color: #4a5568;">
          Position: ${this.state.position} | Segment: ${segmentId}
        </p>
        <p style="font-size: 11px; color: #4a5568; margin-top: 20px;">
          Generate real videos with: python scripts/generate_world_library.py --world ${this.state.worldId}
        </p>
      </div>
    `;
  }

  /**
   * Preload a video segment in the background
   */
  private preloadSegment(segmentId: string): void {
    if (this.preloadedUrls.has(segmentId)) return;
    if (this.loadingSegments.has(segmentId)) return;

    this.loadingSegments.add(segmentId);
    const url = this.getSegmentUrl(segmentId);

    // Use link preload for better browser caching
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'video';
    link.href = url;
    document.head.appendChild(link);

    // Also preload in hidden video element
    if (this.preloadVideo) {
      this.preloadVideo.src = url;
      this.preloadVideo.load();
    }

    this.preloadedUrls.set(segmentId, url);
    this.loadingSegments.delete(segmentId);
  }

  /**
   * Preload next possible segments based on current position
   */
  private preloadNextSegments(): void {
    const actions: Action[] = ['forward', 'backward', 'left', 'right', 'turn_left', 'turn_right'];

    for (const action of actions) {
      const nextSegment = this.getNextSegmentId(action);
      this.preloadSegment(nextSegment);
    }
  }

  /**
   * Get the segment ID for a given action from current position
   */
  private getNextSegmentId(action: Action): string {
    if (this.state.position === 'start') {
      return `from_start_${action}`;
    }

    // For nested positions, use the last action in the path
    const parts = this.state.position.split('_');
    const lastAction = parts[parts.length - 1];
    return `from_${lastAction}_${action}`;
  }

  /**
   * Handle user action (WASD controls)
   */
  async handleAction(action: Action): Promise<void> {
    if (!this.isInitialized || this.state.isLoading) {
      console.warn('Engine not ready, ignoring action');
      return;
    }

    if (action === 'idle') {
      return; // Continue current playback
    }

    await this.transitionToSegment(action);
  }

  /**
   * Transition to the next segment with crossfade
   * Falls back to demo mode if videos aren't available
   */
  private async transitionToSegment(action: Action): Promise<void> {
    const nextSegment = this.getNextSegmentId(action);
    const url = this.getSegmentUrl(nextSegment);

    try {
      this.state.isLoading = true;

      // Swap video elements
      const currentActive = this.activeVideo;
      const nextActive = this.preloadVideo;

      if (!currentActive || !nextActive) return;

      // Try to load next segment into preload video
      const videoLoaded = await new Promise<boolean>((resolve) => {
        const onCanPlay = () => {
          nextActive.removeEventListener('canplay', onCanPlay);
          nextActive.removeEventListener('error', onError);
          resolve(true);
        };
        const onError = () => {
          nextActive.removeEventListener('canplay', onCanPlay);
          nextActive.removeEventListener('error', onError);
          resolve(false); // Video not available, use demo mode
        };

        nextActive.addEventListener('canplay', onCanPlay);
        nextActive.addEventListener('error', onError);
        nextActive.src = url;
        nextActive.load();
      });

      // Update state first (needed for demo placeholder)
      const newPosition = this.state.position === 'start'
        ? action
        : `${this.state.position}_${action}`;

      this.state.currentSegment = nextSegment;
      this.state.position = newPosition;
      this.state.positionHistory.push(newPosition);

      if (videoLoaded) {
        // Real video transition with crossfade
        nextActive.style.opacity = '0';
        nextActive.style.display = 'block';
        await nextActive.play();

        // Animate crossfade
        requestAnimationFrame(() => {
          nextActive.style.opacity = '1';
          currentActive.style.opacity = '0';
        });

        // After transition, swap roles
        setTimeout(() => {
          currentActive.pause();
          currentActive.style.display = 'none';

          // Swap references
          this.activeVideo = nextActive;
          this.preloadVideo = currentActive;
        }, this.transitionDuration);
      } else {
        // Demo mode - update placeholder
        console.warn(`Video not found: ${url}. Updating demo placeholder.`);
        this.createDemoPlaceholder(nextSegment);
      }

      this.state.isLoading = false;

      // Notify callbacks
      this.callbacks.onSegmentStart?.(nextSegment);
      const description = this.getPositionDescription(newPosition);
      this.callbacks.onPositionChange?.(newPosition, description);

      // Preload next segments (will silently fail in demo mode)
      this.preloadNextSegments();
    } catch (error) {
      this.state.isLoading = false;
      const errorMessage = error instanceof Error ? error.message : 'Failed to load segment';
      this.state.error = errorMessage;
      this.callbacks.onError?.(errorMessage);
    }
  }

  /**
   * Get human-readable description of current position
   */
  getPositionDescription(position: string = this.state.position): string {
    if (this.metadata?.positions[position]) {
      return this.metadata.positions[position];
    }

    // Generate description from action path
    const parts = position.split('_');
    const actionDescriptions: Record<string, string> = {
      start: 'Starting position',
      forward: 'moving forward',
      backward: 'moving backward',
      left: 'moving left',
      right: 'moving right',
      turn_left: 'turning left',
      turn_right: 'turning right',
    };

    const descriptions = parts.map((p) => actionDescriptions[p] || p);
    return descriptions.join(', ');
  }

  /**
   * Play the video
   */
  async play(): Promise<void> {
    if (this.activeVideo) {
      await this.activeVideo.play();
      this.state.isPlaying = true;
    }
  }

  /**
   * Pause the video
   */
  pause(): void {
    if (this.activeVideo) {
      this.activeVideo.pause();
      this.state.isPlaying = false;
    }
  }

  /**
   * Get current state
   */
  getState(): Readonly<WorldState> {
    return { ...this.state };
  }

  /**
   * Get world metadata
   */
  getMetadata(): WorldMetadata | null {
    return this.metadata;
  }

  /**
   * Get current position for AI tutor
   */
  getCurrentPosition(): string {
    return this.state.position;
  }

  /**
   * Get position history for AI context
   */
  getPositionHistory(): string[] {
    return [...this.state.positionHistory];
  }

  /**
   * AI tutor can guide to a direction
   */
  async guideTo(action: Action): Promise<void> {
    console.log(`AI guiding to: ${action}`);
    await this.handleAction(action);
  }

  /**
   * AI tutor can play a guided sequence
   */
  async playGuidedSequence(actions: Action[]): Promise<void> {
    console.log('Playing guided sequence:', actions);
    for (const action of actions) {
      await this.handleAction(action);
      await this.waitForSegmentEnd();
    }
  }

  /**
   * Wait for current segment to finish
   */
  private waitForSegmentEnd(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.activeVideo) {
        resolve();
        return;
      }

      const video = this.activeVideo;

      if (video.ended) {
        resolve();
        return;
      }

      const onEnded = () => {
        video.removeEventListener('ended', onEnded);
        resolve();
      };

      video.addEventListener('ended', onEnded);
    });
  }

  /**
   * Get the active video element (for external use like fullscreen)
   */
  getActiveVideoElement(): HTMLVideoElement | null {
    return this.activeVideo;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.activeVideo) {
      this.activeVideo.pause();
      this.activeVideo.src = '';
      this.activeVideo.remove();
    }

    if (this.preloadVideo) {
      this.preloadVideo.pause();
      this.preloadVideo.src = '';
      this.preloadVideo.remove();
    }

    // Remove preload links
    document.querySelectorAll('link[rel="preload"][as="video"]').forEach((el) => el.remove());

    this.preloadedUrls.clear();
    this.loadingSegments.clear();
    this.isInitialized = false;
  }
}

/**
 * Create a WorldPlaybackEngine instance
 */
export function createWorldPlaybackEngine(worldId: string): WorldPlaybackEngine {
  return new WorldPlaybackEngine(worldId);
}

export default WorldPlaybackEngine;
