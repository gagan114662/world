/**
 * useWorldAudio - Audio system for immersive world exploration
 *
 * Provides:
 * - Ambient background music per world
 * - Sound effects for navigation
 * - Audio ducking when AI is speaking
 * - Volume controls
 */

import { useRef, useState, useEffect, useCallback } from 'react';
import { Action } from '../services/worldPlayback';

// Audio CDN URL (configurable)
const AUDIO_CDN_URL = import.meta.env.VITE_AUDIO_CDN_URL || '/audio';

// Ambient tracks for each world type
const AMBIENT_TRACKS: Record<string, string> = {
  solar_system: 'ambient/space.mp3',
  human_cell: 'ambient/cellular.mp3',
  ancient_rome: 'ambient/ancient_city.mp3',
  ocean_deep: 'ambient/underwater.mp3',
  volcano_interior: 'ambient/volcano.mp3',
  rainforest_canopy: 'ambient/rainforest.mp3',
  dna_helix: 'ambient/molecular.mp3',
  medieval_castle: 'ambient/medieval.mp3',
  atom_structure: 'ambient/quantum.mp3',
  egyptian_pyramid: 'ambient/ancient_egypt.mp3',
  // Default fallback
  default: 'ambient/exploration.mp3',
};

// Navigation sound effects
const NAVIGATION_SOUNDS: Record<Action, string> = {
  idle: '',
  forward: 'sfx/whoosh_forward.mp3',
  backward: 'sfx/whoosh_back.mp3',
  left: 'sfx/whoosh_left.mp3',
  right: 'sfx/whoosh_right.mp3',
  turn_left: 'sfx/turn_left.mp3',
  turn_right: 'sfx/turn_right.mp3',
};

export interface AudioSettings {
  ambientVolume: number;
  sfxVolume: number;
  narrationVolume: number;
  masterVolume: number;
  muted: boolean;
}

export interface UseWorldAudioOptions {
  worldId: string;
  autoPlay?: boolean;
  initialSettings?: Partial<AudioSettings>;
}

export interface UseWorldAudioReturn {
  // State
  isReady: boolean;
  isPlaying: boolean;
  settings: AudioSettings;

  // Controls
  playAmbient: () => void;
  pauseAmbient: () => void;
  playNavigationSound: (action: Action) => void;
  toggleMute: () => void;

  // Volume controls
  setAmbientVolume: (volume: number) => void;
  setSfxVolume: (volume: number) => void;
  setNarrationVolume: (volume: number) => void;
  setMasterVolume: (volume: number) => void;

  // Narration integration
  onNarrationStart: () => void;
  onNarrationEnd: () => void;
}

export function useWorldAudio(options: UseWorldAudioOptions): UseWorldAudioReturn {
  const { worldId, autoPlay = true, initialSettings } = options;

  // Audio elements
  const ambientRef = useRef<HTMLAudioElement | null>(null);
  const sfxRefs = useRef<Map<string, HTMLAudioElement>>(new Map());

  // State
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [settings, setSettings] = useState<AudioSettings>({
    ambientVolume: 0.3,
    sfxVolume: 0.5,
    narrationVolume: 1.0,
    masterVolume: 1.0,
    muted: false,
    ...initialSettings,
  });

  // Original ambient volume (for ducking)
  const originalAmbientVolumeRef = useRef(settings.ambientVolume);

  // Initialize ambient audio
  useEffect(() => {
    const ambientTrack = AMBIENT_TRACKS[worldId] || AMBIENT_TRACKS.default;
    const ambientUrl = `${AUDIO_CDN_URL}/${ambientTrack}`;

    const audio = new Audio(ambientUrl);
    audio.loop = true;
    audio.volume = settings.ambientVolume * settings.masterVolume;

    audio.addEventListener('canplaythrough', () => {
      setIsReady(true);
      if (autoPlay && !settings.muted) {
        audio.play().catch((err) => console.warn('Auto-play blocked:', err));
        setIsPlaying(true);
      }
    });

    audio.addEventListener('error', (e) => {
      console.warn('Failed to load ambient audio:', e);
      // Still mark as ready so the app can function
      setIsReady(true);
    });

    ambientRef.current = audio;

    return () => {
      audio.pause();
      audio.src = '';
    };
  }, [worldId]);

  // Update ambient volume when settings change
  useEffect(() => {
    if (ambientRef.current) {
      ambientRef.current.volume = settings.muted
        ? 0
        : settings.ambientVolume * settings.masterVolume;
    }
  }, [settings.ambientVolume, settings.masterVolume, settings.muted]);

  // Preload navigation sound effects
  useEffect(() => {
    const preloadSfx = async () => {
      for (const [action, soundPath] of Object.entries(NAVIGATION_SOUNDS)) {
        if (!soundPath) continue;

        const url = `${AUDIO_CDN_URL}/${soundPath}`;
        const audio = new Audio(url);
        audio.preload = 'auto';
        audio.volume = settings.sfxVolume * settings.masterVolume;
        sfxRefs.current.set(action, audio);
      }
    };

    preloadSfx();

    return () => {
      sfxRefs.current.forEach((audio) => {
        audio.pause();
        audio.src = '';
      });
      sfxRefs.current.clear();
    };
  }, []);

  // Update SFX volumes when settings change
  useEffect(() => {
    sfxRefs.current.forEach((audio) => {
      audio.volume = settings.muted ? 0 : settings.sfxVolume * settings.masterVolume;
    });
  }, [settings.sfxVolume, settings.masterVolume, settings.muted]);

  // Play ambient audio
  const playAmbient = useCallback(() => {
    if (ambientRef.current && !settings.muted) {
      ambientRef.current.play().catch((err) => console.warn('Play failed:', err));
      setIsPlaying(true);
    }
  }, [settings.muted]);

  // Pause ambient audio
  const pauseAmbient = useCallback(() => {
    if (ambientRef.current) {
      ambientRef.current.pause();
      setIsPlaying(false);
    }
  }, []);

  // Play navigation sound effect
  const playNavigationSound = useCallback(
    (action: Action) => {
      if (settings.muted || action === 'idle') return;

      const audio = sfxRefs.current.get(action);
      if (audio) {
        // Clone and play to allow overlapping sounds
        const clone = audio.cloneNode() as HTMLAudioElement;
        clone.volume = settings.sfxVolume * settings.masterVolume;
        clone.play().catch((err) => console.warn('SFX play failed:', err));
      }
    },
    [settings.sfxVolume, settings.masterVolume, settings.muted]
  );

  // Toggle mute
  const toggleMute = useCallback(() => {
    setSettings((prev) => {
      const newMuted = !prev.muted;
      if (ambientRef.current) {
        ambientRef.current.volume = newMuted ? 0 : prev.ambientVolume * prev.masterVolume;
      }
      return { ...prev, muted: newMuted };
    });
  }, []);

  // Volume controls
  const setAmbientVolume = useCallback((volume: number) => {
    const clamped = Math.max(0, Math.min(1, volume));
    setSettings((prev) => ({ ...prev, ambientVolume: clamped }));
    originalAmbientVolumeRef.current = clamped;
  }, []);

  const setSfxVolume = useCallback((volume: number) => {
    const clamped = Math.max(0, Math.min(1, volume));
    setSettings((prev) => ({ ...prev, sfxVolume: clamped }));
  }, []);

  const setNarrationVolume = useCallback((volume: number) => {
    const clamped = Math.max(0, Math.min(1, volume));
    setSettings((prev) => ({ ...prev, narrationVolume: clamped }));
  }, []);

  const setMasterVolume = useCallback((volume: number) => {
    const clamped = Math.max(0, Math.min(1, volume));
    setSettings((prev) => ({ ...prev, masterVolume: clamped }));
  }, []);

  // Narration integration - duck ambient when AI speaks
  const onNarrationStart = useCallback(() => {
    if (ambientRef.current && !settings.muted) {
      // Fade ambient to 30% of its original volume
      const targetVolume = originalAmbientVolumeRef.current * 0.3 * settings.masterVolume;
      fadeAudioTo(ambientRef.current, targetVolume, 500);
    }
  }, [settings.masterVolume, settings.muted]);

  const onNarrationEnd = useCallback(() => {
    if (ambientRef.current && !settings.muted) {
      // Restore ambient to original volume
      const targetVolume = originalAmbientVolumeRef.current * settings.masterVolume;
      fadeAudioTo(ambientRef.current, targetVolume, 500);
    }
  }, [settings.masterVolume, settings.muted]);

  return {
    isReady,
    isPlaying,
    settings,
    playAmbient,
    pauseAmbient,
    playNavigationSound,
    toggleMute,
    setAmbientVolume,
    setSfxVolume,
    setNarrationVolume,
    setMasterVolume,
    onNarrationStart,
    onNarrationEnd,
  };
}

/**
 * Fade audio to target volume over duration
 */
function fadeAudioTo(audio: HTMLAudioElement, targetVolume: number, durationMs: number): void {
  const startVolume = audio.volume;
  const delta = targetVolume - startVolume;
  const steps = 20;
  const stepDuration = durationMs / steps;
  const stepDelta = delta / steps;

  let currentStep = 0;

  const fadeInterval = setInterval(() => {
    currentStep++;
    audio.volume = Math.max(0, Math.min(1, startVolume + stepDelta * currentStep));

    if (currentStep >= steps) {
      clearInterval(fadeInterval);
      audio.volume = targetVolume;
    }
  }, stepDuration);
}

export default useWorldAudio;
