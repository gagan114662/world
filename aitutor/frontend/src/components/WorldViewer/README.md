# WorldViewer - Immersive AI Tutor

This module provides an immersive 3D world exploration experience powered by pre-generated HY-WorldPlay videos with real-time AI tutoring.

## Components

### Core Components

- **`HybridWorldViewer.tsx`** - Main component combining video playback with interactive overlays
- **`SeamlessPlayer.tsx`** - Video player with dual-element switching for seamless transitions
- **`MathOverlay.tsx`** - Interactive overlay for math/abstract topics

### Services

- **`worldPlayback.ts`** - Video playback engine with segment management
- **`worldAI.ts`** - AI tutor integration (position-aware narration)
- **`educationService.ts`** - Curriculum alignment, progress tracking, personalization

### Hooks

- **`useWorldPlayback.ts`** - React hook for playback engine integration

## Usage

### Basic Usage

```tsx
import HybridWorldViewer from './components/WorldViewer/HybridWorldViewer';

<HybridWorldViewer
  worldId="solar_system"
  mode="explore"
  onSessionStart={(session) => console.log('Started', session)}
  onSessionEnd={(session) => console.log('Ended', session)}
/>
```

### With Student Profile (Personalization)

```tsx
const studentProfile = {
  id: 'student-123',
  name: 'Alice',
  gradeLevel: '7',
  age: 12,
  learningStyle: 'visual',
  pacePreference: 'medium',
  interests: ['space', 'science'],
  accommodations: [],
  language: 'en',
};

<HybridWorldViewer
  worldId="solar_system"
  mode="explore"
  studentProfile={studentProfile}
/>
```

### Teacher Mode

```tsx
<HybridWorldViewer
  worldId="solar_system"
  mode="lesson"
  teacherMode={true}
  onTeacherAction={(action, data) => {
    // Handle teacher actions: 'pause', 'guide', 'showProblem'
  }}
/>
```

### Practice Mode (with Math Problems)

```tsx
import { EXAMPLE_ALGEBRA_PROBLEMS } from './MathOverlay';

<HybridWorldViewer
  worldId="solar_system"
  mode="practice"
  problems={EXAMPLE_ALGEBRA_PROBLEMS}
/>
```

## Navigation Controls

| Key | Action |
|-----|--------|
| W / Arrow Up | Move Forward |
| S / Arrow Down | Move Backward |
| A | Strafe Left |
| D | Strafe Right |
| Q / Arrow Left | Turn Left |
| E / Arrow Right | Turn Right |

## Props

### HybridWorldViewerProps

| Prop | Type | Description |
|------|------|-------------|
| `worldId` | `string` | ID of the world to display |
| `mode` | `'explore' \| 'lesson' \| 'practice'` | Interaction mode |
| `studentProfile` | `StudentProfile` | Optional student info for personalization |
| `teacherMode` | `boolean` | Enable teacher controls |
| `problems` | `MathProblem[]` | Problems for practice mode |
| `onSessionStart` | `function` | Called when session starts |
| `onSessionEnd` | `function` | Called when session ends |
| `onProgress` | `function` | Called on position changes |

## Demo

Access the demo page by adding `?demo=world` to the URL:

```
http://localhost:5173/?demo=world
```

## World Library Structure

```
world_library/
├── solar_system/
│   ├── metadata.json
│   ├── start.mp4
│   ├── from_start_forward.mp4
│   ├── from_start_backward.mp4
│   ├── from_start_left.mp4
│   ├── from_start_right.mp4
│   ├── from_start_turn_left.mp4
│   ├── from_start_turn_right.mp4
│   └── from_forward_forward.mp4 (etc...)
├── human_cell/
│   └── ...
└── index.json
```

## Environment Variables

```env
VITE_WORLD_CDN_URL=http://localhost:8010/worlds
```

## Development

1. Create mock world data:
   ```bash
   python3 scripts/create_mock_worlds.py
   ```

2. Start the world library server:
   ```bash
   python3 scripts/serve_worlds.py --dir ./world_library
   ```

3. Start the frontend dev server:
   ```bash
   cd aitutor/frontend && npm run dev
   ```

4. Open the demo: `http://localhost:5173/?demo=world`

## Generating Real Videos

Videos are generated using HY-WorldPlay on a GPU server:

```bash
# Generate a single world
python3 scripts/generate_world_library.py --world solar_system

# Generate a custom world
python3 scripts/generate_world_library.py --custom \
    --name "Black Hole Journey" \
    --subject "astrophysics" \
    --prompt "Flying toward a supermassive black hole..."
```
