/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useRef, useState, useEffect, Suspense, lazy } from "react";
import "./App.scss";
import "./styles/mobile-fixes.css"; // Mobile UI fixes
import { TutorProvider } from "./features/tutor";
import { TutorToolController } from "./features/tutor/TutorToolController";
import AuthGuard from "./components/auth/AuthGuard";
import Header from "./components/header/Header";
import BackgroundShapes from "./components/background-shapes/BackgroundShapes";
import QuestionDisplay from "./components/question-display/QuestionDisplay";
import Scratchpad from "./components/scratchpad/Scratchpad";
import { ThemeProvider } from "./components/theme/theme-provier";
import { HintProvider } from "./contexts/HintContext";
import { Toaster } from "@/components/ui/sonner";
import { useMediaMixer } from "./hooks/useMediaMixer";
import { useMediaCapture } from "./hooks/useMediaCapture";

// Lazy load heavy components
const SidePanel = lazy(() => import("./components/side-panel/SidePanel"));
const GradingSidebar = lazy(() => import("./components/grading-sidebar/GradingSidebar"));
const ScratchpadCapture = lazy(() => import("./components/scratchpad-capture/ScratchpadCapture"));
const FloatingControlPanel = lazy(() => import("./components/floating-control-panel/FloatingControlPanel"));

// Lazy load WorldViewer demo (access via ?demo=world)
const WorldViewerDemo = lazy(() => import("./components/WorldViewer/WorldViewerDemo"));
const ExplainToLearn = lazy(() => import("./components/analysis/ExplainToLearn"));

function App() {
  // Check for demo mode via URL parameter (?demo=world)
  const urlParams = new URLSearchParams(window.location.search);
  const demoMode = urlParams.get('demo');

  // If demo=world, render the WorldViewer demo instead of the main app
  if (demoMode === 'world') {
    return (
      <ThemeProvider defaultTheme="dark" storageKey="ai-tutor-theme">
        <Suspense fallback={
          <div className="min-h-screen bg-gray-900 flex items-center justify-center">
            <div className="text-white text-xl">Loading World Viewer...</div>
          </div>
        }>
          <WorldViewerDemo />
        </Suspense>
      </ThemeProvider>
    );
  }

  // this video reference is used for displaying the active stream, whether that is the webcam or screen capture
  // feel free to style as you see fit
  const videoRef = useRef<HTMLVideoElement>(null);
  const renderCanvasRef = useRef<HTMLCanvasElement>(null);
  // either the screen capture, the video or null, if null we hide it
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [mixerStream, setMixerStream] = useState<MediaStream | null>(null);
  const mixerVideoRef = useRef<HTMLVideoElement>(null);
  const [isScratchpadOpen, setScratchpadOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isGradingSidebarOpen, setIsGradingSidebarOpen] = useState(false);
  const [currentSkill, setCurrentSkill] = useState<string | null>(null);
  const [isExplainToLearnOpen, setIsExplainToLearnOpen] = useState(false);

  // Ref to hold mediaMixer instance for use in callbacks
  const mediaMixerRef = useRef<any>(null);

  // Media capture with frame callbacks - must be called before useMediaMixer
  // Media capture with frame callbacks - must be called before useMediaMixer
  // Optimized: No longer using callbacks for frames, but exposing video refs directly
  const {
    cameraEnabled,
    screenEnabled,
    toggleCamera,
    toggleScreen,
    cameraVideoRef,
    screenVideoRef
  } = useMediaCapture({});

  // MediaMixer hook for local video mixing - uses state from useMediaCapture
  const mediaMixer = useMediaMixer({
    width: 1280,
    height: 2160,
    fps: 2,  // Reduced from 10 to 2 FPS for better performance
    quality: 0.85,
    cameraEnabled: cameraEnabled,
    screenEnabled: screenEnabled,
    cameraVideoRef: cameraVideoRef,
    screenVideoRef: screenVideoRef
  });

  // Store mediaMixer in ref for use in callbacks
  useEffect(() => {
    mediaMixerRef.current = mediaMixer;
  }, [mediaMixer]);

  // Start mixer when component mounts and canvas is available
  useEffect(() => {
    if (mediaMixer.canvasRef.current) {
      mediaMixer.setIsRunning(true);
      return () => {
        mediaMixer.setIsRunning(false);
      };
    }
  }, [mediaMixer]);

  const toggleSidebar = () => {
    if (!isSidebarOpen) setIsGradingSidebarOpen(false);
    setIsSidebarOpen(!isSidebarOpen);
  };

  const toggleGradingSidebar = () => {
    if (!isGradingSidebarOpen) setIsSidebarOpen(false);
    setIsGradingSidebarOpen(!isGradingSidebarOpen);
  };

  useEffect(() => {
    if (mixerVideoRef.current && mixerStream) {
      mixerVideoRef.current.srcObject = mixerStream;
    }
  }, [mixerStream]);

  return (
    <ThemeProvider defaultTheme="light" storageKey="ai-tutor-theme">
      <div className="App">
        <AuthGuard>
          <TutorProvider>
            <HintProvider>
              <Header
                sidebarOpen={isSidebarOpen}
                onToggleSidebar={toggleSidebar}
                onToggleExplainToLearn={() => setIsExplainToLearnOpen(true)}
              />
              <TutorToolController />
              <div className="streaming-console">
                <Suspense fallback={<div className="flex items-center justify-center h-full w-full">Loading...</div>}>
                  <SidePanel
                    open={isSidebarOpen}
                    onToggle={toggleSidebar}
                  />
                  <GradingSidebar
                    open={isGradingSidebarOpen}
                    onToggle={toggleGradingSidebar}
                    currentSkill={currentSkill}
                  />
                  <main style={{
                    marginRight: isSidebarOpen ? "260px" : "0",
                    marginLeft: isGradingSidebarOpen ? "260px" : "40px",
                    transition: "all 0.5s cubic-bezier(0.16, 1, 0.3, 1)"
                  }}>
                    <div className="main-app-area">
                      <div className="question-panel">
                        <BackgroundShapes />
                        <ScratchpadCapture onFrameCaptured={(canvas) => {
                          mediaMixer.updateScratchpadFrame(canvas);
                        }}>
                          <QuestionDisplay onSkillChange={setCurrentSkill} />
                          {isScratchpadOpen && (
                            <div className="scratchpad-container">
                              <Scratchpad />
                            </div>
                          )}
                        </ScratchpadCapture>
                      </div>
                      <FloatingControlPanel
                        renderCanvasRef={mediaMixer.canvasRef}
                        videoRef={videoRef}
                        supportsVideo={true}
                        onVideoStreamChange={setVideoStream}
                        onMixerStreamChange={setMixerStream}
                        enableEditingSettings={true}
                        onPaintClick={() => setScratchpadOpen(!isScratchpadOpen)}
                        isPaintActive={isScratchpadOpen}
                        cameraEnabled={cameraEnabled}
                        screenEnabled={screenEnabled}
                        onToggleCamera={toggleCamera}
                        onToggleScreen={toggleScreen}
                        mediaMixerCanvasRef={mediaMixer.canvasRef}
                      />
                    </div>
                  </main>
                </Suspense>

                {/* Explain to Learn Overlay */}
                {isExplainToLearnOpen && (
                  <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-4 md:p-8 animate-in fade-in duration-300">
                    <div className="absolute inset-0" onClick={() => setIsExplainToLearnOpen(false)} />
                    <div className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto custom-scrollbar shadow-2xl rounded-3xl animate-in zoom-in-95 slide-in-from-bottom-8 duration-500">
                      <Suspense fallback={<div className="bg-slate-900 text-white p-20 text-center rounded-3xl">Loading Lab Experience...</div>}>
                        <ExplainToLearn />
                      </Suspense>
                      <button
                        onClick={() => setIsExplainToLearnOpen(false)}
                        className="absolute top-6 right-6 p-2 bg-white/10 hover:bg-white/20 rounded-full text-white transition-colors"
                      >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                      </button>
                    </div>
                  </div>
                )}
              </div>
              <Toaster richColors closeButton />
            </HintProvider>
          </TutorProvider>
        </AuthGuard>
      </div>
    </ThemeProvider>
  );
}

export default App;
