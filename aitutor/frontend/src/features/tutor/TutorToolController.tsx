import React, { useEffect, useState, useCallback } from 'react';
import { useTutorContext } from './index';
import { TUTOR_TOOLS } from './tools';
import { LiveServerToolCall, Modality } from "@google/genai";
import { apiUtils } from '../../lib/api-utils';
import { toast } from "sonner";
import { Altair } from '../../components/altair/Altair';

const TA_SERVICE_URL = import.meta.env.VITE_TA_SERVICE_URL || 'http://localhost:8002';

export interface Milestone {
    label: string;
    completed: boolean;
    current?: boolean;
}

export interface PythonResult {
    stdout: string;
    stderr: string;
    exit_code: number;
}

export const TutorToolController: React.FC = () => {
    const { client, setConfig, connected } = useTutorContext();
    const [lastPythonResult, setLastPythonResult] = useState<PythonResult | null>(null);
    const [thinkingLog, setThinkingLog] = useState<string[]>([]);
    const [milestones, setMilestones] = useState<Milestone[]>([]);
    const [annotations, setAnnotations] = useState<any[]>([]);
    const [altairGraph, setAltairGraph] = useState<string | null>(null);

    // Initialize config with all tools
    useEffect(() => {
        if (connected) {
            setConfig({
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } },
                },
                tools: [
                    { googleSearch: {} }, // Native Google Search
                    { functionDeclarations: TUTOR_TOOLS },
                ],
            });
        }
    }, [setConfig, connected]);

    const handleToolCall = useCallback(async (toolCall: LiveServerToolCall) => {
        if (!toolCall.functionCalls) return;

        const responses = await Promise.all(toolCall.functionCalls.map(async (fc) => {
            const args: any = fc.args;
            let output: any = { success: true };

            try {
                switch (fc.name) {
                    case "run_python_code":
                        const pyRes = await apiUtils.post(`${TA_SERVICE_URL}/tools/python-exec`, { code: args.code });
                        const pyData = await pyRes.json();
                        setLastPythonResult(pyData);
                        output = pyData;
                        toast.success("Python simulation complete");
                        break;

                    case "render_altair":
                        setAltairGraph(args.json_graph);
                        toast.success("Data visualization rendered");
                        break;

                    case "search_knowledge_base":
                        const sRes = await apiUtils.get(`${TA_SERVICE_URL}/tools/search?query=${encodeURIComponent(args.query)}`);
                        const sData = await sRes.json();
                        output = { results: sData };
                        break;

                    case "draw_overlay":
                        setAnnotations(args.annotations);
                        // Auto-clear annotations after 10 seconds
                        setTimeout(() => setAnnotations([]), 10000);
                        break;

                    case "log_thinking":
                        setThinkingLog(prev => [args.thought, ...prev].slice(0, 10));
                        break;

                    case "update_lesson_plan":
                        setMilestones(args.milestones);
                        break;

                    case "teleport_to_world":
                        toast.info(`Teleporting to: ${args.prompt}`);
                        // Here we would call the world generation service
                        // For now we'll just mock success
                        output = { success: true, world_id: "world_" + Date.now() };
                        break;
                }
            } catch (err) {
                console.error(`Error in tool ${fc.name}:`, err);
                output = { error: String(err) };
            }

            return {
                response: { output },
                id: fc.id,
                name: fc.name,
            };
        }));

        client.sendToolResponse({ functionResponses: responses });
    }, [client]);

    useEffect(() => {
        client.on("toolcall", handleToolCall);
        return () => {
            client.off("toolcall", handleToolCall);
        };
    }, [client, handleToolCall]);

    return (
        <>
            {/* Visual Overlays (Socratic Debugger) */}
            {annotations.length > 0 && (
                <div className="fixed inset-0 pointer-events-none z-[60]">
                    {annotations.map((ann, i) => (
                        <div
                            key={i}
                            style={{
                                position: 'absolute',
                                left: `${ann.x}%`,
                                top: `${ann.y}%`,
                                width: ann.width ? `${ann.width}%` : 'auto',
                                height: ann.height ? `${ann.height}%` : 'auto',
                                transform: 'translate(-50%, -50%)',
                                border: ann.type !== 'text' ? `3px solid ${ann.color || '#ff0000'}` : 'none',
                                borderRadius: ann.type === 'circle' ? '50%' : '4px',
                                background: ann.type === 'box' ? 'rgba(255,0,0,0.1)' : 'transparent',
                                color: ann.color || '#ff0000',
                                padding: '4px',
                                fontWeight: 'bold',
                                textShadow: '0 0 4px rgba(0,0,0,0.5)',
                            }}
                        >
                            {ann.type === 'text' && ann.label}
                            {ann.type === 'arrow' && '➜'}
                        </div>
                    ))}
                </div>
            )}

            {/* Python Output Panel */}
            {lastPythonResult && (
                <div className="fixed bottom-24 right-8 w-80 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-xl p-4 shadow-2xl z-50 animate-in slide-in-from-right duration-300">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-bold text-indigo-400 uppercase tracking-widest">Python Sandbox</span>
                        <button onClick={() => setLastPythonResult(null)} className="text-slate-500 hover:text-white">×</button>
                    </div>
                    {lastPythonResult.stdout && (
                        <pre className="text-xs text-green-400 bg-black/50 p-2 rounded overflow-x-auto max-h-32 mb-2">
                            {lastPythonResult.stdout}
                        </pre>
                    )}
                    {lastPythonResult.stderr && (
                        <pre className="text-xs text-red-400 bg-black/50 p-2 rounded overflow-x-auto max-h-32">
                            {lastPythonResult.stderr}
                        </pre>
                    )}
                </div>
            )}

            {/* Data Visualization (Altair) */}
            {altairGraph && (
                <div className="fixed bottom-24 left-8 w-[400px] bg-white dark:bg-slate-900 rounded-2xl p-4 shadow-2xl z-50 animate-in zoom-in duration-300 border-[3px] border-black dark:border-white">
                    <div className="flex justify-between items-center mb-4">
                        <span className="text-xs font-black uppercase tracking-widest bg-emerald-400 text-black px-2 py-1">Data Visualizer</span>
                        <button onClick={() => setAltairGraph(null)} className="font-bold hover:scale-110 transition-transform">✕</button>
                    </div>
                    <div className="bg-white p-2 rounded">
                        <Altair json_graph={altairGraph} />
                    </div>
                </div>
            )}

            {/* Lesson Plan / Milestones (Planner) */}
            {milestones.length > 0 && (
                <div className="fixed top-20 left-8 w-64 bg-slate-900/80 backdrop-blur border border-slate-700/50 rounded-xl p-4 shadow-xl z-30">
                    <h3 className="text-xs font-black text-slate-500 mb-3 uppercase tracking-tighter">Learning Journey</h3>
                    <div className="space-y-2">
                        {milestones.map((m, i) => (
                            <div key={i} className={`flex items-center gap-3 text-sm ${m.completed ? 'opacity-50' : ''}`}>
                                <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${m.completed ? 'bg-indigo-500 border-indigo-500' : m.current ? 'border-amber-400 animate-pulse' : 'border-slate-600'}`}>
                                    {m.completed && <span className="text-[10px] text-white">✓</span>}
                                </div>
                                <span className={`${m.current ? 'text-amber-400 font-bold' : 'text-slate-300'}`}>{m.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Meta-Cognitive Monologue (Teacher Only) */}
            {thinkingLog.length > 0 && (
                <div className="fixed top-20 right-8 w-64 opacity-30 hover:opacity-100 transition-opacity pointer-events-none hover:pointer-events-auto">
                    <div className="bg-purple-900/20 backdrop-blur border border-purple-500/30 rounded-lg p-3">
                        <span className="text-[10px] font-bold text-purple-400 uppercase mb-1 block">Tutor Internal Monologue</span>
                        <p className="text-xs text-purple-200 italic">{thinkingLog[0]}</p>
                    </div>
                </div>
            )}
        </>
    );
};
