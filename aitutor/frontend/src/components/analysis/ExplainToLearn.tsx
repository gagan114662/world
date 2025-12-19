import React, { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';
import { Spinner } from '../ui/spinner';
import { AnalysisService, AnalysisResult, QuizQuestion } from '../../services/analysisService';
import { Mic, Square, CheckCircle, AlertCircle, Info, RefreshCcw } from 'lucide-react';
import { cn } from '../../lib/utils';

const ExplainToLearn: React.FC = () => {
    const [status, setStatus] = useState<'idle' | 'recording' | 'analyzing' | 'results'>('idle');
    const [recordingTime, setRecordingTime] = useState(0);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [userAnswers, setUserAnswers] = useState<Record<number, number>>({});
    const [error, setError] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            chunksRef.current = [];

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            recorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/mp3' });
                setAudioBlob(blob);
                analyzeAudio(blob);
            };

            recorder.start();
            setStatus('recording');
            setRecordingTime(0);
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);
        } catch (err) {
            console.error('Failed to start recording', err);
            setError('Could not access microphone. Please check permissions.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && status === 'recording') {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            if (timerRef.current) clearInterval(timerRef.current);
            setStatus('analyzing');
        }
    };

    const analyzeAudio = async (blob: Blob) => {
        try {
            const result = await AnalysisService.uploadExplanation(blob);
            setResult(result);
            setStatus('results');
        } catch (err: any) {
            console.error('Analysis failed', err);
            setError(err.message || 'Analysis failed. Please try again.');
            setStatus('idle');
        }
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleAnswerSelect = (questionIndex: number, optionIndex: number) => {
        setUserAnswers(prev => ({ ...prev, [questionIndex]: optionIndex }));
    };

    const reset = () => {
        setStatus('idle');
        setAudioBlob(null);
        setResult(null);
        setUserAnswers({});
        setError(null);
        setRecordingTime(0);
    };

    return (
        <Card className="w-full max-w-4xl mx-auto border-0 shadow-2xl bg-slate-900/50 backdrop-blur-xl text-white overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6">
                <div className="flex justify-between items-center">
                    <div>
                        <CardTitle className="text-2xl font-bold flex items-center gap-2">
                            <Mic className="w-6 h-6" />
                            Explain to Learn
                        </CardTitle>
                        <p className="text-indigo-100 mt-1">
                            The Feynman Technique: Explain a concept to Gemini to identify your knowledge gaps.
                        </p>
                    </div>
                    {status === 'results' && (
                        <Button variant="ghost" className="text-white hover:bg-white/10" onClick={reset}>
                            <RefreshCcw className="w-4 h-4 mr-2" />
                            Start Over
                        </Button>
                    )}
                </div>
            </CardHeader>

            <CardContent className="p-8">
                {error && (
                    <Alert variant="destructive" className="mb-6 bg-red-900/20 border-red-500/50 text-red-200">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                {status === 'idle' && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <div className="w-24 h-24 bg-indigo-500/20 rounded-full flex items-center justify-center mb-6 animate-pulse">
                            <Mic className="w-10 h-10 text-indigo-400" />
                        </div>
                        <h3 className="text-xl font-semibold mb-2">Ready to explain?</h3>
                        <p className="text-slate-400 max-w-md mb-8">
                            Pick a topic you've been learning and explain it out loud as if you were teaching a friend.
                            Gemini will listen and find what you missed.
                        </p>
                        <Button
                            onClick={startRecording}
                            className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-6 rounded-full text-lg font-medium transition-all transform hover:scale-105"
                        >
                            <Mic className="mr-2 h-5 w-5" />
                            Start Recording
                        </Button>
                    </div>
                )}

                {status === 'recording' && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <div className="relative mb-8">
                            <div className="w-32 h-32 bg-red-500/20 rounded-full flex items-center justify-center animate-ping absolute inset-0"></div>
                            <div className="w-32 h-32 bg-red-500/40 rounded-full flex items-center justify-center relative">
                                <Square className="w-12 h-12 text-red-500 fill-current" />
                            </div>
                        </div>
                        <h3 className="text-3xl font-mono font-bold mb-2 text-red-400">{formatTime(recordingTime)}</h3>
                        <p className="text-slate-400 mb-8">Recording your explanation...</p>
                        <Button
                            onClick={stopRecording}
                            variant="destructive"
                            className="px-8 py-6 rounded-full text-lg font-medium"
                        >
                            Stop & Analyze
                        </Button>
                    </div>
                )}

                {status === 'analyzing' && (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                        <div className="relative mb-10">
                            <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500 to-purple-500 rounded-full blur-2xl opacity-30 animate-pulse"></div>
                            <Spinner className="w-20 h-20 text-indigo-500" />
                        </div>
                        <h3 className="text-2xl font-semibold mb-3">Gemini is Thinking...</h3>
                        <div className="flex flex-col gap-2 text-slate-400 animate-pulse">
                            <p>Transcribing your explanation...</p>
                            <p>Identifying knowledge gaps...</p>
                            <p>Generating a custom quiz...</p>
                        </div>
                    </div>
                )}

                {status === 'results' && result && (
                    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
                        {/* Knowledge Gaps Section */}
                        <section>
                            <h3 className="text-xl font-bold mb-4 flex items-center gap-2 text-amber-400">
                                <Info className="w-5 h-5" />
                                Knowledge Gaps Identified
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {result.knowledge_gaps.map((gap, i) => (
                                    <div key={i} className="bg-slate-800/50 border border-slate-700 p-4 rounded-xl flex items-start gap-3">
                                        <div className="mt-1 bg-amber-500/20 text-amber-500 p-1 rounded">
                                            <AlertCircle className="w-4 h-4" />
                                        </div>
                                        <p className="text-slate-200">{gap}</p>
                                    </div>
                                ))}
                            </div>
                        </section>

                        <hr className="border-slate-800" />

                        {/* Quiz Section */}
                        <section className="space-y-8">
                            <h3 className="text-2xl font-bold text-indigo-300">Targeted Quiz</h3>
                            {result.quiz.map((q, qIdx) => {
                                const isAnswered = userAnswers[qIdx] !== undefined;
                                const isCorrect = userAnswers[qIdx] === q.correct_answer_index;

                                return (
                                    <div key={qIdx} className="bg-slate-800/30 rounded-2xl p-6 border border-slate-700/50">
                                        <h4 className="text-lg font-medium mb-6 text-slate-100">
                                            <span className="text-indigo-500 mr-2">Question {qIdx + 1}:</span>
                                            {q.question}
                                        </h4>

                                        <div className="grid grid-cols-1 gap-3">
                                            {q.options.map((opt, oIdx) => {
                                                const isSelected = userAnswers[qIdx] === oIdx;
                                                const isCorrectOption = oIdx === q.correct_answer_index;

                                                let variantStyle = "bg-slate-800 hover:bg-slate-700 border-slate-600";
                                                if (isAnswered) {
                                                    if (isCorrectOption) variantStyle = "bg-green-500/20 border-green-500 text-green-200";
                                                    else if (isSelected) variantStyle = "bg-red-500/20 border-red-500 text-red-200";
                                                    else variantStyle = "bg-slate-800/30 border-slate-700 opacity-50";
                                                } else if (isSelected) {
                                                    variantStyle = "bg-indigo-500/20 border-indigo-500 text-indigo-200";
                                                }

                                                return (
                                                    <button
                                                        key={oIdx}
                                                        onClick={() => !isAnswered && handleAnswerSelect(qIdx, oIdx)}
                                                        disabled={isAnswered}
                                                        className={cn(
                                                            "w-full text-left p-4 rounded-xl border-2 transition-all flex justify-between items-center group",
                                                            variantStyle
                                                        )}
                                                    >
                                                        <span>{opt}</span>
                                                        {isAnswered && isCorrectOption && <CheckCircle className="w-5 h-5 text-green-500" />}
                                                        {isAnswered && isSelected && !isCorrectOption && <AlertCircle className="w-5 h-5 text-red-500" />}
                                                    </button>
                                                );
                                            })}
                                        </div>

                                        {isAnswered && (
                                            <div className={cn(
                                                "mt-6 p-4 rounded-xl animate-in zoom-in-95 duration-300",
                                                isCorrect ? "bg-green-500/10 border border-green-500/30" : "bg-blue-500/10 border border-blue-500/30"
                                            )}>
                                                <div className="flex items-center gap-2 mb-2">
                                                    <Badge className={isCorrect ? "bg-green-500" : "bg-blue-500"}>
                                                        {isCorrect ? "Correct!" : "Explanation"}
                                                    </Badge>
                                                </div>
                                                <p className="text-slate-300 italic">
                                                    {q.explanation}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </section>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};
export default ExplainToLearn;
