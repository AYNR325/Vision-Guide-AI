
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import { ConnectionStatus, TranscriptionItem } from './types';
import { 
  decode, 
  decodeAudioData, 
  createBlobFromAudioData, 
  encode 
} from './utils/audio-utils';

const FRAME_RATE = 2; 
const JPEG_QUALITY = 0.4;

const SYSTEM_INSTRUCTION = `
You are Vision Guide AI — a native multimodal accessibility assistant for visually impaired users.
You operate using a single, continuous perception loop over live visual frames and audio output.

========================
PERCEPTION & REASONING MODES
========================

1) SCANNING MODE:
- Default mode. Object not found or lost.
- Phrase: "Scanning..." or "Checking surfaces..."
- Transition to GUIDING only when target is stable.

2) GUIDING MODE:
- Object found. Use phrases like: "Target Acquired", "I see it", "Found your [object]".
- Directions: "Move left", "Step forward", "At 3 o'clock".

========================
RULES
========================
- Always use the word "FOUND" or "TARGET ACQUIRED" immediately when the object is first seen to trigger the UI lock.
- Always use directional words like "STEP", "MOVE", "LEFT", "RIGHT", "AHEAD" for navigation.
- Be brief and human.
`;

type PerceptionState = 'IDLE' | 'SCANNING' | 'LOCKING' | 'GUIDING';

interface Message {
  role: 'user' | 'model';
  text: string;
}

const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [perceptionState, setPerceptionState] = useState<PerceptionState>('IDLE');
  const [history, setHistory] = useState<Message[]>([]);
  const [currentInput, setCurrentInput] = useState("");
  const [currentOutput, setCurrentOutput] = useState("");
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const audioContextInRef = useRef<AudioContext | null>(null);
  const audioContextOutRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const frameIntervalRef = useRef<number | null>(null);

  const activeInputRef = useRef("");
  const activeOutputRef = useRef("");

  const cleanupSession = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    sourcesRef.current.forEach(source => { try { source.stop(); } catch(e){} });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    setStatus(ConnectionStatus.DISCONNECTED);
    setPerceptionState('IDLE');
    setHistory([]);
    setCurrentInput("");
    setCurrentOutput("");
    activeInputRef.current = "";
    activeOutputRef.current = "";
  }, []);

  const startSession = async () => {
    try {
      setStatus(ConnectionStatus.CONNECTING);
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      
      audioContextInRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextOutRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus(ConnectionStatus.CONNECTED);
            setPerceptionState('SCANNING');
            
            const source = audioContextInRef.current!.createMediaStreamSource(stream);
            const processor = audioContextInRef.current!.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createBlobFromAudioData(inputData);
              sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
            };
            source.connect(processor);
            processor.connect(audioContextInRef.current!.destination);

            frameIntervalRef.current = window.setInterval(() => {
              if (videoRef.current && canvasRef.current) {
                const canvas = canvasRef.current;
                const video = videoRef.current;
                const ctx = canvas.getContext('2d');
                if (ctx) {
                  const scale = 0.5;
                  canvas.width = video.videoWidth * scale;
                  canvas.height = video.videoHeight * scale;
                  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                  canvas.toBlob(async (blob) => {
                    if (blob) {
                      const reader = new FileReader();
                      reader.onloadend = () => {
                        const base64Data = (reader.result as string).split(',')[1];
                        sessionPromise.then(session => session.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } }));
                      };
                      reader.readAsDataURL(blob);
                    }
                  }, 'image/jpeg', JPEG_QUALITY);
                }
              }
            }, 1000 / FRAME_RATE);
          },
          onmessage: async (message) => {
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextOutRef.current) {
              const ctx = audioContextOutRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              const buffer = await decodeAudioData(decode(audioData), ctx, 24000);
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              source.onended = () => sourcesRef.current.delete(source);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => { try { s.stop(); } catch(e){} });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
            }

            if (message.serverContent?.inputTranscription) {
              activeInputRef.current += message.serverContent.inputTranscription.text;
              setCurrentInput(activeInputRef.current);
            }

            if (message.serverContent?.outputTranscription) {
              activeOutputRef.current += message.serverContent.outputTranscription.text;
              setCurrentOutput(activeOutputRef.current);

              const lower = activeOutputRef.current.toLowerCase();
              if (lower.includes("lost sight") || lower.includes("scanning")) setPerceptionState('SCANNING');
              else if (lower.includes("found") || lower.includes("acquire") || lower.includes("see it")) setPerceptionState('LOCKING');
              else if (lower.includes("step") || lower.includes("move") || lower.includes("left") || lower.includes("right") || lower.includes("ahead")) setPerceptionState('GUIDING');
            }

            if (message.serverContent?.turnComplete) {
              const finalInput = activeInputRef.current.trim();
              const finalOutput = activeOutputRef.current.trim();
              setHistory(prev => {
                const newHistory = [...prev];
                if (finalInput) newHistory.push({ role: 'user', text: finalInput });
                if (finalOutput) newHistory.push({ role: 'model', text: finalOutput });
                return newHistory.slice(-20);
              });
              activeInputRef.current = ""; activeOutputRef.current = "";
              setCurrentInput(""); setCurrentOutput("");
            }
          },
          onerror: () => cleanupSession(),
          onclose: () => cleanupSession()
        }
      });
      sessionRef.current = await sessionPromise;
    } catch (err) {
      setStatus(ConnectionStatus.ERROR);
    }
  };

  useEffect(() => cleanupSession, [cleanupSession]);

  return (
    <div className="flex flex-col h-screen bg-[#020408] text-slate-100 overflow-hidden font-sans selection:bg-blue-500/30">
      {/* Responsive Header */}
      <header className="px-4 md:px-8 py-3 md:py-5 border-b border-white/[0.03] flex flex-col sm:flex-row justify-between items-center bg-black/40 backdrop-blur-3xl z-50 gap-3">
        <div className="flex items-center gap-4 md:gap-6">
          <div className="flex items-center gap-2 md:gap-3 bg-white/[0.02] px-3 md:px-4 py-1.5 md:py-2 rounded-2xl border border-white/[0.05]">
            <div className={`w-2 h-2 md:w-2.5 md:h-2.5 rounded-full ${
              status === ConnectionStatus.CONNECTED ? 'bg-emerald-400 shadow-[0_0_12px_#10b981]' : 
              status === ConnectionStatus.CONNECTING ? 'bg-amber-400 animate-pulse' : 'bg-rose-500'
            }`} />
            <span className="text-[9px] md:text-[10px] font-black tracking-[0.1em] md:tracking-[0.2em] text-slate-400 uppercase leading-none">
              {status}
            </span>
          </div>
          <h1 className="text-lg md:text-2xl font-black tracking-tighter text-white uppercase italic leading-none">
            Vision Guide <span className="text-blue-500">AI</span>
          </h1>
        </div>
        <button 
          onClick={status === ConnectionStatus.DISCONNECTED ? startSession : cleanupSession}
          className={`w-full sm:w-auto px-6 md:px-10 py-2.5 md:py-3 rounded-xl md:rounded-2xl font-black text-[10px] md:text-[11px] transition-all active:scale-95 tracking-widest uppercase border ${
            status === ConnectionStatus.DISCONNECTED 
              ? 'bg-blue-600 border-blue-500 text-white shadow-xl hover:bg-blue-500' 
              : 'bg-white/[0.03] border-white/10 text-rose-400 hover:bg-rose-500 hover:text-white'
          }`}
        >
          {status === ConnectionStatus.DISCONNECTED ? 'Demo Activation' : 'Terminate Link'}
        </button>
      </header>

      {/* Main Layout: Column on Mobile, Row on Laptop */}
      <main className="flex-1 flex flex-col lg:flex-row p-3 md:p-6 gap-3 md:gap-6 overflow-hidden">
        
        {/* Core Perception Module (Camera) */}
        <div className="flex-[1.5] lg:flex-[2.5] bg-[#07090d] rounded-[1.5rem] md:rounded-[3rem] overflow-hidden relative border border-white/[0.03] shadow-inner group min-h-[300px] lg:min-h-0">
          {!isCameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 md:p-12 space-y-4 md:space-y-8">
              <div className="w-16 h-16 md:w-24 md:h-24 bg-blue-600/5 rounded-full flex items-center justify-center border border-blue-500/10 animate-pulse">
                <svg className="w-8 h-8 md:w-12 md:h-12 text-blue-500/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div className="space-y-1 md:space-y-2">
                 <h2 className="text-xl md:text-2xl font-black text-white uppercase tracking-tighter italic">Engine Idle</h2>
                 <p className="text-slate-500 text-[9px] md:text-xs font-bold uppercase tracking-widest">Awaiting Neural Link Initialization</p>
              </div>
            </div>
          )}
          
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className={`w-full h-full object-cover transition-all duration-[1.5s] ${isCameraActive ? 'scale-100 opacity-60' : 'scale-110 opacity-0 blur-3xl'}`}
          />
          <canvas ref={canvasRef} className="hidden" />
          
          {/* Spatial Overlays */}
          {status === ConnectionStatus.CONNECTED && (
            <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4 md:p-10">
              <div className="flex justify-between items-start">
                <div className={`px-3 md:px-6 py-1.5 md:py-3 rounded-xl md:rounded-2xl backdrop-blur-3xl border flex items-center gap-2 md:gap-4 transition-all duration-1000 ${
                  perceptionState === 'SCANNING' ? 'bg-blue-500/5 border-blue-500/20 text-blue-400' :
                  perceptionState === 'LOCKING' ? 'bg-amber-500/5 border-amber-500/20 text-amber-400' :
                  'bg-emerald-500/5 border-emerald-500/20 text-emerald-400'
                }`}>
                  <div className={`w-2 md:w-2.5 h-2 md:h-2.5 rounded-full animate-pulse shadow-[0_0_10px_currentColor] ${
                     perceptionState === 'SCANNING' ? 'bg-blue-400' : perceptionState === 'LOCKING' ? 'bg-amber-400' : 'bg-emerald-400'
                  }`} />
                  <span className="text-[8px] md:text-[11px] font-black uppercase tracking-[0.15em] md:tracking-[0.25em]">{perceptionState} MODE</span>
                </div>
                <div className="px-2 md:px-4 py-1 md:py-2 bg-black/40 rounded-lg border border-white/[0.05] text-[8px] md:text-[10px] font-black text-white/30 uppercase tracking-[0.1em] md:tracking-[0.3em]">
                  Real-time Optic Sync
                </div>
              </div>

              {/* Dynamic Focus Center */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                {perceptionState === 'SCANNING' ? (
                  <div className="w-[85%] h-[1px] bg-blue-500/10 shadow-[0_0_40px_rgba(59,130,246,0.2)] animate-[scan_5s_ease-in-out_infinite]" />
                ) : (
                  <div className={`w-24 md:w-40 h-24 md:h-40 border-[2px] md:border-[3px] border-dashed rounded-full animate-spin-slow opacity-20 ${
                    perceptionState === 'LOCKING' ? 'border-amber-500' : 'border-emerald-500'
                  }`} />
                )}
              </div>

              <div className="w-full flex justify-center">
                <div className="bg-black/60 backdrop-blur-3xl border border-white/[0.05] p-4 md:p-8 rounded-[1.5rem] md:rounded-[3rem] shadow-2xl max-w-lg w-full text-center ring-1 ring-white/[0.05]">
                  <div className="flex items-center justify-center gap-2 md:gap-3 mb-1 md:mb-3 text-blue-500/50">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_#3b82f6]" />
                    <span className="text-[8px] md:text-[10px] font-black uppercase tracking-widest">Auditory Link Active</span>
                  </div>
                  <p className="text-base md:text-xl font-black text-white italic tracking-tight leading-snug">
                    {perceptionState === 'SCANNING' ? 'Scanning environment. Move camera slowly.' : 'Follow precise movement guidance.'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Intelligence Log (Adjusted for responsiveness) */}
        <div className="flex-1 lg:flex-none lg:w-[420px] xl:w-[480px] bg-[#07090d] rounded-[1.5rem] md:rounded-[3rem] border border-white/[0.03] flex flex-col shadow-2xl overflow-hidden ring-1 ring-white/[0.02] max-h-[40%] lg:max-h-full">
          <div className="px-6 md:px-8 py-4 md:py-7 border-b border-white/[0.03] flex justify-between items-center bg-white/[0.01]">
            <h2 className="text-[10px] md:text-[12px] font-black text-slate-500 uppercase tracking-[0.2em] md:tracking-[0.4em]">Neural Output</h2>
            <div className="flex gap-1">
               <div className="w-1 h-1 rounded-full bg-blue-500/30" />
               <div className="w-1 h-1 rounded-full bg-blue-500/50" />
               <div className="w-1 h-1 rounded-full bg-blue-500/70" />
            </div>
          </div>
          
          <div className="flex-1 p-4 md:p-8 overflow-y-auto space-y-4 md:space-y-8 scrollbar-hide flex flex-col">
            {history.length === 0 && !currentInput && !currentOutput ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center opacity-10 space-y-4">
                <div className="w-10 h-10 md:w-16 md:h-16 border-2 border-slate-700 rounded-full animate-spin-slow border-t-blue-500" />
                <p className="text-[9px] md:text-[11px] font-black uppercase tracking-widest md:tracking-[0.3em] max-w-[200px] leading-relaxed">System Ready</p>
              </div>
            ) : (
              <>
                {history.map((t, i) => (
                  <div key={i} className={`flex w-full flex-col ${t.role === 'user' ? 'items-end' : 'items-start'} animate-in fade-in duration-500`}>
                    <span className="text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1 px-1">
                      {t.role === 'user' ? 'Input' : 'AI Output'}
                    </span>
                    <div className={`p-3 md:p-5 rounded-2xl md:rounded-3xl text-[12px] md:text-sm leading-relaxed border shadow-xl w-fit max-w-[90%] break-words ${
                      t.role === 'user' 
                        ? 'bg-blue-600 border-blue-500 text-white rounded-tr-none font-bold' 
                        : 'bg-white/[0.02] border-white/[0.05] text-slate-300 rounded-tl-none'
                    }`}>
                      {t.text}
                    </div>
                  </div>
                ))}
                
                {currentOutput && (
                  <div className="flex w-full flex-col items-start animate-in fade-in">
                    <span className="text-[8px] font-black text-emerald-500 uppercase tracking-widest mb-1 px-1">Generating...</span>
                    <div className="p-3 md:p-5 rounded-2xl md:rounded-3xl text-[12px] md:text-sm leading-relaxed border border-emerald-500/20 bg-emerald-500/5 text-emerald-100 rounded-tl-none w-fit max-w-[90%] break-words shadow-lg">
                      {currentOutput}
                      <span className="inline-block w-1 h-3 md:w-1.5 md:h-4 ml-1 bg-emerald-400 animate-pulse align-middle" />
                    </div>
                  </div>
                )}

                {currentInput && (
                  <div className="flex w-full flex-col items-end animate-in fade-in">
                    <span className="text-[8px] font-black text-blue-400 uppercase tracking-widest mb-1 px-1">Capturing...</span>
                    <div className="p-3 md:p-5 rounded-2xl md:rounded-3xl text-[12px] md:text-sm leading-relaxed border border-blue-500/30 bg-blue-500/10 text-white rounded-tr-none w-fit max-w-[90%] break-words italic font-bold">
                      {currentInput}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
          
          <div className="p-4 md:p-8 bg-black/20 border-t border-white/[0.03] space-y-3 md:space-y-6 hidden sm:block">
             <div className="flex gap-3 md:gap-5">
                <div className="flex-1 bg-white/[0.02] p-2 md:p-4 rounded-xl md:rounded-2xl border border-white/[0.05]">
                  <span className="text-[8px] font-black text-slate-600 uppercase block mb-1 tracking-widest">Feed Rate</span>
                  <span className="text-[10px] md:text-sm font-black text-blue-500 italic tracking-tighter">{FRAME_RATE} HZ OPTIC</span>
                </div>
                <div className="flex-1 bg-white/[0.02] p-2 md:p-4 rounded-xl md:rounded-2xl border border-white/[0.05]">
                  <span className="text-[8px] font-black text-slate-600 uppercase block mb-1 tracking-widest">Logic</span>
                  <span className="text-[10px] md:text-sm font-black text-emerald-500 italic tracking-tighter">GEMINI 2.5</span>
                </div>
             </div>
             <p className="text-[8px] md:text-[10px] text-slate-700 font-black uppercase tracking-[0.2em] md:tracking-[0.4em] text-center opacity-40">
               Native Multimodal Architecture
             </p>
          </div>
        </div>
      </main>

      <footer className="px-4 md:px-10 py-3 md:py-4 bg-rose-500/[0.02] border-t border-rose-500/10 flex flex-row justify-between items-center text-[8px] md:text-[10px] font-black uppercase tracking-[0.1em] md:tracking-[0.3em] text-slate-700">
        <div className="flex items-center gap-2 md:gap-3 text-rose-500/40">
          <svg className="w-3 h-3 md:w-4 md:h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
          <span className="hidden xs:inline">Safety Priority: Indoor Phase</span>
        </div>
        <div className="flex gap-3 md:gap-6 items-center">
          <div className="w-1 h-1 md:w-1.5 md:h-1.5 rounded-full bg-slate-800" />
          <span className="opacity-30 tracking-normal">v2.6.5-R</span>
        </div>
      </footer>
      
      <style>{`
        .animate-spin-slow { animation: spin 10s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes scan {
          0%, 100% { transform: translateY(-30vh); opacity: 0; }
          15%, 85% { opacity: 1; }
          50% { transform: translateY(30vh); }
        }
        @media (max-width: 640px) {
          @keyframes scan {
            0%, 100% { transform: translateY(-15vh); opacity: 0; }
            15%, 85% { opacity: 1; }
            50% { transform: translateY(15vh); }
          }
        }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
};

export default App;
