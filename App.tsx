
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import { ConnectionStatus } from './types';
import { 
  decode, 
  decodeAudioData, 
  createBlobFromAudioData, 
  encode 
} from './utils/audio-utils';

const FRAME_RATE = 2; 
const JPEG_QUALITY = 0.5;

const SYSTEM_INSTRUCTION = `
You are Vision Guide AI — a high-performance accessibility assistant for the visually impaired.
Your primary directive is to provide real-time spatial guidance.

========================
CRITICAL LANGUAGE RULE: STICK TO ENGLISH
========================
- YOU MUST ONLY USE ENGLISH. 
- ALWAYS translate any non-English user speech (Hindi, Marathi, etc.) into English context.
- ALL generated text, ALL spoken responses, and ALL reasoning MUST be in English.
- Use clear, plain, and professional English at all times.

========================
PERCEPTION & REASONING
========================
1) SCANNING: Default state. Say "Scanning room..." or "Looking for [object]...".
2) GUIDING: Triggered when target is visible. Say "TARGET ACQUIRED".
3) DIRECTIONS: Provide clear steps: "Move left", "Step forward", "Object at 3 o'clock".

========================
UI KEYWORDS
========================
- Keywords for UI state: FOUND, TARGET, ACQUIRED, SCANNING, LOST SIGHT.
- Keep output extremely concise.
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
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextInRef = useRef<AudioContext | null>(null);
  const audioContextOutRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const frameIntervalRef = useRef<number | null>(null);

  const activeInputRef = useRef("");
  const activeOutputRef = useRef("");

  // Effect to attach stream when video element becomes available or state changes
  useEffect(() => {
    if (isCameraActive && videoRef.current && mediaStreamRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current;
    }
  }, [isCameraActive]);

  const cleanupSession = useCallback(() => {
    if (sessionRef.current) {
      try { sessionRef.current.close(); } catch(e) {}
      sessionRef.current = null;
    }
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    sourcesRef.current.forEach(source => { try { source.stop(); } catch(e){} });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    
    setStatus(ConnectionStatus.DISCONNECTED);
    setPerceptionState('IDLE');
    setIsCameraActive(false);
    activeInputRef.current = "";
    activeOutputRef.current = "";
    setCurrentInput("");
    setCurrentOutput("");
  }, []);

  const startSession = async () => {
    try {
      setErrorMessage(null);
      setStatus(ConnectionStatus.CONNECTING);
      
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          audio: true, 
          video: { 
            facingMode: 'environment',
            width: { ideal: 1280 }, 
            height: { ideal: 720 } 
          } 
        });
      } catch (mediaErr: any) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      }
      
      mediaStreamRef.current = stream;
      setIsCameraActive(true); // Trigger UI change to render video element

      audioContextInRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextOutRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      await audioContextInRef.current.resume();
      await audioContextOutRef.current.resume();

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

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
              sessionPromise.then(session => {
                if (session) session.sendRealtimeInput({ media: pcmBlob });
              }).catch(() => {});
            };
            
            source.connect(processor);
            processor.connect(audioContextInRef.current!.destination);

            frameIntervalRef.current = window.setInterval(() => {
              if (videoRef.current && canvasRef.current) {
                const canvas = canvasRef.current;
                const video = videoRef.current;
                const ctx = canvas.getContext('2d');
                if (ctx && video.videoWidth > 0) {
                  const scale = 0.5;
                  canvas.width = video.videoWidth * scale;
                  canvas.height = video.videoHeight * scale;
                  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                  canvas.toBlob(async (blob) => {
                    if (blob) {
                      const reader = new FileReader();
                      reader.onloadend = () => {
                        const base64Data = (reader.result as string).split(',')[1];
                        sessionPromise.then(session => {
                          if (session) session.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } });
                        }).catch(() => {});
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
              if (lower.includes("found") || lower.includes("acquired") || lower.includes("see it")) {
                setPerceptionState('LOCKING');
              } else if (lower.includes("step") || lower.includes("move") || lower.includes("left") || lower.includes("right") || lower.includes("ahead")) {
                setPerceptionState('GUIDING');
              } else if (lower.includes("scanning") || lower.includes("lost sight") || lower.includes("looking")) {
                setPerceptionState('SCANNING');
              }
            }

            if (message.serverContent?.turnComplete) {
              const finalInput = activeInputRef.current.trim();
              const finalOutput = activeOutputRef.current.trim();
              if (finalInput || finalOutput) {
                setHistory(prev => [...prev, 
                  ...(finalInput ? [{ role: 'user', text: finalInput } as Message] : []),
                  ...(finalOutput ? [{ role: 'model', text: finalOutput } as Message] : [])
                ].slice(-15));
              }
              activeInputRef.current = ""; 
              activeOutputRef.current = "";
              setCurrentInput(""); 
              setCurrentOutput("");
            }
          },
          onerror: (err) => {
            console.error("Gemini Live Error:", err);
            setErrorMessage("Link disrupted. Auto-reconnecting...");
            cleanupSession();
          },
          onclose: () => cleanupSession()
        }
      });
      sessionRef.current = await sessionPromise;
    } catch (err: any) {
      console.error("Initialization Failed:", err);
      if (err.name === 'NotAllowedError' || err.message?.toLowerCase().includes('permission')) {
        setErrorMessage("Permission denied. Enable Camera/Mic.");
      } else {
        setErrorMessage("Link failed. Check network.");
      }
      setStatus(ConnectionStatus.ERROR);
      cleanupSession();
    }
  };

  useEffect(() => cleanupSession, [cleanupSession]);

  return (
    <div className="flex flex-col h-screen bg-[#020408] text-slate-100 overflow-hidden font-sans select-none">
      {/* Header HUD */}
      <header className="px-4 py-3 md:px-6 md:py-4 border-b border-white/[0.05] flex justify-between items-center bg-black/90 backdrop-blur-3xl z-50 shrink-0">
        <div className="flex items-center gap-3 md:gap-6">
          <div className="hidden xs:flex items-center gap-2 bg-white/[0.05] px-3 py-1.5 rounded-xl border border-white/[0.1]">
            <div className={`w-2 h-2 rounded-full ${
              status === ConnectionStatus.CONNECTED ? 'bg-emerald-400 shadow-[0_0_8px_#10b981]' : 
              status === ConnectionStatus.CONNECTING ? 'bg-amber-400 animate-pulse' : 
              status === ConnectionStatus.ERROR ? 'bg-rose-500' : 'bg-slate-700'
            }`} />
            <span className="text-[9px] font-black tracking-widest text-slate-300 uppercase leading-none">
              {status === ConnectionStatus.CONNECTED ? 'LIVE' : status}
            </span>
          </div>
          <h1 className="text-xl md:text-2xl font-black tracking-tighter text-white uppercase italic">
            Vision Guide <span className="text-blue-500">AI</span>
          </h1>
        </div>
        
        <div className="flex items-center gap-2 md:gap-4">
          {errorMessage && (
            <div className="hidden sm:flex items-center gap-2 px-4 py-2 bg-rose-500/10 border border-rose-500/20 rounded-xl">
              <span className="text-[9px] font-bold text-rose-300 uppercase tracking-widest">{errorMessage}</span>
            </div>
          )}
          <button 
            onClick={status === ConnectionStatus.DISCONNECTED || status === ConnectionStatus.ERROR ? startSession : cleanupSession}
            className={`px-5 py-2.5 md:px-8 md:py-3 rounded-xl md:rounded-2xl font-black text-[10px] md:text-[11px] transition-all active:scale-95 tracking-widest uppercase border ${
              status === ConnectionStatus.DISCONNECTED || status === ConnectionStatus.ERROR
                ? 'bg-blue-600 border-blue-500 text-white shadow-xl hover:bg-blue-500' 
                : 'bg-rose-600/10 border-rose-500/30 text-rose-500 hover:bg-rose-500 hover:text-white'
            }`}
          >
            {status === ConnectionStatus.DISCONNECTED || status === ConnectionStatus.ERROR ? 'Connect' : 'Stop'}
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 flex flex-col lg:flex-row p-3 md:p-6 gap-3 md:gap-6 overflow-hidden min-h-0">
        
        {/* Optic View (Camera Window) */}
        <div className="flex-[1.4] lg:flex-[2.5] bg-[#07090d] rounded-2xl md:rounded-[3rem] overflow-hidden relative border border-white/[0.05] shadow-2xl shrink-0 lg:shrink">
          {!isCameraActive && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center p-6 md:p-12 overflow-y-auto scrollbar-hide bg-[#07090d]">
              <div className="w-full max-w-2xl space-y-8 animate-in fade-in zoom-in duration-700">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 md:w-16 md:h-16 bg-blue-600/10 border border-blue-500/30 rounded-2xl flex items-center justify-center">
                    <svg className="w-6 h-6 md:w-8 md:h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-xl md:text-3xl font-black text-white italic tracking-tighter uppercase">Sensory System Overview</h2>
                    <p className="text-[10px] md:text-xs font-bold text-blue-400 uppercase tracking-[0.3em]">Protocol: Vision Guide 2.5</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 md:p-6 bg-white/[0.03] border border-white/[0.05] rounded-[2rem] space-y-2">
                    <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Feature 01</span>
                    <h3 className="text-sm md:text-lg font-black text-white uppercase italic">Spatial Mapping</h3>
                    <p className="text-xs md:text-sm text-slate-400 leading-relaxed font-medium">The AI constructs a real-time 3D model of your environment, identifying furniture, walls, and obstacles.</p>
                  </div>
                  <div className="p-4 md:p-6 bg-white/[0.03] border border-white/[0.05] rounded-[2rem] space-y-2">
                    <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Feature 02</span>
                    <h3 className="text-sm md:text-lg font-black text-white uppercase italic">Object Lock</h3>
                    <p className="text-xs md:text-sm text-slate-400 leading-relaxed font-medium">Ask for specific items. The AI scans the video feed to locate and highlight targets with precision guidance.</p>
                  </div>
                  <div className="p-4 md:p-6 bg-white/[0.03] border border-white/[0.05] rounded-[2rem] space-y-2">
                    <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Feature 03</span>
                    <h3 className="text-sm md:text-lg font-black text-white uppercase italic">Voice Guidance</h3>
                    <p className="text-xs md:text-sm text-slate-400 leading-relaxed font-medium">Step-by-step spatial directions. "Two steps forward," "Object at 3 o'clock," or "Clear path ahead."</p>
                  </div>
                  <div className="p-4 md:p-6 bg-blue-600/10 border border-blue-500/20 rounded-[2rem] flex flex-col justify-center items-center text-center">
                    <div className="w-10 h-10 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mb-3" />
                    <p className="text-[10px] font-black text-blue-300 uppercase tracking-widest">Awaiting Neural Link</p>
                  </div>
                </div>

                <div className="pt-4 border-t border-white/[0.05] flex items-center justify-between">
                  <span className="text-[9px] font-bold text-slate-600 uppercase tracking-widest">Authorizing optic input will enable live stream</span>
                  <div className="flex gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />
                    <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />
                    <div className="w-1.5 h-1.5 rounded-full bg-slate-800" />
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className={`w-full h-full object-cover transition-opacity duration-700 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
          />
          <canvas ref={canvasRef} className="hidden" />
          
          {/* HUD Overlays (Only visible when connected) */}
          {status === ConnectionStatus.CONNECTED && isCameraActive && (
            <div className="absolute inset-0 pointer-events-none z-20 flex flex-col justify-between p-4 md:p-10">
              <div className="flex justify-between items-start">
                <div className={`px-4 py-2 md:px-6 md:py-3 rounded-xl md:rounded-2xl backdrop-blur-3xl border flex items-center gap-2 md:gap-4 shadow-2xl transition-all duration-700 ${
                  perceptionState === 'SCANNING' ? 'bg-blue-600/40 border-blue-400/50 text-white' :
                  perceptionState === 'LOCKING' ? 'bg-amber-600/40 border-amber-400/50 text-white' :
                  'bg-emerald-600/40 border-emerald-400/50 text-white'
                }`}>
                  <div className={`w-2 h-2 md:w-2.5 md:h-2.5 rounded-full animate-pulse ${
                     perceptionState === 'SCANNING' ? 'bg-blue-400' : perceptionState === 'LOCKING' ? 'bg-amber-400' : 'bg-emerald-400'
                  }`} />
                  <span className="text-[9px] md:text-[12px] font-black uppercase tracking-[0.2em]">{perceptionState}</span>
                </div>
              </div>

              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                {perceptionState === 'SCANNING' ? (
                  <div className="w-[85%] h-[1px] bg-blue-500/50 shadow-[0_0_20px_rgba(59,130,246,0.6)] animate-[scan_3s_ease-in-out_infinite]" />
                ) : (
                  <div className={`w-32 h-32 md:w-56 md:h-56 border-[2px] md:border-[4px] border-dashed rounded-full animate-spin-slow opacity-60 ${
                    perceptionState === 'LOCKING' ? 'border-amber-400' : 'border-emerald-400'
                  }`} />
                )}
              </div>

              <div className="w-full flex justify-center">
                <div className="bg-black/80 backdrop-blur-3xl border border-white/10 p-4 md:p-10 rounded-2xl md:rounded-[4rem] shadow-2xl max-w-xl w-full text-center">
                  <p className="text-sm md:text-2xl font-black text-white italic tracking-tight leading-tight">
                    {perceptionState === 'SCANNING' ? 'Analyzing Environment...' : 'Target Identified. Following instructions.'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Data Feed (Chat Window) */}
        <div className="flex-1 lg:flex-none lg:w-[400px] xl:w-[480px] bg-[#07090d] rounded-2xl md:rounded-[3rem] border border-white/[0.05] flex flex-col shadow-2xl overflow-hidden min-h-0">
          <div className="px-5 py-4 md:px-8 md:py-7 border-b border-white/[0.05] flex justify-between items-center shrink-0">
            <h2 className="text-[10px] md:text-[12px] font-black text-slate-500 uppercase tracking-[0.4em]">Neural Feed</h2>
            <div className="flex items-center gap-2">
               <div className="hidden xs:block px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[8px] font-black text-emerald-400 uppercase">Secure Link</div>
               <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            </div>
          </div>
          
          <div className="flex-1 p-4 md:p-8 overflow-y-auto space-y-4 md:space-y-8 scrollbar-hide flex flex-col-reverse">
            <div className="space-y-4 md:space-y-8 flex flex-col">
              {history.length === 0 && !currentInput && !currentOutput ? (
                <div className="flex-1 flex flex-col gap-8 animate-in fade-in slide-in-from-right-4 duration-1000">
                  <div className="space-y-4">
                    <h3 className="text-[10px] font-black text-slate-600 uppercase tracking-[0.3em]">Command Protocol</h3>
                    <div className="space-y-3">
                      {[
                        "What do you see in front of me?",
                        "Find my blue water bottle.",
                        "Describe the room layout.",
                        "Is there anything on the floor?"
                      ].map((cmd, idx) => (
                        <div key={idx} className="p-4 bg-white/[0.02] border border-white/5 rounded-2xl text-[13px] text-slate-400 italic font-medium">
                          "{cmd}"
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="p-5 bg-blue-600/5 border border-blue-500/10 rounded-2xl space-y-3">
                    <div className="flex items-center gap-2">
                       <div className="w-2 h-2 rounded-full bg-blue-500" />
                       <span className="text-[9px] font-black text-blue-400 uppercase tracking-widest">Multi-Lingual Engine</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">Speak in your native tongue (Hindi, Marathi, etc.). The AI will translate your intent and respond with precise English guidance.</p>
                  </div>

                  <div className="mt-auto pt-4 flex flex-col items-center justify-center text-center opacity-30">
                     <div className="w-8 h-8 border border-slate-700 rounded-full mb-3 flex items-center justify-center">
                        <div className="w-1 h-1 bg-slate-500 rounded-full animate-ping" />
                     </div>
                     <p className="text-[9px] font-black uppercase tracking-[0.2em]">Neural Cortex Standing By</p>
                  </div>
                </div>
              ) : (
                <>
                  {history.map((t, i) => (
                    <div key={i} className={`flex w-full flex-col ${t.role === 'user' ? 'items-end' : 'items-start'} animate-in slide-in-from-bottom-2`}>
                      <span className="text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1 px-1">
                        {t.role === 'user' ? 'Input' : 'AI'}
                      </span>
                      <div className={`p-4 md:p-6 rounded-xl md:rounded-[2rem] text-[12px] md:text-[15px] leading-relaxed border w-fit max-w-[90%] break-words whitespace-pre-wrap ${
                        t.role === 'user' 
                          ? 'bg-blue-600 border-blue-500 text-white rounded-tr-none font-bold' 
                          : 'bg-white/[0.03] border-white/10 text-slate-300 rounded-tl-none shadow-xl'
                      }`}>
                        {t.text}
                      </div>
                    </div>
                  ))}
                  
                  {currentOutput && (
                    <div className="flex w-full flex-col items-start animate-in fade-in">
                      <span className="text-[8px] font-black text-emerald-500 uppercase tracking-widest mb-1 px-1">Processing...</span>
                      <div className="p-4 md:p-6 rounded-xl md:rounded-[2rem] text-[12px] md:text-[15px] leading-relaxed border border-emerald-500/30 bg-emerald-500/10 text-emerald-100 rounded-tl-none w-fit max-w-[90%] break-words">
                        {currentOutput}
                        <span className="inline-block w-1.5 h-4 ml-1 bg-emerald-400 animate-pulse" />
                      </div>
                    </div>
                  )}

                  {currentInput && (
                    <div className="flex w-full flex-col items-end animate-in fade-in">
                      <span className="text-[8px] font-black text-blue-400 uppercase tracking-widest mb-1 px-1">Listening...</span>
                      <div className="p-4 md:p-6 rounded-xl md:rounded-[2rem] text-[12px] md:text-[15px] leading-relaxed border border-blue-500/40 bg-blue-500/20 text-white rounded-tr-none w-fit max-w-[90%] italic font-bold">
                        {currentInput}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
          
          <div className="px-5 py-4 md:px-8 md:py-6 bg-black/40 border-t border-white/[0.05] hidden xs:block">
             <div className="grid grid-cols-2 gap-3 md:gap-4">
                <div className="bg-white/[0.02] p-2 md:p-4 rounded-xl border border-white/[0.05]">
                  <span className="text-[8px] font-black text-slate-600 uppercase block mb-0.5">Optics</span>
                  <span className="text-[10px] md:text-[12px] font-black text-blue-500 tracking-tighter uppercase italic">Ready</span>
                </div>
                <div className="bg-white/[0.02] p-2 md:p-4 rounded-xl border border-white/[0.05]">
                  <span className="text-[8px] font-black text-slate-600 uppercase block mb-0.5">Network</span>
                  <span className="text-[10px] md:text-[12px] font-black text-emerald-500 tracking-tighter uppercase italic">Sync</span>
                </div>
             </div>
          </div>
        </div>
      </main>

      <style>{`
        .animate-spin-slow { animation: spin 12s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes scan {
          0%, 100% { transform: translateY(-20vh); opacity: 0; }
          15%, 85% { opacity: 0.6; }
          50% { transform: translateY(20vh); }
        }
        @media (min-width: 1024px) {
          @keyframes scan {
            0%, 100% { transform: translateY(-30vh); opacity: 0; }
            15%, 85% { opacity: 0.8; }
            50% { transform: translateY(30vh); }
          }
        }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
};

export default App;
