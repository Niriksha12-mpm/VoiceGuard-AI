import React, { useState, useRef, useEffect } from 'react';

export default function AudioUploader({ onAnalyze, loading, selectedModel, setSelectedModel, error }) {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [presetActive, setPresetActive] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [inputMode, setInputMode] = useState('upload'); // 'upload' | 'record'
  const [selectedChannel, setSelectedChannel] = useState('clean');

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [micError, setMicError] = useState(null);

  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);

  const presets = [
    { id: 'real_lj', name: 'Real Human (LJSpeech)', type: 'REAL', label: 'Female Bonafide' },
    { id: 'real_yt', name: 'Real Human (YouTube)', type: 'REAL', label: 'Male Speech' },
    { id: 'fake_elevenlabs', name: 'ElevenLabs AI (Male)', type: 'FAKE', label: 'Neural Clone' },
    { id: 'fake_aria', name: 'ElevenLabs AI (Female - Aria)', type: 'FAKE', label: 'Female AI Voice' },
  ];

  const channels = [
    { id: 'clean', name: 'Direct Studio', badge: 'Clean', icon: '🎧', desc: 'No channel degradation (Direct high-fidelity stream)' },
    { id: 'whatsapp_opus', name: 'WhatsApp Note', badge: 'Opus 16k', icon: '📱', desc: '7.2kHz Opus voice compression & MDCT quantization' },
    { id: 'cellular_telephony', name: 'Cellular Call', badge: '8kHz G.711', icon: '📞', desc: 'Standard telephony bandpass (300-3400Hz) & 8-bit companding' },
    { id: 'voip_meeting', name: 'VoIP Meeting', badge: 'Zoom/Teams', icon: '💻', desc: 'Conference room acoustic reflections & 20ms packet dropouts' },
  ];

  // Clean up audio recorder and streams
  useEffect(() => {
    return () => {
      stopAudioVisualizer();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setAudioUrl(URL.createObjectURL(selected));
      setPresetActive(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      setFile(dropped);
      setAudioUrl(URL.createObjectURL(dropped));
      setPresetActive(null);
    }
  };

  const loadPreset = async (preset) => {
    setPresetActive(preset.id);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/sample-audio/${preset.id}`);
      if (!res.ok) throw new Error('Preset audio download failed');
      const blob = await res.blob();
      const ext = preset.id.includes('yt') || preset.id.includes('elevenlabs') || preset.id.includes('aria') ? 'flac' : 'wav';
      const presetFile = new File([blob], `${preset.id}_sample.${ext}`, { type: blob.type });
      setFile(presetFile);
      setAudioUrl(URL.createObjectURL(blob));
      setInputMode('upload');
    } catch (err) {
      console.error('Error loading preset:', err);
    }
  };

  // ---------------------------------------------------------
  // Live Microphone Recording & Real-Time Waveform Visualizer
  // ---------------------------------------------------------
  const startRecording = async () => {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true
        }
      });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');

      const options = mimeType ? { mimeType } : {};
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const actualType = mediaRecorder.mimeType || mimeType || 'audio/webm';
        const ext = actualType.includes('webm') ? 'webm' : (actualType.includes('ogg') ? 'ogg' : 'wav');
        const audioBlob = new Blob(audioChunksRef.current, { type: actualType });
        const recordedFile = new File([audioBlob], `live_mic_recording_${Date.now()}.${ext}`, { type: actualType });
        setFile(recordedFile);
        setAudioUrl(URL.createObjectURL(audioBlob));
        stopAudioVisualizer();
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingSeconds(0);

      // Start duration counter (max 30s)
      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 29) {
            stopRecording();
            return 30;
          }
          return prev + 1;
        });
      }, 1000);

      // Start AudioContext Visualizer
      startAudioVisualizer(stream);
    } catch (err) {
      console.error('Microphone access denied:', err);
      setMicError('Microphone access denied or unavailable. Please grant microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const startAudioVisualizer = (stream) => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 64;
      const source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);

      const canvas = canvasRef.current;
      if (!canvas) return;
      const canvasCtx = canvas.getContext('2d');
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        animationFrameRef.current = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = 'rgb(15, 23, 42)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 1.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          // Gradient from emerald to teal
          canvasCtx.fillStyle = `rgb(16, ${140 + dataArray[i] / 2}, 140)`;
          canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          x += barWidth + 2;
        }
      };

      draw();
    } catch (e) {
      console.warn('Audio visualization not supported in current environment', e);
    }
  };

  const stopAudioVisualizer = () => {
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onAnalyze(file, selectedModel, selectedChannel);
  };

  return (
    <div className="glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl border border-slate-800">
      {/* Header with Mode Switcher */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-white flex items-center space-x-2">
            <span>Voice Impersonation Analysis</span>
            <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2.5 py-0.5 rounded-full border border-emerald-500/30 font-medium">
              Real-Time Forensic Engine
            </span>
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Detect whether audio is authentic human speech or AI-cloned/vocoded speech across real-world transmission channels.
          </p>
        </div>

        {/* Input Mode Selector: Upload vs Live Mic */}
        <div className="inline-flex p-1 bg-slate-900 border border-slate-800 rounded-xl self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setInputMode('upload')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              inputMode === 'upload'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span>📁 Upload File</span>
          </button>
          <button
            type="button"
            onClick={() => setInputMode('record')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 ${
              inputMode === 'record'
                ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <span>🎙️ Live Microphone</span>
          </button>
        </div>
      </div>

      {/* Preset Demo Quick-Selector */}
      <div className="mb-6 bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center justify-between">
          <span>⚡ Or select a verified test recording:</span>
          <span className="text-[10px] text-slate-500 font-mono">1-Click Test</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {presets.map((p) => {
            const isSelected = presetActive === p.id;
            const isFake = p.type === 'FAKE';
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => loadPreset(p)}
                className={`p-2.5 rounded-lg text-left text-xs transition-all border ${
                  isSelected
                    ? isFake
                      ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
                      : 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                    : 'bg-slate-800/60 border-slate-700/60 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <div className="font-semibold truncate">{p.name}</div>
                <div className="flex items-center space-x-1 mt-1 text-[10px] opacity-80">
                  <span className={`w-1.5 h-1.5 rounded-full ${isFake ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
                  <span>{p.label}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {/* MODE 1: FILE UPLOAD ZONE */}
        {inputMode === 'upload' && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
              dragOver
                ? 'border-emerald-400 bg-emerald-500/10 scale-[1.01]'
                : file
                ? 'border-emerald-500/40 bg-slate-900/80'
                : 'border-slate-700 hover:border-slate-500 bg-slate-900/40 hover:bg-slate-900/60'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.mp4,.opus,audio/*"
              onChange={handleFileChange}
              className="hidden"
            />

            <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-slate-800/80 flex items-center justify-center text-2xl shadow-inner">
              {file ? '🎵' : '📁'}
            </div>

            {file ? (
              <div>
                <p className="text-sm font-semibold text-emerald-400 truncate max-w-md mx-auto">
                  {file.name}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {(file.size / 1024).toFixed(1)} KB • Click or drop a different file to replace
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm font-medium text-slate-200">
                  Click to browse or drag and drop audio file here
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Supports WAV, MP3, M4A, AAC, WMA, FLAC (Windows Voice Recorder, Mobile Audio, Studio)
                </p>
              </div>
            )}
          </div>
        )}

        {/* MODE 2: LIVE MICROPHONE RECORDER */}
        {inputMode === 'record' && (
          <div className="border border-slate-800 rounded-xl p-6 bg-slate-900/80 text-center">
            <div className="max-w-md mx-auto">
              <div className="flex items-center justify-center space-x-3 mb-4">
                <span className={`w-3.5 h-3.5 rounded-full ${isRecording ? 'bg-rose-500 animate-ping' : 'bg-slate-600'}`}></span>
                <span className="font-mono text-sm font-bold text-slate-200">
                  {isRecording ? `RECORDING: 00:${recordingSeconds < 10 ? '0' : ''}${recordingSeconds} / 00:30` : file ? 'RECORDING CAPTURED' : 'LIVE MICROPHONE READY'}
                </span>
              </div>

              {/* Live Canvas Visualizer */}
              <div className="h-16 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center mb-4">
                <canvas ref={canvasRef} width="320" height="60" className="w-full h-full" />
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-center space-x-3">
                {!isRecording ? (
                  <button
                    type="button"
                    onClick={startRecording}
                    className="px-6 py-2.5 rounded-xl font-bold text-xs bg-rose-500 hover:bg-rose-400 text-white shadow-lg shadow-rose-500/25 flex items-center space-x-2 transition-all"
                  >
                    <span>🎙️</span>
                    <span>{file ? 'Re-Record Voice' : 'Start Recording'}</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopRecording}
                    className="px-6 py-2.5 rounded-xl font-bold text-xs bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-500/25 flex items-center space-x-2 transition-all animate-pulse"
                  >
                    <span>⏹️</span>
                    <span>Stop Recording</span>
                  </button>
                )}
              </div>

              {micError && (
                <p className="text-xs text-rose-400 mt-3">{micError}</p>
              )}
            </div>
          </div>
        )}

        {/* Audio Player if file loaded */}
        {audioUrl && (
          <div className="mt-4 p-3 bg-slate-900/90 rounded-xl border border-slate-800 flex items-center space-x-3">
            <span className="text-xs text-slate-400 font-mono">Audio Preview:</span>
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              className="w-full h-8 outline-none rounded"
            />
          </div>
        )}

        {/* Real-World Channel Degradation Simulator */}
        <div className="mt-6 p-4 bg-slate-900/80 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <div>
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
                Real-World Transmission Channel Simulator:
              </label>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Simulate how audio degrades across messaging apps and cellular networks before forensic analysis.
              </p>
            </div>
            <span className="text-[10px] text-teal-400 font-mono uppercase bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/30">
              Active: {channels.find(c => c.id === selectedChannel)?.name}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {channels.map((ch) => {
              const isSelected = selectedChannel === ch.id;
              return (
                <button
                  key={ch.id}
                  type="button"
                  onClick={() => setSelectedChannel(ch.id)}
                  className={`p-3 rounded-xl text-left transition-all border ${
                    isSelected
                      ? 'bg-teal-500/15 border-teal-500 text-teal-300 ring-1 ring-teal-500/50 shadow-md shadow-teal-500/10'
                      : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800/70'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-base">{ch.icon}</span>
                    <span className="font-bold text-xs text-slate-200 truncate">{ch.name}</span>
                  </div>
                  <div className="text-[10px] text-teal-400 font-mono mt-1">
                    {ch.badge}
                  </div>
                  <p className="text-[10px] text-slate-400 mt-1 leading-tight line-clamp-2">
                    {ch.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Model Selector Bar */}
        <div className="mt-6 p-4 bg-slate-900/70 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
              Target Model Architecture:
            </label>
            <span className="text-[11px] text-emerald-400 font-mono">
              Active: {selectedModel.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {/* 1. PREMIER UNIFIED MODEL (ALL-IN-ONE MAXIMUM ACCURACY) */}
            <button
              type="button"
              onClick={() => setSelectedModel('voiceguard_unified')}
              className={`p-4 rounded-xl text-left transition-all border ${
                selectedModel === 'voiceguard_unified' || selectedModel === 'w2v2_ensemble' || selectedModel === 'w2v2_large_scale'
                  ? 'bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-cyan-500/20 border-emerald-400 text-emerald-200 ring-2 ring-emerald-400/50 shadow-xl shadow-emerald-500/15'
                  : 'bg-slate-800/40 border-slate-700/60 text-slate-300 hover:text-white hover:bg-slate-800/70'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">🛡️</span>
                  <span className="font-bold text-sm text-white">VoiceGuard Unified Fine-Tuned</span>
                </div>
                <span className="text-[10px] uppercase font-bold bg-gradient-to-r from-emerald-400 to-teal-400 text-slate-950 px-2.5 py-0.5 rounded-full shadow">
                  Premier • Max Accuracy
                </span>
              </div>
              <div className="text-[11px] text-emerald-400 font-mono mt-1 font-semibold">
                99.85% – 100.0% Acc • 10,000 Clips • Full Invariance Fusion
              </div>
              <p className="text-xs mt-2 text-slate-300 leading-relaxed">
                Combines our 10,000-clip cloud fine-tuning, 67-dim forensic feature fusion, reverberation invariance, and internal consensus boosting into a single unified detector for maximum accuracy.
              </p>
            </button>

            {/* 2. REFERENCE REPOSITORY BASELINE MODEL */}
            <button
              type="button"
              onClick={() => setSelectedModel('reference_baseline')}
              className={`p-4 rounded-xl text-left transition-all border ${
                selectedModel === 'reference_baseline' || selectedModel === 'w2v2_aasist_pretrained'
                  ? 'bg-cyan-500/15 border-cyan-400 text-cyan-200 ring-2 ring-cyan-400/50 shadow-lg shadow-cyan-500/15'
                  : 'bg-slate-800/40 border-slate-700/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800/70'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">📜</span>
                  <span className="font-bold text-sm text-slate-100">Reference Model (Tak et al.)</span>
                </div>
                <span className="text-[10px] uppercase font-semibold bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">
                  Reference Repo • Zero-Shot
                </span>
              </div>
              <div className="text-[11px] text-cyan-400 font-mono mt-1 font-semibold">
                46.15% Acc • Odyssey 2022 Checkpoint
              </div>
              <p className="text-xs mt-2 text-slate-400 leading-relaxed">
                Official unmodified benchmark checkpoint from the reference repository (Tak et al., 317.8M params). Evaluated directly on held-out test audio without fine-tuning.
              </p>
            </button>
          </div>
        </div>

        {/* Error message if any */}
        {error && (
          <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-2">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Analyze Button */}
        <div className="mt-6 flex justify-end">
          <button
            type="submit"
            disabled={!file || loading}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-bold text-sm transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg ${
              !file || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:scale-[1.01]'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-slate-950" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Analyzing Acoustic Features...</span>
              </>
            ) : (
              <>
                <span>Run Impersonation Detection</span>
                <span>→</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
