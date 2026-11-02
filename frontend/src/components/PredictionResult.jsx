import React, { useState } from 'react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { prediction, audio_metadata, acoustic_diagnostics, waveform_preview, temporal_analysis } = result;
  const isReal = prediction.label === 'REAL';
  const [selectedSegment, setSelectedSegment] = useState(null);

  const timeline = temporal_analysis?.timeline || [];
  const splicingDetected = temporal_analysis?.splicing_detected || false;
  const channelSim = prediction.channel_simulation;

  return (
    <div className={`mt-8 glass-panel rounded-2xl p-6 sm:p-8 shadow-2xl transition-all duration-300 border ${
      isReal ? 'border-emerald-500/40 glow-emerald' : 'border-rose-500/40 glow-rose'
    }`}>
      {/* Splicing Detection Alert Banner */}
      {splicingDetected && (
        <div className="mb-6 p-4 rounded-xl bg-amber-500/15 border border-amber-500/40 flex items-start space-x-3">
          <span className="text-xl">⚠️</span>
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-amber-300">
              Temporal Splicing Alert • Partial Deepfake Injected
            </div>
            <p className="text-xs text-amber-100/90 mt-1 leading-relaxed">
              {temporal_analysis.splicing_summary} VoiceGuard detected isolated speech intervals with synthetic vocoder artifacts while other segments retain natural human acoustic continuity.
            </p>
          </div>
        </div>
      )}

      {/* Primary Verdict Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <div className="text-xs uppercase font-mono tracking-wider text-slate-400 mb-1 flex items-center space-x-2">
            <span>Overall Classification Verdict</span>
            {channelSim?.applied && (
              <span className="text-[10px] bg-teal-500/20 text-teal-300 border border-teal-500/40 px-2 py-0.5 rounded-full font-mono">
                {channelSim.name}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-3">
            <span className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${
              isReal ? 'text-emerald-400' : 'text-rose-500'
            }`}>
              {prediction.label}
            </span>
            <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
              isReal ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
            }`}>
              {isReal ? 'Human Voice' : 'AI Synthetic / Deepfake'}
            </span>
          </div>
        </div>

        {/* Forensic Verification Status Badge */}
        <div className="bg-slate-900/90 px-5 py-3 rounded-xl border border-slate-800 text-right sm:text-right w-full sm:w-auto">
          <div className="text-xs text-slate-400">Forensic Integrity Status</div>
          <div className="text-lg font-black font-mono flex items-center justify-end space-x-2 mt-0.5">
            <span className={isReal ? 'text-emerald-400' : 'text-rose-400'}>
              {isReal ? 'VERIFIED AUTHENTIC' : 'SYNTHETIC SPOOF DETECTED'}
            </span>
          </div>
        </div>
      </div>

      {/* Forensic Classification Status Banner */}
      <div className="my-5 p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center space-x-2.5 text-xs font-mono">
          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${isReal ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
          <span className="text-slate-300">
            {isReal 
              ? 'Acoustic signature confirms natural human vocal cord dynamics and physiological harmonic structure.'
              : 'Synthetic vocoder artifacts / AI voice impersonation patterns detected across acoustic spectrum.'}
          </span>
        </div>
        <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold uppercase tracking-wider self-start sm:self-auto shrink-0 ${
          isReal ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
        }`}>
          {isReal ? 'PASSED • BONA-FIDE SPEECH' : 'FLAGGED • DEEPFAKE VOICE'}
        </span>
      </div>


      {/* Forensic Ensemble Consensus Breakdown */}
      {prediction.ensemble_breakdown && (
        <div className="mb-6 bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className="text-base">🛡️</span>
              <span className="text-xs font-mono uppercase font-bold text-white">
                Multi-Model Consensus Breakdown
              </span>
            </div>
            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider ${
              prediction.ensemble_breakdown.consensus_status.includes('Unanimous')
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
            }`}>
              {prediction.ensemble_breakdown.consensus_status}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {prediction.ensemble_breakdown.models.map((m, idx) => {
              const isVoteReal = m.vote === 'REAL';
              return (
                <div key={idx} className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-semibold text-slate-200 truncate">{m.name}</span>
                    <span className="text-[10px] text-slate-500 font-mono">Weight: {m.weight}</span>
                  </div>
                  <div className="flex items-center justify-between mt-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      isVoteReal ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                    }`}>
                      {m.vote}
                    </span>
                    <span className="text-xs font-mono text-slate-300">
                      Verdict: <strong className={isVoteReal ? 'text-emerald-400' : 'text-rose-400'}>{isVoteReal ? 'Authentic' : 'Synthetic'}</strong>
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* NEW: Temporal Splicing Timeline & Heatmap */}
      {timeline.length > 0 && (
        <div className="mb-6 bg-slate-900/90 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono uppercase font-bold text-slate-200">
                Temporal Splicing Timeline (Sliding Window)
              </span>
              <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700 font-mono">
                {timeline.length} window(s) • 2.0s slice
              </span>
            </div>
            <div className="flex items-center space-x-3 text-[10px] font-mono">
              <span className="flex items-center space-x-1 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                <span>Authentic</span>
              </span>
              <span className="flex items-center space-x-1 text-rose-400">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                <span>Synthetic</span>
              </span>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 mb-3">
            Click or hover over any slice to inspect second-by-second spoof probability across the recording duration.
          </p>

          {/* Interactive Timeline Heatmap Track */}
          <div className="grid grid-cols-1 gap-1.5">
            <div className="flex w-full h-9 rounded-lg overflow-hidden border border-slate-800 bg-slate-950 p-1 gap-1">
              {timeline.map((seg, idx) => {
                const segIsReal = seg.label === 'REAL';
                const isHovered = selectedSegment?.segment_index === seg.segment_index;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setSelectedSegment(seg)}
                    onMouseEnter={() => setSelectedSegment(seg)}
                    style={{ flex: 1 }}
                    className={`h-full rounded transition-all duration-150 flex items-center justify-center text-[10px] font-mono font-bold ${
                      segIsReal
                        ? isHovered
                          ? 'bg-emerald-400 text-slate-950 scale-105 shadow-md shadow-emerald-500/30'
                          : 'bg-emerald-500/70 hover:bg-emerald-400 text-slate-950'
                        : isHovered
                        ? 'bg-rose-400 text-slate-950 scale-105 shadow-md shadow-rose-500/30'
                        : 'bg-rose-500/80 hover:bg-rose-400 text-slate-950'
                    }`}
                    title={`${seg.start_time}s - ${seg.end_time}s: ${seg.label === 'REAL' ? 'Authentic Human' : 'Synthetic Deepfake'}`}
                  >
                    {idx + 1}
                  </button>
                );
              })}
            </div>

            {/* Selected Segment Details Callout */}
            {selectedSegment ? (
              <div className="mt-2 p-2.5 bg-slate-950/80 rounded-lg border border-slate-800/80 flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-cyan-300 font-bold">
                    Segment #{selectedSegment.segment_index + 1} [{selectedSegment.start_time}s – {selectedSegment.end_time}s]
                  </span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    selectedSegment.label === 'REAL' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
                  }`}>
                    {selectedSegment.label}
                  </span>
                </div>
                <div className="font-mono text-slate-300 text-[11px] space-x-3">
                  <span>Acoustic Integrity: <strong className={selectedSegment.label === 'REAL' ? 'text-emerald-400' : 'text-rose-400'}>{selectedSegment.label === 'REAL' ? 'Natural Speech' : 'Synthesized Spoof'}</strong></span>
                </div>
              </div>
            ) : (
              <div className="text-[10px] text-slate-500 font-mono text-center pt-1">
                Hover over numbered windows above for segment-level diagnostics
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Used & Channel Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400 font-mono uppercase mb-1">Model Evaluator</div>
          <div className="text-sm font-bold text-white flex items-center space-x-2">
            <span>⚙️</span>
            <span>{prediction.model_used}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
            {prediction.model_description}
          </p>
          {channelSim?.applied && (
            <div className="mt-2.5 pt-2 border-t border-slate-800 text-[11px] text-teal-300 font-mono">
              ⚡ Channel Emulation: {channelSim.description}
            </div>
          )}
        </div>

        <div className="bg-amber-500/10 p-4 rounded-xl border border-amber-500/25">
          <div className="text-xs text-amber-400 font-mono uppercase mb-1 flex items-center space-x-1">
            <span>ℹ️</span>
            <span>Confidence & Forensic Disclaimer</span>
          </div>
          <p className="text-xs text-amber-200/90 leading-relaxed">
            {prediction.confidence_disclaimer} Results are evaluated across short spectro-temporal windows to protect against targeted splicing attacks in live scam scenarios.
          </p>
        </div>
      </div>

      {/* Waveform Visualization SVG */}
      {waveform_preview && waveform_preview.length > 0 && (
        <div className="mb-6 bg-slate-900/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono uppercase text-slate-400">Standardized Normalized Waveform (16kHz)</span>
            <span className="text-[10px] text-slate-500 font-mono">120 Downsampled Points</span>
          </div>
          <div className="h-20 w-full flex items-center justify-between space-x-0.5 overflow-hidden">
            {waveform_preview.map((val, idx) => {
              const heightPct = Math.max(8, Math.min(100, Math.abs(val) * 100));
              return (
                <div
                  key={idx}
                  style={{ height: `${heightPct}%` }}
                  className={`w-1 rounded-full transition-all duration-300 ${
                    isReal ? 'bg-emerald-500/70 hover:bg-emerald-400' : 'bg-rose-500/70 hover:bg-rose-400'
                  }`}
                  title={`Point ${idx}: ${val}`}
                ></div>
              );
            })}
          </div>
        </div>
      )}

      {/* Audio Metadata and Forensic Diagnostics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-800">
        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Duration</div>
          <div className="text-sm font-bold font-mono text-white mt-0.5">
            {audio_metadata.duration_seconds}s
          </div>
          <div className="text-[10px] text-slate-500">{audio_metadata.num_segments_analyzed} segment(s) analyzed</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Centroid</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_centroid_hz} Hz
          </div>
          <div className="text-[10px] text-slate-500">Spectral center of mass</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Rolloff</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_rolloff_hz} Hz
          </div>
          <div className="text-[10px] text-slate-500">85% Energy Boundary</div>
        </div>

        <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="text-[11px] text-slate-400">Spectral Flatness</div>
          <div className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
            {acoustic_diagnostics.spectral_flatness}
          </div>
          <div className="text-[10px] text-slate-500">Tonality vs Noise measure</div>
        </div>
      </div>
    </div>
  );
}
