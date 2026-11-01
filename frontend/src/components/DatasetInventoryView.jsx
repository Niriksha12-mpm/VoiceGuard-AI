import React, { useState, useEffect } from 'react';

export default function DatasetInventoryView() {
  const [auditData, setAuditData] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/audit')
      .then((res) => res.json())
      .then((data) => setAuditData(data))
      .catch((err) => console.error('Error fetching audit data:', err));
  }, []);

  if (!auditData) {
    return (
      <div className="glass-panel rounded-2xl p-8 text-center text-slate-400">
        Loading dataset audit and inventory...
      </div>
    );
  }

  const { audit_report, duplicate_removal_report } = auditData;
  const inventory = audit_report.inventory || [];
  const statusCounts = audit_report.status_counts || {};

  const getStatusBadge = (status) => {
    if (status.startsWith('VERIFIED_ACCEPTED')) {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          Verified & Accepted
        </span>
      );
    }
    if (status === 'VERIFIED_WITH_CAUTION') {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
          Caution / Deduplicated
        </span>
      );
    }
    if (status.startsWith('REJECTED')) {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
          Rejected
        </span>
      );
    }
    return (
      <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
        Excluded Duplicate
      </span>
    );
  };

  return (
    <div className="space-y-8">
      {/* Title & Audit Statistics */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="text-xs uppercase font-mono tracking-wider text-emerald-400 mb-1">
              Dataset Governance & Forensic Integrity
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-white">
              Dataset Inventory & Audit Report
            </h2>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl leading-relaxed">
              Every suggested candidate was rigorously audited using official sources and trusted repositories.
              Inaccessible, animal-voice, modality-mismatched, and duplicate datasets were strictly eliminated.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-center">
            <div className="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-400">AUDITED</div>
              <div className="text-xl font-bold text-white mt-0.5">{audit_report.total_investigated}</div>
            </div>
            <div className="bg-slate-900/90 p-3 rounded-xl border border-emerald-500/30">
              <div className="text-[10px] text-emerald-400">ACCEPTED</div>
              <div className="text-xl font-bold text-emerald-400 mt-0.5">{statusCounts.verified_and_accepted}</div>
            </div>
            <div className="bg-slate-900/90 p-3 rounded-xl border border-rose-500/30">
              <div className="text-[10px] text-rose-400">REJECTED</div>
              <div className="text-xl font-bold text-rose-400 mt-0.5">{statusCounts.rejected_unsuitable_or_mismatch}</div>
            </div>
            <div className="bg-slate-900/90 p-3 rounded-xl border border-purple-500/30">
              <div className="text-[10px] text-purple-400">DUPLICATES</div>
              <div className="text-xl font-bold text-purple-400 mt-0.5">{statusCounts.excluded_redundant_duplicate}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Complete Audited Datasets Table */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
        <h3 className="text-lg font-bold text-white mb-4">
          Dataset Inventory & Verification Decisions
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs font-mono text-slate-400 uppercase">
                <th className="py-3 px-4">Dataset Name</th>
                <th className="py-3 px-4">Official Source / Host</th>
                <th className="py-3 px-4">License</th>
                <th className="py-3 px-4">Format / Samples</th>
                <th className="py-3 px-4">Verification Status</th>
                <th className="py-3 px-4">Audit Finding / Rationale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-xs sm:text-sm">
              {inventory.map((ds) => (
                <tr key={ds.id} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-4 px-4 font-bold text-slate-100">
                    <div>{ds.name}</div>
                    <div className="text-[10px] text-slate-500 font-mono mt-0.5">{ds.classes.join(', ')}</div>
                  </td>
                  <td className="py-4 px-4 text-slate-400 text-xs">
                    <div>{ds.official_source}</div>
                    {ds.source_url && (
                      <a
                        href={ds.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-emerald-400 hover:underline font-mono text-[10px] mt-0.5 block truncate max-w-xs"
                      >
                        {ds.source_url}
                      </a>
                    )}
                  </td>
                  <td className="py-4 px-4 font-mono text-xs text-slate-400">{ds.license}</td>
                  <td className="py-4 px-4 font-mono text-xs text-slate-300">
                    <div>{ds.audio_format}</div>
                    <div className="text-[10px] text-slate-500">{ds.sample_count?.toLocaleString()} samples</div>
                  </td>
                  <td className="py-4 px-4 whitespace-nowrap">
                    {getStatusBadge(ds.verification_status)}
                  </td>
                  <td className="py-4 px-4 text-xs text-slate-300 max-w-sm leading-relaxed">
                    {ds.decision_reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Duplicate & Near-Duplicate Removal Report */}
      {duplicate_removal_report && (
        <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <div>
              <div className="text-xs font-mono uppercase text-emerald-400">De-duplication Engine</div>
              <h3 className="text-lg font-bold text-white">
                Audio Duplicate & Near-Duplicate Removal Audit
              </h3>
            </div>
            <div className="text-xs font-mono bg-slate-900/90 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-800">
              Raw Files: {duplicate_removal_report.initial_sample_count} → Retained Canonical: {duplicate_removal_report.clean_sample_count} ({duplicate_removal_report.removed_count} Removed)
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            {/* Exact Hash Duplicates */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <h4 className="text-xs font-mono uppercase text-slate-400 mb-2">
                1. Exact Bit-for-Bit Hash Matches (SHA-256)
              </h4>
              <p className="text-xs text-slate-400 mb-3">
                Detected identical audio files across dataset folders with identical SHA-256 byte sums:
              </p>
              <div className="space-y-2 text-xs font-mono">
                {duplicate_removal_report.exact_duplicates?.groups.map((grp, idx) => (
                  <div key={idx} className="p-2.5 bg-slate-950 rounded border border-slate-800">
                    <div className="text-emerald-400 font-bold truncate">Primary: {grp.primary}</div>
                    <div className="text-rose-400 text-[11px] mt-0.5 truncate">
                      Duplicates Pruned: {grp.duplicates.join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Acoustic Near-Duplicates / Re-encoded */}
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800">
              <h4 className="text-xs font-mono uppercase text-slate-400 mb-2">
                2. Cross-Format & Re-Encoded Near-Duplicates (Acoustic Fingerprint)
              </h4>
              <p className="text-xs text-slate-400 mb-3">
                Detected identical speech recordings re-encoded into WAV, MP3, and FLAC containers (Similarity &gt; 0.985):
              </p>
              <div className="space-y-2 text-xs font-mono max-h-48 overflow-y-auto pr-1">
                {duplicate_removal_report.near_duplicates_reencoded?.pairs.slice(0, 4).map((pair, idx) => (
                  <div key={idx} className="p-2.5 bg-slate-950 rounded border border-slate-800">
                    <div className="flex items-center justify-between text-slate-300">
                      <span className="truncate">{pair.sample_a} ↔ {pair.sample_b}</span>
                      <span className="text-emerald-400 text-[11px] font-bold">{(pair.acoustic_similarity * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-1">{pair.action_taken}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Leakage Prevention & Bias Statement */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel rounded-2xl p-6 border border-slate-800">
          <div className="text-xs font-mono uppercase text-emerald-400 mb-1">Methodology</div>
          <h4 className="text-base font-bold text-white mb-2">
            Data Leakage Prevention Protocol
          </h4>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li className="flex items-start space-x-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Group-Based Partitioning:</strong> Splitting was grouped on <code className="text-emerald-300 font-mono">parent_file</code> and <code className="text-emerald-300 font-mono">speaker_id</code>. All segments from the same speaker and recording stay strictly together.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Held-Out Isolation:</strong> The 20% test split was never touched during training, fine-tuning, preprocessing choices, or hyperparameter tuning.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-emerald-400">✓</span>
              <span><strong>Deterministic Seed:</strong> Seed 42 was fixed across all data shuffles, train-test splits, and augmentations.</span>
            </li>
          </ul>
        </div>

        <div className="glass-panel rounded-2xl p-6 border border-slate-800">
          <div className="text-xs font-mono uppercase text-amber-400 mb-1">Ethics & Forensics</div>
          <h4 className="text-base font-bold text-white mb-2">
            Limitations & Potential Bias Analysis
          </h4>
          <ul className="text-xs text-slate-300 space-y-2 leading-relaxed">
            <li className="flex items-start space-x-2">
              <span className="text-amber-400">•</span>
              <span><strong>Vocoder Domain Shift:</strong> Detectors trained on older vocoders (e.g. WaveGlow) may experience performance drops when exposed to newer zero-shot voice cloners (e.g. VITS, XTTS, ElevenLabs v3).</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-amber-400">•</span>
              <span><strong>Lossy Codec Compression:</strong> Telephony compression (GSM, AMR) and MP3 compression at low bitrates (&lt;64kbps) can mask subtle high-frequency vocoder artifacts.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-amber-400">•</span>
              <span><strong>Linguistic Diversity:</strong> The current benchmark primarily reflects English utterances; non-English tonal languages require cross-lingual calibration.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
