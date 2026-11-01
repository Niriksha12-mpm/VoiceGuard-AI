import React from 'react';

export default function Navbar({ activeTab, setActiveTab, backendStatus }) {
  const tabs = [
    { id: 'predict', label: 'Audio Detection', icon: '🎙️' },
    { id: 'comparison', label: 'Model Comparison', icon: '⚖️' },
    { id: 'metrics', label: 'Evaluation Metrics', icon: '📊' },
    { id: 'datasets', label: 'Dataset Audit & Inventory', icon: '🗄️' },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Project Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('predict')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 via-teal-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <span className="text-xl">🛡️</span>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                  VoiceGuard AI
                </span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  v1.0
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Voice Impersonation & Audio Deepfake Detection
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-3 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all duration-200 flex items-center space-x-1.5 ${
                    isActive
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span className="hidden md:inline">{tab.label}</span>
                  <span className="md:hidden">{tab.label.split(' ')[0]}</span>
                </button>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className="hidden lg:flex items-center space-x-2 bg-slate-900/80 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
            <span className={`w-2 h-2 rounded-full ${backendStatus ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span className="text-slate-400 font-mono">
              FastAPI: <span className={backendStatus ? 'text-emerald-400 font-medium' : 'text-amber-400'}>
                {backendStatus ? 'Online' : 'Checking...'}
              </span>
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
