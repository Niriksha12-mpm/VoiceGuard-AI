import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import AudioUploader from './components/AudioUploader';
import PredictionResult from './components/PredictionResult';
import ModelComparisonView from './components/ModelComparisonView';
import MetricsDashboard from './components/MetricsDashboard';
import DatasetInventoryView from './components/DatasetInventoryView';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [selectedModel, setSelectedModel] = useState('voiceguard_unified');
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState(false);
  const [comparisonData, setComparisonData] = useState(null);
  const [phase3Data, setPhase3Data] = useState(null);

  // Check backend health and fetch model comparison data
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/health');
        if (res.ok) {
          setBackendStatus(true);
        } else {
          setBackendStatus(false);
        }
      } catch (err) {
        setBackendStatus(false);
      }
    };

    const fetchComparison = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/comparison');
        if (res.ok) {
          const data = await res.json();
          setComparisonData(data);
        }
      } catch (err) {
        console.error('Failed to fetch comparison data:', err);
      }
    };

    const fetchPhase3 = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/phase3-experiments');
        if (res.ok) {
          const data = await res.json();
          setPhase3Data(data);
        }
      } catch (err) {
        console.error('Failed to fetch phase 3 data:', err);
      }
    };

    checkBackend();
    fetchComparison();
    fetchPhase3();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle Audio Analysis
  const handleAnalyze = async (file, modelType, channelSim = 'clean') => {
    setLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`http://127.0.0.1:8000/api/predict?model_type=${modelType}&channel_sim=${channelSim}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.detail || 'Prediction failed.');
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during audio classification.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col justify-between">
      <div>
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          backendStatus={backendStatus}
        />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Tab 1: Audio Prediction */}
          {activeTab === 'predict' && (
            <div className="space-y-6">
              <AudioUploader
                onAnalyze={handleAnalyze}
                loading={loading}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                error={error}
              />
              <PredictionResult result={predictionResult} />
            </div>
          )}

          {/* Tab 2: Model Comparison */}
          {activeTab === 'comparison' && (
            <ModelComparisonView comparisonData={comparisonData} phase3Data={phase3Data} />
          )}

          {/* Tab 3: Evaluation Metrics Dashboard */}
          {activeTab === 'metrics' && (
            <MetricsDashboard comparisonData={comparisonData} />
          )}

          {/* Tab 4: Dataset Audit & Inventory */}
          {activeTab === 'datasets' && (
            <DatasetInventoryView />
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="glass-panel border-t border-slate-800/80 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <span className="font-bold text-slate-400">VoiceGuard AI</span> • Machine Learning Voice Impersonation Detection System
          </div>
          <div className="font-mono text-[11px] text-slate-500">
            PyTorch • FastAPI • React.js • Tailwind CSS • Librosa
          </div>
        </div>
      </footer>
    </div>
  );
}
