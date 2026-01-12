'use client';

import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';


export default function Home() {
  const [activeTab, setActiveTab] = useState('intro');
  const [deliverables, setDeliverables] = useState([]);
  const [results, setResults] = useState({ images: [], csvs: [], latex: [] });
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documentContent, setDocumentContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);
  const [csvData, setCsvData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [delRes, resRes] = await Promise.all([
          fetch('/api/deliverables'),
          fetch('/api/results')
        ]);
        const delData = await delRes.json();
        const resData = await resRes.json();
        setDeliverables(delData);
        setResults(resData);
      } catch (err) {
        console.error('Failed to fetch data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  async function loadDocument(filename) {
    setLoading(true);
    try {
      const res = await fetch(`/api/deliverables/${filename}`);
      const data = await res.json();
      setDocumentContent(data.content);
      setSelectedDocument(filename);
      setActiveTab('documents');
    } catch (err) {
      console.error('Failed to load document:', err);
    } finally {
      setLoading(false);
    }
  }

  async function loadCsv(filename) {
    try {
      const res = await fetch(`/api/results/${filename}`);
      const data = await res.json();
      setCsvData({ filename, content: data.content });
    } catch (err) {
      console.error('Failed to load CSV:', err);
    }
  }

  function parseCSV(csvString) {
    const lines = csvString.trim().split('\n');
    const headers = lines[0].split(',');
    const rows = lines.slice(1).map(line => line.split(','));
    return { headers, rows };
  }

  const stats = [
    { label: 'Deliverables', value: deliverables.length, icon: '📄' },
    { label: 'Result Images', value: results.images?.length || 0, icon: '📊' },
    { label: 'CSV Files', value: results.csvs?.length || 0, icon: '📈' },
    { label: 'Datasets', value: 4, icon: '🛩️' },
  ];

  // Group images by dataset
  const imagesByDataset = results.images?.reduce((acc, img) => {
    const match = img.match(/^(FD00\d|all_datasets)/);
    const key = match ? match[1] : 'Other';
    if (!acc[key]) acc[key] = [];
    acc[key].push(img);
    return acc;
  }, {}) || {};

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="sidebar w-64 fixed h-full p-4 flex flex-col">
        <div className="mb-8">
          <h1 className="text-xl font-bold gradient-text">NASA Data</h1>
          <p className="text-sm text-zinc-500">RUL Analysis Dashboard</p>
        </div>

        <nav className="flex-1">
          <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Navigation</div>

          <div
            className={`sidebar-item ${activeTab === 'intro' ? 'active' : ''}`}
            onClick={() => setActiveTab('intro')}
          >
            <span className="mr-2">👋</span> Introduction
          </div>

          <div
            className={`sidebar-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <span className="mr-2">🏠</span> Overview
          </div>

          <div
            className={`sidebar-item ${activeTab === 'documents' ? 'active' : ''}`}
            onClick={() => setActiveTab('documents')}
          >
            <span className="mr-2">📄</span> Deliverables
          </div>

          <div
            className={`sidebar-item ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            <span className="mr-2">📊</span> Results Hub
          </div>

          <div
            className={`sidebar-item ${activeTab === 'data' ? 'active' : ''}`}
            onClick={() => setActiveTab('data')}
          >
            <span className="mr-2">📈</span> Data Explorer
          </div>

          <div className="text-xs text-zinc-500 uppercase tracking-wider mt-6 mb-2">Quick Access</div>
          {deliverables.map((file) => (
            <div
              key={file}
              className={`sidebar-item text-sm ${selectedDocument === file ? 'active' : ''}`}
              onClick={() => loadDocument(file)}
            >
              {file.replace('.md', '').replace(/_/g, ' ')}
            </div>
          ))}
        </nav>

        <div className="glass-card p-3 text-center">
          <div className="text-xs text-zinc-400">Powered by</div>
          <div className="text-sm gradient-text font-semibold">Next.js + Tailwind</div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64 flex-1 p-8">
        {loading && (
          <div className="flex items-center justify-center h-64">
            <div className="spinner"></div>
          </div>
        )}

        {!loading && activeTab === 'intro' && (
          <div className="animate-fadeIn max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h1 className="text-5xl font-bold gradient-text mb-4">Advanced RUL Analysis Toolkit</h1>
              <p className="text-xl text-zinc-400">
                Predicting the Remaining Useful Life (RUL) of Aircraft Engines
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
              <div className="glass-card p-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
                  <span className="text-3xl">🛩️</span> What is this?
                </h2>
                <p className="text-zinc-400 leading-relaxed mb-4">
                  This dashboard presents the results of a comprehensive study on <strong>Predictive Maintenance</strong>.
                  We use advanced Artificial Intelligence (AI) to predict when an aircraft engine will fail so maintenance can be scheduled
                  <em>before</em> a problem occurs.
                </p>
                <p className="text-zinc-400 leading-relaxed">
                  The dataset used is the famous <strong>NASA C-MAPSS</strong> (Commercial Modular Aero-Propulsion System Simulation),
                  which simulates realistic engine degradation under various conditions.
                </p>
              </div>

              <div className="glass-card p-8">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
                  <span className="text-3xl">🛡️</span> Why it matters?
                </h2>
                <p className="text-zinc-400 leading-relaxed mb-4">
                  <strong>Safety & Efficiency:</strong> Accurately predicting engine life prevents catastrophic failures and reduces unnecessary maintenance costs.
                </p>
                <p className="text-zinc-400 leading-relaxed">
                  <strong>Cyber-Security Focus:</strong> A unique aspect of this project is testing <strong>Robustness</strong>.
                  We evaluate how well our AI models perform even when sensor data is "noisy" or potentially compromised by cyber-attacks.
                  A model that fails under slight noise is a security risk!
                </p>
              </div>
            </div>

            <h2 className="text-2xl font-bold mb-6 text-center text-zinc-200">What You Can Explore</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              <div className="glass-card p-6 hover:border-[var(--accent)] cursor-pointer transition-colors" onClick={() => setActiveTab('documents')}>
                <div className="text-4xl mb-4">📄</div>
                <h3 className="text-xl font-semibold mb-2">Deliverables</h3>
                <p className="text-zinc-500 text-sm">Read the Research Paper and Lightning Talk notes explaining the methodology.</p>
              </div>

              <div className="glass-card p-6 hover:border-[var(--accent)] cursor-pointer transition-colors" onClick={() => setActiveTab('results')}>
                <div className="text-4xl mb-4">📊</div>
                <h3 className="text-xl font-semibold mb-2">Results Visualization</h3>
                <p className="text-zinc-500 text-sm">See the charts proving our AI models (like TCN) outperform standard approaches.</p>
              </div>

              <div className="glass-card p-6 hover:border-[var(--accent)] cursor-pointer transition-colors" onClick={() => setActiveTab('data')}>
                <div className="text-4xl mb-4">📈</div>
                <h3 className="text-xl font-semibold mb-2">Raw Data</h3>
                <p className="text-zinc-500 text-sm">Dive into the actual CSV numbers and statistics generated by the code.</p>
              </div>
            </div>

            <div className="text-center">
              <button
                className="btn-primary text-xl px-8 py-4 shadow-lg shadow-indigo-500/20"
                onClick={() => setActiveTab('overview')}
              >
                Start Exploring Dashboard
              </button>
            </div>
          </div>
        )}

        {!loading && activeTab === 'overview' && (
          <div className="animate-fadeIn">
            <h1 className="text-3xl font-bold gradient-text mb-2">Welcome to the NASA Data Dashboard</h1>
            <p className="text-zinc-400 mb-8">
              Explore C-MAPSS turbofan engine degradation data, assessment deliverables, and analysis results.
            </p>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4 mb-8">
              {stats.map((stat, i) => (
                <div key={i} className="glass-card stat-card">
                  <div className="text-3xl mb-2">{stat.icon}</div>
                  <div className="stat-value">{stat.value}</div>
                  <div className="stat-label">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* Quick Actions */}
            <h2 className="text-xl font-semibold mb-4 text-zinc-200">Quick Actions</h2>
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div
                className="glass-card p-6 cursor-pointer"
                onClick={() => deliverables[0] && loadDocument(deliverables[0])}
              >
                <h3 className="font-semibold mb-2">📝 View Lightning Talk</h3>
                <p className="text-sm text-zinc-400">Read the S1 assessment deliverable</p>
              </div>
              <div
                className="glass-card p-6 cursor-pointer"
                onClick={() => deliverables[1] && loadDocument(deliverables[1])}
              >
                <h3 className="font-semibold mb-2">📚 View Research Paper</h3>
                <p className="text-sm text-zinc-400">Read the S2 assessment deliverable</p>
              </div>
              <div
                className="glass-card p-6 cursor-pointer"
                onClick={() => setActiveTab('results')}
              >
                <h3 className="font-semibold mb-2">📊 Explore Results</h3>
                <p className="text-sm text-zinc-400">View analysis plots and data</p>
              </div>
            </div>

            {/* Featured Results Preview */}
            <h2 className="text-xl font-semibold mb-4 text-zinc-200">Featured Results</h2>
            <div className="grid grid-cols-2 gap-4">
              {results.images?.slice(0, 4).map((img, i) => (
                <div
                  key={i}
                  className="glass-card image-card"
                  onClick={() => setSelectedImage(img)}
                >
                  <img src={`/api/results/${img}`} alt={img} />
                  <div className="overlay">
                    <p className="text-sm">{img.replace(/_/g, ' ').replace('.png', '')}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && activeTab === 'documents' && (
          <div className="animate-fadeIn">
            <h1 className="text-3xl font-bold gradient-text mb-2">Assessment Deliverables</h1>
            <p className="text-zinc-400 mb-6">View formatted markdown documents</p>

            {selectedDocument && documentContent ? (
              <div className="glass-card p-8">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold">
                    {selectedDocument.replace('.md', '').replace(/_/g, ' ')}
                  </h2>
                  <button
                    className="btn-secondary text-sm"
                    onClick={() => { setSelectedDocument(null); setDocumentContent(''); }}
                  >
                    ← Back to List
                  </button>
                </div>
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {documentContent}
                  </ReactMarkdown>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {deliverables.map((file) => (
                  <div
                    key={file}
                    className="glass-card p-6 cursor-pointer flex items-center justify-between"
                    onClick={() => loadDocument(file)}
                  >
                    <div>
                      <h3 className="font-semibold text-lg">
                        {file.replace('.md', '').replace(/_/g, ' ')}
                      </h3>
                      <p className="text-sm text-zinc-400">Click to view formatted content</p>
                    </div>
                    <span className="text-2xl">📄</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!loading && activeTab === 'results' && (
          <div className="animate-fadeIn">
            <h1 className="text-3xl font-bold gradient-text mb-2">Results Hub</h1>
            <p className="text-zinc-400 mb-6">Explore analysis plots and visualizations</p>

            {selectedImage && (
              <div
                className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-8"
                onClick={() => setSelectedImage(null)}
              >
                <div className="max-w-6xl max-h-full" onClick={(e) => e.stopPropagation()}>
                  <img
                    src={`/api/results/${selectedImage}`}
                    alt={selectedImage}
                    className="max-w-full max-h-[80vh] object-contain rounded-lg"
                  />
                  <p className="text-center mt-4 text-lg">
                    {selectedImage.replace(/_/g, ' ').replace('.png', '')}
                  </p>
                  <button
                    className="btn-primary mt-4 w-full"
                    onClick={() => setSelectedImage(null)}
                  >
                    Close
                  </button>
                </div>
              </div>
            )}

            {Object.entries(imagesByDataset).map(([dataset, images]) => (
              <div key={dataset} className="mb-8">
                <h2 className="text-xl font-semibold mb-4 text-zinc-200">
                  {dataset === 'all_datasets' ? '📊 Cross-Dataset Analysis' : `🛩️ ${dataset}`}
                </h2>
                <div className="image-grid">
                  {images.map((img) => (
                    <div
                      key={img}
                      className="glass-card image-card"
                      onClick={() => setSelectedImage(img)}
                    >
                      <img src={`/api/results/${img}`} alt={img} />
                      <div className="overlay">
                        <p className="text-sm">{img.replace(/_/g, ' ').replace('.png', '').replace(dataset + ' ', '')}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && activeTab === 'data' && (
          <div className="animate-fadeIn">
            <h1 className="text-3xl font-bold gradient-text mb-2">Data Explorer</h1>
            <p className="text-zinc-400 mb-6">Explore CSV result files</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {results.csvs?.map((csv) => (
                <div
                  key={csv}
                  className={`glass-card p-4 cursor-pointer ${csvData?.filename === csv ? 'pulse-glow' : ''}`}
                  onClick={() => loadCsv(csv)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">📈</span>
                    <div>
                      <h4 className="font-medium">{csv.replace('.csv', '').replace(/_/g, ' ')}</h4>
                      <p className="text-xs text-zinc-500">Click to load</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {csvData && (
              <div className="glass-card p-6">
                <h3 className="text-xl font-semibold mb-4">
                  {csvData.filename.replace('.csv', '').replace(/_/g, ' ')}
                </h3>
                <div className="overflow-x-auto">
                  {(() => {
                    const { headers, rows } = parseCSV(csvData.content);
                    return (
                      <table className="w-full">
                        <thead>
                          <tr>
                            {headers.map((h, i) => (
                              <th key={i} className="text-left p-3 bg-zinc-800/50 border border-zinc-700">
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row, i) => (
                            <tr key={i} className="hover:bg-zinc-800/30">
                              {row.map((cell, j) => (
                                <td key={j} className="p-3 border border-zinc-700/50">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    );
                  })()}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
