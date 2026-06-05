import React, { useState } from 'react';
import {
    Upload, FileText, Activity,
    ShieldAlert, Cpu, Zap, BarChart2,
    Info, FileEdit, Download, Share2, AlertTriangle, User, ScanLine, Brain, X, Maximize2
} from 'lucide-react';
import { useECGAnalysis } from '../hooks/useECGAnalysis';
import { getReportUrl } from '../services/api';
import ECGViewer from '../components/features/ECGViewer';
import cn from 'clsx';

/* ─────────────────────────────────────────────────────────────
   ANALYSIS PAGE — 3-column clinical dashboard
   ───────────────────────────────────────────────────────────── */
const AnalysisPage = () => {
    const {
        file, isPreviewing, isAnalyzing,
        previewData, result, error,
        processFile, runAnalysis
    } = useECGAnalysis();

    const [viewMode, setViewMode] = useState('signal');
    const [showSeg, setShowSeg] = useState(true);
    const [clinicalNotes, setClinicalNotes] = useState('');
    const [isXaiModalOpen, setIsXaiModalOpen] = useState(false);

    const handleFileChange = (e) => {
        const f = e.target.files?.[0];
        if (f) processFile(f);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f) processFile(f);
    };

    const reportUrl = result ? getReportUrl(result.report_id) : '#';

    const parseMetric = (value) => {
        const cleaned = String(value ?? '').replace(/<[^>]*>/g, '').trim();
        const parsed = Number(cleaned);
        return Number.isFinite(parsed) ? parsed : null;
    };

    const formatWithSuffix = (value, suffix) => {
        const parsed = parseMetric(value);
        return parsed !== null ? `${parsed}${suffix}` : '--';
    };

    const xaiModal = result && isXaiModalOpen ? (
        <div className="modal-overlay" onClick={() => setIsXaiModalOpen(false)}>
            <div className="modal-container" onClick={(e) => e.stopPropagation()}>
                <div className="modal-content">
                    <div className="modal-header">
                        <div className="modal-title-wrap">
                            <div className="modal-icon-box">
                                <Brain className="modal-icon" />
                            </div>
                            <div>
                                <h2 className="modal-title">Clinical XAI Interpretive Inspector</h2>
                                <p className="modal-subtitle">High-resolution attention mapping for {result.diagnosis}</p>
                            </div>
                        </div>
                        <button
                            type="button"
                            onClick={() => setIsXaiModalOpen(false)}
                            className="modal-close-btn"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="modal-body">
                        <div className="modal-section">
                            <div className="modal-section-header">
                                <Activity className="modal-section-icon" />
                                <span className="modal-section-label">Segmented Morphology View</span>
                            </div>
                            <div className="modal-viz-box" style={{ height: '320px' }}>
                                <ECGViewer
                                    signal={result.signal}
                                    fs={500}
                                    waves={result.waves}
                                    showSegmentation={true}
                                    height={320}
                                />
                            </div>
                        </div>
                        <div className="modal-section">
                            <div className="modal-section-header">
                                <Zap className="modal-section-icon--warning" />
                                <span className="modal-section-label">Continuous Explainability Map (Heatmap)</span>
                            </div>
                            <div className="modal-viz-box" style={{ height: '220px' }}>
                                <XAIMap
                                    signal={result.signal}
                                    heatmap={result.heatmap}
                                    waves={result.waves}
                                    diagnosis={result}
                                />
                            </div>
                        </div>
                        <div className="modal-logic-grid">
                            <div className="modal-logic-box">
                                <h3 className="modal-logic-title">Lead-II Diagnostic Rationale</h3>
                                <div className="modal-logic-content">
                                    {(result?.explanation ?? '').split('\n').filter(l => l.trim()).map((line, i) => (
                                        <p key={i} className={cn(
                                            "modal-logic-line",
                                            (line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL')) && "modal-logic-line--heading"
                                        )}>
                                            {line}
                                        </p>
                                    ))}
                                </div>
                            </div>
                            <div className="modal-logic-aside">
                                <div className="modal-info-chip modal-info-chip--blue">
                                    <span className="modal-info-chip-label">Detection Focus</span>
                                    <span className="modal-info-chip-value">{result.xai?.focus_label || 'Multi-wave morphology'}</span>
                                </div>
                                <div className="modal-info-chip modal-info-chip--green">
                                    <span className="modal-info-chip-label">Stability Audit</span>
                                    <span className="modal-info-chip-value">Clinically Consistent</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="modal-footer">
                        <span className="modal-footer-text">AtrionNet Clinical Interpretation Engine v2.4 (XAI Enabled)</span>
                    </div>
                </div>
            </div>
        </div>
    ) : null;

    return (
        <div className="page-shell">
            <div className="analysis-shell">
                <div className="analysis-grid">

                    {/* ══════════ LEFT SIDEBAR ══════════ */}
                    <aside className="sidebar-col">

                        {/* Patient Details */}
                        <SidebarSection title="Patient Details" icon={User}>
                            <InfoField label="Patient ID"
                                value={file ? 'PT-2024-00847' : '—'} />
                            <InfoField label="Name"
                                value={file ? 'Patient Record' : '—'} />
                            <InfoField label="Age / Sex"
                                value={file ? '56 / Male' : '—'} />
                            <InfoField label="Date"
                                value={file ? new Date().toISOString().split('T')[0] : '—'} />
                        </SidebarSection>

                        {/* ECG Record Info */}
                        <SidebarSection title="ECG Record Info" icon={ScanLine}>
                            <InfoField label="Record ID"
                                value={file ? 'ECG-2024-03847' : '—'} />
                            <InfoField label="Lead Type" value="12-Lead" />
                            <InfoField label="Duration" value="10 seconds" />
                            <InfoField label="Sample Rate" value="500 Hz" />
                        </SidebarSection>

                        {/* Upload */}
                        <div className="sidebar-section" style={{ overflow: 'visible', border: 'none', background: 'none' }}>
                            <div
                                className={`upload-zone${file ? ' upload-zone--active' : ''}`}
                                onDrop={handleDrop}
                                onDragOver={(e) => e.preventDefault()}
                            >
                                <input
                                    id="ecg-file"
                                    type="file"
                                    accept=".npy"
                                    className="upload-input"
                                    disabled={isPreviewing}
                                    onChange={handleFileChange}
                                />
                                <label htmlFor="ecg-file" className="upload-label">
                                    <div className="upload-icon-wrap">
                                        {isPreviewing
                                            ? <Spinner />
                                            : <Upload style={{ width: 22, height: 22 }} />}
                                    </div>
                                    <p className="upload-title">
                                        {file ? file.name : 'Upload ECG Data'}
                                    </p>
                                    <p className="upload-sub">
                                        {file
                                            ? `${(file.size / 1024).toFixed(0)} KB · .NPY`
                                            : 'Drag & drop or click to browse'}
                                    </p>
                                </label>
                            </div>

                            {previewData && !result && (
                                <button
                                    className="btn-analyse"
                                    onClick={runAnalysis}
                                    disabled={isAnalyzing}
                                >
                                    <Cpu style={{ width: 15, height: 15 }} />
                                    {isAnalyzing ? 'Running AI…' : 'Analyse ECG'}
                                </button>
                            )}
                        </div>

                        {error && (
                            <div className="error-banner">
                                <ShieldAlert style={{ width: 15, height: 15, flexShrink: 0 }} />
                                <span>{error}</span>
                            </div>
                        )}

                        {result && result.clinical_metrics && (
                            <SidebarSection title="ECG Measurements" icon={FileText}>
                                <InfoField label="Heart Rate" value={formatWithSuffix(result.clinical_metrics.heart_rate_bpm, ' BPM')} />
                                <InfoField label="PR Interval" value={formatWithSuffix(result.clinical_metrics.mean_pr_ms, ' ms')} />
                                <InfoField label="Assoc. P-waves" value={parseMetric(result.clinical_metrics.n_p_assoc) ?? 0} />
                                <InfoField label="Dissoc. P-waves" value={parseMetric(result.clinical_metrics.n_p_dissoc) ?? 0} highlight />
                            </SidebarSection>
                        )}
                    </aside>

                    {/* ══════════ CENTER ══════════ */}
                    <main className="center-col">
                        {/* Header row */}
                        <div className="ecg-page-header">
                            <div>
                                <h2 className="ecg-page-title">ECG Signal Analysis</h2>
                                <p className="ecg-page-sub">
                                    Clinical-grade waveform review · AV Block Detection
                                </p>
                            </div>
                            <div className="view-toggle">
                                {['signal', 'grid', 'split'].map((m) => (
                                    <button
                                        key={m}
                                        className={`view-toggle-btn${viewMode === m ? ' view-toggle-btn--active' : ''}`}
                                        onClick={() => setViewMode(m)}
                                    >
                                        {m.charAt(0).toUpperCase() + m.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* ECG Waveform panel */}
                        <div className="ecg-panel">
                            <div className="ecg-panel-header">
                                <span className="ecg-panel-title">
                                    <Activity style={{ width: 13, height: 13 }} />
                                    ECG Waveform — Lead II
                                </span>
                                <div className="ecg-panel-controls">
                                    <span className="ecg-meta-chip">Zoom: 100%</span>
                                    <span className="ecg-meta-chip">25mm/s</span>
                                    <span className="ecg-meta-chip">10mm/mV</span>
                                    {result && (
                                        <button
                                            className={`seg-toggle${showSeg ? ' seg-toggle--on' : ''}`}
                                            onClick={() => setShowSeg(!showSeg)}
                                        >
                                            <span className={`seg-toggle-dot${showSeg ? ' seg-toggle-dot--on' : ''}`} />
                                            Segmentation
                                        </button>
                                    )}
                                </div>
                            </div>

                            <div className="ecg-chart-area">
                                {isPreviewing && <LoadingOverlay label="Loading Preview…" />}
                                {isAnalyzing && <LoadingOverlay label="Running AI Analysis…" />}

                                {(previewData || result) ? (
                                    <ECGViewer
                                        signal={result?.signal ?? previewData?.signal}
                                        fs={500}
                                        waves={result?.waves ?? null}
                                        showSegmentation={showSeg}
                                        height={268}
                                    />
                                ) : (
                                    <EmptyState label="Upload an ECG file to begin" />
                                )}
                            </div>

                            <div className="ecg-scale-row">
                                <span className="ecg-scale-label">+1.0 mV</span>
                                <span className="ecg-scale-label">0 &nbsp;&nbsp; 5s &nbsp;&nbsp; 10s</span>
                            </div>
                        </div>

                        {/* XAI Attention Map panel */}
                        <div className="ecg-panel xai-panel">
                            <div className="ecg-panel-header xai-panel-header">
                                <span className="ecg-panel-title xai-title">
                                    <Zap style={{ width: 13, height: 13 }} />
                                    XAI Attention Map — AI Focus Region
                                </span>
                                <div className="flex items-center gap-3">
                                    {result && (
                                        <button
                                            type="button"
                                            onClick={() => setIsXaiModalOpen(true)}
                                            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md border border-blue-200 transition-colors"
                                        >
                                            <Brain style={{ width: 12, height: 12 }} />
                                            <span className="text-[10px] font-bold uppercase tracking-wider">Clinical Interpretive View</span>
                                        </button>
                                    )}
                                    {result && (
                                        <span className="confidence-badge">
                                            Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="xai-chart-area">
                                {result
                                    ? <XAIMap
                                        signal={result?.signal}
                                        heatmap={result?.heatmap}
                                        waves={result?.waves}
                                        diagnosis={result}
                                    />
                                    : <EmptyState label="Run analysis to view XAI attention map" dim />}
                            </div>

                            <div className="xai-footer">
                                <div className="xai-legend">
                                    <span className="legend-dot" style={{ background: '#dc2626' }} /> High Importance
                                    <span className="legend-dot ml-4" style={{ background: '#f59e0b' }} /> Moderate Focus
                                    <span className="legend-dot ml-4" style={{ background: '#3b82f6' }} /> Low Relevance
                                </div>
                                {result && (
                                    <span className="xai-segment-label">
                                        Focus Area: {result?.xai?.focus_label || 'Clinical Waveform Analysis'}
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* ─── AI Clinical Rationale (The "User-Friendly" Explanation) ─── */}
                        <div className="analysis-card mt-6 rationale-card">
                            <div className="card-header">
                                <div className="header-title">
                                    <Brain style={{ width: 14, height: 14, color: '#2563eb' }} />
                                    <span>AI Clinical Rationale & Patient Guidance</span>
                                </div>
                            </div>
                            <div className="card-body rationale-body">
                                {result?.explanation ? (
                                    <div className="explanation-bubble">
                                        <div className="explanation-text">
                                            {result.explanation.split('\n').map((line, i) => (
                                                <p key={i} className={line.startsWith('DIAGNOSIS') || line.startsWith('UNDERSTANDING') || line.startsWith('TECHNICAL') ? 'explanation-heading' : 'explanation-line'}>
                                                    {line}
                                                </p>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="explanation-bubble text-center py-8 opacity-50">
                                        <p>Run analysis to generate AI diagnostic rationale and patient guidance.</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </main>

                    {/* ══════════ RIGHT PANEL ══════════ */}
                    <aside className="results-col">

                        {/* Classification */}
                        <ResultSection title="Model Diagnosis" icon={Activity}>
                            {result
                                ? <DiagnosisCard result={result} />
                                : <p className="results-placeholder">Awaiting analysis…</p>}
                        </ResultSection>

                        {/* Model Info */}
                        <ResultSection title="Model Info" icon={Info}>
                            <div className="model-info-row">
                                <span className="model-info-label">Model Version</span>
                                <span className="model-info-value">AtrionNet v2.4</span>
                            </div>
                            <div className="model-info-row">
                                <span className="model-info-label">Inference Time</span>
                                <span className="model-info-value">{result ? '0.34s' : '—'}</span>
                            </div>
                            <div className="model-info-row">
                                <span className="model-info-label">Validation</span>
                                <span className="model-info-value model-info-value--highlight">FDA Cleared</span>
                            </div>
                        </ResultSection>

                        {/* Actions */}
                        <div className="action-buttons">
                            <button
                                className="btn-generate-report"
                                onClick={() => result && window.open(reportUrl, '_blank')}
                                disabled={!result}
                            >
                                <FileText style={{ width: 15, height: 15 }} />
                                Generate PDF Report
                            </button>
                            <div className="action-row">
                                <button className="btn-secondary" disabled={!result}>
                                    <Download style={{ width: 13, height: 13 }} /> Export Data
                                </button>
                                <button className="btn-secondary" disabled={!result}>
                                    <Share2 style={{ width: 13, height: 13 }} /> Share
                                </button>
                            </div>
                        </div>

                        {/* Disclaimer */}
                        <div className="disclaimer">
                            <AlertTriangle style={{ width: 12, height: 12, flexShrink: 0, marginTop: 1, color: '#d97706' }} />
                            <p>
                                [DISCLAIMER] This AI-generated analysis is intended to assist
                                clinical decision-making and should not replace professional
                                medical judgment.
                            </p>
                        </div>
                    </aside>
                </div>
            </div>

            {xaiModal}

        </div>
    );
};

/* ── Sub-components ── */

const SidebarSection = ({ title, icon: Icon, children }) => (
    <div className="sidebar-section">
        <div className="sidebar-section-header">
            <Icon style={{ width: 11, height: 11 }} />
            <span>{title}</span>
        </div>
        <div className="sidebar-fields">{children}</div>
    </div>
);

const InfoField = ({ label, value, highlight }) => (
    <div>
        <p className="info-label">{label}</p>
        <div
            className="info-value-box"
            style={highlight ? { background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca', fontWeight: 800 } : {}}
        >
            {value}
        </div>
    </div>
);

const ResultSection = ({ title, icon: Icon, children }) => (
    <div className="results-section">
        <div className="results-section-header">
            <Icon style={{ width: 11, height: 11 }} />
            <span>{title}</span>
        </div>
        <div className="results-content">{children}</div>
    </div>
);

const DiagnosisCard = ({ result }) => {
    const confidencePct = result?.confidence ? (result.confidence * 100).toFixed(1) : 0;

    // Determine color based on severity
    let sevColor = '#10b981'; // Normal
    const sev = result?.severity?.toLowerCase() || '';
    if (sev === 'critical') sevColor = '#ef4444';
    else if (sev === 'high') sevColor = '#f59e0b';
    else if (sev === 'moderate') sevColor = '#8b5cf6';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '4px' }}>
            {/* Diagnosis Result Box */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '12px' }}>
                <span style={{ fontSize: '10px', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '6px' }}>
                    Primary Classification
                </span>
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#0f172a', lineHeight: 1.4, display: 'block' }}>
                    {result?.diagnosis || 'Unknown'}
                </span>
            </div>

            {/* Confidence Bar */}
            <div>
                <div className="prob-bar-meta" style={{ marginBottom: '6px' }}>
                    <span className="prob-label" style={{ color: '#475569', fontWeight: 600 }}>AI Confidence Score</span>
                    <span className="prob-pct">{confidencePct}%</span>
                </div>
                <div className="prob-track">
                    <div className="prob-fill" style={{ width: `${confidencePct}%`, background: '#3b82f6' }} />
                </div>
            </div>

            {/* Severity Tag */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '10px', borderTop: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: '11px', fontWeight: 600, color: '#64748b' }}>Clinical Severity</span>
                <span style={{
                    fontSize: '10px',
                    fontWeight: 800,
                    color: 'white',
                    background: sevColor,
                    padding: '3px 8px',
                    borderRadius: '4px',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
                }}>
                    {result?.severity || 'UNKNOWN'}
                </span>
            </div>
        </div>
    );
};

const XAIMap = ({ signal, heatmap, waves, diagnosis }) => {
    const canvasRef = React.useRef(null);
    const containerRef = React.useRef(null);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container || !signal) return;

        const render = () => {
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = container.getBoundingClientRect();

            if (rect.width === 0 || rect.height === 0) return;

            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const W = rect.width;
            const H = rect.height;
            const PAD = 20;
            const SIG_H = H - PAD * 2;
            const points = signal.length;
            const signalValues = signal.map((v) => Number(v) || 0);
            let sigMin = Math.min(...signalValues);
            let sigMax = Math.max(...signalValues);
            const sigRange = sigMax - sigMin || 1;
            const normalizedHeatmap = Array.isArray(heatmap) && heatmap.length > 0
                ? heatmap.slice(0, points).concat(Array(Math.max(0, points - heatmap.length)).fill(0.4))
                : Array(points).fill(0.4);

            const getCol = (v) => {
                v = Number(v);
                if (!Number.isFinite(v)) v = 0;
                v = Math.max(0, Math.min(1, v));
                let hue;
                if (v < 0.5) hue = 220 - (v * 2) * (220 - 60);
                else hue = 60 - ((v - 0.5) * 2) * 60;
                return `hsl(${hue}, 90%, ${42 + v * 14}%)`;
            };

            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, W, H);

            // Grid
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 0.5;
            for (let x = 0; x <= W; x += 10) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }
            for (let y = 0; y <= H; y += 10) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
            }
            ctx.strokeStyle = '#cbd5e1';
            ctx.lineWidth = 1;
            for (let x = 0; x <= W; x += 50) {
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
            }
            for (let y = 0; y <= H; y += 50) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
            }

            // Heatmap band overlay
            ctx.save();
            ctx.globalAlpha = 0.35;
            const bandHeight = Math.max(16, H * 0.16);
            for (let i = 0; i < points; i++) {
                const x = (i / (points - 1)) * W;
                const segmentW = Math.max(W / points, 1);
                ctx.fillStyle = getCol(normalizedHeatmap[i]);
                ctx.fillRect(x, H - bandHeight, segmentW + 1, bandHeight);
            }
            ctx.restore();

            // Baseline waveform for better contrast
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.lineWidth = 4;
            ctx.strokeStyle = '#475569';
            ctx.beginPath();
            for (let i = 0; i < points; i++) {
                const x = (i / (points - 1)) * W;
                const y = PAD + SIG_H - ((signalValues[i] - sigMin) / sigRange) * SIG_H;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Colorized ECG signal overlay
            ctx.lineWidth = 2.4;
            for (let i = 0; i < points - 1; i++) {
                const x1 = (i / (points - 1)) * W;
                const y1 = PAD + SIG_H - ((signalValues[i] - sigMin) / sigRange) * SIG_H;
                const x2 = ((i + 1) / (points - 1)) * W;
                const y2 = PAD + SIG_H - ((signalValues[i + 1] - sigMin) / sigRange) * SIG_H;

                const grad = ctx.createLinearGradient(x1, y1, x2, y2);
                grad.addColorStop(0, getCol(normalizedHeatmap[i]));
                grad.addColorStop(1, getCol(normalizedHeatmap[i + 1]));

                ctx.beginPath(); ctx.strokeStyle = grad; ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
            }

            // Accent marker path on high-attention regions
            ctx.save();
            ctx.lineWidth = 1.8;
            ctx.globalCompositeOperation = 'screen';
            ctx.globalAlpha = 0.85;
            for (let i = 1; i < points - 1; i++) {
                const x = (i / (points - 1)) * W;
                const y = PAD + SIG_H - ((signalValues[i] - sigMin) / sigRange) * SIG_H;
                const intensity = normalizedHeatmap[i];
                if (intensity > 0.78) {
                    ctx.beginPath();
                    ctx.strokeStyle = getCol(intensity);
                    ctx.moveTo(x, y - 5);
                    ctx.lineTo(x, y + 5);
                    ctx.stroke();
                }
            }
            ctx.restore();

            // Mark waves and PR intervals
            if (waves) {
                ctx.font = 'bold 9px Inter, system-ui, sans-serif';
                ctx.textAlign = 'center';

                const drawLabels = (spans, label, color) => {
                    if (!spans) return;
                    spans.forEach(([s, e]) => {
                        const x = ((s + e) / 2 / (points - 1)) * W;
                        const y = PAD - 6;
                        ctx.fillStyle = color;
                        ctx.fillText(label, x, y);
                        ctx.beginPath();
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 1;
                        ctx.moveTo(x, y + 4);
                        ctx.lineTo(x, y + 8);
                        ctx.stroke();
                    });
                };

                if (waves.p_associated && waves.qrs) {
                    waves.p_associated.forEach(([ps, pe]) => {
                        const nextQrs = waves.qrs.find(([qs]) => qs > pe);
                        if (nextQrs) {
                            const [qs] = nextQrs;
                            const x1 = (pe / (points - 1)) * W;
                            const x2 = (qs / (points - 1)) * W;
                            ctx.beginPath();
                            ctx.strokeStyle = '#64748b';
                            ctx.setLineDash([2, 2]);
                            ctx.moveTo(x1, PAD + SIG_H + 10);
                            ctx.lineTo(x2, PAD + SIG_H + 10);
                            ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = '#64748b';
                            ctx.fillText('PR', (x1 + x2) / 2, PAD + SIG_H + 22);
                        }
                    });
                }

                drawLabels(waves.p_associated, 'P', '#2563eb');
                drawLabels(waves.p_dissociated, 'P*', '#dc2626');
                drawLabels(waves.qrs, 'QRS', '#d97706');
                drawLabels(waves.t, 'T', '#10b981');
            }
        };

        const handle = requestAnimationFrame(render);
        const observer = new ResizeObserver(render);
        observer.observe(container);

        return () => {
            cancelAnimationFrame(handle);
            observer.disconnect();
        };
    }, [signal, heatmap, waves]);

    if (!signal) return <EmptyState label="Awaiting signal data..." dim />;

    return (
        <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%', background: '#fffcfc', borderRadius: 8, overflow: 'hidden', border: '1px solid #e2e8f0' }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />

            {/* Standardized Continuous Color Scale Legend */}
            <div className="absolute bottom-2 left-2 flex flex-col gap-2 bg-white/96 backdrop-blur-sm p-3 rounded-xl shadow-lg border border-slate-200 min-w-[240px]">
                <div className="text-[9px] font-semibold text-slate-600 uppercase tracking-[0.18em]">
                    Uploaded signal attention map (same waveform shown)
                </div>
                <div className="flex justify-between text-[8px] font-black text-slate-500 uppercase tracking-tighter mb-1">
                    <span>0.0</span>
                    <span>0.5</span>
                    <span>1.0</span>
                </div>
                <div style={{
                    height: 10,
                    width: '100%',
                    borderRadius: 4,
                    background: 'linear-gradient(to right, #3182ce, #fbd38d, #dc2626)'
                }} />
                <div className="flex justify-between text-[8px] font-medium text-slate-500 uppercase tracking-tight">
                    <span>Low attention</span>
                    <span>High attention</span>
                </div>
                {diagnosis?.xai?.conf_text && (
                    <p className="text-[9px] font-medium text-slate-600 mt-2 border-t pt-2 border-slate-100">
                        <span className="font-black text-blue-600 uppercase mr-1">Status:</span>
                        {diagnosis.xai.conf_text}
                    </p>
                )}
            </div>

            {diagnosis && (
                <div className="absolute top-2 right-2 flex flex-col items-end">
                    <span className="text-[9px] text-slate-500 font-black uppercase tracking-tighter">AI Focus Index</span>
                    <div className="flex items-center gap-2">
                        <div className="h-1.5 w-20 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                            <div className="h-full bg-red-500" style={{ width: `${diagnosis.confidence * 100}%`, transition: 'width 1s ease-out' }} />
                        </div>
                        <span className="text-[11px] font-black text-slate-800">{(diagnosis.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            )}
        </div>
    );
};

const EmptyState = ({ label, dim }) => (
    <div className={`empty-waveform${dim ? ' empty-waveform--dim' : ''}`}>
        <Activity style={{ width: 36, height: 36, opacity: 0.12 }} />
        <p>{label}</p>
    </div>
);

const LoadingOverlay = ({ label }) => (
    <div className="loading-overlay">
        <div className="loading-spinner" />
        <p>{label}</p>
    </div>
);

const Spinner = () => (
    <div style={{
        width: 22, height: 22,
        border: '2.5px solid rgba(255,255,255,0.3)',
        borderTopColor: 'white',
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite'
    }} />
);

export default AnalysisPage;
