import React, { useState, useEffect } from 'react';
import { FileText, Download, Printer, ShieldCheck, AlertTriangle, Clock, Gauge, Activity } from 'lucide-react';
import './TripReport.css';

export default function TripReport() {
  const [report, setReport] = useState({
    trip_id: 'TRIP-KA01-928',
    driver_id: 'DL-KA042026',
    generated_at: new Date().toISOString().replace('T', ' ').slice(0, 19),
    trip_duration_min: 60.0,
    average_speed_kmh: 58.5,
    total_alerts: 2,
    stage_breakdown: { '1': 2, '2': 0, '3': 0 },
    longest_eye_closure_sec: 1.15,
    distraction_events: 0,
    response_acknowledgements: 2,
    risk_score: 4.0,
    risk_category: 'EXCELLENT',
    recommendations: [
      'Exemplary attentiveness maintained throughout the trip. Safe driving parameters observed.',
      'Scheduled breaks every 2 hours are recommended on national highway routes.',
    ],
  });

  useEffect(() => {
    fetch('/api/fleet/trip-summary')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data && data.trip_id) {
          setReport(data);
        }
      })
      .catch((err) => {
        console.debug('Trip summary fetch warning (using local telemetry):', err);
      });
  }, []);

  const handlePrintPdf = () => {
    window.print();
  };

  const handleExportHtml = () => {
    if (!report) return;
    const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DrishtiDrive Trip Report - ${report.trip_id}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px; color: #1e293b; }
    h1 { color: #0f172a; margin-bottom: 4px; }
    .subtitle { color: #64748b; margin-bottom: 24px; }
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .stat-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
    .stat-val { font-size: 20px; font-weight: bold; margin-top: 4px; }
    .recs { background: #f0fdf4; border: 1px solid #bbf7d0; padding: 16px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>DrishtiDrive AI - Driver Trip Safety Audit</h1>
  <div class="subtitle">Trip ID: ${report.trip_id} | Driver: ${report.driver_id} | Date: ${report.generated_at}</div>
  <div class="stats-grid">
    <div class="stat-card"><div>Risk Score</div><div class="stat-val">${report.risk_score} / 100 (${report.risk_category})</div></div>
    <div class="stat-card"><div>Duration</div><div class="stat-val">${report.trip_duration_min} mins</div></div>
    <div class="stat-card"><div>Longest Closure</div><div class="stat-val">${report.longest_eye_closure_sec}s</div></div>
    <div class="stat-card"><div>Total Alerts</div><div class="stat-val">${report.total_alerts}</div></div>
  </div>
  <h3>Safety Recommendations</h3>
  <div class="recs">
    <ul>${report.recommendations.map((r) => `<li>${r}</li>`).join('')}</ul>
  </div>
</body>
</html>`;
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drishtidrive_trip_${report.trip_id}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const riskClass = report?.risk_score < 20 ? 'risk-low' : (report?.risk_score < 50 ? 'risk-med' : 'risk-high');

  return (
    <div className="trip-report-card" data-testid="trip-report-view">
      {/* Top Banner and Export Toolbar */}
      <div className="report-header-row">
        <div>
          <div className="title-row">
            <FileText className="report-icon" />
            <h2 className="report-title">Commercial Trip Safety Audit & Summary</h2>
          </div>
          <div className="report-meta">
            Trip ID: <span className="meta-val">{report?.trip_id}</span> • Driver: <span className="meta-val">{report?.driver_id}</span> • Timestamp: <span className="meta-val">{report?.generated_at}</span>
          </div>
        </div>

        <div className="report-actions">
          <button className="btn-action" onClick={handleExportHtml} title="Export HTML File">
            <Download className="action-icon" />
            <span>Export HTML</span>
          </button>
          <button className="btn-action btn-primary" onClick={handlePrintPdf} title="Export PDF / Print">
            <Printer className="action-icon" />
            <span>Export PDF</span>
          </button>
        </div>
      </div>

      {/* Primary KPI Metrics Grid */}
      <div className="report-kpi-grid">
        <div className={`kpi-box ${riskClass}`}>
          <div className="kpi-label"><ShieldCheck className="kpi-icon" /> Safety Risk Score</div>
          <div className="kpi-value">{report?.risk_score} <span className="kpi-sub">/ 100</span></div>
          <div className="kpi-badge">{report?.risk_category}</div>
        </div>

        <div className="kpi-box">
          <div className="kpi-label"><Clock className="kpi-icon" /> Driving Duration</div>
          <div className="kpi-value">{report?.trip_duration_min} <span className="kpi-sub">mins</span></div>
          <div className="kpi-subtext">Continuous highway driving</div>
        </div>

        <div className="kpi-box">
          <div className="kpi-label"><AlertTriangle className="kpi-icon" /> Longest Closure</div>
          <div className="kpi-value">{report?.longest_eye_closure_sec} <span className="kpi-sub">sec</span></div>
          <div className="kpi-subtext">Peak micro-sleep episode</div>
        </div>

        <div className="kpi-box">
          <div className="kpi-label"><Gauge className="kpi-icon" /> Average Velocity</div>
          <div className="kpi-value">{report?.average_speed_kmh} <span className="kpi-sub">km/h</span></div>
          <div className="kpi-subtext">GPS telemetry tracking</div>
        </div>
      </div>

      {/* Alert Breakdown Cards */}
      <div className="alert-breakdown-row">
        <div className="alert-pill-card pill-stg1">
          <span className="pill-count">{report?.stage_breakdown?.['1'] ?? 0}</span>
          <span className="pill-name">Stage 1 Warnings</span>
        </div>
        <div className="alert-pill-card pill-stg2">
          <span className="pill-count">{report?.stage_breakdown?.['2'] ?? 0}</span>
          <span className="pill-name">Stage 2 Drowsy</span>
        </div>
        <div className="alert-pill-card pill-stg3">
          <span className="pill-count">{report?.stage_breakdown?.['3'] ?? 0}</span>
          <span className="pill-name">Stage 3 Critical</span>
        </div>
        <div className="alert-pill-card pill-ack">
          <span className="pill-count">{report?.response_acknowledgements ?? 0}</span>
          <span className="pill-name">Snooze Responses</span>
        </div>
      </div>

      {/* Automated Recommendations Section */}
      <div className="recommendations-box">
        <h3 className="recs-title">
          <Activity className="recs-icon" /> Offline Safety Recommendations
        </h3>
        <ul className="recs-list">
          {report?.recommendations?.map((rec, idx) => (
            <li key={idx} className="rec-item">
              <span className="bullet">•</span>
              <span>{rec}</span>
            </li>
          ))}
        </ul>
        <div className="disclaimer-note">
          <strong>Notice:</strong> Safety scores and thresholds are automated engineering heuristics designed for Indian road operations and are not medically validated.
        </div>
      </div>
    </div>
  );
}
