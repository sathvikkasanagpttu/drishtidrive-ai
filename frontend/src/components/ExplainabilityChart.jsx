import React, { useMemo } from 'react';
import { Activity, Eye, Compass, AlertCircle, Info } from 'lucide-react';
import './ExplainabilityChart.css';

export default function ExplainabilityChart({ telemetry }) {
  const ear = telemetry?.ear ?? null;
  const perclos = telemetry?.perclos ?? null;
  const mar = telemetry?.mar ?? null;
  const pitch = telemetry?.pitch ?? null;
  const yaw = telemetry?.yaw ?? null;
  const riskScore = telemetry?.risk_score ?? 0;
  const stage = telemetry?.stage ?? 0;

  // Compute transparent signal weight contributions (Phase 2 fusion layer)
  const breakdown = useMemo(() => {
    const eyeWeight = (ear !== null && ear < 0.22) ? 40 : (ear !== null && ear < 0.25 ? 20 : 5);
    const perclosWeight = (perclos !== null && perclos > 0.15) ? 25 : (perclos !== null && perclos > 0.08 ? 12 : 2);
    const poseWeight = (pitch !== null && pitch > 18) ? 20 : (yaw !== null && Math.abs(yaw) > 25 ? 15 : 4);
    const yawnWeight = (mar !== null && mar > 0.65) ? 15 : 0;
    const total = eyeWeight + perclosWeight + poseWeight + yawnWeight;

    return {
      eye: eyeWeight,
      perclos: perclosWeight,
      pose: poseWeight,
      yawn: yawnWeight,
      total: Math.min(100, total),
    };
  }, [ear, perclos, mar, pitch, yaw]);

  const diagnosticReason = useMemo(() => {
    if (stage === 0) return 'All signals within safe threshold limits. Attentive monitoring active.';
    const reasons = [];
    if (ear !== null && ear < 0.22) reasons.push(`Sustained eye closure (EAR ${ear.toFixed(2)} < 0.22)`);
    if (perclos !== null && perclos > 0.15) reasons.push(`Elevated PERCLOS (${(perclos * 100).toFixed(0)}% > 15%)`);
    if (pitch !== null && pitch > 18) reasons.push(`Head nodding forward (Pitch +${pitch.toFixed(0)}°)`);
    if (yaw !== null && Math.abs(yaw) > 25) reasons.push(`Looking away from road (Yaw ${yaw.toFixed(0)}°)`);
    if (mar !== null && mar > 0.65) reasons.push(`Mouth yawn detected (MAR ${mar.toFixed(2)} >= 0.65)`);
    return reasons.length > 0 ? reasons.join(' • ') : 'Composite risk index exceeded stage safety margin.';
  }, [stage, ear, perclos, pitch, yaw, mar]);

  return (
    <div className="explainability-card">
      <div className="card-header-row">
        <div className="title-with-icon">
          <Activity className="section-icon" />
          <h2 className="section-title">Alert Explainability & Fusion Diagnostics</h2>
        </div>
        <span className={`stage-pill stage-${stage}`}>
          {stage === 0 ? 'Normal' : `Stage ${stage} Alert Active`}
        </span>
      </div>

      <p className="section-subtitle">
        Transparent multi-signal breakdown showing real-time contributions from eye closure, PERCLOS, head pose, and yawns.
      </p>

      {/* Primary Diagnostic Banner */}
      <div className={`diagnostic-banner stage-${stage}`}>
        <AlertCircle className="diagnostic-icon" />
        <div className="diagnostic-content">
          <div className="diagnostic-title">
            {stage === 0 ? 'Normal Monitoring State' : `Alert Trigger Explanation`}
          </div>
          <div className="diagnostic-text">{diagnosticReason}</div>
        </div>
      </div>

      {/* 4 Multi-Signal Gauge Grid */}
      <div className="signal-gauges-grid">
        {/* EAR Signal */}
        <div className="gauge-box">
          <div className="gauge-label-row">
            <span className="gauge-name"><Eye className="inline-icon" /> EAR (Eye Aspect)</span>
            <span className="gauge-value">{ear !== null ? ear.toFixed(2) : 'No Data'}</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${ear !== null && ear < 0.22 ? 'critical' : 'nominal'}`}
              style={{ width: `${Math.min(100, Math.max(0, ((ear ?? 0.3) / 0.4) * 100))}%` }}
            />
          </div>
          <div className="gauge-footnote">Threshold: &lt; 0.22 (Weight: 40%)</div>
        </div>

        {/* PERCLOS Signal */}
        <div className="gauge-box">
          <div className="gauge-label-row">
            <span className="gauge-name"><Activity className="inline-icon" /> PERCLOS (30s)</span>
            <span className="gauge-value">{perclos !== null ? `${(perclos * 100).toFixed(0)}%` : 'No Data'}</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${perclos !== null && perclos > 0.15 ? 'critical' : 'nominal'}`}
              style={{ width: `${Math.min(100, ((perclos ?? 0) / 0.4) * 100)}%` }}
            />
          </div>
          <div className="gauge-footnote">Threshold: &gt; 15% (Weight: 25%)</div>
        </div>

        {/* Head Nodding Signal */}
        <div className="gauge-box">
          <div className="gauge-label-row">
            <span className="gauge-name"><Compass className="inline-icon" /> Head Nod (Pitch)</span>
            <span className="gauge-value">{pitch !== null ? `+${pitch.toFixed(0)}°` : '0°'}</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${pitch !== null && pitch > 18 ? 'critical' : 'nominal'}`}
              style={{ width: `${Math.min(100, ((pitch ?? 0) / 35) * 100)}%` }}
            />
          </div>
          <div className="gauge-footnote">Threshold: &gt; +18° (Weight: 20%)</div>
        </div>

        {/* Yawn MAR Signal */}
        <div className="gauge-box">
          <div className="gauge-label-row">
            <span className="gauge-name"><Info className="inline-icon" /> Yawn MAR</span>
            <span className="gauge-value">{mar !== null ? mar.toFixed(2) : '0.20'}</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${mar !== null && mar > 0.65 ? 'critical' : 'nominal'}`}
              style={{ width: `${Math.min(100, ((mar ?? 0) / 0.9) * 100)}%` }}
            />
          </div>
          <div className="gauge-footnote">Threshold: &gt; 0.65 (Weight: 15%)</div>
        </div>
      </div>

      {/* Composite Risk Contribution Stack */}
      <div className="fusion-breakdown-panel">
        <h3 className="breakdown-heading">Signal Weight Allocation (Phase 2 Transparent Fusion)</h3>
        <div className="stacked-bar">
          <div className="segment segment-eye" style={{ width: `${breakdown.eye}%` }} title={`Eye Closure: ${breakdown.eye}%`}>
            {breakdown.eye > 10 ? `Eye ${breakdown.eye}%` : ''}
          </div>
          <div className="segment segment-perclos" style={{ width: `${breakdown.perclos}%` }} title={`PERCLOS: ${breakdown.perclos}%`}>
            {breakdown.perclos > 10 ? `PERCLOS ${breakdown.perclos}%` : ''}
          </div>
          <div className="segment segment-pose" style={{ width: `${breakdown.pose}%` }} title={`Head Pose: ${breakdown.pose}%`}>
            {breakdown.pose > 10 ? `Pose ${breakdown.pose}%` : ''}
          </div>
          <div className="segment segment-yawn" style={{ width: `${breakdown.yawn}%` }} title={`Yawn: ${breakdown.yawn}%`}>
            {breakdown.yawn > 5 ? `Yawn` : ''}
          </div>
        </div>
        <div className="legend-row">
          <span className="legend-item"><span className="legend-color color-eye"></span> Eye Closure (40%)</span>
          <span className="legend-item"><span className="legend-color color-perclos"></span> PERCLOS Window (25%)</span>
          <span className="legend-item"><span className="legend-color color-pose"></span> Head Pitch/Yaw (20%)</span>
          <span className="legend-item"><span className="legend-color color-yawn"></span> Yawn Frequency (15%)</span>
        </div>
      </div>
    </div>
  );
}
