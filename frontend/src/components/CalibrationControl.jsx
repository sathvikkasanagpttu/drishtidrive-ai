import React, { useState, useEffect } from 'react';
import { UserCheck, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';
import './CalibrationControl.css';

/**
 * Driver calibration control component.
 * Allows drivers to calibrate their personal baseline EAR (7s open, 3s closed)
 * to set adaptive eye-closure thresholds.
 */
export default function CalibrationControl() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/calibration/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.warn('Failed to fetch calibration status:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const isCalibrating = status?.state === 'IN_PROGRESS';

  useEffect(() => {
    let interval;
    if (isCalibrating) {
      interval = setInterval(fetchStatus, 400);
    }
    return () => clearInterval(interval);
  }, [isCalibrating]);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ driver_id: 'default_driver', duration_sec: 10.0 }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setStatus((prev) => ({ ...prev, ...data, state: 'IN_PROGRESS' }));
    } catch (err) {
      setError('Could not start calibration. Ensure camera is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      await fetch('/api/calibration/reset', { method: 'POST' });
      await fetchStatus();
    } catch (err) {
      console.warn('Failed to reset calibration:', err);
    } finally {
      setLoading(false);
    }
  };

  const profile = status?.profile;
  const progressPercent = status?.progress_percent || 0;
  const instruction = status?.instruction || 'Calibrate for personalized eye openness baseline.';

  return (
    <div className="calibration-control-box" id="calibration-control">
      <div className="calib-header">
        <div className="calib-title-group">
          <UserCheck size={16} className="text-blue-500" />
          <span className="calib-title">Per-Driver EAR Calibration</span>
        </div>
        {status?.is_calibrated && !isCalibrating && (
          <span className="calib-badge calibrated">
            <CheckCircle2 size={12} /> Calibrated
          </span>
        )}
      </div>

      {error && (
        <div className="calib-error">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {isCalibrating ? (
        <div className="calib-progress-wrap">
          <div className="calib-instruction-box">
            <span className="calib-phase-tag">{status?.phase}</span>
            <span className="calib-instruction">{instruction}</span>
          </div>
          <div className="calib-progress-track">
            <div
              className="calib-progress-fill"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>
          <div className="calib-telemetry-row">
            <span>Progress: {progressPercent}%</span>
            <span>Live EAR: {status?.measured_ear?.toFixed(3) ?? '--'}</span>
          </div>
        </div>
      ) : profile ? (
        <div className="calib-profile-summary">
          <div className="profile-metrics-grid">
            <div className="profile-metric">
              <span className="metric-k">Open Baseline</span>
              <span className="metric-v">{profile.open_ear_baseline?.toFixed(3) ?? '--'}</span>
            </div>
            <div className="profile-metric">
              <span className="metric-k">Closed Baseline</span>
              <span className="metric-v">{profile.closed_ear_baseline?.toFixed(3) ?? '--'}</span>
            </div>
            <div className="profile-metric">
              <span className="metric-k">Close Threshold</span>
              <span className="metric-v highlight">{profile.ear_close_threshold?.toFixed(3) ?? '--'}</span>
            </div>
          </div>
          <div className="calib-actions-row">
            <button
              className="btn-calib-start"
              onClick={handleStart}
              disabled={loading}
              id="recalibrate-btn"
            >
              <RefreshCw size={12} className={loading ? 'spin' : ''} />
              <span>Recalibrate (10s)</span>
            </button>
            <button
              className="btn-calib-reset"
              onClick={handleReset}
              disabled={loading}
              id="reset-calib-btn"
            >
              Reset
            </button>
          </div>
        </div>
      ) : (
        <div className="calib-start-wrap">
          <p className="calib-desc">
            Adapts detection to your eye shape: 7s normal open gaze, then 3s gentle eye closure.
          </p>
          <button
            className="btn-calib-start"
            onClick={handleStart}
            disabled={loading}
            id="start-calibration-btn"
          >
            <UserCheck size={14} />
            <span>{loading ? 'Starting...' : 'Start Driver Calibration (10s)'}</span>
          </button>
        </div>
      )}
    </div>
  );
}
