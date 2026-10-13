import React, { useState } from 'react';
import Header from './components/Header';
import LiveMonitoring from './components/LiveMonitoring';
import CameraControls from './components/CameraControls';
import AlertSystem from './components/AlertSystem';
import DriverStatus from './components/DriverStatus';
import DriverMetrics from './components/DriverMetrics';
import EventLog from './components/EventLog';
import ExplainabilityChart from './components/ExplainabilityChart';
import TripReport from './components/TripReport';
import GradCamView from './components/GradCamView';
import { useCameraStream } from './hooks/useCameraStream';
import './App.css';

export default function App() {
  const {
    streamData,
    devices,
    selectedDevice,
    setSelectedDevice,
    sensitivity,
    updateSensitivity,
    events,
    startCamera,
    stopCamera,
    previewAlert,
    clearLog,
  } = useCameraStream();

  const [activeTab, setActiveTab] = useState('monitoring');
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [resolution, setResolution] = useState('1280x720');

  const activeStage = streamData?.telemetry?.stage ?? 0;
  const driverStatus = streamData?.telemetry?.status_text || 'Normal';
  const attentionScore = streamData?.telemetry?.attention_score ?? 95;
  const voiceMessage = streamData?.telemetry?.voice_message || '';
  const audioStatus = streamData?.telemetry?.audio_status || 'Ready';

  return (
    <div className="drishti-app">
      {/* Top Header with Tab Switcher */}
      <Header
        systemOnline={true}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {/* Main Container by Active Tab */}
      <main className="dashboard-layout">
        {activeTab === 'monitoring' && (
          <>
            {/* Left Column: Live Monitoring + Controls + Alert System */}
            <div className="dashboard-left-col">
              <LiveMonitoring
                streamData={streamData}
                onStartCamera={() => startCamera(selectedDevice)}
                onStopCamera={stopCamera}
                showLandmarks={showLandmarks}
              />

              <div className="bottom-cards-grid">
                <CameraControls
                  devices={devices}
                  selectedDevice={selectedDevice}
                  onSelectDevice={(id) => {
                    setSelectedDevice(id);
                    if (streamData.active) startCamera(id);
                  }}
                  sensitivity={sensitivity}
                  onChangeSensitivity={updateSensitivity}
                  showLandmarks={showLandmarks}
                  onToggleLandmarks={setShowLandmarks}
                  resolution={resolution}
                  onChangeResolution={setResolution}
                />

                <AlertSystem
                  activeStage={activeStage}
                  onPreviewAlert={previewAlert}
                  voiceMessage={voiceMessage}
                  audioStatus={audioStatus}
                />
              </div>
            </div>

            {/* Right Column: Driver Status + Metrics + Event Log */}
            <div className="dashboard-right-col">
              <DriverStatus
                status={driverStatus}
                attentionScore={attentionScore}
              />

              <DriverMetrics
                telemetry={streamData?.telemetry}
              />

              <EventLog
                events={events}
                onClearLog={clearLog}
              />
            </div>
          </>
        )}

        {activeTab === 'explainability' && (
          <div className="dashboard-full-width">
            <ExplainabilityChart telemetry={streamData?.telemetry} />
            <div style={{ marginTop: '20px' }}>
              <LiveMonitoring
                streamData={streamData}
                onStartCamera={() => startCamera(selectedDevice)}
                onStopCamera={stopCamera}
                showLandmarks={showLandmarks}
              />
            </div>
          </div>
        )}

        {activeTab === 'tripreport' && (
          <div className="dashboard-full-width">
            <TripReport />
          </div>
        )}

        {activeTab === 'gradcam' && (
          <div className="dashboard-full-width">
            <GradCamView />
          </div>
        )}
      </main>
    </div>
  );
}
