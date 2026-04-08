"use client";

import dynamic from "next/dynamic";
import ControlPanel from "@/components/ControlPanel";
import TelemetryPanel from "@/components/TelemetryPanel";
import EpisodeTimeline from "@/components/EpisodeTimeline";
import { useEpisodeStore } from "@/lib/useEpisodeStore";

const SimulationMap = dynamic(() => import("@/components/SimulationMap"), {
    ssr: false,
    loading: () => (
        <div
            style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                background: "var(--bg-secondary)",
                color: "var(--text-muted)",
                fontSize: 13,
            }}
        >
            Loading map…
        </div>
    ),
});

export default function ConsolePage() {
    const {
        state,
        currentFrame,
        loadEpisode,
        setLoading,
        setError,
        stepForward,
        play,
        pause,
        reset,
        setSpeed,
    } = useEpisodeStore();

    return (
        <div className="console-layout" data-testid="console-layout">
            {/* Left sidebar */}
            <ControlPanel
                status={state.status}
                onEpisodeLoaded={loadEpisode}
                onLoading={setLoading}
                onError={setError}
                onPlay={play}
                onPause={pause}
                onStep={stepForward}
                onReset={reset}
                onSpeedChange={setSpeed}
            />

            {/* Right: map + telemetry */}
            <div className="main-area">
                {/* Top: Map */}
                <div className="map-area" data-testid="map-area">
                    <SimulationMap
                        frame={currentFrame}
                        metadata={state.metadata}
                        allFrames={state.frames}
                    />
                </div>

                {/* Bottom: Timeline + Telemetry */}
                <div className="bottom-area">
                    <EpisodeTimeline
                        allFrames={state.frames}
                        currentFrameIndex={state.currentFrameIndex}
                    />
                    <div className="telemetry-area" data-testid="telemetry-area">
                        <TelemetryPanel
                            frame={currentFrame}
                            allFrames={state.frames}
                            currentFrameIndex={state.currentFrameIndex}
                            outcomeSummary={state.outcomeSummary}
                        />
                    </div>
                </div>

                {/* Error overlay */}
                {state.error && (
                    <div className="error-overlay" data-testid="error-overlay">
                        <div className="error-box" data-testid="error-box">
                            <span className="error-icon">⚠</span>
                            <span>{state.error}</span>
                        </div>
                    </div>
                )}

                {/* Idle state */}
                {state.status === "idle" && !state.error && (
                    <div className="idle-overlay" data-testid="idle-overlay">
                        <div className="idle-content">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" opacity="0.3">
                                <circle cx="12" cy="12" r="10" stroke="#06b6d4" strokeWidth="1.5" />
                                <circle cx="12" cy="12" r="4" fill="#06b6d4" opacity="0.5" />
                                <path d="M12 2C12 2 14 8 18 12C14 16 12 22 12 22" stroke="#06b6d4" strokeWidth="1" opacity="0.3" />
                                <path d="M12 2C12 2 10 8 6 12C10 16 12 22 12 22" stroke="#06b6d4" strokeWidth="1" opacity="0.3" />
                            </svg>
                            <p className="idle-title">Tsunami Warning RL Console</p>
                            <p className="idle-subtitle">
                                Configure an episode in the left panel and press Start
                            </p>
                        </div>
                    </div>
                )}
            </div>

            <style jsx>{`
        .console-layout {
          display: flex;
          height: 100vh;
          overflow: hidden;
        }
        .main-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          position: relative;
        }
        .map-area {
          flex: 1;
          min-height: 300px;
          position: relative;
        }
        .bottom-area {
          display: flex;
          flex-direction: column;
          height: 50%;
          min-height: 280px;
          max-height: 55%;
          border-top: 1px solid var(--border-color);
        }
        .telemetry-area {
          flex: 1;
          overflow: hidden;
        }
        .error-overlay {
          position: absolute;
          top: 12px;
          right: 12px;
          z-index: 100;
        }
        .error-box {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid rgba(239, 68, 68, 0.4);
          border-radius: 8px;
          color: #fca5a5;
          font-size: 13px;
        }
        .error-icon {
          font-size: 18px;
        }
        .idle-overlay {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-primary);
          z-index: 50;
        }
        .idle-content {
          text-align: center;
        }
        .idle-title {
          font-size: 18px;
          font-weight: 600;
          color: var(--text-secondary);
          margin: 16px 0 4px;
        }
        .idle-subtitle {
          font-size: 13px;
          color: var(--text-muted);
          margin: 0;
        }
      `}</style>
        </div>
    );
}
