"use client";

import { useCallback, useEffect, useState } from "react";
import type { CatalogEvent, CheckpointInfo, AgentType, DangerFilter } from "@/lib/types";
import { fetchCatalog, fetchCheckpoints, simulateEpisode } from "@/lib/api";
import type { EpisodeResponse } from "@/lib/types";

interface ControlPanelProps {
    status: string;
    onEpisodeLoaded: (response: EpisodeResponse) => void;
    onLoading: () => void;
    onError: (msg: string) => void;
    onPlay: (speed?: number) => void;
    onPause: () => void;
    onStep: () => void;
    onReset: () => void;
    onSpeedChange: (ms: number) => void;
}

const SPEED_OPTIONS = [
    { label: "0.5×", ms: 2000 },
    { label: "1×", ms: 1000 },
    { label: "2×", ms: 500 },
    { label: "4×", ms: 250 },
    { label: "8×", ms: 125 },
];

export default function ControlPanel({
    status,
    onEpisodeLoaded,
    onLoading,
    onError,
    onPlay,
    onPause,
    onStep,
    onReset,
    onSpeedChange,
}: ControlPanelProps) {
    const [catalog, setCatalog] = useState<CatalogEvent[]>([]);
    const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
    const [dangerFilter, setDangerFilter] = useState<DangerFilter>("All");
    const [selectedEvent, setSelectedEvent] = useState<string>("random");
    const [agentType, setAgentType] = useState<AgentType>("ppo");
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
    const [seed, setSeed] = useState(42);
    const [speedMs, setSpeedMs] = useState(1000);

    useEffect(() => {
        let mounted = true;

        async function loadInitialData() {
            const [catalogResult, checkpointResult] = await Promise.allSettled([
                fetchCatalog(),
                fetchCheckpoints(),
            ]);

            if (!mounted) return;

            const startupErrors: string[] = [];

            if (catalogResult.status === "fulfilled") {
                setCatalog(catalogResult.value);
            } else {
                console.error("Failed to load catalog:", catalogResult.reason);
                startupErrors.push("Failed to load event catalog.");
            }

            if (checkpointResult.status === "fulfilled") {
                setCheckpoints(checkpointResult.value);
                if (checkpointResult.value.length > 0) {
                    setSelectedCheckpoint(checkpointResult.value[0].name);
                }
            } else {
                console.error("Failed to load checkpoints:", checkpointResult.reason);
                startupErrors.push("Failed to load available model checkpoints.");
            }

            if (startupErrors.length > 0) {
                onError(startupErrors.join(" "));
            }
        }

        void loadInitialData();

        return () => {
            mounted = false;
        };
    }, [onError]);

    const filteredCatalog =
        dangerFilter === "All"
            ? catalog
            : catalog.filter((e) => e.danger_label === dangerFilter);

    useEffect(() => {
        if (selectedEvent === "random") return;
        const selectedStillVisible = filteredCatalog.some(
            (event) => event.event_group_id === selectedEvent,
        );
        if (!selectedStillVisible) {
            setSelectedEvent("random");
        }
    }, [filteredCatalog, selectedEvent]);

    const handleStart = useCallback(async () => {
        onLoading();
        try {
            const response = await simulateEpisode({
                event_group_id: selectedEvent === "random" ? null : selectedEvent,
                danger_filter: dangerFilter,
                agent_type: agentType,
                checkpoint_name: agentType === "ppo" ? selectedCheckpoint || null : null,
                seed,
            });
            onEpisodeLoaded(response);
        } catch (e: unknown) {
            onError(e instanceof Error ? e.message : "Simulation failed");
        }
    }, [selectedEvent, dangerFilter, agentType, selectedCheckpoint, seed, onEpisodeLoaded, onLoading, onError]);

    const handleSpeedChange = (ms: number) => {
        setSpeedMs(ms);
        onSpeedChange(ms);
    };

    const isActive = status === "playing" || status === "paused" || status === "done";

    return (
        <aside className="control-panel" data-testid="control-panel">
            <div className="panel-header">
                <div className="panel-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="#06b6d4" strokeWidth="2" />
                        <circle cx="12" cy="12" r="4" fill="#06b6d4" />
                        <path d="M12 2C12 2 14 8 18 12C14 16 12 22 12 22" stroke="#06b6d4" strokeWidth="1.5" opacity="0.5" />
                        <path d="M12 2C12 2 10 8 6 12C10 16 12 22 12 22" stroke="#06b6d4" strokeWidth="1.5" opacity="0.5" />
                    </svg>
                    <span>TSUNAMI RL CONSOLE</span>
                </div>
            </div>

            <div className="panel-section">
                <h3 className="section-title">EPISODE SETUP</h3>

                <label className="field-label">Threat Filter</label>
                <select
                    value={dangerFilter}
                    onChange={(e) => setDangerFilter(e.target.value as DangerFilter)}
                    className="field-select"
                    data-testid="danger-filter-select"
                >
                    <option value="All">All Events</option>
                    <option value="No Threat">No Threat</option>
                    <option value="Potential Threat">Potential Threat</option>
                    <option value="Confirmed Threat">Confirmed Threat</option>
                </select>

                <label className="field-label">Scenario</label>
                <select
                    value={selectedEvent}
                    onChange={(e) => setSelectedEvent(e.target.value)}
                    className="field-select"
                    data-testid="scenario-select"
                >
                    <option value="random">🎲 Random from catalog</option>
                    {filteredCatalog.map((ev) => (
                        <option key={ev.event_group_id} value={ev.event_group_id}>
                            {ev.event_group_id} — M{ev.max_magnitude?.toFixed(1) ?? "?"}{" "}
                            {ev.danger_label === "Confirmed Threat" ? "🔴" : ev.danger_label === "Potential Threat" ? "🟡" : "🟢"}
                        </option>
                    ))}
                </select>

                <label className="field-label">Agent Type</label>
                <select
                    value={agentType}
                    onChange={(e) => setAgentType(e.target.value as AgentType)}
                    className="field-select"
                    data-testid="agent-type-select"
                >
                    <option value="ppo">PPO-LSTM (Deep RL)</option>
                    <option value="rule">Rule-Based</option>
                </select>

                {agentType === "ppo" && checkpoints.length > 0 && (
                    <>
                        <label className="field-label">Checkpoint</label>
                        <select
                            value={selectedCheckpoint}
                            onChange={(e) => setSelectedCheckpoint(e.target.value)}
                            className="field-select"
                            data-testid="checkpoint-select"
                        >
                            {checkpoints.map((cp) => (
                                <option key={cp.name} value={cp.name}>
                                    {cp.label}
                                </option>
                            ))}
                        </select>
                    </>
                )}

                <label className="field-label">Seed</label>
                <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                    className="field-input"
                    data-testid="seed-input"
                />
            </div>

            <div className="panel-section">
                <h3 className="section-title">PLAYBACK</h3>

                <label className="field-label">Speed</label>
                <div className="speed-buttons">
                    {SPEED_OPTIONS.map((opt) => (
                        <button
                            key={opt.ms}
                            onClick={() => handleSpeedChange(opt.ms)}
                            className={`speed-btn ${speedMs === opt.ms ? "active" : ""}`}
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="panel-section control-buttons">
                <button
                    onClick={handleStart}
                    disabled={status === "loading"}
                    className="btn btn-primary"
                    data-testid="start-btn"
                >
                    {status === "loading" ? (
                        <span className="loading-spinner" />
                    ) : (
                        <>▶ Start Episode</>
                    )}
                </button>

                <div className="btn-row">
                    <button
                        onClick={() => onPlay(speedMs)}
                        disabled={!isActive || status === "playing"}
                        className="btn btn-secondary"
                        data-testid="play-btn"
                    >
                        ▶ Play
                    </button>
                    <button
                        onClick={onPause}
                        disabled={status !== "playing"}
                        className="btn btn-secondary"
                        data-testid="pause-btn"
                    >
                        ⏸ Pause
                    </button>
                </div>
                <div className="btn-row">
                    <button
                        onClick={onStep}
                        disabled={!isActive || status === "playing"}
                        className="btn btn-secondary"
                        data-testid="step-btn"
                    >
                        ⏭ Step
                    </button>
                    <button onClick={onReset} disabled={!isActive} className="btn btn-secondary" data-testid="reset-btn">
                        ↺ Reset
                    </button>
                </div>
            </div>

            <style jsx>{`
        .control-panel {
          width: 340px;
          min-width: 340px;
          height: 100vh;
          background: var(--bg-panel);
          border-right: 1px solid var(--border-color);
          display: flex;
          flex-direction: column;
          overflow-y: auto;
        }
        .panel-header {
          padding: 16px 20px;
          border-bottom: 1px solid var(--border-color);
        }
        .panel-logo {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 1.5px;
          color: #06b6d4;
        }
        .panel-section {
          padding: 16px 20px;
          border-bottom: 1px solid var(--border-color);
        }
        .section-title {
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 1.5px;
          color: var(--text-muted);
          margin: 0 0 12px 0;
          text-transform: uppercase;
        }
        .field-label {
          display: block;
          font-size: 11px;
          color: var(--text-secondary);
          margin-bottom: 4px;
          margin-top: 10px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .field-label:first-of-type {
          margin-top: 0;
        }
        .field-select,
        .field-input {
          width: 100%;
          padding: 8px 10px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 6px;
          color: var(--text-primary);
          font-size: 13px;
          outline: none;
          transition: border-color 0.2s;
        }
        .field-select:focus,
        .field-input:focus {
          border-color: var(--accent-cyan);
        }
        .field-select option {
          background: var(--bg-secondary);
          color: var(--text-primary);
        }
        .speed-buttons {
          display: flex;
          gap: 4px;
        }
        .speed-btn {
          flex: 1;
          padding: 6px 0;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          color: var(--text-secondary);
          font-size: 12px;
          cursor: pointer;
          transition: all 0.15s;
        }
        .speed-btn:hover {
          border-color: var(--accent-cyan);
          color: var(--text-primary);
        }
        .speed-btn.active {
          background: rgba(6, 182, 212, 0.15);
          border-color: var(--accent-cyan);
          color: var(--accent-cyan);
        }
        .control-buttons {
          padding: 20px;
        }
        .btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 6px;
          padding: 10px 16px;
          border: none;
          border-radius: 8px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.15s;
          width: 100%;
        }
        .btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .btn-primary {
          background: linear-gradient(135deg, #06b6d4, #3b82f6);
          color: white;
          margin-bottom: 10px;
        }
        .btn-primary:hover:not(:disabled) {
          filter: brightness(1.1);
        }
        .btn-secondary {
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          color: var(--text-secondary);
        }
        .btn-secondary:hover:not(:disabled) {
          border-color: var(--accent-cyan);
          color: var(--text-primary);
        }
        .btn-row {
          display: flex;
          gap: 6px;
          margin-bottom: 6px;
        }
        .loading-spinner {
          display: inline-block;
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.6s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
        </aside>
    );
}
