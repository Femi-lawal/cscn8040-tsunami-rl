"use client";

import { useMemo } from "react";
import type { EpisodeFrame } from "@/lib/types";
import { ALERT_COLORS, ACTION_ICONS } from "@/lib/types";

interface TelemetryPanelProps {
    frame: EpisodeFrame | null;
    allFrames: EpisodeFrame[];
    currentFrameIndex: number;
    outcomeSummary: string;
}

export default function TelemetryPanel({
    frame,
    allFrames,
    currentFrameIndex,
    outcomeSummary,
}: TelemetryPanelProps) {
    const visibleFrames = useMemo(
        () => allFrames.slice(0, currentFrameIndex + 1),
        [allFrames, currentFrameIndex]
    );

    if (!frame) {
        return (
            <div className="telemetry-panel">
                <div className="empty-state">
                    <p>Start an episode to see telemetry data</p>
                </div>
                <style jsx>{`
          .telemetry-panel {
            height: 100%;
            background: var(--bg-panel);
            display: flex;
            align-items: center;
            justify-content: center;
          }
          .empty-state {
            color: var(--text-muted);
            font-size: 13px;
          }
        `}</style>
            </div>
        );
    }

    const obs = frame.observation;
    const alertColor = ALERT_COLORS[frame.alert_level] || "#6b7280";
    const showPolicyDiagnostics = frame.value_estimate != null;

    return (
        <div className="telemetry-panel" data-testid="telemetry-panel">
            {/* Top row: Key metrics */}
            <div className="metrics-row" data-testid="metrics-row">
                <MetricCard
                    label="STEP"
                    value={`${frame.t + 1} / 12`}
                    sub={`T+${frame.time_min}min`}
                />
                <MetricCard
                    label="ALERT"
                    value={frame.alert_level.toUpperCase()}
                    color={alertColor}
                />
                <MetricCard
                    label="ACTION"
                    value={`${ACTION_ICONS[frame.action] || ""} ${frame.action}`}
                />
                <MetricCard
                    label={frame.terminal_bonus != null ? "STEP REWARD" : "REWARD"}
                    value={frame.terminal_bonus != null ? frame.step_reward.toFixed(2) : frame.reward.toFixed(2)}
                    color={frame.reward >= 0 ? "#22c55e" : "#ef4444"}
                />
                {frame.terminal_bonus != null && (
                    <MetricCard
                        label="TERMINAL BONUS"
                        value={`${frame.terminal_bonus >= 0 ? "+" : ""}${frame.terminal_bonus.toFixed(2)}`}
                        color={frame.terminal_bonus >= 0 ? "#22c55e" : "#ef4444"}
                    />
                )}
                <MetricCard
                    label="CUMULATIVE"
                    value={frame.cumulative_reward.toFixed(2)}
                    color={frame.cumulative_reward >= 0 ? "#22c55e" : "#ef4444"}
                />
                {frame.value_estimate != null && (
                    <MetricCard
                        label="V(s)"
                        value={frame.value_estimate.toFixed(2)}
                        color="#06b6d4"
                        sub="expected return"
                    />
                )}
            </div>

            {/* Three-column layout */}
            <div className="telemetry-grid" data-testid="telemetry-grid">
                {/* Col 1: Observation */}
                <div className="telemetry-col">
                    <h4 className="col-title">OBSERVATIONS</h4>
                    <div className="obs-grid">
                        <ObsRow label="Magnitude" value={obs.magnitude_estimate?.toFixed(2)} highlight={obs.magnitude_estimate >= 7.5} />
                        <ObsRow label="Depth (km)" value={obs.depth_estimate_km?.toFixed(1)} />
                        <ObsRow label="Coastal Index" value={obs.coastal_proximity_index?.toFixed(2)} />
                        <ObsRow label="Wave (m)" value={obs.wave_estimate_m?.toFixed(4)} highlight={obs.wave_estimate_m >= 0.10} />
                        <ObsRow label="Buoy" value={obs.buoy_confirmation > 0.5 ? "CONFIRMED" : "—"} highlight={obs.buoy_confirmation > 0.5} />
                        <ObsRow label="Tide" value={obs.tide_confirmation > 0.5 ? "CONFIRMED" : "—"} highlight={obs.tide_confirmation > 0.5} />
                        <ObsRow label="Uncertainty" value={obs.uncertainty?.toFixed(3)} warn={obs.uncertainty >= 0.6} />
                        <ObsRow label="Δ Magnitude" value={obs.delta_magnitude?.toFixed(3)} />
                        <ObsRow label="Δ Wave (m)" value={obs.delta_wave_m?.toFixed(4)} />
                        <ObsRow label="Δ Uncertainty" value={obs.delta_uncertainty?.toFixed(3)} />
                    </div>
                </div>

                {/* Col 2: Agent Diagnostics */}
                <div className="telemetry-col">
                    <h4 className="col-title">AGENT DIAGNOSTICS</h4>
                    <div className="diag-section">
                        <span className="diag-label">Rule Recommendation <span className="diag-hint">(baseline policy)</span></span>
                        <span className="diag-value">{ACTION_ICONS[frame.rule_recommendation]} {frame.rule_recommendation}</span>
                    </div>
                    <div className="diag-section">
                        <span className="diag-label">Valid Actions</span>
                        <div className="valid-actions">
                            {frame.valid_actions.map((a) => (
                                <span
                                    key={a}
                                    className={`action-tag ${a === frame.action ? "active" : ""}`}
                                >
                                    {ACTION_ICONS[a]} {a}
                                </span>
                            ))}
                        </div>
                    </div>
                    {showPolicyDiagnostics && (
                        <div className="diag-section" data-testid="policy-probabilities">
                            <span className="diag-label">Policy Probabilities</span>
                            <div className="prob-bars">
                                {Object.entries(frame.agent_probabilities).map(([action, prob]) => (
                                    <div key={action} className="prob-row">
                                        <span className="prob-action">{action}</span>
                                        <div className="prob-bar-track">
                                            <div
                                                className="prob-bar-fill"
                                                style={{
                                                    width: `${(prob * 100).toFixed(0)}%`,
                                                    background: action === frame.action ? "#06b6d4" : "#3b82f6",
                                                }}
                                            />
                                        </div>
                                        <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    <div className="diag-section state-summary">
                        <span className="diag-label">State Summary</span>
                        <span className="diag-value-sm">{frame.state_summary}</span>
                    </div>
                </div>

                {/* Col 3: Charts */}
                <div className="telemetry-col">
                    <h4 className="col-title">EPISODE CHARTS</h4>
                    <MiniChart
                        title="Reward"
                        data={visibleFrames.map((f) => f.reward)}
                        labels={visibleFrames.map((f) => `T+${f.time_min}`)}
                        color="#22c55e"
                        negColor="#ef4444"
                    />
                    <MiniChart
                        title="Cumulative Return"
                        data={visibleFrames.map((f) => f.cumulative_reward)}
                        labels={visibleFrames.map((f) => `T+${f.time_min}`)}
                        color="#06b6d4"
                    />
                    <MiniChart
                        title="Uncertainty"
                        data={visibleFrames.map((f) => f.observation.uncertainty)}
                        labels={visibleFrames.map((f) => `T+${f.time_min}`)}
                        color="#f59e0b"
                    />
                    <MiniChart
                        title="Wave (m)"
                        data={visibleFrames.map((f) => f.observation.wave_estimate_m)}
                        labels={visibleFrames.map((f) => `T+${f.time_min}`)}
                        color="#8b5cf6"
                    />
                </div>
            </div>

            {/* Outcome summary */}
            {frame.done && outcomeSummary && (
                <div className="outcome-banner" data-testid="outcome-banner" style={{
                    borderColor: frame.missed_severe ? "#ef4444" : frame.false_warning ? "#f59e0b" : "#22c55e"
                }}>
                    {outcomeSummary}
                </div>
            )}

            <style jsx>{`
        .telemetry-panel {
          height: 100%;
          background: var(--bg-panel);
          display: flex;
          flex-direction: column;
          overflow-y: auto;
          padding: 12px;
          gap: 12px;
        }
        .metrics-row {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .telemetry-grid {
          display: grid;
          grid-template-columns: 1fr 1.2fr 1fr;
          gap: 12px;
          flex: 1;
          min-height: 0;
          overflow-y: auto;
        }
        .telemetry-col {
          background: var(--bg-card);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 12px;
          overflow-y: auto;
        }
        .col-title {
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 1.5px;
          color: var(--text-muted);
          margin: 0 0 10px 0;
        }
        .obs-grid {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .diag-section {
          margin-bottom: 12px;
        }
        .diag-label {
          display: block;
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.5px;
          text-transform: uppercase;
          margin-bottom: 4px;
        }
        .diag-hint {
          text-transform: none;
          font-weight: 400;
          opacity: 0.7;
          font-size: 9px;
        }
        .diag-value {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
        }
        .diag-value-sm {
          font-size: 12px;
          color: var(--text-secondary);
          line-height: 1.5;
        }
        .valid-actions {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
        .action-tag {
          padding: 3px 8px;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          font-size: 11px;
          color: var(--text-secondary);
        }
        .action-tag.active {
          background: rgba(6, 182, 212, 0.15);
          border-color: #06b6d4;
          color: #06b6d4;
          font-weight: 600;
        }
        .prob-bars {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .prob-row {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .prob-action {
          font-size: 10px;
          color: var(--text-secondary);
          width: 80px;
          text-align: right;
        }
        .prob-bar-track {
          flex: 1;
          height: 6px;
          background: var(--bg-secondary);
          border-radius: 3px;
          overflow: hidden;
        }
        .prob-bar-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.3s;
        }
        .prob-pct {
          font-size: 10px;
          color: var(--text-muted);
          width: 36px;
          text-align: right;
        }
        .state-summary {
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid var(--border-color);
        }
        .outcome-banner {
          padding: 10px 16px;
          border: 2px solid;
          border-radius: 8px;
          background: rgba(0, 0, 0, 0.3);
          font-size: 13px;
          font-weight: 600;
          text-align: center;
          color: var(--text-primary);
        }
      `}</style>
        </div>
    );
}

function MetricCard({
    label,
    value,
    sub,
    color,
}: {
    label: string;
    value: string;
    sub?: string;
    color?: string;
}) {
    return (
        <div className="metric-card">
            <span className="mc-label">{label}</span>
            <span className="mc-value" style={{ color: color || "var(--text-primary)" }}>
                {value}
            </span>
            {sub && <span className="mc-sub">{sub}</span>}
            <style jsx>{`
        .metric-card {
          background: var(--bg-card);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 8px 14px;
          display: flex;
          flex-direction: column;
          min-width: 100px;
        }
        .mc-label {
          font-size: 9px;
          font-weight: 700;
          color: var(--text-muted);
          letter-spacing: 1px;
        }
        .mc-value {
          font-size: 16px;
          font-weight: 700;
          margin-top: 2px;
        }
        .mc-sub {
          font-size: 10px;
          color: var(--text-muted);
        }
      `}</style>
        </div>
    );
}

function ObsRow({
    label,
    value,
    highlight,
    warn,
}: {
    label: string;
    value: string | undefined;
    highlight?: boolean;
    warn?: boolean;
}) {
    const color = highlight ? "#22c55e" : warn ? "#f59e0b" : "var(--text-primary)";
    return (
        <div className="obs-row">
            <span className="obs-label">{label}</span>
            <span className="obs-value" style={{ color }}>
                {value ?? "—"}
            </span>
            <style jsx>{`
        .obs-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 3px 0;
          border-bottom: 1px solid rgba(45, 58, 77, 0.4);
        }
        .obs-label {
          font-size: 11px;
          color: var(--text-secondary);
        }
        .obs-value {
          font-size: 12px;
          font-weight: 600;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
      `}</style>
        </div>
    );
}

function MiniChart({
    title,
    data,
    labels,
    color,
    negColor,
}: {
    title: string;
    data: number[];
    labels: string[];
    color: string;
    negColor?: string;
}) {
    if (data.length === 0) return null;

    const max = Math.max(...data, 0.01);
    const min = Math.min(...data, 0);
    const range = max - min || 1;
    const height = 50;
    const width = 200;

    const points = data.map((v, i) => {
        const x = data.length === 1 ? width / 2 : (i / (data.length - 1)) * width;
        const y = height - ((v - min) / range) * height;
        return `${x},${y}`;
    });

    const zeroY = height - ((0 - min) / range) * height;

    return (
        <div className="mini-chart">
            <span className="chart-title">{title}</span>
            <div className="chart-values">
                <span style={{ color }}>{data[data.length - 1]?.toFixed(3)}</span>
            </div>
            <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg">
                {negColor && <line x1="0" y1={zeroY} x2={width} y2={zeroY} stroke="#2d3a4d" strokeWidth="0.5" />}
                <polyline
                    points={points.join(" ")}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                {data.map((v, i) => {
                    const x = data.length === 1 ? width / 2 : (i / (data.length - 1)) * width;
                    const y = height - ((v - min) / range) * height;
                    return (
                        <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r="3"
                            fill={negColor && v < 0 ? negColor : color}
                        />
                    );
                })}
            </svg>
            <style jsx>{`
        .mini-chart {
          margin-bottom: 12px;
        }
        .chart-title {
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.5px;
          text-transform: uppercase;
        }
        .chart-values {
          font-size: 14px;
          font-weight: 700;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
        .chart-svg {
          width: 100%;
          height: 50px;
          margin-top: 4px;
        }
      `}</style>
        </div>
    );
}
