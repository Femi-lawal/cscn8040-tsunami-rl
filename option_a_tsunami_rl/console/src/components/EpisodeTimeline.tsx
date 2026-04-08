"use client";

import { useMemo, useState } from "react";
import type { EpisodeFrame } from "@/lib/types";
import { ALERT_COLORS, ACTION_ICONS } from "@/lib/types";

interface EpisodeTimelineProps {
  allFrames: EpisodeFrame[];
  currentFrameIndex: number;
}

export default function EpisodeTimeline({
  allFrames,
  currentFrameIndex,
}: EpisodeTimelineProps) {
  const [modalFrame, setModalFrame] = useState<EpisodeFrame | null>(null);

  const visibleFrames = useMemo(
    () => allFrames.slice(0, currentFrameIndex + 1),
    [allFrames, currentFrameIndex]
  );

  if (visibleFrames.length === 0) {
    return (
      <div className="timeline-panel" data-testid="timeline-panel-empty">
        <div className="timeline-empty" data-testid="timeline-empty">Waiting for episode data…</div>
        <style jsx>{`
          .timeline-panel {
            background: var(--bg-panel);
            border-top: 1px solid var(--border-color);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            min-height: 90px;
          }
          .timeline-empty {
            color: var(--text-muted);
            font-size: 12px;
          }
        `}</style>
      </div>
    );
  }

  return (
    <div className="timeline-panel" data-testid="timeline-panel">
      <div className="timeline-header">
        <h4 className="timeline-title">EPISODE TIMELINE</h4>
        <span className="timeline-progress">
          {currentFrameIndex + 1} / {allFrames.length} steps
        </span>
      </div>

      <div className="timeline-track">
        {/* Progress bar with dots */}
        <div className="track-rail">
          <div className="track-bar">
            <div
              className="track-fill"
              style={{
                width: `${((currentFrameIndex + 1) / allFrames.length) * 100}%`,
              }}
            />
          </div>
          {allFrames.map((f, i) => {
            const pct = ((i + 0.5) / allFrames.length) * 100;
            const isVisible = i <= currentFrameIndex;
            const isCurrent = i === currentFrameIndex;
            const alertColor = ALERT_COLORS[f.alert_level] || "#6b7280";
            return (
              <div
                key={i}
                className={`track-node ${isVisible ? "visible" : ""} ${isCurrent ? "current" : ""}`}
                style={{ left: `${pct}%` }}
              >
                <div
                  className="node-dot"
                  style={{
                    background: isVisible ? alertColor : "var(--bg-secondary)",
                    borderColor: isVisible ? alertColor : "var(--border-color)",
                  }}
                />
                <span className="node-label">T+{f.time_min}</span>
              </div>
            );
          })}
        </div>

        {/* Event cards — grid matches dot positions */}
        <div
          className="event-cards"
          data-testid="event-cards"
          style={{ gridTemplateColumns: `repeat(${allFrames.length}, 1fr)` }}
        >
          {allFrames.map((f, i) => {
            const isVisible = i <= currentFrameIndex;
            return isVisible ? (
              <TimelineCard
                key={i}
                frame={f}
                isCurrent={i === currentFrameIndex}
                onClick={() => setModalFrame(f)}
              />
            ) : (
              <div key={i} />
            );
          })}
        </div>
      </div>

      {modalFrame && (
        <FrameDetailModal frame={modalFrame} onClose={() => setModalFrame(null)} />
      )}

      <style jsx>{`
        .timeline-panel {
          background: var(--bg-panel);
          border-top: 1px solid var(--border-color);
          padding: 10px 16px;
          min-height: 90px;
          max-height: 200px;
          overflow: hidden;
        }
        .timeline-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }
        .timeline-title {
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 1.5px;
          color: var(--text-muted);
          margin: 0;
        }
        .timeline-progress {
          font-size: 11px;
          color: var(--text-secondary);
        }
        .timeline-track {
          position: relative;
        }
        .track-rail {
          position: relative;
          height: 32px;
          margin-bottom: 6px;
          padding-top: 10px;
        }
        .track-bar {
          position: absolute;
          left: 0;
          right: 0;
          top: 10px;
          height: 4px;
          background: var(--bg-secondary);
          border-radius: 2px;
        }
        .track-fill {
          height: 100%;
          background: linear-gradient(90deg, #06b6d4, #3b82f6);
          border-radius: 2px;
          transition: width 0.3s;
        }
        .track-node {
          position: absolute;
          top: 10px;
          transform: translate(-50%, -50%);
          display: flex;
          flex-direction: column;
          align-items: center;
          z-index: 1;
        }
        .node-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          border: 2px solid;
          transition: all 0.3s;
        }
        .track-node.current .node-dot {
          width: 14px;
          height: 14px;
          box-shadow: 0 0 8px rgba(6, 182, 212, 0.6);
        }
        .node-label {
          font-size: 8px;
          color: var(--text-muted);
          margin-top: 4px;
          white-space: nowrap;
        }
        .event-cards {
          display: grid;
          gap: 4px;
          padding: 2px 0;
        }
      `}</style>
    </div>
  );
}

function TimelineCard({
  frame,
  isCurrent,
  onClick,
}: {
  frame: EpisodeFrame;
  isCurrent: boolean;
  onClick: () => void;
}) {
  const alertColor = ALERT_COLORS[frame.alert_level] || "#6b7280";

  return (
    <div
      className={`tcard ${isCurrent ? "current" : ""}`}
      data-testid="timeline-card"
      style={{ borderTopColor: alertColor }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onClick(); }}
    >
      <div className="tcard-time">T+{frame.time_min}m</div>
      <div className="tcard-action">
        {ACTION_ICONS[frame.action]} {frame.action}
      </div>
      <div className="tcard-alert" style={{ color: alertColor }}>
        {frame.alert_level}
      </div>
      <div className={`tcard-reward ${frame.reward >= 0 ? "pos" : "neg"}`}>
        {frame.terminal_bonus != null ? "★ " : ""}
        {frame.reward >= 0 ? "+" : ""}
        {frame.reward.toFixed(1)}
      </div>

      <style jsx>{`
        .tcard {
          padding: 6px 8px;
          background: var(--bg-card);
          border: 1px solid var(--border-color);
          border-top: 3px solid;
          border-radius: 6px;
          font-size: 10px;
          overflow: hidden;
          transition: all 0.2s;
          cursor: pointer;
          user-select: none;
        }
        .tcard:hover {
          background: #253245;
          border-color: rgba(6, 182, 212, 0.3);
        }
        .tcard.current {
          box-shadow: 0 0 8px rgba(6, 182, 212, 0.3);
          border-color: rgba(6, 182, 212, 0.4);
        }
        .tcard-time {
          color: var(--text-muted);
          font-weight: 600;
          margin-bottom: 2px;
        }
        .tcard-action {
          color: var(--text-primary);
          font-weight: 600;
          margin-bottom: 2px;
        }
        .tcard-alert {
          font-weight: 700;
          text-transform: uppercase;
          font-size: 9px;
          letter-spacing: 0.5px;
          margin-bottom: 2px;
        }
        .tcard-reward {
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          font-weight: 600;
        }
        .tcard-reward.pos {
          color: #22c55e;
        }
        .tcard-reward.neg {
          color: #ef4444;
        }
      `}</style>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Frame Detail Modal                                                  */
/* ------------------------------------------------------------------ */

function FrameDetailModal({
  frame,
  onClose,
}: {
  frame: EpisodeFrame;
  onClose: () => void;
}) {
  const alertColor = ALERT_COLORS[frame.alert_level] || "#6b7280";
  const obs = frame.observation;

  // Sort probabilities descending
  const sortedProbs = Object.entries(frame.agent_probabilities)
    .sort(([, a], [, b]) => b - a);

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="frame-detail-modal">
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-row">
            <h3 className="modal-title">
              Step {frame.t + 1} — T+{frame.time_min}min
            </h3>
            <span className="modal-badge" style={{ background: alertColor }}>
              {frame.alert_level.toUpperCase()}
            </span>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close modal">✕</button>
        </div>

        {/* Summary */}
        <div className="modal-summary">{frame.state_summary}</div>

        <div className="modal-grid">
          {/* Column 1: Action & Reward */}
          <div className="modal-section">
            <h4 className="section-title">RL AGENT ACTION</h4>
            <div className="detail-rows">
              <DetailRow label="Chosen Action" value={`${ACTION_ICONS[frame.action] || ""} ${frame.action}`} />
              {frame.value_estimate != null && (
                <DetailRow label="V(s) Expected Return" value={frame.value_estimate.toFixed(3)} color="#06b6d4" />
              )}
            </div>

            <h4 className="section-title" style={{ marginTop: 12 }}>RULE BASELINE</h4>
            <div className="detail-rows">
              <DetailRow label="Rule Recommendation" value={`${ACTION_ICONS[frame.rule_recommendation] || ""} ${frame.rule_recommendation}`} />
            </div>

            <h4 className="section-title" style={{ marginTop: 12 }}>ENVIRONMENT REWARD</h4>
            <div className="detail-rows">
              <DetailRow label="Step Reward" value={frame.step_reward.toFixed(3)} color={frame.step_reward >= 0 ? "#22c55e" : "#ef4444"} />
              {frame.terminal_bonus != null && (
                <DetailRow label="Terminal Bonus" value={frame.terminal_bonus >= 0 ? `+${frame.terminal_bonus.toFixed(3)}` : frame.terminal_bonus.toFixed(3)} color={frame.terminal_bonus >= 0 ? "#22c55e" : "#ef4444"} />
              )}
              <DetailRow label="Total Reward" value={frame.reward.toFixed(3)} color={frame.reward >= 0 ? "#22c55e" : "#ef4444"} />
              <DetailRow label="Cumulative Return" value={frame.cumulative_reward.toFixed(3)} color={frame.cumulative_reward >= 0 ? "#22c55e" : "#ef4444"} />
            </div>

            <h4 className="section-title" style={{ marginTop: 12 }}>EPISODE</h4>
            <div className="detail-rows">
              <DetailRow label="Danger Tier" value={`${frame.danger_tier} — ${frame.danger_label}`} />
              <DetailRow label="Done" value={frame.done ? "Yes" : "No"} />
              {frame.missed_severe && <DetailRow label="Missed Severe" value="YES" color="#ef4444" />}
              {frame.false_warning && <DetailRow label="False Warning" value="YES" color="#f59e0b" />}
            </div>

            <h4 className="section-title" style={{ marginTop: 12 }}>VALID ACTIONS</h4>
            <div className="action-mask-row">
              {frame.action_mask.map((m, i) => {
                const names = ["hold", "escalate", "deescalate", "issue_watch", "issue_warning", "cancel"];
                return (
                  <span
                    key={i}
                    className={`mask-chip ${m > 0 ? "valid" : "blocked"} ${names[i] === frame.action ? "chosen" : ""}`}
                  >
                    {ACTION_ICONS[names[i]] || ""} {names[i]}
                  </span>
                );
              })}
            </div>
          </div>

          {/* Column 2: Observation */}
          <div className="modal-section">
            <h4 className="section-title">OBSERVATION STATE</h4>
            <div className="detail-rows">
              <DetailRow label="Magnitude" value={obs.magnitude_estimate?.toFixed(3)} highlight={obs.magnitude_estimate >= 7.5} />
              <DetailRow label="Depth (km)" value={obs.depth_estimate_km?.toFixed(2)} />
              <DetailRow label="Coastal Index" value={obs.coastal_proximity_index?.toFixed(3)} />
              <DetailRow label="Wave (m)" value={obs.wave_estimate_m?.toFixed(5)} highlight={obs.wave_estimate_m >= 0.10} />
              <DetailRow label="Buoy" value={obs.buoy_confirmation > 0.5 ? "CONFIRMED" : "—"} color={obs.buoy_confirmation > 0.5 ? "#22c55e" : undefined} />
              <DetailRow label="Tide" value={obs.tide_confirmation > 0.5 ? "CONFIRMED" : "—"} color={obs.tide_confirmation > 0.5 ? "#22c55e" : undefined} />
              <DetailRow label="Uncertainty" value={obs.uncertainty?.toFixed(4)} highlight={obs.uncertainty >= 0.6} />
              <DetailRow label="Time Fraction" value={obs.time_fraction?.toFixed(3)} />
              <DetailRow label="Alert Level Norm" value={obs.alert_level_norm?.toFixed(3)} />
              <DetailRow label="Cancel Flag" value={obs.cancel_issued_flag?.toFixed(0)} />
              <DetailRow label="Δ Magnitude" value={obs.delta_magnitude?.toFixed(5)} />
              <DetailRow label="Δ Wave (m)" value={obs.delta_wave_m?.toFixed(5)} />
              <DetailRow label="Δ Uncertainty" value={obs.delta_uncertainty?.toFixed(5)} />
              <DetailRow label="Time Since Buoy" value={obs.time_since_buoy_norm != null && obs.time_since_buoy_norm >= 0 ? obs.time_since_buoy_norm.toFixed(3) : "N/A"} />
              <DetailRow label="Time Since Tide" value={obs.time_since_tide_norm != null && obs.time_since_tide_norm >= 0 ? obs.time_since_tide_norm.toFixed(3) : "N/A"} />
            </div>
          </div>

          {/* Column 3: Policy */}
          <div className="modal-section">
            <h4 className="section-title">POLICY PROBABILITIES</h4>
            <div className="prob-bars">
              {sortedProbs.map(([action, prob]) => (
                <div key={action} className="prob-row">
                  <span className="prob-label">
                    {ACTION_ICONS[action] || ""} {action}
                  </span>
                  <div className="prob-track">
                    <div
                      className="prob-fill"
                      style={{
                        width: `${Math.max(prob * 100, 0.5)}%`,
                        background: action === frame.action ? "#06b6d4" : "#3b82f6",
                      }}
                    />
                  </div>
                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <h4 className="section-title" style={{ marginTop: 12 }}>SENSORS</h4>
            <div className="sensor-list">
              {frame.sensors.map((s) => (
                <div key={s.id} className="sensor-row">
                  <span className={`sensor-dot ${s.status}`} />
                  <span className="sensor-id">{s.id}</span>
                  <span className="sensor-type">{s.type.replace("_", " ")}</span>
                  <span className={`sensor-status ${s.status}`}>{s.status}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(4px);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
        }
        .modal-content {
          background: var(--bg-panel);
          border: 1px solid var(--border-color);
          border-radius: 12px;
          padding: 20px 24px;
          max-width: 900px;
          width: 100%;
          max-height: 85vh;
          overflow-y: auto;
          box-shadow: 0 24px 48px rgba(0, 0, 0, 0.4);
        }
        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 8px;
        }
        .modal-title-row {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .modal-title {
          margin: 0;
          font-size: 16px;
          font-weight: 700;
          color: var(--text-primary);
        }
        .modal-badge {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 1px;
          color: #fff;
        }
        .modal-close {
          background: none;
          border: 1px solid var(--border-color);
          color: var(--text-secondary);
          font-size: 16px;
          cursor: pointer;
          padding: 4px 8px;
          border-radius: 6px;
          line-height: 1;
        }
        .modal-close:hover {
          background: var(--bg-secondary);
          color: var(--text-primary);
        }
        .modal-summary {
          color: var(--text-secondary);
          font-size: 12px;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid var(--border-color);
        }
        .modal-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 16px;
        }
        .modal-section {
          min-width: 0;
        }
        .section-title {
          font-size: 9px;
          font-weight: 700;
          letter-spacing: 1.5px;
          color: var(--text-muted);
          margin: 0 0 8px;
        }
        .detail-rows {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .action-mask-row {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
          margin-top: 4px;
        }
        .mask-chip {
          font-size: 9px;
          padding: 2px 6px;
          border-radius: 4px;
          border: 1px solid var(--border-color);
          color: var(--text-muted);
        }
        .mask-chip.valid {
          color: var(--text-secondary);
          border-color: var(--text-muted);
        }
        .mask-chip.blocked {
          opacity: 0.35;
          text-decoration: line-through;
        }
        .mask-chip.chosen {
          background: rgba(6, 182, 212, 0.15);
          border-color: #06b6d4;
          color: #06b6d4;
          font-weight: 700;
        }
        .prob-bars {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .prob-row {
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .prob-label {
          font-size: 9px;
          color: var(--text-secondary);
          width: 80px;
          flex-shrink: 0;
          text-align: right;
        }
        .prob-track {
          flex: 1;
          height: 6px;
          background: var(--bg-secondary);
          border-radius: 3px;
          overflow: hidden;
        }
        .prob-fill {
          height: 100%;
          border-radius: 3px;
          transition: width 0.3s;
        }
        .prob-value {
          font-size: 9px;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          color: var(--text-muted);
          width: 38px;
          text-align: right;
        }
        .sensor-list {
          display: flex;
          flex-direction: column;
          gap: 3px;
        }
        .sensor-row {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
        }
        .sensor-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
        }
        .sensor-dot.triggered {
          background: var(--sensor-triggered);
        }
        .sensor-dot.monitoring {
          background: var(--sensor-monitoring);
        }
        .sensor-dot.inactive {
          background: var(--sensor-inactive);
        }
        .sensor-id {
          font-weight: 700;
          color: var(--text-primary);
          width: 24px;
        }
        .sensor-type {
          color: var(--text-muted);
          flex: 1;
        }
        .sensor-status {
          font-size: 9px;
          text-transform: uppercase;
        }
        .sensor-status.triggered {
          color: var(--sensor-triggered);
        }
        .sensor-status.monitoring {
          color: var(--sensor-monitoring);
        }
        .sensor-status.inactive {
          color: var(--sensor-inactive);
        }
      `}</style>
    </div>
  );
}

function DetailRow({
  label,
  value,
  color,
  highlight,
}: {
  label: string;
  value?: string | number | null;
  color?: string;
  highlight?: boolean;
}) {
  return (
    <div className="drow">
      <span className="drow-label">{label}</span>
      <span
        className={`drow-value ${highlight ? "highlight" : ""}`}
        style={color ? { color } : undefined}
      >
        {value ?? "—"}
      </span>
      <style jsx>{`
        .drow {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 10px;
          padding: 1px 0;
        }
        .drow-label {
          color: var(--text-muted);
        }
        .drow-value {
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          color: var(--text-primary);
          font-weight: 500;
        }
        .drow-value.highlight {
          color: #f59e0b;
          font-weight: 700;
        }
      `}</style>
    </div>
  );
}
