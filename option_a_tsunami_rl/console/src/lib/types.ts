export interface SensorState {
  id: string;
  type: "buoy" | "tide_gauge";
  status: "triggered" | "monitoring" | "inactive";
  lat: number | null;
  lon: number | null;
}

export interface EpisodeFrame {
  t: number;
  time_min: number;
  epicenter: { lat: number; lon: number };
  wave_radius_km: number;
  sensors: SensorState[];
  observation: Record<string, number>;
  state_summary: string;
  action: string;
  action_index: number;
  reward: number;
  step_reward: number;
  terminal_bonus: number | null;
  cumulative_reward: number;
  alert_level: string;
  alert_level_index: number;
  danger_tier: number;
  danger_label: string;
  valid_actions: string[];
  action_mask: number[];
  agent_probabilities: Record<string, number>;
  value_estimate: number | null;
  rule_recommendation: string;
  done: boolean;
  missed_severe: boolean;
  false_warning: boolean;
  hidden_trace: Record<string, number[]>;
}

export interface EventMetadata {
  event_group_id: string;
  danger_tier: number;
  danger_label: string;
  location_name: string | null;
  origin_time_utc: string | null;
  latitude: number | null;
  longitude: number | null;
  target_alert_level: string;
  bulletin_count: number;
  first_bulletin_delay_min: number;
  final_bulletin_delay_min: number;
  sea_level_confirmed_flag: boolean;
  has_threat_assessment: boolean;
  wave_imputed_flag: boolean;
  observed_max_wave_m: number;
  coastal_proximity_index: number;
}

export interface EpisodeResponse {
  event_metadata: EventMetadata;
  frames: EpisodeFrame[];
  total_return: number;
  outcome_summary: string;
}

export interface CatalogEvent {
  event_group_id: string;
  danger_tier: number;
  danger_label: string;
  location_name: string | null;
  latitude: number | null;
  longitude: number | null;
  initial_magnitude: number | null;
  max_magnitude: number | null;
  initial_depth_km: number | null;
  coastal_proximity_index: number | null;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  label: string;
}

export type AgentType = "ppo" | "rule" | "manual";
export type DangerFilter =
  | "All"
  | "No Threat"
  | "Potential Threat"
  | "Confirmed Threat";

export const ALERT_COLORS: Record<string, string> = {
  monitor: "#6b7280",
  info: "#3b82f6",
  watch: "#f59e0b",
  advisory: "#f97316",
  warning: "#ef4444",
};

export const DANGER_COLORS: Record<number, string> = {
  0: "#22c55e",
  1: "#f59e0b",
  2: "#ef4444",
};

export const ACTION_ICONS: Record<string, string> = {
  hold: "⏸",
  escalate: "⬆",
  deescalate: "⬇",
  issue_watch: "👁",
  issue_warning: "🚨",
  cancel: "✕",
};
