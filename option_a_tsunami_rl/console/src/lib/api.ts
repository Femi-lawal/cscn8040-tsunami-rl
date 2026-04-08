import type {
  EpisodeResponse,
  CatalogEvent,
  CheckpointInfo,
  AgentType,
  DangerFilter,
} from "./types";

const API_BASE = "/api";

const CURATED_CHECKPOINTS = [
  {
    filename: "ppo_lstm_recommended.pt",
    label: "Recommended PPO Policy",
  },
  {
    filename: "ppo_lstm_stable.pt",
    label: "Stable PPO Policy",
  },
  {
    filename: "ppo_lstm_baseline.pt",
    label: "Baseline PPO Policy",
  },
];

type RawCheckpointInfo = {
  name: string;
  path: string;
};

function checkpointFilename(path: string): string {
  const segments = path.split(/[\\/]/);
  return segments[segments.length - 1] || "";
}

function checkpointStem(filename: string): string {
  return filename.replace(/\.pt$/i, "");
}

function curateCheckpoints(rawCheckpoints: RawCheckpointInfo[]): CheckpointInfo[] {
  const curated = CURATED_CHECKPOINTS.flatMap((spec) => {
    const match = rawCheckpoints.find((checkpoint) => {
      const rawName = checkpoint.name.trim();
      return (
        checkpointFilename(checkpoint.path) === spec.filename ||
        rawName === spec.label ||
        rawName === checkpointStem(spec.filename)
      );
    });

    return match
      ? [
          {
            name: match.name,
            path: match.path,
            label: spec.label,
          },
        ]
      : [];
  });

  if (curated.length > 0) {
    return curated;
  }

  return rawCheckpoints.slice(0, 3).map((checkpoint, index) => ({
    name: checkpoint.name,
    path: checkpoint.path,
    label: `PPO Checkpoint ${index + 1}`,
  }));
}

export async function fetchCatalog(): Promise<CatalogEvent[]> {
  const res = await fetch(`${API_BASE}/catalog`);
  if (!res.ok) throw new Error(`Catalog fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchCheckpoints(): Promise<CheckpointInfo[]> {
  const res = await fetch(`${API_BASE}/checkpoints`);
  if (!res.ok) throw new Error(`Checkpoints fetch failed: ${res.statusText}`);
  const raw = (await res.json()) as RawCheckpointInfo[];
  return curateCheckpoints(raw);
}

export async function simulateEpisode(params: {
  event_group_id?: string | null;
  danger_filter?: DangerFilter;
  agent_type?: AgentType;
  checkpoint_name?: string | null;
  seed?: number;
}): Promise<EpisodeResponse> {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      event_group_id: params.event_group_id || null,
      danger_filter: params.danger_filter || "All",
      agent_type: params.agent_type || "ppo",
      checkpoint_name: params.checkpoint_name || null,
      seed: params.seed ?? 42,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Simulation failed: ${detail}`);
  }
  return res.json();
}
