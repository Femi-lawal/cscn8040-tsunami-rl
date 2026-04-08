import { useCallback, useRef, useState } from "react";
import type { EpisodeFrame, EpisodeResponse, EventMetadata } from "./types";

export interface EpisodeState {
  status: "idle" | "loading" | "playing" | "paused" | "done";
  metadata: EventMetadata | null;
  frames: EpisodeFrame[];
  currentFrameIndex: number;
  totalReturn: number;
  outcomeSummary: string;
  error: string | null;
}

const INITIAL: EpisodeState = {
  status: "idle",
  metadata: null,
  frames: [],
  currentFrameIndex: -1,
  totalReturn: 0,
  outcomeSummary: "",
  error: null,
};

export function useEpisodeStore() {
  const [state, setState] = useState<EpisodeState>(INITIAL);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const speedRef = useRef(1000); // ms per frame

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const loadEpisode = useCallback(
    (response: EpisodeResponse) => {
      clearTimer();
      setState({
        status: "paused",
        metadata: response.event_metadata,
        frames: response.frames,
        currentFrameIndex: 0,
        totalReturn: response.total_return,
        outcomeSummary: response.outcome_summary,
        error: null,
      });
    },
    [clearTimer],
  );

  const setLoading = useCallback(() => {
    clearTimer();
    setState((s) => ({ ...s, status: "loading", error: null }));
  }, [clearTimer]);

  const setError = useCallback(
    (msg: string) => {
      clearTimer();
      setState((s) => ({ ...s, status: "idle", error: msg }));
    },
    [clearTimer],
  );

  const stepForward = useCallback(() => {
    setState((prev) => {
      if (prev.currentFrameIndex >= prev.frames.length - 1) {
        clearTimer();
        return { ...prev, status: "done" };
      }
      return { ...prev, currentFrameIndex: prev.currentFrameIndex + 1 };
    });
  }, [clearTimer]);

  const play = useCallback(
    (speedMs?: number) => {
      clearTimer();
      if (speedMs) speedRef.current = speedMs;
      setState((s) => ({ ...s, status: "playing" }));

      timerRef.current = setInterval(() => {
        setState((prev) => {
          if (prev.currentFrameIndex >= prev.frames.length - 1) {
            clearTimer();
            return { ...prev, status: "done" };
          }
          return { ...prev, currentFrameIndex: prev.currentFrameIndex + 1 };
        });
      }, speedRef.current);
    },
    [clearTimer],
  );

  const pause = useCallback(() => {
    clearTimer();
    setState((s) => ({ ...s, status: "paused" }));
  }, [clearTimer]);

  const reset = useCallback(() => {
    clearTimer();
    setState((s) => ({
      ...s,
      status: "paused",
      currentFrameIndex: 0,
    }));
  }, [clearTimer]);

  const setSpeed = useCallback(
    (ms: number) => {
      speedRef.current = ms;
      if (state.status === "playing") {
        play(ms);
      }
    },
    [state.status, play],
  );

  const currentFrame =
    state.currentFrameIndex >= 0 &&
    state.currentFrameIndex < state.frames.length
      ? state.frames[state.currentFrameIndex]
      : null;

  return {
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
    clearTimer,
  };
}
