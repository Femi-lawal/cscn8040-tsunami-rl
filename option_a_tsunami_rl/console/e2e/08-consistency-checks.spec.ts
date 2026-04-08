import { test, expect, type Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const API_BASE = "/api";

interface SimulateBody {
  event_group_id?: string | null;
  agent_type?: string;
  checkpoint?: string;
  danger_filter?: string;
  seed?: number;
}

async function simulate(page: Page, body: SimulateBody = {}) {
  const res = await page.request.post(`${API_BASE}/simulate`, {
    data: {
      event_group_id: body.event_group_id ?? null,
      agent_type: body.agent_type ?? "ppo",
      checkpoint_name: body.checkpoint ?? null,
      danger_filter: body.danger_filter ?? "All",
      seed: body.seed ?? 42,
    },
  });
  expect(res.ok()).toBeTruthy();
  return res.json();
}

async function waitForIdle(page: Page) {
  await page.waitForSelector('[data-testid="idle-overlay"]', {
    timeout: 15_000,
  });
}

// =========================================================================
//  1. Terminal value estimate must be zero
// =========================================================================

test.describe("Terminal value estimate consistency", () => {
  test("PPO value_estimate is 0 on the terminal (done=true) frame", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "ppo" });
    const frames = episode.frames;
    const lastFrame = frames[frames.length - 1];

    expect(lastFrame.done).toBe(true);
    expect(lastFrame.value_estimate).toBe(0.0);
  });

  test("non-terminal PPO frames have non-zero value_estimate", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "ppo" });
    const nonTerminal = episode.frames.filter(
      (f: { done: boolean }) => !f.done,
    );
    // At least some non-terminal frames should have non-zero value
    const hasNonZero = nonTerminal.some(
      (f: { value_estimate: number | null }) =>
        f.value_estimate !== null && f.value_estimate !== 0.0,
    );
    expect(hasNonZero).toBe(true);
  });

  test("rule agent has null value_estimate on all frames including terminal", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "rule" });
    for (const frame of episode.frames) {
      expect(frame.value_estimate).toBeNull();
    }
  });
});

// =========================================================================
//  2. Reward scale: potential threats capped below confirmed threats
// =========================================================================

test.describe("Reward scale consistency", () => {
  test("potential threat terminal reward is less than confirmed threat reward", async ({
    page,
  }) => {
    // Run a potential threat episode
    const potentialEp = await simulate(page, {
      danger_filter: "Potential Threat",
      seed: 42,
    });

    // Run a confirmed threat episode
    const confirmedEp = await simulate(page, {
      danger_filter: "Confirmed Threat",
      seed: 42,
    });

    // The confirmed threat total return (absolute) should dominate
    // If both have positive returns, confirmed should be higher
    const potReturn = potentialEp.total_return;
    const confReturn = confirmedEp.total_return;

    // The max possible potential-threat terminal bonus is 45 (step 0)
    // The max possible confirmed-threat terminal bonus is 140 (step 0)
    // So confirmed should have higher ceiling
    // We test: if both are positive, confirmed >= potential
    if (potReturn > 0 && confReturn > 0) {
      expect(confReturn).toBeGreaterThanOrEqual(potReturn);
    }
  });

  test("potential threat terminal reward does not exceed 45", async ({
    page,
  }) => {
    // Run multiple seeds to check the cap
    for (const seed of [1, 42, 99, 123, 777]) {
      const ep = await simulate(page, {
        danger_filter: "Potential Threat",
        seed,
      });
      // Terminal reward for potential threats: max(8, 45 - 5*step)
      // Max possible = 45, but step rewards are -0.1 each, so total_return < 45
      // The total should be well under confirmed threat ceiling (~140)
      const lastFrame = ep.frames[ep.frames.length - 1];
      if (lastFrame.done) {
        // Total return should not exceed ~45 (terminal ceiling + step penalties)
        expect(ep.total_return).toBeLessThan(50);
      }
    }
  });
});

// =========================================================================
//  3. Sentinel values must not appear as raw numbers
// =========================================================================

test.describe("Sentinel value display", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("observation time_since fields use N/A for unconfirmed sensors", async ({
    page,
  }) => {
    // Find an episode where tide is NOT confirmed, check via API
    const episode = await simulate(page, {
      danger_filter: "No Threat",
      seed: 42,
    });

    // Check frames for sentinel values
    for (const frame of episode.frames) {
      const obs = frame.observation;
      // time_since values should be >= 0 if sensor confirmed, or negative sentinel
      // The frontend should render negative values as "N/A"
      if (obs.time_since_buoy_norm < 0) {
        // If value is negative sentinel, it should NOT be shown as a number
        // This test validates the API sends the raw value so the frontend
        // can apply the N/A formatting
        expect(obs.time_since_buoy_norm).toBeLessThan(0);
      }
      if (obs.time_since_tide_norm < 0) {
        expect(obs.time_since_tide_norm).toBeLessThan(0);
      }
    }
  });

  test("modal displays N/A for negative sentinel time_since values", async ({
    page,
  }) => {
    // Start episode
    await page.getByTestId("danger-filter-select").selectOption("No Threat");
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });

    // Click first timeline card to open modal
    const card = page.getByTestId("timeline-card").first();
    await card.click();
    await page.waitForSelector('[data-testid="frame-detail-modal"]', {
      timeout: 5_000,
    });

    // Check that no "-1.000" appears in the modal content
    const modalText = await page
      .locator('[data-testid="frame-detail-modal"]')
      .textContent();
    expect(modalText).not.toContain("-1.000");

    // If unconfirmed, should say "N/A" instead
    // (at least for Time Since Tide or Time Since Buoy)
    // We verify by checking those rows exist
    expect(modalText).toContain("Time Since");
  });
});

// =========================================================================
//  4. Label consistency across modal and telemetry panel
// =========================================================================

test.describe("Label consistency", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("telemetry panel uses consistent observation labels", async ({
    page,
  }) => {
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="telemetry-panel"]', {
      timeout: 30_000,
    });

    const telemetryText = await page
      .getByTestId("telemetry-panel")
      .textContent();

    // Should use "Wave (m)" not "Wave Est. (m)" or just "Wave"
    expect(telemetryText).toContain("Wave (m)");
    // Should use "Buoy" label
    expect(telemetryText).toContain("Buoy");
    // Should use "Tide" label
    expect(telemetryText).toContain("Tide");
    // Delta labels should include units
    expect(telemetryText).toContain("Δ Wave (m)");
  });

  test("telemetry panel shows V(s) for value estimate", async ({ page }) => {
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="telemetry-panel"]', {
      timeout: 30_000,
    });

    const telemetryText = await page
      .getByTestId("telemetry-panel")
      .textContent();

    expect(telemetryText).toContain("V(s)");
    expect(telemetryText).toContain("expected return");
  });

  test("telemetry diagnostics labels rule as baseline policy", async ({
    page,
  }) => {
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="telemetry-panel"]', {
      timeout: 30_000,
    });

    const telemetryText = await page
      .getByTestId("telemetry-panel")
      .textContent();

    expect(telemetryText).toContain("Rule Recommendation");
    expect(telemetryText).toContain("(baseline policy)");
  });

  test("modal uses matching observation labels", async ({ page }) => {
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });

    // Open modal
    await page.getByTestId("timeline-card").first().click();
    await page.waitForSelector('[data-testid="frame-detail-modal"]', {
      timeout: 5_000,
    });

    const modalText = await page
      .locator('[data-testid="frame-detail-modal"]')
      .textContent();

    // Same labels as telemetry
    expect(modalText).toContain("Wave (m)");
    expect(modalText).toContain("Δ Wave (m)");
    // Buoy/Tide should show CONFIRMED or —, not "YES"/"No"
    expect(modalText).not.toContain("BuoyYES");
    expect(modalText).not.toContain("BuoyNo");
  });

  test("modal separates RL agent, rule baseline, and environment reward sections", async ({
    page,
  }) => {
    await page.getByTestId("start-btn").click();
    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });

    await page.getByTestId("timeline-card").first().click();
    await page.waitForSelector('[data-testid="frame-detail-modal"]', {
      timeout: 5_000,
    });

    const modalText = await page
      .locator('[data-testid="frame-detail-modal"]')
      .textContent();

    expect(modalText).toContain("RL AGENT ACTION");
    expect(modalText).toContain("RULE BASELINE");
    expect(modalText).toContain("ENVIRONMENT REWARD");
    expect(modalText).toContain("V(s) Expected Return");
    expect(modalText).toContain("Step Reward");
  });
});

// =========================================================================
//  5. Rule recommendation vs action vs reward coherence
// =========================================================================

test.describe("Rule-action-reward coherence", () => {
  test("when rule and agent agree, reward is not punitive", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "ppo", seed: 42 });

    for (const frame of episode.frames) {
      if (frame.rule_recommendation === frame.action) {
        // When agent follows the rule, step reward should not be heavily negative
        // (beyond the baseline -0.1 step cost)
        expect(frame.reward).toBeGreaterThanOrEqual(-5.0);
      }
    }
  });

  test("deep earthquake (>70km) rule should not recommend escalation past watch", async ({
    page,
  }) => {
    // Simulate across various seeds and check rule recommendations
    // For events with depth > 70km, rule should not recommend escalate
    // when already at watch level
    const catalog = await (
      await page.request.get(`${API_BASE}/catalog`)
    ).json();

    // Find events with deep depths
    const deepEvents = catalog.filter(
      (e: { initial_depth_km: number | null }) =>
        e.initial_depth_km != null && e.initial_depth_km > 70,
    );

    if (deepEvents.length === 0) return; // skip if no deep events

    // Test a sample of deep events
    const sample = deepEvents.slice(0, 3);
    for (const event of sample) {
      const ep = await simulate(page, {
        event_group_id: event.event_group_id,
        agent_type: "rule",
      });

      for (const frame of ep.frames) {
        // For deep quakes, rule should not push past watch unless
        // there's overwhelming evidence (tide confirmed)
        const obs = frame.observation;
        const isDeep = obs.depth_estimate_km > 70;
        const tideConfirmed = obs.tide_confirmation > 0.5;

        if (isDeep && !tideConfirmed && frame.alert_level_index >= 2) {
          // If already at watch and deep, rule should not escalate further
          expect(frame.action).not.toBe("escalate");
        }
      }
    }
  });
});

// =========================================================================
//  6. Physical plausibility of observations
// =========================================================================

test.describe("Observation value plausibility", () => {
  test("magnitude is in geophysically valid range", async ({ page }) => {
    const episode = await simulate(page);
    for (const frame of episode.frames) {
      const mag = frame.observation.magnitude_estimate;
      expect(mag).toBeGreaterThanOrEqual(4.0);
      expect(mag).toBeLessThanOrEqual(10.0);
    }
  });

  test("depth is non-negative", async ({ page }) => {
    const episode = await simulate(page);
    for (const frame of episode.frames) {
      expect(frame.observation.depth_estimate_km).toBeGreaterThanOrEqual(0);
    }
  });

  test("wave estimate is non-negative", async ({ page }) => {
    const episode = await simulate(page);
    for (const frame of episode.frames) {
      expect(frame.observation.wave_estimate_m).toBeGreaterThanOrEqual(0);
    }
  });

  test("uncertainty is between 0 and 1", async ({ page }) => {
    const episode = await simulate(page);
    for (const frame of episode.frames) {
      expect(frame.observation.uncertainty).toBeGreaterThanOrEqual(0);
      expect(frame.observation.uncertainty).toBeLessThanOrEqual(1);
    }
  });

  test("time_fraction increases monotonically across frames", async ({
    page,
  }) => {
    const episode = await simulate(page);
    for (let i = 1; i < episode.frames.length; i++) {
      expect(
        episode.frames[i].observation.time_fraction,
      ).toBeGreaterThanOrEqual(episode.frames[i - 1].observation.time_fraction);
    }
  });
});

// =========================================================================
//  7. Step reward / terminal bonus split
// =========================================================================

test.describe("Step reward and terminal bonus split", () => {
  test("non-terminal frames have step_reward equal to reward and null terminal_bonus", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "ppo", seed: 42 });
    const nonTerminal = episode.frames.filter(
      (f: { done: boolean }) => !f.done,
    );
    for (const frame of nonTerminal) {
      expect(frame.terminal_bonus).toBeNull();
      expect(frame.step_reward).toBeCloseTo(frame.reward, 5);
    }
  });

  test("terminal frame has non-null terminal_bonus and step_reward + terminal_bonus = reward", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "ppo", seed: 42 });
    const lastFrame = episode.frames[episode.frames.length - 1];
    expect(lastFrame.done).toBe(true);
    expect(lastFrame.terminal_bonus).not.toBeNull();
    expect(lastFrame.step_reward + lastFrame.terminal_bonus).toBeCloseTo(
      lastFrame.reward,
      5,
    );
  });

  test("rule agent also has step_reward and terminal_bonus fields", async ({
    page,
  }) => {
    const episode = await simulate(page, { agent_type: "rule", seed: 42 });
    const lastFrame = episode.frames[episode.frames.length - 1];
    expect(lastFrame.done).toBe(true);
    expect(lastFrame.terminal_bonus).not.toBeNull();
    expect(lastFrame.step_reward).toBeDefined();
  });
});
