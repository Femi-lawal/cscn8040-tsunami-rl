import { test, expect, type Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForIdle(page: Page) {
  await page.waitForSelector('[data-testid="idle-overlay"]', {
    timeout: 15_000,
  });
}

async function startEpisodeWithOptions(
  page: Page,
  opts: {
    agentType?: string;
    dangerFilter?: string;
    seed?: string;
    scenario?: string;
    checkpoint?: string;
  } = {},
) {
  if (opts.dangerFilter) {
    await page
      .getByTestId("danger-filter-select")
      .selectOption(opts.dangerFilter);
  }
  if (opts.scenario) {
    await page.getByTestId("scenario-select").selectOption(opts.scenario);
  }
  if (opts.agentType) {
    await page.getByTestId("agent-type-select").selectOption(opts.agentType);
  }
  if (opts.checkpoint) {
    await page.getByTestId("checkpoint-select").selectOption(opts.checkpoint);
  }
  if (opts.seed !== undefined) {
    await page.getByTestId("seed-input").fill(opts.seed);
  }

  await page.getByTestId("start-btn").click();
  await page.waitForSelector('[data-testid="timeline-panel"]', {
    timeout: 30_000,
  });
}

// =========================================================================
//  SPECIFIC EVENT SELECTION (event_group_id lookup)
// =========================================================================

test.describe("Specific event_group_id lookup", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("simulate API accepts a specific event_group_id string", async ({
    page,
  }) => {
    // Fetch catalog to get a real event ID
    const catalogRes = await page.request.get("/api/catalog");
    const catalog = await catalogRes.json();
    const eventId = String(catalog[0].event_group_id);

    const response = await page.request.post("/api/simulate", {
      data: {
        agent_type: "rule",
        seed: 42,
        event_group_id: eventId,
      },
    });
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(String(data.event_metadata.event_group_id)).toBe(eventId);
    expect(data.frames).toHaveLength(12);
  });

  test("simulate API returns 404 for nonexistent event_group_id", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: {
        agent_type: "rule",
        seed: 1,
        event_group_id: "00000000000000",
      },
    });
    expect(response.status()).toBe(404);
    const body = await response.json();
    expect(body.detail).toContain("not found");
  });

  test("selecting a specific scenario in UI sends correct event_group_id", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);

    const scenarioSelect = page.getByTestId("scenario-select");
    const options = scenarioSelect.locator("option");
    const count = await options.count();

    // Select first non-random event
    if (count > 1) {
      const eventValue = await options.nth(1).getAttribute("value");
      await scenarioSelect.selectOption(eventValue!);

      // Set up request interception BEFORE clicking start
      const requestPromise = page.waitForRequest(
        (req) => req.url().includes("/api/simulate") && req.method() === "POST",
      );

      await page.getByTestId("start-btn").click();

      const req = await requestPromise;
      const body = req.postDataJSON();
      expect(body.event_group_id).toBe(eventValue);

      await page.waitForSelector('[data-testid="timeline-panel"]', {
        timeout: 30_000,
      });
    }
  });

  test("multiple specific events can be simulated in sequence", async ({
    page,
  }) => {
    const catalogRes = await page.request.get("/api/catalog");
    const catalog = await catalogRes.json();

    // Simulate three different events sequentially
    for (const event of catalog.slice(0, 3)) {
      const response = await page.request.post("/api/simulate", {
        data: {
          agent_type: "rule",
          seed: 42,
          event_group_id: String(event.event_group_id),
        },
      });
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data.frames).toHaveLength(12);
      expect(String(data.event_metadata.event_group_id)).toBe(
        String(event.event_group_id),
      );
    }
  });
});

// =========================================================================
//  DANGER FILTER + EVENT LOOKUP COMBINATIONS
// =========================================================================

test.describe("Danger filter and event lookup combinations", () => {
  test("specific event with matching danger filter works", async ({ page }) => {
    const catalogRes = await page.request.get("/api/catalog");
    const catalog = await catalogRes.json();

    // Find a Confirmed Threat event
    const confirmed = catalog.find(
      (e: { danger_tier: number }) => e.danger_tier === 2,
    );
    if (confirmed) {
      const response = await page.request.post("/api/simulate", {
        data: {
          agent_type: "rule",
          seed: 1,
          event_group_id: String(confirmed.event_group_id),
          danger_filter: "Confirmed Threat",
        },
      });
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data.event_metadata.danger_tier).toBe(2);
    }
  });

  test("specific event with non-matching danger filter returns 404", async ({
    page,
  }) => {
    const catalogRes = await page.request.get("/api/catalog");
    const catalog = await catalogRes.json();

    // Find a No Threat event and request it with Confirmed Threat filter
    const noThreat = catalog.find(
      (e: { danger_tier: number }) => e.danger_tier === 0,
    );
    if (noThreat) {
      const response = await page.request.post("/api/simulate", {
        data: {
          agent_type: "rule",
          seed: 1,
          event_group_id: String(noThreat.event_group_id),
          danger_filter: "Confirmed Threat",
        },
      });
      expect(response.status()).toBe(404);
    }
  });

  test("each danger filter returns events of the correct tier", async ({
    page,
  }) => {
    const filters = [
      { label: "No Threat", tier: 0 },
      { label: "Potential Threat", tier: 1 },
      { label: "Confirmed Threat", tier: 2 },
    ];

    for (const { label, tier } of filters) {
      const response = await page.request.post("/api/simulate", {
        data: {
          agent_type: "rule",
          seed: 42,
          danger_filter: label,
        },
      });
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data.event_metadata.danger_tier).toBe(tier);
    }
  });
});

// =========================================================================
//  CATALOG DATA INTEGRITY
// =========================================================================

test.describe("Catalog data integrity", () => {
  test("all catalog events have string event_group_id", async ({ page }) => {
    const response = await page.request.get("/api/catalog");
    const catalog = await response.json();

    for (const event of catalog) {
      expect(typeof event.event_group_id).toBe("string");
      expect(event.event_group_id.length).toBeGreaterThan(0);
    }
  });

  test("all catalog events have valid danger_tier (0, 1, or 2)", async ({
    page,
  }) => {
    const response = await page.request.get("/api/catalog");
    const catalog = await response.json();

    for (const event of catalog) {
      expect([0, 1, 2]).toContain(event.danger_tier);
      expect(["No Threat", "Potential Threat", "Confirmed Threat"]).toContain(
        event.danger_label,
      );
    }
  });

  test("catalog events have valid coordinates", async ({ page }) => {
    const response = await page.request.get("/api/catalog");
    const catalog = await response.json();

    for (const event of catalog) {
      if (event.latitude !== null) {
        expect(event.latitude).toBeGreaterThanOrEqual(-90);
        expect(event.latitude).toBeLessThanOrEqual(90);
      }
      if (event.longitude !== null) {
        expect(event.longitude).toBeGreaterThanOrEqual(-180);
        expect(event.longitude).toBeLessThanOrEqual(180);
      }
    }
  });

  test("catalog events have non-negative magnitudes", async ({ page }) => {
    const response = await page.request.get("/api/catalog");
    const catalog = await response.json();

    for (const event of catalog) {
      if (event.initial_magnitude !== null) {
        expect(event.initial_magnitude).toBeGreaterThan(0);
      }
    }
  });

  test("repeated catalog calls return the same data", async ({ page }) => {
    const res1 = await page.request.get("/api/catalog");
    const res2 = await page.request.get("/api/catalog");
    const cat1 = await res1.json();
    const cat2 = await res2.json();

    expect(cat1.length).toBe(cat2.length);
    expect(cat1[0].event_group_id).toBe(cat2[0].event_group_id);
    expect(cat1[cat1.length - 1].event_group_id).toBe(
      cat2[cat2.length - 1].event_group_id,
    );
  });
});

// =========================================================================
//  SIMULATION RESPONSE INTEGRITY
// =========================================================================

test.describe("Simulation response integrity", () => {
  test("all frames have monotonically non-decreasing time_min", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    for (let i = 1; i < data.frames.length; i++) {
      expect(data.frames[i].time_min).toBeGreaterThanOrEqual(
        data.frames[i - 1].time_min,
      );
    }
  });

  test("frame t values range from 0 to 11", async ({ page }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    for (let i = 0; i < data.frames.length; i++) {
      expect(data.frames[i].t).toBe(i);
    }
  });

  test("action names are valid", async ({ page }) => {
    const validActions = [
      "hold",
      "escalate",
      "deescalate",
      "issue_watch",
      "issue_warning",
      "cancel",
    ];

    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "ppo", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      expect(validActions).toContain(frame.action);
    }
  });

  test("alert_level values are valid", async ({ page }) => {
    const validAlerts = ["monitor", "info", "watch", "advisory", "warning"];

    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "ppo", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      expect(validAlerts).toContain(frame.alert_level);
    }
  });

  test("PPO frames include agent_probabilities summing to ~1", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "ppo", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      const probs = Object.values(frame.agent_probabilities) as number[];
      expect(probs.length).toBe(6);
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });

  test("PPO frames include value_estimate as a number", async ({ page }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "ppo", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      expect(typeof frame.value_estimate).toBe("number");
    }
  });

  test("rule agent frames have null value_estimate", async ({ page }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      expect(frame.value_estimate).toBeNull();
    }
  });

  test("epicenter coordinates are consistent across all frames", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    const firstEpicenter = data.frames[0].epicenter;
    for (const frame of data.frames) {
      expect(frame.epicenter.lat).toBe(firstEpicenter.lat);
      expect(frame.epicenter.lon).toBe(firstEpicenter.lon);
    }
  });

  test("last frame done flag is true", async ({ page }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    const lastFrame = data.frames[data.frames.length - 1];
    expect(lastFrame.done).toBe(true);
  });

  test("cumulative_reward in last frame matches total_return", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    const lastFrame = data.frames[data.frames.length - 1];
    expect(lastFrame.cumulative_reward).toBeCloseTo(data.total_return, 4);
  });

  test("observation dict has all expected fields", async ({ page }) => {
    const expectedFields = [
      "magnitude_estimate",
      "depth_estimate_km",
      "coastal_proximity_index",
      "wave_estimate_m",
      "buoy_confirmation",
      "tide_confirmation",
      "uncertainty",
      "time_fraction",
      "alert_level_norm",
      "cancel_issued_flag",
      "delta_magnitude",
      "delta_wave_m",
      "delta_uncertainty",
      "time_since_buoy_norm",
      "time_since_tide_norm",
    ];

    const response = await page.request.post("/api/simulate", {
      data: { agent_type: "rule", seed: 42 },
    });
    const data = await response.json();

    for (const frame of data.frames) {
      for (const field of expectedFields) {
        expect(frame.observation).toHaveProperty(field);
        expect(typeof frame.observation[field]).toBe("number");
      }
    }
  });
});

// =========================================================================
//  CONCURRENT / REPEATED REQUESTS
// =========================================================================

test.describe("Repeated and concurrent requests", () => {
  test("same seed same event gives identical results across calls", async ({
    page,
  }) => {
    const body = {
      agent_type: "rule",
      seed: 77,
      danger_filter: "All",
    };

    const r1 = await page.request.post("/api/simulate", { data: body });
    const r2 = await page.request.post("/api/simulate", { data: body });
    const d1 = await r1.json();
    const d2 = await r2.json();

    expect(d1.event_metadata.event_group_id).toBe(
      d2.event_metadata.event_group_id,
    );
    expect(d1.total_return).toBe(d2.total_return);
    expect(d1.frames.length).toBe(d2.frames.length);

    // All frame actions should match
    for (let i = 0; i < d1.frames.length; i++) {
      expect(d1.frames[i].action).toBe(d2.frames[i].action);
      expect(d1.frames[i].reward).toBe(d2.frames[i].reward);
    }
  });

  test("catalog is not mutated after simulate call", async ({ page }) => {
    // Get catalog before
    const catBefore = await (await page.request.get("/api/catalog")).json();

    // Simulate with specific event
    await page.request.post("/api/simulate", {
      data: {
        agent_type: "rule",
        seed: 42,
        event_group_id: String(catBefore[0].event_group_id),
      },
    });

    // Get catalog after
    const catAfter = await (await page.request.get("/api/catalog")).json();

    // Catalog should be identical
    expect(catAfter.length).toBe(catBefore.length);
    for (let i = 0; i < catBefore.length; i++) {
      expect(catAfter[i].event_group_id).toBe(catBefore[i].event_group_id);
      expect(catAfter[i].danger_tier).toBe(catBefore[i].danger_tier);
    }
  });

  test("rapid sequential simulations all succeed", async ({ page }) => {
    const seeds = [1, 2, 3, 4, 5];
    for (const seed of seeds) {
      const response = await page.request.post("/api/simulate", {
        data: { agent_type: "rule", seed },
      });
      expect(response.ok()).toBeTruthy();
      const data = await response.json();
      expect(data.frames).toHaveLength(12);
    }
  });
});

// =========================================================================
//  UI: SPECIFIC EVENT SELECTION FLOW
// =========================================================================

test.describe("UI specific event selection flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("selecting event from dropdown starts episode without error", async ({
    page,
  }) => {
    const scenarioSelect = page.getByTestId("scenario-select");
    const options = scenarioSelect.locator("option");
    const count = await options.count();

    // Skip "random" option, pick the second one
    if (count > 1) {
      const eventValue = await options.nth(1).getAttribute("value");
      await scenarioSelect.selectOption(eventValue!);
      await startEpisodeWithOptions(page, { seed: "42" });

      // Should show event info and timeline without errors
      await expect(page.getByTestId("event-info")).toBeVisible();
      await expect(page.getByTestId("timeline-panel")).toContainText(
        "1 / 12 steps",
      );

      // No error overlay should appear
      const errorOverlay = page.locator('[data-testid="error-overlay"]');
      await expect(errorOverlay).not.toBeVisible();
    }
  });

  test("selecting event then changing danger filter resets scenario", async ({
    page,
  }) => {
    const catalogRes = await page.request.get("/api/catalog");
    const catalog = await catalogRes.json();
    const noThreat = catalog.find((e: { danger_tier: number }) => e.danger_tier === 0);

    // Select a specific event
    const scenarioSelect = page.getByTestId("scenario-select");
    expect(noThreat).toBeTruthy();

    await scenarioSelect.selectOption(String(noThreat.event_group_id));

    // Change danger filter so the selected event is no longer valid.
    await page
      .getByTestId("danger-filter-select")
      .selectOption("Confirmed Threat");

    // UI state should reset to the random option, not keep a stale hidden value.
    await expect(scenarioSelect).toHaveValue("random");

    const requestPromise = page.waitForRequest(
      (req) => req.url().includes("/api/simulate") && req.method() === "POST",
    );

    await page.getByTestId("start-btn").click();

    const req = await requestPromise;
    const body = req.postDataJSON();
    expect(body.event_group_id).toBeNull();

    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });
    await expect(page.getByTestId("error-overlay")).not.toBeVisible();
  });

  test("complete episode flow with specific event and PPO agent", async ({
    page,
  }) => {
    const scenarioSelect = page.getByTestId("scenario-select");
    const options = scenarioSelect.locator("option");
    const count = await options.count();

    if (count > 1) {
      const eventValue = await options.nth(1).getAttribute("value");
      await scenarioSelect.selectOption(eventValue!);

      await startEpisodeWithOptions(page, { agentType: "ppo", seed: "42" });

      // Step through all frames
      for (let i = 0; i < 11; i++) {
        await page.getByTestId("step-btn").click();
      }

      // Should reach completion
      await expect(page.getByTestId("timeline-panel")).toContainText(
        "12 / 12 steps",
      );
      await expect(page.getByTestId("outcome-banner")).toBeVisible();
    }
  });
});

// =========================================================================
//  CHECKPOINT SWITCHING
// =========================================================================

test.describe("Checkpoint edge cases", () => {
  test("switching checkpoints between episodes works", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const cpSelect = page.getByTestId("checkpoint-select");
    const options = cpSelect.locator("option");
    const count = await options.count();

    if (count >= 2) {
      // Run with first checkpoint
      await cpSelect.selectOption(
        (await options.nth(0).getAttribute("value"))!,
      );
      await startEpisodeWithOptions(page, { agentType: "ppo", seed: "42" });
      await expect(page.getByTestId("timeline-panel")).toContainText(
        "1 / 12 steps",
      );

      // Switch checkpoint and run again
      await cpSelect.selectOption(
        (await options.nth(1).getAttribute("value"))!,
      );
      await startEpisodeWithOptions(page, { agentType: "ppo", seed: "42" });
      await expect(page.getByTestId("timeline-panel")).toContainText(
        "1 / 12 steps",
      );
    }
  });
});
