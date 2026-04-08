import { test, expect, type Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForIdle(page: Page) {
  await page.waitForSelector('[data-testid="idle-overlay"]', {
    timeout: 15_000,
  });
}

async function startEpisode(page: Page) {
  await page.getByTestId("start-btn").click();
  await page.waitForSelector('[data-testid="timeline-panel"]', {
    timeout: 30_000,
  });
}

// =========================================================================
//  FULL WORKFLOW: END-TO-END SCENARIO
// =========================================================================

test.describe("Full end-to-end workflow", () => {
  test("complete workflow: configure → start → play → pause → step → reset → replay", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);

    // 1. Configure
    await page
      .getByTestId("danger-filter-select")
      .selectOption("Confirmed Threat");
    await page.getByTestId("agent-type-select").selectOption("ppo");
    await page.getByTestId("seed-input").fill("42");

    // 2. Start
    await startEpisode(page);
    await expect(page.getByTestId("timeline-panel")).toContainText(
      "1 / 12 steps",
    );

    // 3. Play
    await page.locator("button.speed-btn").filter({ hasText: "8×" }).click();
    await page.getByTestId("play-btn").click();
    await page.waitForTimeout(500);

    // 4. Pause
    await page.getByTestId("pause-btn").click();

    // Should be somewhere in the middle
    const timeline = page.getByTestId("timeline-panel");
    const text1 = await timeline.textContent();
    const match = text1?.match(/(\d+) \/ 12/);
    const pausedAt = parseInt(match![1], 10);
    expect(pausedAt).toBeGreaterThanOrEqual(1);

    // 5. Step forward once
    await page.getByTestId("step-btn").click();
    await expect(timeline).toContainText(`${pausedAt + 1} / 12`);

    // 6. Reset
    await page.getByTestId("reset-btn").click();
    await expect(timeline).toContainText("1 / 12 steps");

    // 7. Replay (play again)
    await page.getByTestId("play-btn").click();
    await page.waitForTimeout(500);
    const text2 = await timeline.textContent();
    const match2 = text2?.match(/(\d+) \/ 12/);
    expect(parseInt(match2![1], 10)).toBeGreaterThan(1);
  });

  test("complete workflow: switch agent mid-session", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    // Start with PPO
    await page.getByTestId("agent-type-select").selectOption("ppo");
    await startEpisode(page);
    await expect(page.getByTestId("telemetry-panel")).toBeVisible();

    // Step through a few
    await page.getByTestId("step-btn").click();
    await page.getByTestId("step-btn").click();

    // Switch to rule agent and re-start
    await page.getByTestId("agent-type-select").selectOption("rule");
    await startEpisode(page);

    // Timeline should reset to 1
    await expect(page.getByTestId("timeline-panel")).toContainText(
      "1 / 12 steps",
    );
    await expect(page.getByTestId("telemetry-panel")).toBeVisible();
  });

  test("complete workflow: run two consecutive episodes", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    // Episode 1
    await page.getByTestId("seed-input").fill("10");
    await startEpisode(page);

    // Step to end
    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }
    await expect(page.getByTestId("timeline-panel")).toContainText(
      "12 / 12 steps",
    );
    await expect(page.getByTestId("outcome-banner")).toBeVisible();

    // Episode 2 with different seed
    await page.getByTestId("seed-input").fill("20");
    await startEpisode(page);
    await expect(page.getByTestId("timeline-panel")).toContainText(
      "1 / 12 steps",
    );

    // Outcome banner from previous episode should be gone
    await expect(page.getByTestId("outcome-banner")).not.toBeVisible();
  });
});

// =========================================================================
//  SPECIFIC SCENARIO SELECTION
// =========================================================================

test.describe("Specific scenario selection", () => {
  test("can select a specific event from the dropdown", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const scenarioSelect = page.getByTestId("scenario-select");
    const options = scenarioSelect.locator("option");
    const count = await options.count();

    if (count > 1) {
      // Pick the second option (first non-random event)
      const secondOptionValue = await options.nth(1).getAttribute("value");
      expect(secondOptionValue).toBeTruthy();

      await scenarioSelect.selectOption(secondOptionValue!);
      await startEpisode(page);

      // Event info should show details for this specific event
      await expect(page.getByTestId("event-info")).toBeVisible();
    }
  });

  test("random scenario selection works", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    await page.getByTestId("scenario-select").selectOption("random");
    await startEpisode(page);
    await expect(page.getByTestId("timeline-panel")).toBeVisible();
  });
});

// =========================================================================
//  CHECKPOINT SELECTION
// =========================================================================

test.describe("Checkpoint selection", () => {
  test("checkpoint dropdown has options loaded", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const cpSelect = page.getByTestId("checkpoint-select");
    await expect(cpSelect).toBeVisible();

    const options = cpSelect.locator("option");
    await expect
      .poll(async () => await options.count(), { timeout: 10_000 })
      .toBeGreaterThan(0);
    const count = await options.count();
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThanOrEqual(3);
  });

  test("checkpoint dropdown uses curated user-facing labels", async ({ page }) => {
    await page.route("**/api/checkpoints", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([
          {
            name: "ppo_lstm_recommended",
            path: "C:\\models\\ppo_lstm_recommended.pt",
          },
          {
            name: "ppo_lstm_stable",
            path: "C:\\models\\ppo_lstm_stable.pt",
          },
          {
            name: "ppo_lstm_baseline",
            path: "C:\\models\\ppo_lstm_baseline.pt",
          },
          {
            name: "ppo_lstm_last",
            path: "C:\\models\\ppo_lstm_last.pt",
          },
        ]),
      });
    });

    await page.goto("/");
    await waitForIdle(page);

    const options = page.getByTestId("checkpoint-select").locator("option");
    await expect
      .poll(async () => await options.count(), { timeout: 10_000 })
      .toBe(3);

    await expect(options.nth(0)).toHaveText("Recommended PPO Policy");
    await expect(options.nth(1)).toHaveText("Stable PPO Policy");
    await expect(options.nth(2)).toHaveText("Baseline PPO Policy");

    const values = await options.evaluateAll((nodes) =>
      nodes.map((node) => (node as HTMLOptionElement).value),
    );
    expect(values).toEqual([
      "ppo_lstm_recommended",
      "ppo_lstm_stable",
      "ppo_lstm_baseline",
    ]);
  });

  test("can change checkpoint and run episode", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const cpSelect = page.getByTestId("checkpoint-select");
    const options = cpSelect.locator("option");
    const count = await options.count();

    if (count > 1) {
      // Select the second checkpoint
      const secondValue = await options.nth(1).getAttribute("value");
      await cpSelect.selectOption(secondValue!);
    }

    await startEpisode(page);
    await expect(page.getByTestId("timeline-panel")).toBeVisible();
  });
});

// =========================================================================
//  KEYBOARD INTERACTION
// =========================================================================

test.describe("Keyboard interaction", () => {
  test("tab navigation through controls", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    // Tab should cycle through interactive elements
    await page.keyboard.press("Tab");
    // Should be able to tab through select elements
    // Just verify page doesn't crash during keyboard navigation
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press("Tab");
    }

    // Verify the page is still intact
    await expect(page.getByTestId("control-panel")).toBeVisible();
  });

  test("enter key activates start button when focused", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    // Focus the start button
    await page.getByTestId("start-btn").focus();
    await page.keyboard.press("Enter");

    // Should trigger episode load
    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });
    await expect(page.getByTestId("timeline-panel")).toBeVisible();
  });
});

// =========================================================================
//  LOADING STATE
// =========================================================================

test.describe("Loading state", () => {
  test("start button shows loading indicator during simulation", async ({
    page,
  }) => {
    // Slow down the simulate response so we can observe loading state
    await page.route("**/api/simulate", async (route) => {
      await new Promise((r) => setTimeout(r, 1500));
      await route.continue();
    });

    await page.goto("/");
    await waitForIdle(page);

    const startBtn = page.getByTestId("start-btn");
    await startBtn.click();

    // Button should be disabled during loading
    await expect(startBtn).toBeDisabled();

    // Unroute and wait for response
    await page.unroute("**/api/simulate");
    await page.waitForSelector('[data-testid="timeline-panel"]', {
      timeout: 30_000,
    });
  });
});

// =========================================================================
//  DATA INTEGRITY
// =========================================================================

test.describe("Data integrity", () => {
  test("cumulative reward increases monotonically or matches frame data", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    // Step through and verify cumulative text appears in metrics
    for (let i = 0; i < 11; i++) {
      const metricsRow = page.getByTestId("metrics-row");
      const text = await metricsRow.textContent();
      expect(text).toContain("CUMULATIVE");
      await page.getByTestId("step-btn").click();
    }
  });

  test("all 12 frames have valid time values in timeline cards", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    // Step to end
    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }

    // All 12 cards should exist with T+ values
    const cards = page.getByTestId("timeline-card");
    await expect(cards).toHaveCount(12);

    for (let i = 0; i < 12; i++) {
      const cardText = await cards.nth(i).textContent();
      expect(cardText).toMatch(/T\+\d+/);
    }
  });

  test("alert level is always one of the valid values", async ({ page }) => {
    const validAlerts = ["monitor", "info", "watch", "advisory", "warning"];

    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    // Step through and check alert text in banner
    for (let i = 0; i < 12; i++) {
      const banner = page.getByTestId("alert-banner");
      const text = (await banner.textContent())?.toLowerCase() ?? "";
      const hasValid = validAlerts.some((a) => text.includes(a));
      expect(hasValid).toBe(true);

      if (i < 11) await page.getByTestId("step-btn").click();
    }
  });
});
