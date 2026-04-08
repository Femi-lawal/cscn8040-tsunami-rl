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
//  SIMULATION: START, PLAY, PAUSE, STEP, RESET
// =========================================================================

test.describe("Episode simulation flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("clicking Start loads an episode and enters paused state", async ({
    page,
  }) => {
    await startEpisode(page);

    // Idle overlay should vanish
    await expect(page.getByTestId("idle-overlay")).not.toBeVisible();

    // Timeline panel should appear with step count
    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toBeVisible();
    await expect(timeline).toContainText("1 / 12 steps");

    // Map should display alert banner
    await expect(page.getByTestId("alert-banner")).toBeVisible();

    // Telemetry should be populated
    const telemetry = page.getByTestId("telemetry-panel");
    await expect(telemetry).toBeVisible();
    await expect(telemetry).toContainText("STEP");
    await expect(telemetry).toContainText("ALERT");
    await expect(telemetry).toContainText("REWARD");

    // Play/Step/Reset should be enabled, Pause should remain disabled
    await expect(page.getByTestId("play-btn")).toBeEnabled();
    await expect(page.getByTestId("step-btn")).toBeEnabled();
    await expect(page.getByTestId("reset-btn")).toBeEnabled();
    await expect(page.getByTestId("pause-btn")).toBeDisabled();
  });

  test("Step button advances one frame at a time", async ({ page }) => {
    await startEpisode(page);

    // Should be at step 1
    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("1 / 12 steps");

    // Step forward
    await page.getByTestId("step-btn").click();
    await expect(timeline).toContainText("2 / 12 steps");

    // Step forward again
    await page.getByTestId("step-btn").click();
    await expect(timeline).toContainText("3 / 12 steps");
  });

  test("Step button advances through all 12 frames to done state", async ({
    page,
  }) => {
    await startEpisode(page);

    // Step through all 12 frames (starting at 1, so 11 more steps)
    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("12 / 12 steps");

    // In done state: pause is disabled, reset is enabled
    await expect(page.getByTestId("pause-btn")).toBeDisabled();
    await expect(page.getByTestId("reset-btn")).toBeEnabled();
  });

  test("Reset button returns to frame 1", async ({ page }) => {
    await startEpisode(page);

    // Advance a few steps
    await page.getByTestId("step-btn").click();
    await page.getByTestId("step-btn").click();
    await page.getByTestId("step-btn").click();

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("4 / 12 steps");

    // Reset
    await page.getByTestId("reset-btn").click();
    await expect(timeline).toContainText("1 / 12 steps");
  });

  test("Play button auto-advances frames", async ({ page }) => {
    await startEpisode(page);

    const timeline = page.getByTestId("timeline-panel");

    // Click play
    await page.getByTestId("play-btn").click();

    // Pause should become enabled
    await expect(page.getByTestId("pause-btn")).toBeEnabled();

    // Wait and check we're advancing
    await page.waitForTimeout(3000);

    // Should have advanced beyond step 1
    const text = await timeline.textContent();
    const match = text?.match(/(\d+) \/ 12/);
    expect(match).not.toBeNull();
    const currentStep = parseInt(match![1], 10);
    expect(currentStep).toBeGreaterThan(1);
  });

  test("Pause button stops auto-advance", async ({ page }) => {
    await startEpisode(page);

    const timeline = page.getByTestId("timeline-panel");

    // Play
    await page.getByTestId("play-btn").click();
    await page.waitForTimeout(2000);

    // Pause
    await page.getByTestId("pause-btn").click();
    await expect(page.getByTestId("pause-btn")).toBeDisabled();

    // Record current step
    const textAfterPause = await timeline.textContent();
    const matchAfter = textAfterPause?.match(/(\d+) \/ 12/);
    const pausedStep = parseInt(matchAfter![1], 10);

    // Wait and check it doesn't advance
    await page.waitForTimeout(1500);
    const textLater = await timeline.textContent();
    const matchLater = textLater?.match(/(\d+) \/ 12/);
    const laterStep = parseInt(matchLater![1], 10);

    expect(laterStep).toBe(pausedStep);
  });

  test("Play runs to completion and reaches done state", async ({ page }) => {
    await startEpisode(page);

    // Play at fastest speed by selecting the highest speed first
    // Speed buttons are not individually testid'd, so click the 8× button
    await page.locator("button.speed-btn").filter({ hasText: "8×" }).click();
    await page.getByTestId("play-btn").click();

    // Wait for it to complete – at 125ms per frame × 12 = ~1.5s + margin
    await page.waitForTimeout(4000);

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("12 / 12 steps");

    // In done state: pause is disabled
    await expect(page.getByTestId("pause-btn")).toBeDisabled();
  });

  test("Reset after done allows replay", async ({ page }) => {
    await startEpisode(page);

    // Step to end
    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }

    // Reset
    await page.getByTestId("reset-btn").click();

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("1 / 12 steps");

    // Buttons should be enabled again
    await expect(page.getByTestId("play-btn")).toBeEnabled();
    await expect(page.getByTestId("step-btn")).toBeEnabled();
  });

  test("Starting a new episode replaces the previous one", async ({ page }) => {
    await startEpisode(page);

    // Step a few times
    await page.getByTestId("step-btn").click();
    await page.getByTestId("step-btn").click();

    // Start again
    await startEpisode(page);

    // Should be back at step 1
    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("1 / 12 steps");
  });
});
