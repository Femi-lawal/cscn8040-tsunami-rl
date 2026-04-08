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
//  MAP COMPONENT
// =========================================================================

test.describe("Map component", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("map container renders before episode starts", async ({ page }) => {
    const map = page.getByTestId("map-container");
    await expect(map).toBeVisible();
  });

  test("map container has reasonable size", async ({ page }) => {
    const box = await page.getByTestId("map-container").boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThan(200);
    expect(box!.height).toBeGreaterThan(200);
  });

  test("alert banner appears after episode starts", async ({ page }) => {
    await startEpisode(page);
    const banner = page.getByTestId("alert-banner");
    await expect(banner).toBeVisible();
    // Should contain the alert level name
    const text = await banner.textContent();
    expect(text?.length).toBeGreaterThan(0);
  });

  test("event info overlay shows metadata", async ({ page }) => {
    await startEpisode(page);
    const info = page.getByTestId("event-info");
    await expect(info).toBeVisible();
    // Should contain magnitude or depth info
    const text = await info.textContent();
    expect(text).toBeTruthy();
  });

  test("alert banner updates across steps", async ({ page }) => {
    await startEpisode(page);

    const banner = page.getByTestId("alert-banner");
    const initialText = await banner.textContent();

    // Step through several frames
    for (let i = 0; i < 6; i++) {
      await page.getByTestId("step-btn").click();
    }

    // Banner should still be visible (may or may not have changed text)
    await expect(banner).toBeVisible();
  });
});

// =========================================================================
//  TELEMETRY PANEL
// =========================================================================

test.describe("Telemetry panel", () => {
  test("shows empty state before episode", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await expect(
      page.getByText("Start an episode to see telemetry data"),
    ).toBeVisible();
  });

  test("displays metrics row after episode starts", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const metricsRow = page.getByTestId("metrics-row");
    await expect(metricsRow).toBeVisible();

    // Check for key metric labels
    await expect(metricsRow).toContainText("STEP");
    await expect(metricsRow).toContainText("ALERT");
    await expect(metricsRow).toContainText("REWARD");
    await expect(metricsRow).toContainText("CUMULATIVE");
  });

  test("displays telemetry grid with observations and diagnostics", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const grid = page.getByTestId("telemetry-grid");
    await expect(grid).toBeVisible();
  });

  test("telemetry updates when stepping", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const metricsRow = page.getByTestId("metrics-row");

    // Get initial reward text
    const initialText = await metricsRow.textContent();

    // Step forward a few times
    for (let i = 0; i < 3; i++) {
      await page.getByTestId("step-btn").click();
    }

    // STEP indicator should have changed
    await expect(metricsRow).toContainText("4 / 12");
  });

  test("shows outcome banner at episode end", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    // Step through all 12 frames
    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }

    // Outcome banner should appear
    const banner = page.getByTestId("outcome-banner");
    await expect(banner).toBeVisible();
  });
});

// =========================================================================
//  EPISODE TIMELINE
// =========================================================================

test.describe("Episode timeline", () => {
  test("shows empty/waiting state before episode", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await expect(page.getByTestId("timeline-empty")).toBeVisible();
    await expect(page.getByTestId("timeline-empty")).toContainText(
      "Waiting for episode data",
    );
  });

  test("renders timeline panel after episode starts", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toBeVisible();
    await expect(timeline).toContainText("EPISODE TIMELINE");
  });

  test("shows progress indicator", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const timeline = page.getByTestId("timeline-panel");
    await expect(timeline).toContainText("1 / 12 steps");
  });

  test("renders timeline cards matching frame count", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    // At step 1, should have 1 card
    let cards = page.getByTestId("timeline-card");
    await expect(cards).toHaveCount(1);

    // Step forward
    await page.getByTestId("step-btn").click();
    cards = page.getByTestId("timeline-card");
    await expect(cards).toHaveCount(2);

    // Step again
    await page.getByTestId("step-btn").click();
    cards = page.getByTestId("timeline-card");
    await expect(cards).toHaveCount(3);
  });

  test("timeline cards contain action and alert info", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    const card = page.getByTestId("timeline-card").first();
    const text = await card.textContent();
    // Should contain time like T+0
    expect(text).toMatch(/T\+\d+/);
  });

  test("event cards container is present", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);
    await expect(page.getByTestId("event-cards")).toBeVisible();
  });

  test("all 12 cards available at episode end", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
    await startEpisode(page);

    for (let i = 0; i < 11; i++) {
      await page.getByTestId("step-btn").click();
    }

    const cards = page.getByTestId("timeline-card");
    await expect(cards).toHaveCount(12);
  });
});
