import { test, expect, type Page, type Locator } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Wait for the idle overlay to be visible – signals app has fully rendered. */
async function waitForIdle(page: Page) {
  await page.waitForSelector('[data-testid="idle-overlay"]', {
    timeout: 15_000,
  });
}

/** Click Start Episode and wait for the simulation to load (idle overlay disappears). */
async function startEpisode(page: Page) {
  await page.getByTestId("start-btn").click();
  // Wait for loading state to finish – idle overlay goes away and
  // timeline-panel appears.
  await page.waitForSelector('[data-testid="timeline-panel"]', {
    timeout: 30_000,
  });
}

// =========================================================================
//  1 ─ PAGE LOAD & LAYOUT
// =========================================================================

test.describe("Page load & layout", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("renders the console layout container", async ({ page }) => {
    const layout = page.getByTestId("console-layout");
    await expect(layout).toBeVisible();
  });

  test("renders the control panel sidebar", async ({ page }) => {
    const panel = page.getByTestId("control-panel");
    await expect(panel).toBeVisible();
    // Width should be around 340px
    const box = await panel.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(300);
    expect(box!.width).toBeLessThanOrEqual(400);
  });

  test("renders the map area", async ({ page }) => {
    const mapArea = page.getByTestId("map-area");
    await expect(mapArea).toBeVisible();
  });

  test("shows idle overlay with instructions", async ({ page }) => {
    const overlay = page.getByTestId("idle-overlay");
    await expect(overlay).toBeVisible();
    await expect(overlay).toContainText("Tsunami Warning RL Console");
    await expect(overlay).toContainText("Configure an episode");
  });

  test("does not show timeline or telemetry before an episode starts", async ({
    page,
  }) => {
    // The timeline should show the empty state
    await expect(page.getByTestId("timeline-panel")).not.toBeVisible();
    // Telemetry should be in empty state
    await expect(
      page.getByText("Start an episode to see telemetry data"),
    ).toBeVisible();
  });

  test("page title contains app name", async ({ page }) => {
    await expect(page).toHaveTitle(/Tsunami/i);
  });
});

// =========================================================================
//  2 ─ CONTROL PANEL: RENDERING & DEFAULTS
// =========================================================================

test.describe("Control panel rendering", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("danger filter defaults to 'All'", async ({ page }) => {
    const select = page.getByTestId("danger-filter-select");
    await expect(select).toHaveValue("All");
  });

  test("lists all four threat-filter options", async ({ page }) => {
    const select = page.getByTestId("danger-filter-select");
    const options = select.locator("option");
    await expect(options).toHaveCount(4);
    await expect(options.nth(0)).toHaveText("All Events");
    await expect(options.nth(1)).toHaveText("No Threat");
    await expect(options.nth(2)).toHaveText("Potential Threat");
    await expect(options.nth(3)).toHaveText("Confirmed Threat");
  });

  test("scenario selector defaults to random", async ({ page }) => {
    const select = page.getByTestId("scenario-select");
    await expect(select).toHaveValue("random");
  });

  test("scenario selector has catalog events loaded", async ({ page }) => {
    const select = page.getByTestId("scenario-select");
    // Wait for the catalog to load (async fetch on mount)
    await expect(select.locator("option")).not.toHaveCount(1, {
      timeout: 10_000,
    });
    const count = await select.locator("option").count();
    expect(count).toBeGreaterThan(1);
  });

  test("agent type defaults to ppo", async ({ page }) => {
    const select = page.getByTestId("agent-type-select");
    await expect(select).toHaveValue("ppo");
  });

  test("agent type has ppo and rule options", async ({ page }) => {
    const select = page.getByTestId("agent-type-select");
    const options = select.locator("option");
    await expect(options).toHaveCount(2);
  });

  test("checkpoint selector is visible when ppo is selected", async ({
    page,
  }) => {
    await expect(page.getByTestId("checkpoint-select")).toBeVisible();
  });

  test("seed input defaults to 42", async ({ page }) => {
    const seedInput = page.getByTestId("seed-input");
    await expect(seedInput).toHaveValue("42");
  });

  test("start button is visible and enabled in idle state", async ({
    page,
  }) => {
    const btn = page.getByTestId("start-btn");
    await expect(btn).toBeVisible();
    await expect(btn).toBeEnabled();
    await expect(btn).toContainText("Start Episode");
  });

  test("play/pause/step/reset buttons are disabled in idle state", async ({
    page,
  }) => {
    await expect(page.getByTestId("play-btn")).toBeDisabled();
    await expect(page.getByTestId("pause-btn")).toBeDisabled();
    await expect(page.getByTestId("step-btn")).toBeDisabled();
    await expect(page.getByTestId("reset-btn")).toBeDisabled();
  });
});

// =========================================================================
//  3 ─ CONTROL PANEL: INTERACTIONS
// =========================================================================

test.describe("Control panel interactions", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("changing danger filter updates filter value", async ({ page }) => {
    const select = page.getByTestId("danger-filter-select");
    await select.selectOption("Confirmed Threat");
    await expect(select).toHaveValue("Confirmed Threat");
  });

  test("changing danger filter filters scenario dropdown", async ({ page }) => {
    const dangerFilter = page.getByTestId("danger-filter-select");
    const scenarioSelect = page.getByTestId("scenario-select");

    // Wait for catalog to load first
    await expect(scenarioSelect.locator("option")).not.toHaveCount(1, {
      timeout: 10_000,
    });

    // Count with All
    await dangerFilter.selectOption("All");
    await page.waitForTimeout(300);
    const allCount = await scenarioSelect.locator("option").count();

    // Filter to confirmed threat
    await dangerFilter.selectOption("Confirmed Threat");
    await page.waitForTimeout(300);
    const confirmedCount = await scenarioSelect.locator("option").count();

    // Confirmed should have fewer than all (but at least 1 for random)
    expect(confirmedCount).toBeLessThanOrEqual(allCount);
    expect(confirmedCount).toBeGreaterThanOrEqual(1);
  });

  test("switching agent type to rule hides checkpoint selector", async ({
    page,
  }) => {
    await page.getByTestId("agent-type-select").selectOption("rule");
    await expect(page.getByTestId("checkpoint-select")).not.toBeVisible();
  });

  test("switching back to ppo shows checkpoint selector", async ({ page }) => {
    await page.getByTestId("agent-type-select").selectOption("rule");
    await expect(page.getByTestId("checkpoint-select")).not.toBeVisible();
    await page.getByTestId("agent-type-select").selectOption("ppo");
    await expect(page.getByTestId("checkpoint-select")).toBeVisible();
  });

  test("seed input accepts custom values", async ({ page }) => {
    const seedInput = page.getByTestId("seed-input");
    await seedInput.fill("123");
    await expect(seedInput).toHaveValue("123");
  });

  test("seed input accepts zero", async ({ page }) => {
    const seedInput = page.getByTestId("seed-input");
    await seedInput.fill("0");
    await expect(seedInput).toHaveValue("0");
  });
});
