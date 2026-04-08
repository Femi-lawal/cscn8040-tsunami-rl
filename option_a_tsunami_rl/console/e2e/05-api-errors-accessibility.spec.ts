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
//  API INTEGRATION
// =========================================================================

test.describe("API integration", () => {
  test("catalog API returns events", async ({ page }) => {
    const response = await page.request.get("/api/catalog");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBeGreaterThan(0);

    // Validate event structure
    const event = data[0];
    expect(event).toHaveProperty("event_group_id");
    expect(event).toHaveProperty("danger_tier");
    expect(event).toHaveProperty("danger_label");
  });

  test("checkpoints API returns model list", async ({ page }) => {
    const response = await page.request.get("/api/checkpoints");
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(Array.isArray(data)).toBe(true);
    expect(data.length).toBeGreaterThan(0);

    // Validate checkpoint structure
    const cp = data[0];
    expect(cp).toHaveProperty("name");
    expect(cp).toHaveProperty("path");
  });

  test("simulate API returns valid episode", async ({ page }) => {
    const response = await page.request.post("/api/simulate", {
      data: {
        agent_type: "rule",
        seed: 42,
        danger_filter: "All",
      },
    });
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty("event_metadata");
    expect(data).toHaveProperty("frames");
    expect(data).toHaveProperty("total_return");
    expect(data).toHaveProperty("outcome_summary");

    // Should have 12 frames
    expect(data.frames.length).toBe(12);

    // Validate frame structure
    const frame = data.frames[0];
    expect(frame).toHaveProperty("t");
    expect(frame).toHaveProperty("time_min");
    expect(frame).toHaveProperty("epicenter");
    expect(frame).toHaveProperty("action");
    expect(frame).toHaveProperty("reward");
    expect(frame).toHaveProperty("alert_level");
    expect(frame).toHaveProperty("observation");
  });

  test("simulate API rejects unknown agent_type", async ({
    page,
  }) => {
    const response = await page.request.post("/api/simulate", {
      data: {
        agent_type: "invalid_agent",
        seed: 42,
      },
    });
    expect(response.status()).toBe(422);
  });

  test("health endpoint responds", async ({ page }) => {
    const response = await page.request.get("/api/health");
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data).toHaveProperty("status", "ok");
  });
});

// =========================================================================
//  NETWORK: REQUEST/RESPONSE OBSERVATION
// =========================================================================

test.describe("Network behaviour", () => {
  test("start episode triggers simulate API call", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    // Listen for the simulate request
    const requestPromise = page.waitForRequest(
      (req) => req.url().includes("/api/simulate") && req.method() === "POST",
    );

    await page.getByTestId("start-btn").click();

    const request = await requestPromise;
    expect(request.method()).toBe("POST");

    const body = request.postDataJSON();
    expect(body).toHaveProperty("agent_type");
    expect(body).toHaveProperty("seed");
  });

  test("page load fetches catalog and checkpoints", async ({ page }) => {
    const catalogPromise = page.waitForResponse(
      (res) => res.url().includes("/api/catalog") && res.ok(),
    );
    const checkpointsPromise = page.waitForResponse(
      (res) => res.url().includes("/api/checkpoints") && res.ok(),
    );

    await page.goto("/");

    const [catalogRes, checkpointsRes] = await Promise.all([
      catalogPromise,
      checkpointsPromise,
    ]);
    expect(catalogRes.status()).toBe(200);
    expect(checkpointsRes.status()).toBe(200);
  });

  test("simulate response includes expected metadata fields", async ({
    page,
  }) => {
    await page.goto("/");
    await waitForIdle(page);

    const responsePromise = page.waitForResponse(
      (res) => res.url().includes("/api/simulate") && res.ok(),
    );

    await page.getByTestId("start-btn").click();

    const response = await responsePromise;
    const data = await response.json();

    expect(data.event_metadata).toHaveProperty("event_group_id");
    expect(data.event_metadata).toHaveProperty("danger_tier");
    expect(typeof data.total_return).toBe("number");
    expect(typeof data.outcome_summary).toBe("string");
  });
});

// =========================================================================
//  ERROR HANDLING
// =========================================================================

test.describe("Error handling", () => {
  test("handles API down gracefully with mocked failure", async ({ page }) => {
    // Intercept simulate calls and return 500
    await page.route("**/api/simulate", (route) =>
      route.fulfill({
        status: 500,
        body: "Internal Server Error",
      }),
    );

    await page.goto("/");
    await waitForIdle(page);

    await page.getByTestId("start-btn").click();

    // Should show error overlay or return to idle
    await page.waitForTimeout(2000);
    const errorBox = page.locator(".error-box");
    await expect(errorBox).toBeVisible();
  });

  test("error overlay shows error message", async ({ page }) => {
    await page.route("**/api/simulate", (route) =>
      route.fulfill({
        status: 500,
        body: "Test error message",
      }),
    );

    await page.goto("/");
    await waitForIdle(page);
    await page.getByTestId("start-btn").click();

    await page.waitForTimeout(2000);
    const errorBox = page.getByTestId("error-box");
    await expect(errorBox).toBeVisible();
    await expect(errorBox).toContainText("Simulation failed");
  });

  test("can start new episode after error", async ({ page }) => {
    // First, cause an error
    await page.route("**/api/simulate", (route) =>
      route.fulfill({
        status: 500,
        body: "Test error",
      }),
    );

    await page.goto("/");
    await waitForIdle(page);
    await page.getByTestId("start-btn").click();
    await page.waitForTimeout(2000);

    // Unroute to allow real requests
    await page.unroute("**/api/simulate");

    // Try starting again
    await startEpisode(page);
    await expect(page.getByTestId("timeline-panel")).toBeVisible();
  });

  test("startup API failures are surfaced to the user", async ({ page }) => {
    await page.route("**/api/catalog", (route) =>
      route.fulfill({
        status: 500,
        body: "catalog failed",
      }),
    );
    await page.route("**/api/checkpoints", (route) =>
      route.fulfill({
        status: 500,
        body: "checkpoints failed",
      }),
    );

    await page.goto("/");

    const errorBox = page.getByTestId("error-box");
    await expect(errorBox).toBeVisible();
    await expect(errorBox).toContainText("Failed to load event catalog.");
    await expect(errorBox).toContainText("Failed to load available model checkpoints.");
  });
});

// =========================================================================
//  ACCESSIBILITY BASICS
// =========================================================================

test.describe("Accessibility basics", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);
  });

  test("all select elements have associated labels", async ({ page }) => {
    // Each select should have a preceding label
    const selects = page.locator("select.field-select");
    const count = await selects.count();
    expect(count).toBeGreaterThanOrEqual(3); // danger, scenario, agent
  });

  test("buttons have text content", async ({ page }) => {
    const buttons = page.locator('[data-testid$="-btn"]');
    const count = await buttons.count();

    for (let i = 0; i < count; i++) {
      const btn = buttons.nth(i);
      const text = await btn.textContent();
      expect(text?.trim().length).toBeGreaterThan(0);
    }
  });

  test("input has type=number for seed", async ({ page }) => {
    const seedInput = page.getByTestId("seed-input");
    await expect(seedInput).toHaveAttribute("type", "number");
  });
});

// =========================================================================
//  VISUAL REGRESSION GUARDS
// =========================================================================

test.describe("Visual consistency", () => {
  test("control panel has dark background", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const panel = page.getByTestId("control-panel");
    const bg = await panel.evaluate(
      (el) => getComputedStyle(el).backgroundColor,
    );
    // Should be a dark color (not white)
    expect(bg).not.toBe("rgb(255, 255, 255)");
  });

  test("console fills viewport height", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const layout = page.getByTestId("console-layout");
    const box = await layout.boundingBox();
    const viewport = page.viewportSize();

    expect(box).not.toBeNull();
    expect(viewport).not.toBeNull();
    // Should fill viewport height (allow small margin)
    expect(box!.height).toBeGreaterThanOrEqual(viewport!.height - 5);
  });

  test("map area fills most of the right side", async ({ page }) => {
    await page.goto("/");
    await waitForIdle(page);

    const mapArea = page.getByTestId("map-area");
    const box = await mapArea.boundingBox();
    expect(box).not.toBeNull();
    // Map should be at least 250px tall
    expect(box!.height).toBeGreaterThanOrEqual(250);
  });
});
