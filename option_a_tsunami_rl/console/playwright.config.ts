import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.BASE_URL || "http://localhost:3000";
const manageLocalServers = !process.env.BASE_URL;
const playwrightApiPort = 8010;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["html", { open: "never" }], ["list"]],
  timeout: 60_000,
  expect: { timeout: 15_000 },

  use: {
    baseURL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: manageLocalServers
    ? [
        {
          command:
            `python -m uvicorn option_a_tsunami_rl.api.server:app --host 127.0.0.1 --port ${playwrightApiPort}`,
          cwd: "../..",
          port: playwrightApiPort,
          timeout: 30_000,
          reuseExistingServer: true,
        },
        {
          command: `cmd /c \"set API_URL=http://127.0.0.1:${playwrightApiPort}&& npm run dev\"`,
          port: 3000,
          timeout: 30_000,
          reuseExistingServer: true,
        },
      ]
    : undefined,
});
