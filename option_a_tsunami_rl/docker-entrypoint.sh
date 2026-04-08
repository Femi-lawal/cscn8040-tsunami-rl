#!/bin/sh
set -e

# Start FastAPI backend in the background
cd /workspace
python -m uvicorn option_a_tsunami_rl.api.server:app \
  --host 0.0.0.0 --port 8000 &

# Start Next.js standalone server in the foreground
cd /workspace/option_a_tsunami_rl/console
PORT=3000 HOSTNAME=0.0.0.0 node server.js
