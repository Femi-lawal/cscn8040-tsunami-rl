# ---------- Stage 1: Build the Next.js standalone bundle ----------
FROM node:22-alpine AS console-build

WORKDIR /build/console
COPY console/package.json console/package-lock.json* ./
RUN npm ci --ignore-scripts

COPY console/ ./
ENV API_URL=http://localhost:8000
RUN npm run build -- --webpack


# ---------- Stage 2: Python API + Next.js runner ----------
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl nodejs && \
    rm -rf /var/lib/apt/lists/*

# Layout: /workspace/option_a_tsunami_rl/  (mirrors the repo structure)
WORKDIR /workspace/option_a_tsunami_rl

# Python dependencies (CPU-only torch to keep the image smaller)
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt fastapi uvicorn[standard]

# Copy project source and data
COPY __init__.py ./
COPY src/ src/
COPY api/ api/
COPY data/processed/ data/processed/
COPY outputs/models/ outputs/models/

# Copy Next.js standalone output
COPY --from=console-build /build/console/.next/standalone/ /workspace/option_a_tsunami_rl/console/
COPY --from=console-build /build/console/.next/static /workspace/option_a_tsunami_rl/console/.next/static

# Entrypoint
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 3000 8000

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

CMD ["/workspace/option_a_tsunami_rl/docker-entrypoint.sh"]
