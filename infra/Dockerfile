# Single image for the API and the worker (the dashboard is built in a Node stage).
FROM node:22-slim AS web
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/apps
COPY pyproject.toml ./
COPY src/ src/
COPY apps/api/ apps/api/
COPY apps/worker/ apps/worker/
COPY configs/ configs/
COPY third_party/ third_party/
RUN pip install --no-cache-dir ".[postgres]"
COPY --from=web /web/dist apps/web/dist
# Research data and model artifacts are mounted at runtime (never baked into the image).
VOLUME ["/app/data", "/app/artifacts"]
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
