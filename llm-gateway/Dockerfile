FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY llm_gateway ./llm_gateway
COPY providers.yaml ./

RUN pip install --no-cache-dir . \
    && mkdir -p /app/data \
    && useradd --system --uid 10001 --home /app appuser \
    && chown -R appuser:appuser /app

USER appuser

ENV GATEWAY_CONFIG=/app/providers.yaml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1

CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
