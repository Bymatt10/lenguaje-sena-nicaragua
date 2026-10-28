FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
# solo lo que usa server.py: la inferencia corre en el navegador
COPY requirements-api.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements-api.txt

FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LSN_RETENTION_DAYS=30 \
    GUNICORN_WORKERS=2 \
    GUNICORN_BIND=0.0.0.0:5000 \
    FLASK_ENV=production

RUN apt-get update && apt-get install --no-install-recommends -y tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r -g 10001 lsp \
    && useradd -r -u 10001 -g lsp -d /app -s /usr/sbin/nologin lsp

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=root:lsp gunicorn.conf.py server.py validate_contrib.py logging_config.py ./
COPY --chown=root:lsp tools/purge_dataset_contrib.py tools/validate_contrib_folder.py tools/

RUN mkdir -p /app/dataset_contrib /app/logs /app/tmp \
    && chown -R lsp:lsp /app

USER lsp

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "-c", "gunicorn.conf.py", "server:app"]
