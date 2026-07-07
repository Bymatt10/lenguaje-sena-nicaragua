FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/app/tools:/usr/local/bin:$PATH \
    LSN_RETENTION_DAYS=30

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tini \
    dcron \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r -g 10002 cron \
    && useradd -r -u 10002 -g cron -d /app -s /usr/sbin/nologin cron

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=root:cron constants.py helpers.py validate_contrib.py ./
COPY --chown=root:cron tools/ tools/
COPY --chown=root:cron auto_retrain.py train_v2.py evaluate_loso.py ./

RUN mkdir -p /app/dataset_contrib /app/logs /var/spool/cron/crontabs \
    && chown -R cron:cron /app \
    && touch /var/spool/cron/crontabs/cron \
    && chown cron:cron /var/spool/cron/crontabs/cron

COPY --chown=root:cron docker/cron/crontab.txt /etc/cron.d/lsp
RUN chmod 0644 /etc/cron.d/lsp && crontab -u cron /etc/cron.d/lsp

USER cron

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["cron", "-f"]