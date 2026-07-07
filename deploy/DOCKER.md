# Despliegue con Docker + Jenkins

## Arquitectura

```
                    ┌──────────────────────────────────┐
   Internet ────▶  │  nginx (80/443 + TLS)            │
                    │  - Sirve web/ estático           │
                    │  - Reverse proxy /contribute,    │
                    │    /session, /feedback, /health  │
                    │  - Cabeceras de seguridad        │
                    └──────────┬───────────────────────┘
                               │ lsp_internal
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌───────────────┐            ┌───────────────┐
        │ app           │            │ cron          │
        │ Flask +       │            │ dcron         │
        │ gunicorn      │            │ - purge (3am) │
        │ :5000         │            │ - retrain (3:30)│
        │ - /contribute │            │ - validate    │
        │ - /session    │            │   (cada 15m)  │
        │ - /feedback   │            └───────────────┘
        └───────┬───────┘
                │
        ┌───────┴────────┬────────────┐
        ▼                ▼            ▼
   lsp_data         lsp_logs     lsp_www
   (samples)        (logs)       (certbot)
```

## Servicios

| Servicio | Imagen base | Puerto | Función |
|---|---|---|---|
| `nginx` | `nginx:1.27-alpine` | 80, 443 | TLS, estáticos, reverse proxy |
| `app` | `python:3.10-slim-bookworm` | 5000 (interno) | Flask + gunicorn |
| `cron` | `python:3.10-slim-bookworm` + dcron | — | auto_retrain + purge |

## Variables de entorno

Crea un `.env` en el directorio del compose (NUNCA en la imagen):

```dotenv
TAG=latest
LSN_SECRET=<openssl rand -hex 32>
LSN_SESSION_TTL=86400
LSN_RETENTION_DAYS=30
GUNICORN_WORKERS=2
```

```bash
chmod 600 .env
```

## Despliegue manual (sin Jenkins)

```bash
# 1. Clonar
git clone git@github.com:Bymatt10/lenguaje-sena-nicaragua.git /srv/lengua-lsp
cd /srv/lengua-lsp

# 2. Variables de entorno
cp .env.example .env
sed -i "s|replace-with-openssl-rand-hex-32|$(openssl rand -hex 32)|" .env
chmod 600 .env

# 3. TLS con Let's Encrypt (en el host, fuera de Docker)
sudo apt install certbot
sudo certbot certonly --webroot -w /var/www/certbot -d lenguaje.chepeonline.com
sudo mkdir -p /srv/lengua-lsp/letsencrypt
sudo cp -RL /etc/letsencrypt /srv/lengua-lsp/letsencrypt/
# Renovar automáticamente (cron del host):
echo "0 3 * * * certbot renew --webroot -w /var/www/certbot && cp -RL /etc/letsencrypt /srv/lengua-lsp/letsencrypt/ && cd /srv/lengua-lsp && docker compose restart nginx" | sudo crontab -

# 4. Levantar
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f app
```

## Jenkins

### Credenciales requeridas en Jenkins

| ID | Tipo | Uso |
|---|---|---|
| `github-container-registry` | Username/Password | Usuario PAT de GitHub con scope `write:packages` |
| `vps-ssh-key` | SSH Username/Password | Clave SSH del VPS (`${DEPLOY_ENV}@${host}`) |

### Stages del pipeline

1. **Checkout** — clona el repo y etiqueta por branch+commit.
2. **Lint** — `ruff` sobre el código Python + `docker compose config -q`.
3. **Audit deps** (sólo PRs) — `pip-audit`.
4. **Build** — tres imágenes en paralelo (`app`, `nginx`, `cron`).
5. **Push** — a `ghcr.io/bymatt10/lengua-lsp/{app,nginx,cron}` con tag + `:latest`.
6. **Deploy** — vía SSH al VPS, hace `docker compose pull && up -d`.
7. **Smoke test** — `GET /health` espera 200 con 5 reintentos.

### Disparadores

- **Push a `main`** → build + push + deploy a producción.
- **Pull request** → build + lint + audit (no deploy).
- **Tag `v*.*.*`** → build + push + deploy.
- **Manual** con parámetros `TAG`, `DEPLOY_ENV`, `DEPLOY`.

### Despliegue con tag manual

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Persistencia

| Volumen | Ruta contenedor | Backup recomendado |
|---|---|---|
| `lsp_dataset_contrib` | `/app/dataset_contrib` | semanal (contiene muestras) |
| `lsp_logs` | `/app/logs` | rotación interna (10 MB × 3) |
| `lsp_certbot_www` | `/var/www/certbot` | no necesario (regenerable) |

Backup sugerido:

```bash
docker run --rm \
    -v lsp_dataset_contrib:/data:ro \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/dataset_contrib-$(date +%F).tar.gz -C /data .
```

## Healthchecks

```bash
curl -fsS https://lenguaje.chepeonline.com/health
# {"ok":true,"service":"lsp-contrib","version":"2.0"}
```

## Hardening aplicado

- `no-new-privileges:true` en los tres servicios.
- `cap_drop: [ALL]` + capacidades mínimas necesarias.
- Lectura-escritura sólo en volúmenes nombrados.
- `tini` como PID 1 para recoger zombies.
- Usuarios no-root (UIDs 10001, 10002, 10003).
- Red interna `lsp_internal` (no accesible desde internet salvo vía nginx).
- `LSN_SECRET` nunca en la imagen — viene de `.env` del host.

## Actualizar el despliegue

```bash
cd /srv/lengua-lsp
docker compose pull
docker compose up -d
docker system prune -f
```

O disparar el pipeline en Jenkins sobre `main`.