# Traductor LSN — Lengua de Señas Nicaragüense

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Security](https://img.shields.io/badge/security-policy-blue.svg)](SECURITY.md)

Red neuronal LSTM que traduce **Lengua de Señas Nicaragüense (LSN)** a texto y voz. Detecta keypoints con **MediaPipe Holistic**, entrena con **TensorFlow/Keras** y corre también 100% en el navegador con **TensorFlow.js**.

- Web pública: <https://lenguaje.chepeonline.com>
- Repositorio: `git@github.com:Bymatt10/lenguaje-sena-nicaragua.git`

## Arquitectura

```
┌──────────────┐    WebRTC    ┌────────────────┐
│   Cámara     │─────────────▶│  Navegador     │
└──────────────┘              │  (TFJS +       │
                              │   MediaPipe)   │
                              └───────┬────────┘
                                      │ POST /contribute
                                      ▼
                            ┌─────────────────────┐
                            │  Flask + gunicorn   │
                            │  (rate-limit, auth, │
                            │   validación)       │
                            └───────┬─────────────┘
                                    ▼
                            ┌─────────────────────┐
                            │  dataset_contrib/   │
                            │  (30 días, .gitignore)│
                            └───────┬─────────────┘
                                    │ cron
                                    ▼
                            ┌─────────────────────┐
                            │  auto_retrain.py    │
                            │  (LOSO quality lock)│
                            └─────────────────────┘
```

## Privacidad

- La cámara **nunca se sube** al servidor. La inferencia corre en el navegador.
- Las muestras de feedback/contribución son keypoints numéricos de **manos + pose** (los puntos del rostro se descartan antes de transmitir).
- Las muestras se almacenan máximo **30 días** (configurable con `LSN_RETENTION_DAYS`) y se eliminan automáticamente.
- Ver [`SECURITY.md`](SECURITY.md) para la política completa.

## Estructura del proyecto

```
.
├── server.py                 # Flask endurecido (rate-limit, auth, validación)
├── validate_contrib.py       # Esquema estricto de /contribute
├── logging_config.py         # Logs rotando en logs/security.log
├── auto_retrain.py           # Reentrenamiento con candado LOSO + cuarentena
├── constants.py              # Hiperparámetros
├── helpers.py                # Utilidades de MediaPipe + keypoints
├── model.py                  # Arquitectura LSTM
├── train_v2.py               # Entrenamiento
├── evaluate_model.py         # Inferencia local (GUI / consola)
├── evaluate_loso.py          # Validación Leave-One-Subject-Out
├── main.py                   # GUI PyQt5
├── requirements.txt          # Deps producción (pinned)
├── requirements-dev.txt      # Deps desarrollo (pip-audit, linters)
├── gunicorn.conf.py          # WSGI production server
├── .env.example              # Variables de entorno
├── Dockerfile                # Backend Flask multi-stage
├── docker-compose.yml        # nginx + app + cron
├── Jenkinsfile               # CI/CD: build, push, deploy
├── .dockerignore             # Exclusiones para build
├── models/                   # Modelos .keras + MODEL_HASHES.txt
├── web/                      # Frontend (HTML/CSS/JS + TFJS)
├── docker/
│   ├── nginx/
│   │   ├── Dockerfile
│   │   └── nginx.conf        # Reverse proxy con cabeceras de seguridad
│   └── cron/
│       ├── Dockerfile
│       └── crontab.txt       # auto_retrain + purge + validate
├── deploy/
│   ├── nginx.conf            # Reverse proxy sin Docker (systemd)
│   ├── systemd/
│   │   └── lengua-lsp.service # Servicio systemd endurecido
│   └── DOCKER.md             # Guía de despliegue con Docker
├── tools/
│   ├── audit_deps.sh         # pip-audit
│   ├── purge_dataset_contrib.py # Limpieza por antigüedad
│   └── validate_contrib_folder.py # Validación manual de dataset_contrib/
├── SECURITY.md               # Política de seguridad
├── CONTRIBUTING.md           # Cómo contribuir
├── LICENSE                   # Apache 2.0
└── NOTICE                    # Aviso de licencia (Apache 2.0)
```

## SCRIPTS PRINCIPALES (uso local / GUI)

- `capture_samples.py` → captura muestras y las guarda en `frame_actions/`.
- `normalize_samples.py` → normaliza muestras a la misma cantidad de frames.
- `create_keypoints.py` → genera los `.h5` por palabra para entrenar.
- `training_model.py` / `train_v2.py` → entrena la LSTM.
- `evaluate_model.py` → prueba el modelo desde un video o webcam.
- `main.py` → GUI PyQt5 del traductor.

## Pipeline web (captura → reentrenamiento)

1. Usuario entra a `https://lenguaje.chepeonline.com` y hace "Contribuir muestras".
2. La web obtiene token de sesión (`GET /session`) firmado HttpOnly+SameSite+Lax+Secure.
3. Captura keypoints de manos+pose, **descarta los del rostro**, los empaqueta y envía (`POST /contribute`) con `X-LSN-Token`.
4. El servidor valida esquema, rate-limit (30/min), tamaño (≤512 KB persistido) y persiste en `dataset_contrib/`.
5. `auto_retrain.py` corre como cron, valida todo `dataset_contrib/`, mueve inválidos a `quarantine/`, mide LOSO, y reentrena+sube a `web/model/` **sólo si el LOSO no cae**.

## Despliegue (VPS con nginx + HTTPS)

### Opción A: Docker + Jenkins (recomendado)

```bash
# VPS, primera vez
sudo apt install docker.io docker-compose-plugin certbot
sudo mkdir -p /srv/lengua-lsp/letsencrypt
sudo certbot certonly --standalone -d lenguaje.chepeonline.com
sudo cp -RL /etc/letsencrypt /srv/lengua-lsp/letsencrypt/
cd /srv/lengua-lsp && git clone git@github.com:Bymatt10/lenguaje-sena-nicaragua.git .
cp .env.example .env && sed -i "s|replace-with-openssl-rand-hex-32|$(openssl rand -hex 32)|" .env && chmod 600 .env
docker compose up -d
docker compose ps
curl -fsS https://lenguaje.chepeonline.com/health
```

El `Jenkinsfile` en la raíz hace: lint → audit deps → build 3 imágenes → push a `ghcr.io/bymatt10/lengua-lsp/{app,nginx,cron}` → deploy vía SSH al VPS → smoke test en `/health`. Credenciales Jenkins necesarias: `github-container-registry` (PAT) y `vps-ssh-key`.

Documentación detallada: [`deploy/DOCKER.md`](deploy/DOCKER.md).

### Opción B: Bare-metal (systemd)

### 1. Clonar

```bash
git clone git@github.com:Bymatt10/lenguaje-sena-nicaragua.git /var/www/lengua-lsp
cd /var/www/lengua-lsp
```

### 2. Entorno virtual

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar entorno

```bash
cp .env.example .env
sed -i "s|replace-with-openssl-rand-hex-32|$(openssl rand -hex 32)|" .env
chmod 600 .env
```

### 4. Copiar frontend al docroot de nginx

```bash
sudo mkdir -p /var/www/lengua-lsp/web
sudo rsync -a web/ /var/www/lengua-lsp/web/
```

### 5. TLS con Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d lenguaje.chepeonline.com
```

### 6. nginx

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/lengua-lsp.conf
sudo ln -sf /etc/nginx/sites-available/lengua-lsp.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 7. Servicio systemd

```bash
sudo cp deploy/systemd/lengua-lsp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lengua-lsp
sudo systemctl status lengua-lsp
```

### 8. Cron de limpieza

```cron
0 3 * * * /var/www/lengua-lsp/.venv/bin/python /var/www/lengua-lsp/tools/purge_dataset_contrib.py >> /var/log/lengua-lsp.log 2>&1
```

### 9. (Opcional) Reentrenamiento automático

```cron
30 3 * * * /var/www/lengua-lsp/.venv/bin/python /var/www/lengua-lsp/auto_retrain.py >> /var/log/lengua-lsp.log 2>&1
```

### 10. Auditoría de dependencias

```bash
pip install -r requirements-dev.txt
./tools/audit_deps.sh
```

## Desarrollo local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# servidor de desarrollo
FLASK_ENV=development LSN_SECRET=dev-secret .venv/bin/python server.py

# frontend: abrir web/index.html directamente o servirlo con
python3 -m http.server 8000 --directory web
```

## Cómo añadir una nueva seña

1. Graba muestras (`capture_samples.py`, 30+ por palabra).
2. Normaliza (`normalize_samples.py`) y crea keypoints (`create_keypoints.py`).
3. Reentrena (`train_v2.py`) y mide LOSO (`evaluate_loso.py`).
4. Si el LOSO no cae → push a `models/actions_<N>.keras` + actualiza `models/MODEL_HASHES.txt` con el nuevo SHA-256.
5. Ejecuta `auto_retrain.py` para desplegar a `web/model/`.

## Licencia

[Apache License 2.0](LICENSE). Copyright 2026 bymatt.

## Contribuir

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md). Las contribuciones de código requieren un `Signed-off-by` (DCO). Las contribuciones de muestras están sujetas a la política de privacidad en [`SECURITY.md`](SECURITY.md).

## Reportar vulnerabilidades

**No abras un issue público.** Sigue el procedimiento en [`SECURITY.md`](SECURITY.md).