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

## Cómo correr el proyecto

### 1. Probar la web (sin backend, solo el traductor en el navegador)

El frontend hace toda la inferencia con TFJS + MediaPipe en el navegador, así que puedes abrirlo sin servidor.

```bash
git clone git@github.com:Bymatt10/lenguaje-sena-nicaragua.git
cd lenguaje-sena-nicaragua
python3 -m http.server 8000 --directory web
# Abre http://localhost:8000 en Chrome/Edge (HTTPS no es necesario para getUserMedia en localhost)
```

> ⚠️ La cámara puede no funcionar si abres `file://` directamente en algunos navegadores. Usa siempre `http://localhost`.

### 2. Correr backend + web juntos (desarrollo local)

Necesitas dos procesos: Flask en `:5000` y la web estática en `:8000`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Genera un secret aleatorio (cualquier valor sirve en dev)
export LSN_SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
export FLASK_ENV=development

# Terminal 1: backend Flask (incluye /contribute, /session, /feedback)
python server.py

# Terminal 2: frontend estático
python -m http.server 8000 --directory web
```

La web en `http://localhost:8000` automáticamente hará `fetch('http://localhost:5000/contribute')` que, por la política same-origin con CORS abierto en modo dev, funcionará.

> En producción ambos servicios los sirve nginx desde el mismo dominio (`https://lenguaje.chepeonline.com`), por eso no hay CORS.

### 3. Todo con Docker (recomendado para producción)

```bash
cp .env.example .env
sed -i "s|replace-with-openssl-rand-hex-32|$(openssl rand -hex 32)|" .env
chmod 600 .env

docker compose up -d --build
docker compose ps
curl -fsS http://localhost/health
# Abre http://localhost en el navegador
```

Para HTTPS real con Let's Encrypt, ver la sección **Despliegue → Opción A** más abajo.

### 4. Probar un script individual (sin servidor)

```bash
source .venv/bin/activate

# Capturar muestras con la webcam (genera frame_actions/<palabra>/<n>.jpg)
python capture_samples.py

# Normalizar muestras a la misma cantidad de frames
python normalize_samples.py

# Convertir muestras a .h5 (keypoints) por palabra
python create_keypoints.py

# Entrenar el modelo
python train_v2.py

# Medir calidad con Leave-One-Subject-Out (debe dar >85% para desplegar)
python evaluate_loso.py

# Probar el modelo con la webcam (interfaz OpenCV, sin servidor)
python evaluate_model.py

# Lanzar la GUI PyQt5
python main.py
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

Pipeline para entrenar el modelo desde cero, en orden:

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `capture_samples.py` | Captura fotogramas de la webcam por cada palabra → `frame_actions/<palabra>/<n>.jpg` |
| 2 | `normalize_samples.py` | Normaliza todas las muestras a la misma cantidad de frames |
| 3 | `create_keypoints.py` | Extrae los 1662 keypoints por frame con MediaPipe → `data/keypoints/*.h5` |
| 4 | `train_v2.py` (o `training_model.py`) | Entrena la LSTM → `models/actions_v2.keras` |
| 5 | `evaluate_loso.py` | Mide calidad con Leave-One-Subject-Out (objetivo: ≥85%) |
| 6 | `evaluate_model.py` | Prueba el modelo en vivo (consola, sin servidor) |
| 7 | `main.py` | GUI PyQt5 del traductor |

```bash
source .venv/bin/activate

python capture_samples.py    # seguir el menú interactivo
python normalize_samples.py
python create_keypoints.py
python train_v2.py
python evaluate_loso.py
python evaluate_model.py     # webcam en vivo
# o
python main.py               # GUI
```

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

Ver la sección **Cómo correr el proyecto → Opción 2** más arriba. Resumen rápido:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Genera LSN_SECRET aleatorio (en dev puede ser cualquiera)
export LSN_SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
export FLASK_ENV=development

# Terminal 1
python server.py

# Terminal 2
python -m http.server 8000 --directory web
```

### Verificar la instalación

```bash
# Backend responde
curl -fsS http://localhost:5000/health
# → {"ok":true,"service":"lsp-contrib","version":"2.0"}

# Token de sesión (necesario para POST /contribute)
curl -fsS -c /tmp/cookies http://localhost:5000/session
# → {"expires":"...","token":"..."}

# Validar el dataset actual
python tools/validate_contrib_folder.py dataset_contrib/

# Auditoría de vulnerabilidades en deps
./tools/audit_deps.sh
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