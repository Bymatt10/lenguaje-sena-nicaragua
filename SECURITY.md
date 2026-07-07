# Política de seguridad

## Datos personales y privacidad

El proyecto **no publica datos biométricos** (videos, keypoints de voluntarios) en este repositorio. Todo lo recibido en `dataset_contrib/` se trata como dato personal sensible.

- Los voluntarios pueden borrar sus muestras en cualquier momento desde la web (botón "Borrar mis datos") o enviando un correo al mantenedor.
- Las muestras se eliminan automáticamente al cabo de **30 días** (`LSN_RETENTION_DAYS`).
- La web **elimina los keypoints del rostro** antes de transmitir. Sólo se almacenan pose + manos.
- Los videos de muestra en `dataset_videos/` son del equipo del proyecto y no se distribuyen públicamente.

## Reporte de vulnerabilidades

Por favor **no abras un issue público**. Escribe un correo a **bymatt** (usuario de GitHub) con:

- Descripción del problema y pasos para reproducirlo.
- Impacto estimado.
- (Opcional) Parche o sugerencia.

Tiempo de respuesta objetivo: **7 días**. Las vulnerabilidades confirmadas se divulgan tras parchearse.

## Hardening aplicado

| Capa | Medida |
|---|---|
| Transporte | TLS 1.2+ detrás de nginx (HSTS, HSTS preload ready) |
| Aplicación | Flask 3 + gunicorn, `debug=False`, `MAX_CONTENT_LENGTH=1MB` |
| Auth | Token de sesión firmado (`itsdangerous`) HttpOnly + SameSite=Lax + Secure |
| Rate-limit | 30/min y 500/h por IP en `/contribute` (flask-limiter, memory backend) |
| Validación | Esquema estricto: palabras en allowlist, valores numéricos en `[-10, 10]`, sin NaN/Inf, sin path traversal |
| CORS | Eliminado: web servida desde mismo origen (`lenguaje.chepeonline.com`) |
| Modelos | SHA-256 de los `.keras` y `weights.bin` registrados para detectar reemplazos |
| Dependencias | Versiones pinneadas (`==`), `pip-audit` recomendado en CI |
| Sistema | Servicio systemd con `NoNewPrivileges`, `ProtectSystem=strict`, `ReadWritePaths` mínimos |
| Logs | `logs/security.log` rotando, sin PII en logs |

## Historial de avisos

| Fecha | CVE | Severidad | Descripción | Parche |
|---|---|---|---|---|
| _ninguno publicado_ | | | | |

## Alcance

Están en alcance:

- `server.py` y sus rutas (`/`, `/health`, `/session`, `/contribute`, `/feedback/<token>`).
- La aplicación web en `web/` (`index.html`, `app.js`).
- El pipeline de auto-reentrenamiento (`auto_retrain.py`).

No está en alcance:

- Vulnerabilidades reportadas en versiones de `tensorflow`, `mediapipe`, `flask`, `opencv` y otras dependencias (reportar upstream).
- Denegación de servicio por volumen extremo de tráfico (mitigar con un WAF/CDN).