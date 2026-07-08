#!/usr/bin/env bash
# Sirve la web del Traductor LSN y la expone con un túnel HTTPS de Cloudflare.
# Uso:  ./compartir.sh          (Ctrl+C para detener todo)
set -euo pipefail

PORT=8000
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/web"

command -v cloudflared >/dev/null 2>&1 || { echo "Falta cloudflared. Instálalo con: brew install cloudflared"; exit 1; }
[ -d "$DIR" ] || { echo "No encuentro la carpeta web/ en $DIR"; exit 1; }

LOG="$(mktemp -t lsn-tunnel)"
SERVER_PID=""
TUNNEL_PID=""

cleanup() {
  echo ""
  echo "Deteniendo…"
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$LOG"
  exit 0
}
trap cleanup INT TERM

# Libera el puerto si quedó algo colgado de una sesión anterior
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Puerto $PORT ocupado; liberándolo…"
  lsof -nP -tiTCP:$PORT -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 1
fi

echo "Sirviendo $DIR en http://localhost:$PORT …"
python3 -m http.server "$PORT" --directory "$DIR" >/dev/null 2>&1 &
SERVER_PID=$!
sleep 1

echo "Abriendo túnel HTTPS (puede tardar unos segundos)…"
cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate >"$LOG" 2>&1 &
TUNNEL_PID=$!

# Espera a que aparezca la URL pública
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done

echo ""
if [ -n "$URL" ]; then
  echo "======================================================================"
  echo "  App disponible en:  $URL"
  echo "  (compártela; funciona en redes con DNS normal)"
  echo "======================================================================"
else
  echo "No pude leer la URL del túnel. Revisa el log:"
  tail -20 "$LOG"
fi
echo ""
echo "Ctrl+C para detener el servidor y el túnel."

# Mantén el script vivo mientras el túnel siga corriendo
wait "$TUNNEL_PID"
cleanup
