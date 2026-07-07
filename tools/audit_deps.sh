#!/usr/bin/env bash
# Auditoría de vulnerabilidades en dependencias. Salida no-bloqueante por defecto.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet -r requirements-dev.txt

echo "=== pip-audit ==="
pip-audit -r requirements.txt || true
echo
echo "=== safety (opcional) ==="
if command -v safety >/dev/null 2>&1; then
    safety check --file requirements.txt || true
else
    echo "safety no instalado; omitiendo"
fi