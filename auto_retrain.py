"""Reentrenamiento automático con candado de calidad.

Detecta datos nuevos en dataset_videos/ y dataset_contrib/, mide LOSO, y solo
reentrena y despliega a web/model/ si el número NO empeoró respecto al último
despliegue. Así el ciclo de auto-aprendizaje nunca puede degradar el modelo
en silencio.

Antes del LOSO valida cada archivo de dataset_contrib/ con validate_contribution
(esquema, palabras en allowlist, sin NaN/Inf, sin keypoints faciales) y mueve
los inválidos a dataset_contrib/quarantine/ para auditoría.

Uso:
    .venv/bin/python auto_retrain.py           # una pasada (ideal para cron)
"""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from validate_contrib import validate_contribution  # noqa: E402

STATE_PATH = os.path.join(ROOT, 'auto_retrain_state.json')
CONTRIB_PATH = os.path.join(ROOT, 'dataset_contrib')
QUARANTINE_PATH = os.path.join(CONTRIB_PATH, 'quarantine')
TOLERANCE = 0.02  # caída de LOSO permitida (ruido del benchmark)


def dataset_fingerprint():
    h = hashlib.sha256()
    for folder, exts in (('dataset_videos', ('.mp4',)), ('dataset_contrib', ('.json',))):
        base = os.path.join(ROOT, folder)
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in sorted(os.walk(base)):
            for f in sorted(files):
                if f.endswith(exts):
                    p = os.path.join(dirpath, f)
                    st = os.stat(p)
                    h.update(f'{p}:{st.st_mtime_ns}:{st.size}'.encode())
    return h.hexdigest()


def quarantine_invalid():
    """Valida cada .json de dataset_contrib/ y mueve los inválidos a quarantine/.

    Devuelve el número de archivos aceptados (quedan en dataset_contrib/).
    """
    if not os.path.isdir(CONTRIB_PATH):
        return 0
    os.makedirs(QUARANTINE_PATH, exist_ok=True)
    accepted = 0
    for name in sorted(os.listdir(CONTRIB_PATH)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(CONTRIB_PATH, name)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f'quarantine (json inválido): {name} — {e}')
            shutil.move(path, os.path.join(QUARANTINE_PATH, name))
            continue
        ok, err = validate_contribution(data)
        if not ok:
            print(f'quarantine (esquema): {name} — {err}')
            shutil.move(path, os.path.join(QUARANTINE_PATH, name))
            continue
        accepted += 1
    return accepted


def main():
    state = {}
    if os.path.exists(STATE_PATH):
        state = json.load(open(STATE_PATH))

    fp = dataset_fingerprint()
    if fp == state.get('fingerprint'):
        print('Sin datos nuevos: nada que hacer.')
        return

    accepted = quarantine_invalid()
    if not accepted:
        print('Todos los archivos nuevos son inválidos; en cuarentena. No se reentrena.')
        return

    print(f'Datos nuevos detectados: {accepted} muestras válidas tras cuarentena.')
    print('Midiendo LOSO (candado de calidad)...')
    from evaluate_loso import loso_score
    res = loso_score()
    prev = state.get('loso_mean')
    print(f"LOSO: {res['mean']*100:.1f}%"
          + (f" (anterior: {prev*100:.1f}%)" if prev is not None else ''))

    if prev is not None and res['mean'] < prev - TOLERANCE:
        print(f'\n✗ CANDADO: el LOSO cayó más de {TOLERANCE*100:.0f} puntos. '
              'NO se despliega. Revisa las muestras más recientes de dataset_contrib/ '
              '(¿alguna mal etiquetada?) y vuelve a correr.')
        sys.exit(1)

    print('Candado aprobado. Entrenando modelo de producción...')
    from train_v2 import main as train_main
    train_main()

    json.dump({'fingerprint': fp, 'loso_mean': res['mean'],
               'date': datetime.now().isoformat(timespec='seconds')},
              open(STATE_PATH, 'w'), indent=2)
    print('\n✓ Modelo desplegado a web/model/ y estado actualizado.')


if __name__ == '__main__':
    main()
