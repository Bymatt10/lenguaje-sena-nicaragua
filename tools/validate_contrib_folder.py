#!/usr/bin/env python3
"""Valida archivos JSON en dataset_contrib/ contra el esquema de /contribute.

Útil como pre-filtro antes de correr auto_retrain.py: descarta archivos que
contengan keypoints del rostro, NaN/Inf, palabras fuera de allowlist, etc.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validate_contrib import validate_contribution  # noqa: E402


def main(folder):
    if not os.path.isdir(folder):
        print(f'carpeta no existe: {folder}')
        return 1
    invalid = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            invalid.append((name, f'JSON inválido: {e}'))
            continue
        ok, err = validate_contribution(data)
        if not ok:
            invalid.append((name, err))
    if invalid:
        print(f'{len(invalid)} archivos inválidos:')
        for n, e in invalid:
            print(f'  - {n}: {e}')
        return 1
    print(f'todos los {len([f for f in os.listdir(folder) if f.endswith(".json")])} archivos válidos')
    return 0


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'dataset_contrib')
    sys.exit(main(target))