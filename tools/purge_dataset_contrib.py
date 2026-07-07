#!/usr/bin/env python3
"""Borra archivos de dataset_contrib/ con más de LSN_RETENTION_DAYS días.

Pensado para correr como cron diario:
    0 3 * * * /var/www/lengua-lsp/.venv/bin/python /var/www/lengua-lsp/tools/purge_dataset_contrib.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRIB_PATH = os.path.join(ROOT, 'dataset_contrib')
QUARANTINE_PATH = os.path.join(CONTRIB_PATH, 'quarantine')

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
except ImportError:
    pass

RETENTION_DAYS = int(os.environ.get('LSN_RETENTION_DAYS', '30'))
RETENTION_SECS = RETENTION_DAYS * 86400


def purge():
    if not os.path.isdir(CONTRIB_PATH):
        return 0
    now = time.time()
    removed = 0
    for entry in os.listdir(CONTRIB_PATH):
        if entry == 'quarantine' or entry.startswith('.'):
            continue
        path = os.path.join(CONTRIB_PATH, entry)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if now - mtime > RETENTION_SECS:
            try:
                os.remove(path)
                removed += 1
                print(f'purged: {entry}')
            except OSError as e:
                print(f'error: {entry}: {e}', file=sys.stderr)
    return removed


if __name__ == '__main__':
    n = purge()
    print(f'purge done: {n} archivos eliminados (retención {RETENTION_DAYS} días)')