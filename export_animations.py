"""Exporta las secuencias de keypoints de cada seña a web/signs_anim.json para el
traductor inverso (texto → señas): una figura animada reproduce el esqueleto de la
señante. Usa los cachés .npy generados por train_v2 (mismas features de 225 dims,
relativas al centro de hombros)."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT, 'dataset_videos')
OUT_PATH = os.path.join(ROOT, 'web', 'signs_anim.json')

FPS_OUT = 15  # los videos van a 30; la mitad basta para una animación fluida

words = {}
for word in sorted(os.listdir(DATASET_PATH)):
    word_dir = os.path.join(DATASET_PATH, word)
    if not os.path.isdir(word_dir):
        continue
    # el primer clip de elizabeth como animación de referencia
    clips = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4.npy'))
    if not clips:
        print(f'⚠ {word}: sin caché .npy (corre train_v2.py primero), omitida')
        continue
    seq = np.load(os.path.join(word_dir, clips[0]))
    seq = seq[::2]  # 30 → 15 fps
    words[word] = np.round(seq, 3).tolist()
    print(f'{word}: {len(seq)} frames')

with open(OUT_PATH, 'w') as f:
    json.dump({'fps': FPS_OUT, 'words': words}, f)

size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f'\n{len(words)} animaciones → {OUT_PATH} ({size_mb:.1f} MB)')
