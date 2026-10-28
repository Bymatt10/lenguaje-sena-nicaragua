"""Exporta las secuencias de keypoints de cada seña a web/signs_anim.json para el
traductor inverso (texto → señas) y la referencia "esqueleto" del modo Aprender:
una figura animada reproduce el esqueleto del señante. Usa las mismas features
de 225 dims que train_v2 (relativas al centro de hombros, corregidas por aspecto).

Uso:
    .venv/bin/python export_animations.py --vocab vocab_demo.txt
Con --vocab exporta ese vocabulario más todas las letras del abecedario (para
deletrear); las animaciones ya existentes de otras palabras se conservan.
"""
import argparse
import json
import os

import numpy as np

from train_v2 import DATASET_PATH, read_vocab, video_to_sequence

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, 'web', 'signs_anim.json')

FPS_IN = 30
STEP = 3          # 30 → 10 fps: basta para una animación fluida (se interpola)
LETTERS = list('abcdefghijklmnopqrstuvwxyz') + ['ch', 'll', 'rr', 'nn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab', help='archivo de vocabulario (default: todas las carpetas)')
    args = ap.parse_args()

    if args.vocab:
        wanted = read_vocab(args.vocab) + LETTERS
    else:
        wanted = sorted(os.listdir(DATASET_PATH))
    wanted = [w for w in dict.fromkeys(wanted) if os.path.isdir(os.path.join(DATASET_PATH, w))]

    words = {}
    if os.path.exists(OUT_PATH):
        old = json.load(open(OUT_PATH))
        ratio = old.get('fps', FPS_IN // STEP) / (FPS_IN // STEP)
        # las conservadas se remuestrean al fps nuevo para que no cambie su velocidad
        for w, frames in old['words'].items():
            idx = np.arange(0, len(frames), ratio).astype(int)
            words[w] = [frames[i] for i in idx]

    from mediapipe.python.solutions.holistic import Holistic
    with Holistic() as holistic:
        for word in wanted:
            word_dir = os.path.join(DATASET_PATH, word)
            # la primera repetición del primer señante como referencia
            clips = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4'))
            seq = video_to_sequence(os.path.join(word_dir, clips[0]), holistic)
            if len(seq) < 5:
                print(f'⚠ {word}: sin manos detectadas, omitida')
                continue
            words[word] = np.round(seq[::STEP], 3).tolist()
            print(f'{word}: {len(words[word])} frames')

    with open(OUT_PATH, 'w') as f:
        json.dump({'fps': FPS_IN // STEP, 'words': words}, f)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f'\n{len(words)} animaciones → {OUT_PATH} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
