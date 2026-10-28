"""Evaluación por repetición sobre los clips del curso (dataset_videos/<palabra>/<señante>_<video>r<n>.mp4).

En el curso cada seña se muestra dos veces seguidas (r1, r2) por la misma persona.
Se entrena con una repetición (aumentada) y se evalúa con la otra, y viceversa.
Mide si el modelo aprende la seña y no la grabación; NO mide generalización a
otra persona (eso es evaluate_loso.py con señantes nuevos).

Uso:
    .venv/bin/python tools/eval_reps.py --words-file vocab_demo.txt
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_v2 import (DATASET_PATH, MODEL_FRAMES, SEED, augment, build_model,  # noqa: E402
                      normalize_length, prepare, video_to_sequence)


def load_reps(words):
    '''{palabra: {'r1': seq, 'r2': seq}} con la primera pareja r1/r2 de cada palabra.'''
    from mediapipe.python.solutions.holistic import Holistic
    data = {}
    with Holistic() as holistic:
        for word in words:
            word_dir = os.path.join(DATASET_PATH, word)
            clips = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4'))
            for clip in clips:
                rep = os.path.splitext(clip)[0][-2:]
                seq = video_to_sequence(os.path.join(word_dir, clip), holistic)
                if len(seq) >= 5 and rep not in data.get(word, {}):
                    data.setdefault(word, {})[rep] = seq
    return {w: d for w, d in data.items() if 'r1' in d and 'r2' in d}


def run_fold(data, words, train_rep, test_rep, rng, aug):
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
    X, y = [], []
    for label, word in enumerate(words):
        seq = data[word][train_rep]
        X.append(prepare(normalize_length(seq, MODEL_FRAMES)))
        y.append(label)
        for _ in range(aug):
            X.append(prepare(augment(seq, rng)))
            y.append(label)
    X, y = np.array(X, dtype=np.float32), to_categorical(y, len(words))
    model = build_model(len(words))
    model.fit(X, y, epochs=150, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])
    Xt = np.array([prepare(normalize_length(data[w][test_rep], MODEL_FRAMES)) for w in words])
    pred = model.predict(Xt, verbose=0)
    top1 = pred.argmax(1) == np.arange(len(words))
    top3 = np.array([i in np.argsort(-p)[:3] for i, p in enumerate(pred)])
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--words-file', required=True)
    ap.add_argument('--aug', type=int, default=60)
    ap.add_argument('--out', help='escribe las métricas en JSON (p. ej. web/model/metrics.json)')
    args = ap.parse_args()

    wanted = [w.strip() for w in open(args.words_file) if w.strip() and not w.startswith('#')]
    missing = [w for w in wanted if not os.path.isdir(os.path.join(DATASET_PATH, w))]
    if missing:
        print(f'⚠ sin clips: {missing}')
    data = load_reps([w for w in wanted if w not in missing])
    words = sorted(data)
    print(f'{len(words)} palabras con r1 y r2')

    rng = np.random.default_rng(SEED)
    t1a, t3a = run_fold(data, words, 'r1', 'r2', rng, args.aug)
    t1b, t3b = run_fold(data, words, 'r2', 'r1', rng, args.aug)
    top1 = (t1a.mean() + t1b.mean()) / 2
    top3 = (t3a.mean() + t3b.mean()) / 2
    print(f'\nTop-1: {top1 * 100:.1f}%   Top-3: {top3 * 100:.1f}%   (azar: {100 / len(words):.1f}%)')
    fails = [w for i, w in enumerate(words) if not (t1a[i] and t1b[i])]
    print(f'Falla en al menos un sentido ({len(fails)}): {fails}')

    if args.out:
        import json
        signers = {f.rsplit('_', 1)[0] for w in words
                   for f in os.listdir(os.path.join(DATASET_PATH, w)) if f.endswith('.mp4')}
        metrics = {
            'method': 'repeticiones',
            'headline': {'value': round(float(top1), 4),
                         'label': 'top-1 en repetición no vista (mismo señante)'},
            'top3': round(float(top3), 4),
            'chance': round(1 / len(words), 4),
            'classes': len(words),
            'signers': len(signers),
            'folds': {'r1 → r2': round(float(t1a.mean()), 4), 'r2 → r1': round(float(t1b.mean()), 4)},
            'confused': fails,
            'data_summary': (
                f'{len(words)} señas del Curso Básico de Lenguaje de Señas (CD-SINAPRED/ANSNIC), '
                f'{len(signers)} señantes; cada seña aparece dos veces seguidas por la misma persona. '
                'Los clips se cortaron automáticamente leyendo con OCR el cartel con la palabra '
                f'y se aumentaron ×{args.aug} (recortes temporales, espejo, rotación, escala y ruido).'),
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1)
        print(f'Métricas → {args.out}')


if __name__ == '__main__':
    main()
