"""Evaluación leave-one-signer-out (LOSO): entrena con un señante y evalúa con el
otro. Es la medida honesta de generalización a personas nuevas — el val_accuracy
del entrenamiento normal está inflado porque valida con variantes de los mismos takes.

`loso_score(cfg)` es reutilizable por ablation.py. Opciones de cfg:
- las de augment(): rot3d, hand_dropout, fps_decimation, time_warp
- las de build_model(): bidirectional, label_smoothing
- frames (default 15), ensemble (n modelos, default 1), complexity (MediaPipe, default 1)
"""
import json
import os

import numpy as np

from train_v2 import (video_to_sequence, normalize_length, augment, prepare,
                      build_model, DATASET_PATH, CONTRIB_PATH, AUG_PER_CLIP,
                      MODEL_FRAMES, SEED)


def load_signers(complexity=1):
    '''Devuelve {señante: {palabra: secuencia_cruda}}.'''
    from mediapipe.python.solutions.holistic import Holistic
    signers = {}

    suffix = '' if complexity == 1 else f'.mc{complexity}'
    with Holistic(model_complexity=complexity) as holistic:
        for word in sorted(os.listdir(DATASET_PATH)):
            word_dir = os.path.join(DATASET_PATH, word)
            if not os.path.isdir(word_dir):
                continue
            for clip in sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4')):
                signer = clip.rsplit('_', 1)[0]  # elizabeth_01.mp4 → elizabeth
                seq = video_to_sequence(os.path.join(word_dir, clip), holistic, suffix)
                if len(seq) >= 5:
                    signers.setdefault(signer, {})[word] = seq

    if os.path.isdir(CONTRIB_PATH):
        for fname in sorted(os.listdir(CONTRIB_PATH)):
            if not fname.endswith('.json'):
                continue
            data = json.load(open(os.path.join(CONTRIB_PATH, fname)))
            signer = data.get('contributor', fname)
            for s in data.get('samples', []):
                if s['word'].startswith('_'):
                    continue  # p.ej. '_nada': reservada para la futura clase negativa
                seq = np.array(s['frames'], dtype=np.float32)
                if len(seq) >= 5:
                    signers.setdefault(signer, {})[s['word']] = seq
    return signers


def train_fold(train_data, words, rng, cfg):
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical

    frames = cfg.get('frames', MODEL_FRAMES)
    X, y = [], []
    for word, seq in train_data.items():
        label = words.index(word)
        X.append(prepare(normalize_length(seq, frames)))
        y.append(label)
        for _ in range(AUG_PER_CLIP):
            X.append(prepare(augment(seq, rng, cfg)))
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=len(words))

    model = build_model(len(words), frames=frames,
                        bidirectional=cfg.get('bidirectional', False),
                        label_smoothing=cfg.get('label_smoothing', 0.0))
    early = EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True)
    model.fit(X, y, epochs=200, batch_size=16, callbacks=[early], verbose=0)
    return model


def loso_score(cfg=None, signers=None, verbose=True):
    '''Entrena con N-1 señantes y evalúa con el restante, para cada señante.
    Devuelve {señante: accuracy, 'mean': promedio}.'''
    cfg = cfg or {}
    frames = cfg.get('frames', MODEL_FRAMES)
    if signers is None:
        signers = load_signers(complexity=cfg.get('complexity', 1))
    names = sorted(signers)
    words = sorted({w for data in signers.values() for w in data})

    results = {}
    for test_signer in names:
        train_data = {}
        for s in names:
            if s != test_signer:
                train_data.update(signers[s])

        models = [
            train_fold(train_data, words, np.random.default_rng(SEED + k), cfg)
            for k in range(cfg.get('ensemble', 1))
        ]

        correct, total, errors = 0, 0, []
        for word, seq in sorted(signers[test_signer].items()):
            if word not in words:
                continue
            x = prepare(normalize_length(seq, frames))[None]
            probs = np.mean([m.predict(x, verbose=0)[0] for m in models], axis=0)
            pred = words[int(probs.argmax())]
            total += 1
            if pred == word:
                correct += 1
            else:
                errors.append(f'{word}→{pred}')
        acc = correct / total if total else 0.0
        results[test_signer] = acc
        if verbose:
            print(f'  evalúa {test_signer}: {correct}/{total} = {acc*100:.1f}%'
                  + (f'  errores: {", ".join(errors)}' if errors else ''))

    results['mean'] = float(np.mean([results[n] for n in names]))
    return results


if __name__ == '__main__':
    res = loso_score()
    print(f"\npromedio LOSO: {res['mean']*100:.1f}%")
