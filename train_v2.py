"""Entrenamiento v2: desde videos etiquetados en dataset_videos/<palabra>/*.mp4

Mejoras sobre el pipeline original:
- Features de 225 dims (pose 33x3 + mano izq 21x3 + mano der 21x3), sin cara.
- Normalización espacial: coordenadas relativas al punto medio de los hombros,
  escaladas por el ancho de hombros → invariante a posición y distancia de cámara.
- Aumentación: recortes temporales, espejo, rotación, escala y ruido.

Genera:
- models/actions_v2.keras + models/words_v2.json
- web/model/weights.bin + weights_spec.json + reference.json + web/words.json

La extracción de features debe mantenerse IDÉNTICA a extractFeatures() en web/app.js.
"""
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT, 'dataset_videos')
CONTRIB_PATH = os.path.join(ROOT, 'dataset_contrib')  # JSONs del modo captura de la web
MODEL_V2_PATH = os.path.join(ROOT, 'models', 'actions_v2.keras')
WORDS_V2_PATH = os.path.join(ROOT, 'models', 'words_v2.json')
WEB_MODEL_DIR = os.path.join(ROOT, 'web', 'model')
WEB_WORDS_PATH = os.path.join(ROOT, 'web', 'words.json')

# 25 frames ganó en el estudio de ablación (+4.5 pts LOSO vs 15) — ver ABLATION.md
MODEL_FRAMES = 25
N_POSE, N_HAND = 33, 21
FEATURES = N_POSE * 3 + N_HAND * 3 * 2          # 225 features base por frame
STATIC_EXPANDED = FEATURES + N_HAND * 3 * 2     # 351: + manos relativas a su muñeca
MODEL_FEATURES = STATIC_EXPANDED * 2            # 702: + velocidades entre frames
AUG_PER_CLIP = 80
SEED = 42

# pares izquierda/derecha de pose para el espejo (índices MediaPipe)
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
             (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

LH_START = N_POSE * 3          # 99
RH_START = LH_START + N_HAND * 3  # 162


# ---------- extracción de features (espejo exacto de web/app.js) ----------
def extract_features(results):
    '''225 features normalizadas, o None si no hay pose o no hay manos.'''
    if not results.pose_landmarks:
        return None
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return None

    pose = results.pose_landmarks.landmark
    cx = (pose[11].x + pose[12].x) / 2
    cy = (pose[11].y + pose[12].y) / 2
    s = max(np.hypot(pose[11].x - pose[12].x, pose[11].y - pose[12].y), 1e-3)

    feats = np.zeros(FEATURES, dtype=np.float32)
    for i, p in enumerate(pose):
        feats[i * 3] = (p.x - cx) / s
        feats[i * 3 + 1] = (p.y - cy) / s
        feats[i * 3 + 2] = p.z / s
    for start, hand in ((LH_START, results.left_hand_landmarks),
                        (RH_START, results.right_hand_landmarks)):
        if hand:
            for i, p in enumerate(hand.landmark):
                feats[start + i * 3] = (p.x - cx) / s
                feats[start + i * 3 + 1] = (p.y - cy) / s
                feats[start + i * 3 + 2] = p.z / s
    return feats


def video_to_sequence(video_path, holistic, cache_suffix=''):
    # caché: extraer keypoints con MediaPipe es lo lento; se guarda junto al video
    cache = video_path + cache_suffix + '.npy'
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(video_path):
        return np.load(cache)

    from helpers import mediapipe_detection
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feats = extract_features(mediapipe_detection(frame, holistic))
        if feats is not None:
            seq.append(feats)
    cap.release()
    seq = np.array(seq, dtype=np.float32)
    np.save(cache, seq)
    return seq


# ---------- features v3: manos locales + velocidades (espejo exacto de app.js) ----------
def expand_static(seq):
    '''(n, 225) → (n, 351): agrega cada mano relativa a su muñeca, escalada por el
    tamaño de la mano (muñeca → nudillo medio). Amplifica la geometría de dedos,
    clave para el abecedario.'''
    n = len(seq)
    out = np.zeros((n, STATIC_EXPANDED), dtype=np.float32)
    out[:, :FEATURES] = seq
    for start, dst in ((LH_START, FEATURES), (RH_START, FEATURES + N_HAND * 3)):
        block = seq[:, start:start + N_HAND * 3].reshape(n, N_HAND, 3)
        wrist = block[:, 0:1, :]
        scale = np.maximum(np.linalg.norm(block[:, 9] - block[:, 0], axis=1), 1e-3)
        out[:, dst:dst + N_HAND * 3] = ((block - wrist) / scale[:, None, None]).reshape(n, -1)
    return out


def prepare(sample):
    '''(15, 225) → (15, 702): expande manos locales y concatena velocidades.'''
    e = expand_static(np.asarray(sample, dtype=np.float32))
    v = np.zeros_like(e)
    v[1:] = e[1:] - e[:-1]
    return np.concatenate([e, v], axis=1)


# ---------- normalización temporal (misma semántica que evaluate_model) ----------
def normalize_length(seq, target=MODEL_FRAMES):
    n = len(seq)
    if n == target:
        return np.array(seq, dtype=np.float32)
    if n > target:
        idx = np.arange(0, n, n / target).astype(int)[:target]
        return np.array([seq[i] for i in idx], dtype=np.float32)
    out = []
    for t in np.linspace(0, n - 1, target):
        lo, hi = int(np.floor(t)), int(np.ceil(t))
        w = t - lo
        out.append(seq[lo] if lo == hi else (1 - w) * seq[lo] + w * seq[hi])
    return np.array(out, dtype=np.float32)


# ---------- aumentación ----------
def mirror(sample):
    out = sample.copy()
    # intercambiar bloques de manos
    lh = out[:, LH_START:LH_START + N_HAND * 3].copy()
    out[:, LH_START:LH_START + N_HAND * 3] = out[:, RH_START:RH_START + N_HAND * 3]
    out[:, RH_START:RH_START + N_HAND * 3] = lh
    # intercambiar pares izquierda/derecha de pose
    for a, b in POSE_SWAP:
        pa, pb = out[:, a * 3:a * 3 + 3].copy(), out[:, b * 3:b * 3 + 3].copy()
        out[:, a * 3:a * 3 + 3], out[:, b * 3:b * 3 + 3] = pb, pa
    # negar todas las x (cada 3ra columna empezando en 0)
    out[:, 0::3] *= -1
    return out


def time_warp(seq, rng):
    '''Remuestreo temporal no uniforme: curva monótona aleatoria con 2–3 nodos.
    Simula ritmos distintos dentro de la misma seña.'''
    n = len(seq)
    if n < 6:
        return seq
    k = int(rng.integers(2, 4))
    src = np.concatenate([[0], np.sort(rng.uniform(0.15, 0.85, k)), [1]])
    dst = np.concatenate([[0], np.sort(rng.uniform(0.15, 0.85, k)), [1]])
    t = np.interp(np.linspace(0, 1, n), src, dst) * (n - 1)
    lo, hi = np.floor(t).astype(int), np.ceil(t).astype(int)
    w = (t - lo)[:, None]
    return (1 - w) * seq[lo] + w * seq[hi]


def augment(seq, rng, cfg=None):
    cfg = cfg or {}
    frames = cfg.get('frames', MODEL_FRAMES)
    n = len(seq)

    # decimación de fps: simula webcams lentas (13–18 fps vs videos a 30)
    if cfg.get('fps_decimation') and rng.random() < 0.3 and n >= 12:
        idx = np.arange(0, n, rng.uniform(1.5, 2.5)).astype(int)
        seq = seq[idx]
        n = len(seq)

    # recorte temporal aleatorio (70–100% del clip)
    length = max(5, int(n * rng.uniform(0.7, 1.0)))
    start = rng.integers(0, n - length + 1)
    sub = seq[start:start + length]

    if cfg.get('time_warp'):
        sub = time_warp(sub, rng)

    sample = normalize_length(sub, frames)

    if rng.random() < 0.5:
        sample = mirror(sample)

    # rotación 3D de punto de vista: gira (x,z) alrededor del eje vertical,
    # simulando cámaras en ángulos distintos
    if cfg.get('rot3d'):
        theta = rng.uniform(-0.44, 0.44)  # ±25°
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x, z = sample[:, 0::3].copy(), sample[:, 2::3].copy()
        sample[:, 0::3] = x * cos_t + z * sin_t
        sample[:, 2::3] = -x * sin_t + z * cos_t

    # rotación leve en el plano xy
    theta = rng.uniform(-0.14, 0.14)  # ±8°
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y = sample[:, 0::3].copy(), sample[:, 1::3].copy()
    sample[:, 0::3] = x * cos_t - y * sin_t
    sample[:, 1::3] = x * sin_t + y * cos_t

    sample *= rng.uniform(0.9, 1.1)                   # escala
    sample += rng.normal(0, 0.02, sample.shape)       # ruido

    # hand-dropout: borra una mano en un tramo (simula pérdida de MediaPipe);
    # va al final para que los ceros queden exactos, como en inferencia real
    if cfg.get('hand_dropout') and rng.random() < 0.15:
        hand = LH_START if rng.random() < 0.5 else RH_START
        span = int(rng.integers(int(frames * 0.3), int(frames * 0.7) + 1))
        a = int(rng.integers(0, frames - span + 1))
        sample[a:a + span, hand:hand + N_HAND * 3] = 0

    return sample.astype(np.float32)


# ---------- modelo ----------
def build_model(n_classes, frames=MODEL_FRAMES, features=MODEL_FEATURES,
                bidirectional=False, label_smoothing=0.0):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from keras.losses import CategoricalCrossentropy

    l1 = LSTM(64, return_sequences=True)
    l2 = LSTM(128)
    if bidirectional:
        l1, l2 = Bidirectional(l1), Bidirectional(l2)

    model = Sequential([
        Input(shape=(frames, features)),
        l1,
        Dropout(0.4),
        l2,
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
                  metrics=['accuracy'])
    return model


# ---------- open-set: centroides de embeddings para rechazar gestos desconocidos ----------
def export_openset(model, X, y_labels, out_path):
    '''Guarda el centroide del embedding (capa penúltima) de cada clase y su radio
    de aceptación (percentil 95 de las distancias intra-clase). En inferencia, un
    gesto cuyo embedding queda fuera del radio de su clase predicha se rechaza.'''
    from keras.models import Model
    embed_model = Model(model.inputs, model.layers[-2].output)
    emb = embed_model.predict(X, verbose=0)

    centroids, radii = [], []
    for label in range(int(y_labels.max()) + 1):
        class_emb = emb[y_labels == label]
        c = class_emb.mean(axis=0)
        d = np.linalg.norm(class_emb - c, axis=1)
        centroids.append(c.round(5).tolist())
        radii.append(float(np.percentile(d, 95)))
    with open(out_path, 'w') as f:
        json.dump({'centroids': centroids, 'radii': radii}, f)
    return np.array(centroids), np.array(radii)


# ---------- entrenamiento ----------
def main(cfg=None):
    from mediapipe.python.solutions.holistic import Holistic
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    cfg = cfg or {}
    frames = cfg.get('frames', MODEL_FRAMES)
    rng = np.random.default_rng(SEED)

    words = sorted(w for w in os.listdir(DATASET_PATH)
                   if os.path.isdir(os.path.join(DATASET_PATH, w)))
    print(f'Palabras ({len(words)}): {words}')

    X, y = [], []
    with Holistic() as holistic:
        for label, word in enumerate(words):
            word_dir = os.path.join(DATASET_PATH, word)
            clips = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4'))
            for clip in clips:
                seq = video_to_sequence(os.path.join(word_dir, clip), holistic)
                print(f'  {word}/{clip}: {len(seq)} frames útiles')
                if len(seq) < 5:
                    print(f'  ⚠ {word}/{clip}: muy corto, omitido')
                    continue
                X.append(prepare(normalize_length(seq, frames)))  # el clip original, sin aumentar
                y.append(label)
                for _ in range(AUG_PER_CLIP):
                    X.append(prepare(augment(seq, rng, cfg)))
                    y.append(label)

    # muestras contribuidas desde el modo captura de la web
    if os.path.isdir(CONTRIB_PATH):
        for fname in sorted(os.listdir(CONTRIB_PATH)):
            if not fname.endswith('.json'):
                continue
            data = json.load(open(os.path.join(CONTRIB_PATH, fname)))
            added = 0
            for sample in data.get('samples', []):
                word = sample['word']
                if word not in words:
                    print(f'  ⚠ {fname}: palabra desconocida "{word}", omitida')
                    continue
                seq = np.array(sample['frames'], dtype=np.float32)
                if len(seq) < 5 or seq.shape[1] != FEATURES:
                    continue
                label = words.index(word)
                X.append(prepare(normalize_length(seq, frames)))
                y.append(label)
                for _ in range(AUG_PER_CLIP):
                    X.append(prepare(augment(seq, rng, cfg)))
                    y.append(label)
                added += 1
            print(f'  {fname} ({data.get("contributor", "?")}): {added} muestras')

    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=len(words))
    print(f'Dataset: {X.shape} (muestras, frames, features)')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y.argmax(axis=1))

    model = build_model(len(words), frames=frames,
                        bidirectional=cfg.get('bidirectional', False),
                        label_smoothing=cfg.get('label_smoothing', 0.0))
    early = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=300, batch_size=16, callbacks=[early], verbose=2)

    val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f'\nval_accuracy: {val_acc:.3f}')

    model.save(MODEL_V2_PATH)
    with open(WORDS_V2_PATH, 'w') as f:
        json.dump({'word_ids': words}, f, indent=2)

    # ---------- export a la web ----------
    os.makedirs(WEB_MODEL_DIR, exist_ok=True)
    weights = model.get_weights()
    spec = []
    with open(os.path.join(WEB_MODEL_DIR, 'weights.bin'), 'wb') as f:
        for w in weights:
            arr = w.astype('float32')
            spec.append(list(arr.shape))
            f.write(arr.tobytes())
    with open(os.path.join(WEB_MODEL_DIR, 'weights_spec.json'), 'w') as f:
        json.dump({'frames': frames, 'keypoints': MODEL_FEATURES,
                   'static': FEATURES, 'version': 3, 'shapes': spec}, f)

    export_openset(model, X, y.argmax(axis=1),
                   os.path.join(WEB_MODEL_DIR, 'centroids.json'))

    # referencia 1: entrada directa al modelo (valida los pesos en TFJS)
    x_ref = rng.random((1, frames, MODEL_FEATURES)).astype('float32')
    y_ref = model.predict(x_ref, verbose=0)[0]
    # referencia 2: secuencia estática cruda → valida el pipeline completo de JS
    # (normalización temporal + manos locales + velocidades) de punta a punta
    static_ref = (rng.random((20, FEATURES)).astype('float32') * 2 - 1)
    y_static = model.predict(prepare(normalize_length(static_ref, frames))[None], verbose=0)[0]
    with open(os.path.join(WEB_MODEL_DIR, 'reference.json'), 'w') as f:
        json.dump({'input': x_ref[0].round(6).tolist(),
                   'expected_output': y_ref.round(6).tolist(),
                   'static_input': static_ref.round(6).tolist(),
                   'static_expected_output': y_static.round(6).tolist()}, f)
    with open(WEB_WORDS_PATH, 'w') as f:
        json.dump({'word_ids': words}, f, indent=2)

    print(f'Exportado a {WEB_MODEL_DIR} y {WEB_WORDS_PATH}')


if __name__ == '__main__':
    main()
