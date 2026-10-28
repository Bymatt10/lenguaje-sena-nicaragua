"""Corta el Curso Básico de LSN de CD-SINAPRED en clips etiquetados por seña.

Cada video del curso muestra la palabra en un cartel fijo en la franja inferior
(p. ej. "CULEBRA"). Se lee ese cartel con OCR (Vision de macOS) solo cuando su
contenido cambia, se agrupan los tramos con la misma palabra y se cortan con ffmpeg:

    dataset_raw/NNN_<id>.mp4  →  dataset_videos/<palabra>/<señante>_NN.mp4

Además escribe dataset_raw/segments.json (palabra, video de YouTube, inicio, fin,
señante), que usa el modo "Aprender" de la web para incrustar el video original en
el segundo exacto sin volver a publicar los clips.

Los señantes se asignan por video en dataset_raw/signers.json ({"003": "carlos"});
los videos sin asignar quedan como "cds<NNN>". Videos sin cartel (reportajes,
introducción) se omiten solos.

Uso (solo macOS):
    .venv/bin/python tools/segment_course.py               # todo dataset_raw/
    .venv/bin/python tools/segment_course.py 003 032       # solo esos videos
    .venv/bin/python tools/segment_course.py --dry-run     # solo segments.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import unicodedata

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, 'dataset_raw')
OUT_PATH = os.path.join(ROOT, 'dataset_videos')
SEGMENTS_PATH = os.path.join(RAW_PATH, 'segments.json')
SIGNERS_PATH = os.path.join(RAW_PATH, 'signers.json')
FIXES_PATH = os.path.join(RAW_PATH, 'label_fixes.json')
FFMPEG = os.environ.get('FFMPEG', '/opt/homebrew/bin/ffmpeg')

# cartel de la palabra, en fracciones del frame (medido sobre videos 640x360)
BANNER = (0.32, 0.76, 0.70, 0.89)  # x0, y0, x1, y1
SAMPLE_EVERY = 3        # frames: ~10 lecturas por segundo a 30 fps
CHANGE_THRESHOLD = 6.0  # diferencia media de gris para volver a leer el cartel
MIN_SEGMENT_S = 1.0
MAX_SEGMENT_S = 6.0     # una seña dura ~3 s; los créditos finales duran más
PAD_S = 0.15            # margen al cortar


def ocr(img):
    '''Texto del recorte con Vision (preciso, sin corrección de idioma).'''
    import Vision
    from Foundation import NSData
    ok, buf = cv2.imencode('.png', img)
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(['es-ES'])
    req.setUsesLanguageCorrection_(False)
    handler.performRequests_error_([req], None)
    lines = [o.topCandidates_(1)[0].string() for o in (req.results() or [])]
    return ' '.join(lines).strip()


def slugify(text):
    '''"ATENCIÓN/ATENDER" → "atencion_atender" (mismo formato que words.json).'''
    s = unicodedata.normalize('NFD', text.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.replace('ñ', 'n')
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')


def read_banners(video_path):
    '''Lista de (t_segundos, texto) cada vez que cambia el cartel.'''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    events, last_crop, last_text, idx = [], None, None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % SAMPLE_EVERY == 0:
            _, frame = cap.retrieve()
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = BANNER
            crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if last_crop is None or np.abs(gray.astype(int) - last_crop).mean() > CHANGE_THRESHOLD:
                big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                text = ocr(big)
                if text != last_text:
                    events.append((idx / fps, text))
                    last_text = text
                last_crop = gray.astype(int)
        idx += 1
    cap.release()
    return events, idx / fps


def to_segments(events, duration, fixes=None):
    '''Tramos continuos con la misma palabra legible. `fixes` corrige lecturas
    erróneas del OCR revisadas a mano, por texto exacto o por slug
    ({"Ññ": "NN", "owsis": "SISMO"}); null descarta el cartel (títulos).'''
    fixes = fixes or {}
    segs = []
    for (t, text), nxt in zip(events, events[1:] + [(duration, None)]):
        key = text if text in fixes else slugify(text)
        if key in fixes:
            if fixes[key] is None:
                continue
            text = fixes[key]
        label = slugify(text)
        if len(label) < 1 or len(text) > 40:
            continue
        start, end = t, nxt[0]
        if segs and segs[-1]['word'] == label and start - segs[-1]['end'] < 0.5:
            segs[-1]['end'] = end  # parpadeo del OCR: se une con el anterior
            continue
        segs.append({'word': label, 'text': text, 'start': start, 'end': end})
    return [s for s in segs if MIN_SEGMENT_S <= s['end'] - s['start'] <= MAX_SEGMENT_S]


def cut(src, start, end, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run([
        FFMPEG, '-y', '-loglevel', 'error',
        '-ss', f'{max(0, start - PAD_S):.2f}', '-to', f'{end + PAD_S:.2f}', '-i', src,
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', dst,
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='*', help='prefijos NNN de dataset_raw/ (default: todos)')
    ap.add_argument('--dry-run', action='store_true', help='no cortar, solo segments.json')
    args = ap.parse_args()

    signers = json.load(open(SIGNERS_PATH)) if os.path.exists(SIGNERS_PATH) else {}
    manifest = json.load(open(SEGMENTS_PATH)) if os.path.exists(SEGMENTS_PATH) else {}
    fixes = json.load(open(FIXES_PATH)) if os.path.exists(FIXES_PATH) else {}

    paths = sorted(glob.glob(os.path.join(RAW_PATH, '[0-9][0-9][0-9]_*.mp4')))
    if args.videos:
        paths = [p for p in paths if os.path.basename(p)[:3] in args.videos]
    if not paths:
        sys.exit('No hay videos en dataset_raw/ (descarga primero la playlist).')

    for path in paths:
        name = os.path.basename(path)
        num, yt_id = name[:3], os.path.splitext(name)[0][4:]
        info_path = os.path.splitext(path)[0] + '.info.json'
        title = json.load(open(info_path)).get('title', '') if os.path.exists(info_path) else ''
        signer = signers.get(num, f'cds{num}')

        events, duration = read_banners(path)
        segs = to_segments(events, duration, fixes)
        print(f'{num} [{signer}] {title[:60]}: {len(segs)} señas')

        reps = {}
        for s in segs:
            # cada seña se muestra dos veces seguidas: r1, r2
            reps[s['word']] = reps.get(s['word'], 0) + 1
            s.update({'video': num, 'youtube_id': yt_id, 'signer': signer,
                      'start': round(s['start'], 2), 'end': round(s['end'], 2)})
            # <señante>_<video>r<n>.mp4: evaluate_loso toma el señante con rsplit('_', 1)
            s['clip'] = os.path.join(s['word'], f"{signer}_{num}r{reps[s['word']]}.mp4")
            print(f"   {s['start']:7.1f}–{s['end']:6.1f}s  {s['text']}")
            if not args.dry_run:
                cut(path, s['start'], s['end'], os.path.join(OUT_PATH, s['clip']))
        manifest[num] = {'title': title, 'youtube_id': yt_id, 'signer': signer, 'segments': segs}

    with open(SEGMENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    total = sum(len(v['segments']) for v in manifest.values())
    words = {s['word'] for v in manifest.values() for s in v['segments']}
    print(f'\n{total} clips, {len(words)} palabras distintas → {SEGMENTS_PATH}')


if __name__ == '__main__':
    main()
