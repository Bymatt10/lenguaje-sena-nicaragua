"""Genera web/lessons.json y los clips de referencia del modo "Aprender".

Por cada palabra de dataset_raw/segments.json toma la primera repetición (r1)
y la guarda reducida en web/clips/<palabra>.mp4 (H.264, 480 px de ancho, sin
audio), que la web reproduce localmente. Nada apunta a sitios externos.

Uso:
    .venv/bin/python tools/export_lessons.py              # lecciones + clips
    .venv/bin/python tools/export_lessons.py --no-clips   # solo lessons.json
"""
import argparse
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTS_PATH = os.path.join(ROOT, 'dataset_raw', 'segments.json')
DATASET_PATH = os.path.join(ROOT, 'dataset_videos')
OUT_PATH = os.path.join(ROOT, 'web', 'lessons.json')
CLIPS_DIR = os.path.join(ROOT, 'web', 'clips')
FFMPEG = os.environ.get('FFMPEG', '/opt/homebrew/bin/ffmpeg')


def make_clip(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    subprocess.run([
        FFMPEG, '-y', '-loglevel', 'error', '-i', src, '-an',
        '-vf', 'scale=480:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst,
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-clips', action='store_true')
    args = ap.parse_args()

    manifest = json.load(open(SEGMENTS_PATH, encoding='utf-8'))
    os.makedirs(CLIPS_DIR, exist_ok=True)
    words = {}
    for num in sorted(manifest):
        video = manifest[num]
        for s in video['segments']:
            # primera aparición de cada palabra (la r1 del primer video que la enseña)
            if s['word'] in words:
                continue
            entry = {'text': s['text'].strip().upper(), 'signer': s['signer'], 'title': video['title']}
            src = os.path.join(DATASET_PATH, s['clip'])
            if not args.no_clips and os.path.exists(src):
                make_clip(src, os.path.join(CLIPS_DIR, f"{s['word']}.mp4"))
            if os.path.exists(os.path.join(CLIPS_DIR, f"{s['word']}.mp4")):
                entry['clip'] = f"clips/{s['word']}.mp4"
            words[s['word']] = entry

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'source': 'Curso Básico de Lenguaje de Señas — CD-SINAPRED / ANSNIC',
                   'words': dict(sorted(words.items()))}, f, ensure_ascii=False, indent=1)
    n_clips = sum('clip' in w for w in words.values())
    print(f'{len(words)} lecciones ({n_clips} con clip) → {OUT_PATH}')


if __name__ == '__main__':
    main()
