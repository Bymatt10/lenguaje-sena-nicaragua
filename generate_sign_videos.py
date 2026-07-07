import os
import shutil
import subprocess
import tempfile

from constants import FRAME_ACTIONS_PATH
from helpers import sort_frame_names

WEB_VIDEOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'videos')
FPS = 10


def best_sample(word_path):
    '''Elige la muestra con más frames (la más fluida para mostrar).'''
    best, best_count = None, 0
    for sample in os.listdir(word_path):
        sample_path = os.path.join(word_path, sample)
        if not os.path.isdir(sample_path):
            continue
        count = len([f for f in os.listdir(sample_path) if f.endswith('.jpg')])
        if count > best_count:
            best, best_count = sample_path, count
    return best


def make_video(sample_path, output_path):
    '''Genera un mp4 H.264 (compatible con navegadores) desde los frames de una muestra.'''
    frames = sort_frame_names(f for f in os.listdir(sample_path) if f.endswith('.jpg'))
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(frames, start=1):
            shutil.copy(os.path.join(sample_path, name), os.path.join(tmp, f'{i:03}.jpg'))
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(tmp, '%03d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            output_path,
        ], check=True, capture_output=True)


if __name__ == '__main__':
    if not os.path.isdir(FRAME_ACTIONS_PATH):
        raise SystemExit(f'No existe {FRAME_ACTIONS_PATH}: copia aquí tus muestras (frame_actions/) primero.')

    os.makedirs(WEB_VIDEOS_PATH, exist_ok=True)
    done = set()
    for word in sorted(os.listdir(FRAME_ACTIONS_PATH)):
        word_path = os.path.join(FRAME_ACTIONS_PATH, word)
        if not os.path.isdir(word_path):
            continue
        base = word.split('-')[0]  # hola-der / hola-izq → hola
        if base in done:
            continue
        sample = best_sample(word_path)
        if not sample:
            print(f'"{word}": sin muestras, omitida')
            continue
        output = os.path.join(WEB_VIDEOS_PATH, f'{base}.mp4')
        make_video(sample, output)
        done.add(base)
        print(f'"{base}": {output}')

    print(f'\n{len(done)} videos generados en web/videos/')
