import math

MAX_PAYLOAD_BYTES = 1 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024
MAX_SAMPLES_PER_FILE = 500
MIN_FRAMES = 3
MAX_FRAMES = 60
MIN_FEATURES = 1
MAX_FEATURES = 1000
MIN_VALUE = -10.0
MAX_VALUE = 10.0
MAX_CONTRIBUTOR_LEN = 30

VALID_WORDS = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'adios', 'bien', 'bienvenidos', 'buenas_noches', 'buenas_tardes',
    'buenos_dias', 'como_estas', 'disculpa', 'gracias', 'hola', 'mal',
    'mas_o_menos', 'me_ayudas', 'por_favor', 'saludar',
    'ch', 'll', 'rr', 'nn',
}


def _is_finite_number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return math.isfinite(x) and MIN_VALUE <= x <= MAX_VALUE


def validate_contribution(data):
    if not isinstance(data, dict):
        return False, 'payload debe ser un objeto JSON'
    if not isinstance(data.get('language'), str) or not data['language']:
        return False, 'language requerido'
    contributor = data.get('contributor')
    if not isinstance(contributor, str) or not contributor:
        return False, 'contributor requerido'
    if len(contributor) > MAX_CONTRIBUTOR_LEN:
        return False, f'contributor debe tener <= {MAX_CONTRIBUTOR_LEN} caracteres'

    samples = data.get('samples')
    if not isinstance(samples, list) or not samples:
        return False, 'samples debe ser una lista no vacía'
    if len(samples) > MAX_SAMPLES_PER_FILE:
        return False, f'máximo {MAX_SAMPLES_PER_FILE} muestras por archivo'

    features = data.get('features')
    if not isinstance(features, int) or features < MIN_FEATURES or features > MAX_FEATURES:
        return False, f'features debe ser entero entre {MIN_FEATURES} y {MAX_FEATURES}'

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            return False, f'sample[{i}] debe ser un objeto'
        word = sample.get('word')
        if not isinstance(word, str) or word not in VALID_WORDS:
            return False, f'sample[{i}].word inválido'
        frames = sample.get('frames')
        if not isinstance(frames, list) or not (MIN_FRAMES <= len(frames) <= MAX_FRAMES):
            return False, f'sample[{i}].frames debe tener entre {MIN_FRAMES} y {MAX_FRAMES} frames'
        for j, frame in enumerate(frames):
            if not isinstance(frame, list) or len(frame) != features:
                return False, f'sample[{i}].frames[{j}] debe tener {features} valores'
            for k, v in enumerate(frame):
                if not _is_finite_number(v):
                    return False, f'sample[{i}].frames[{j}][{k}] fuera de rango o no numérico'

    return True, None