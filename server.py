"""Servidor Flask endurecido para recibir muestras de Lengua de Señas.

- DEBUG desactivado en producción.
- Tamaño máximo de payload (1 MB) y archivo persistido (512 KB).
- Rate-limit por IP en /contribute.
- Token de sesión firmado (HttpOnly + SameSite=Lax + Secure) requerido en POST.
- Validación de esquema estricto (sin cara, sin NaN/Inf, palabras en allowlist).
- Sin endpoint /upload_video: la inferencia corre en el navegador con TFJS.
"""
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from flask import Flask, abort, jsonify, make_response, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from itsdangerous import BadSignature, SignatureExpired, URLSafeSerializer

from logging_config import configure_logging
from validate_contrib import MAX_FILE_BYTES, MAX_PAYLOAD_BYTES, validate_contribution

app = Flask(__name__)
configure_logging(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_PAYLOAD_BYTES

SECRET = os.environ.get('LSN_SECRET')
if not SECRET:
    if os.environ.get('FLASK_ENV') == 'development':
        SECRET = secrets.token_hex(32)
        app.logger.warning('LSN_SECRET no configurado; usando token efímero (solo dev)')
    else:
        raise SystemExit('LSN_SECRET es obligatorio en producción.')

TOKEN_SALT = 'lsn-volunteer-session-v1'
TOKEN_MAX_AGE = int(os.environ.get('LSN_SESSION_TTL', '86400'))
serializer = URLSafeSerializer(SECRET, salt=TOKEN_SALT)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CONTRIB_PATH = os.path.join(ROOT_PATH, 'dataset_contrib')
QUARANTINE_PATH = os.path.join(CONTRIB_PATH, 'quarantine')
SAFE_NAME = re.compile(r'[^a-z0-9]+')

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=['200 per hour'],
    storage_uri='memory://',
)


def _client_ip():
    fwd = request.headers.get('X-Forwarded-For')
    if fwd:
        return fwd.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def _require_token():
    token = request.headers.get('X-LSN-Token') or request.cookies.get('lsn_session')
    if not token:
        app.logger.warning('rejected: missing token ip=%s', _client_ip())
        abort(401)
    try:
        serializer.loads(token, max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        app.logger.warning('rejected: bad/expired token ip=%s', _client_ip())
        abort(401)


@app.route('/')
def hello():
    return 'LSP Translate'


@app.route('/health')
def health():
    return jsonify({'ok': True, 'service': 'lsp-contrib', 'version': '2.0'})


@app.route('/session', methods=['GET'])
def session():
    raw = secrets.token_urlsafe(24)
    signed = serializer.dumps(raw)
    expires = datetime.now(timezone.utc) + timedelta(seconds=TOKEN_MAX_AGE)
    resp = make_response(jsonify({'token': signed, 'expires': expires.isoformat()}))
    resp.set_cookie(
        'lsn_session', signed,
        max_age=TOKEN_MAX_AGE, httponly=True, samesite='Lax',
        secure=os.environ.get('FLASK_ENV') != 'development', path='/',
    )
    return resp


@app.route('/contribute', methods=['POST'])
@limiter.limit('30 per minute;500 per hour')
def contribute():
    _require_token()

    raw = request.get_data()
    if len(raw) > MAX_PAYLOAD_BYTES:
        app.logger.warning('rejected: oversized payload ip=%s bytes=%d', _client_ip(), len(raw))
        return jsonify({'ok': False, 'error': 'payload demasiado grande'}), 413

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        app.logger.warning('rejected: invalid json ip=%s err=%s', _client_ip(), e)
        return jsonify({'ok': False, 'error': f'JSON inválido: {e}'}), 400

    ok, err = validate_contribution(data)
    if not ok:
        app.logger.warning('rejected: schema ip=%s err=%s', _client_ip(), err)
        return jsonify({'ok': False, 'error': err}), 400

    serialized = json.dumps(data, ensure_ascii=False)
    if len(serialized.encode('utf-8')) > MAX_FILE_BYTES:
        app.logger.warning('rejected: serialized too large ip=%s', _client_ip())
        return jsonify({'ok': False, 'error': 'archivo demasiado grande'}), 413

    os.makedirs(CONTRIB_PATH, exist_ok=True)
    os.makedirs(QUARANTINE_PATH, exist_ok=True)

    contributor = str(data.get('contributor') or 'anonimo')
    safe = SAFE_NAME.sub('_', contributor.lower()).strip('_') or 'anonimo'
    safe = safe[:30]
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    fname = f'muestras_lsn_{safe}_{ts}.json'

    fpath = os.path.join(CONTRIB_PATH, fname)
    if not os.path.realpath(fpath).startswith(os.path.realpath(CONTRIB_PATH) + os.sep):
        app.logger.error('rejected: path traversal ip=%s contributor=%s', _client_ip(), contributor)
        return jsonify({'ok': False, 'error': 'contributor inválido'}), 400

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(serialized)

    app.logger.info('accepted: contributor=%s file=%s samples=%d ip=%s',
                    contributor, fname, len(data['samples']), _client_ip())
    return jsonify({'ok': True, 'file': fname, 'samples': len(data['samples'])})


@app.route('/feedback/<token>', methods=['DELETE'])
def delete_feedback(token):
    try:
        serializer.loads(token, max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        abort(401)

    contributor = str(request.args.get('contributor') or '').strip()
    if contributor:
        safe = SAFE_NAME.sub('_', contributor.lower()).strip('_') or 'anonimo'
        safe = safe[:30]
        removed = 0
        for f in os.listdir(CONTRIB_PATH):
            if not f.endswith('.json'):
                continue
            if f.startswith(f'muestras_lsn_{safe}_'):
                try:
                    os.remove(os.path.join(CONTRIB_PATH, f))
                    removed += 1
                except OSError:
                    pass
        app.logger.info('feedback delete: contributor=%s removed=%d ip=%s',
                        contributor, removed, _client_ip())
        return jsonify({'ok': True, 'removed': removed})
    return jsonify({'ok': True, 'removed': 0})


@app.errorhandler(401)
def _unauth(e):
    return jsonify({'ok': False, 'error': 'unauthorized'}), 401


@app.errorhandler(413)
def _toobig(e):
    return jsonify({'ok': False, 'error': 'payload too large'}), 413


@app.errorhandler(429)
def _ratelimit(e):
    app.logger.warning('rate-limit hit ip=%s path=%s', _client_ip(), request.path)
    return jsonify({'ok': False, 'error': 'rate limit exceeded'}), 429


@app.errorhandler(500)
def _ise(e):
    app.logger.exception('internal error')
    return jsonify({'ok': False, 'error': 'internal error'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)