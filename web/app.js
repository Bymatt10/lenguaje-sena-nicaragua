/* Traductor LSP en el navegador.
 * Réplica exacta del pipeline Python: MediaPipe Holistic (legacy, mismos 1662
 * keypoints) → normalización a 15 frames → LSTM (TFJS) → texto y voz.
 */

// ---- Constantes (deben coincidir con constants.py) ----
const MIN_LENGTH_FRAMES = 5;
const MARGIN_FRAME = 1;
const DELAY_FRAMES = 3;

const WORDS_TEXT = {
  adios: 'ADIÓS',
  bienvenidos: 'BIENVENIDOS',
  saludar: 'SALUDAR',
  bien: 'BIEN',
  buenas_noches: 'BUENAS NOCHES',
  buenas_tardes: 'BUENAS TARDES',
  buenos_dias: 'BUENOS DÍAS',
  como_estas: 'COMO ESTÁS',
  disculpa: 'DISCULPA',
  gracias: 'GRACIAS',
  hola: 'HOLA',
  mal: 'MAL',
  mas_o_menos: 'MAS O MENOS',
  me_ayudas: 'ME AYUDAS',
  por_favor: 'POR FAVOR',
};

// ---- Estado global ----
let model = null;
let embedModel = null;   // salida de la capa penúltima, para rechazo open-set
let centroids = null;    // centroide del embedding de cada clase
let radii = null;        // radio de aceptación por clase (p95 intra-clase)
let wordIds = [];
let modelFrames = 15;
let lengthKeypoints = 702;  // dimensión de entrada del modelo (features expandidas)
let staticLength = 225;     // dimensión de las features crudas por frame

let kpSeq = [];
let sentence = [];
let countFrame = 0;
let fixFrames = 0;
let recording = false;

// ---- Elementos de la UI ----
const videoEl = document.getElementById('input-video');
const canvasEl = document.getElementById('output-canvas');
const ctx = canvasEl.getContext('2d');
const statusPill = document.getElementById('status-pill');
const fpsBadge = document.getElementById('fps-badge');
const flashWord = document.getElementById('flash-word');
const confidenceBars = document.getElementById('confidence-bars');
const sentenceEl = document.getElementById('sentence');
const modelCheck = document.getElementById('model-check');
const thresholdInput = document.getElementById('threshold');
const thresholdValue = document.getElementById('threshold-value');
const chkVoice = document.getElementById('chk-voice');
const chkFace = document.getElementById('chk-face');
const signsList = document.getElementById('signs-list');
const signCount = document.getElementById('sign-count');
const abcList = document.getElementById('abc-list');
const abcCount = document.getElementById('abc-count');

thresholdInput.addEventListener('input', () => {
  thresholdValue.textContent = `${thresholdInput.value}%`;
});
const opensetInput = document.getElementById('openset');
const opensetValue = document.getElementById('openset-value');
opensetInput.addEventListener('input', () => {
  opensetValue.textContent = `${(opensetInput.value / 10).toFixed(1)}×`;
});
document.getElementById('btn-clear').addEventListener('click', () => { sentence = []; renderSentence(); });
document.getElementById('btn-undo').addEventListener('click', () => { sentence.shift(); renderSentence(); });
document.getElementById('btn-copy').addEventListener('click', () => {
  navigator.clipboard.writeText([...sentence].reverse().join(' '));
});
document.getElementById('btn-speak-sentence').addEventListener('click', () => {
  speak([...sentence].reverse().join('. '));
});

const STATUS_ICONS = {
  loading: 'fa-spinner fa-spin',
  waiting: 'fa-hand',
  capturing: 'fa-circle-dot fa-beat',
};

function setStatus(kind, text) {
  statusPill.className = `status-pill ${kind}`;
  statusPill.innerHTML = `<i class="fa-solid ${STATUS_ICONS[kind] ?? 'fa-circle-info'}"></i> ${text}`;
}

function speak(text) {
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text.toLowerCase());
  u.lang = 'es-ES';
  speechSynthesis.speak(u);
}

// ---- Modelo: misma arquitectura que model.py, pesos desde weights.bin ----
async function loadModelFromBin() {
  const [specRes, binRes, wordsRes, centRes] = await Promise.all([
    fetch('model/weights_spec.json'),
    fetch('model/weights.bin'),
    fetch('words.json'),
    fetch('model/centroids.json'),
  ]);
  const spec = await specRes.json();
  const buf = await binRes.arrayBuffer();
  wordIds = (await wordsRes.json()).word_ids;
  const cent = await centRes.json();
  centroids = cent.centroids;
  radii = cent.radii;
  modelFrames = spec.frames;
  lengthKeypoints = spec.keypoints;
  staticLength = spec.static ?? spec.keypoints;

  // misma arquitectura que train_v2.py
  // recurrentActivation explícito: TFJS usa hardSigmoid por defecto, Keras usa sigmoid
  const m = tf.sequential();
  m.add(tf.layers.lstm({ units: 64, returnSequences: true, recurrentActivation: 'sigmoid', inputShape: [modelFrames, lengthKeypoints] }));
  m.add(tf.layers.dropout({ rate: 0.4 }));
  m.add(tf.layers.lstm({ units: 128, returnSequences: false, recurrentActivation: 'sigmoid' }));
  m.add(tf.layers.dropout({ rate: 0.4 }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: wordIds.length, activation: 'softmax' }));

  let offset = 0;
  const tensors = spec.shapes.map((shape) => {
    const size = shape.reduce((a, b) => a * b, 1);
    const arr = new Float32Array(buf, offset, size);
    offset += size * 4;
    return tf.tensor(arr, shape);
  });
  m.setWeights(tensors);
  tensors.forEach((t) => t.dispose());

  // modelo auxiliar: embedding de la capa penúltima (Dense 64)
  embedModel = tf.model({ inputs: m.inputs, outputs: m.layers[m.layers.length - 2].output });
  return m;
}

// Self-test doble: (1) pesos del modelo, (2) pipeline completo de features
// (normalización temporal + manos locales + velocidades) — ambos vs Python.
async function selfTest() {
  const ref = await (await fetch('model/reference.json')).json();
  const out = tf.tidy(() => model.predict(tf.tensor3d([ref.input])).dataSync());
  const diffModel = Math.max(...ref.expected_output.map((e, i) => Math.abs(e - out[i])));

  let diffPipe = 0;
  if (ref.static_input) {
    const prepared = prepareSequence(
      normalizeKeypoints(ref.static_input.map((f) => Float32Array.from(f)), modelFrames));
    const out2 = tf.tidy(() => model.predict(tf.tensor3d([prepared])).dataSync());
    diffPipe = Math.max(...ref.static_expected_output.map((e, i) => Math.abs(e - out2[i])));
  }

  const ok = diffModel < 1e-2 && diffPipe < 1e-2;
  console.log(`[self-test] modelo: ${diffModel.toExponential(2)} · pipeline: ${diffPipe.toExponential(2)} → ${ok ? 'OK' : 'FALLO'}`);
  modelCheck.textContent = ok ? 'modelo y pipeline verificados ✓'
    : `⚠ difiere de Python (modelo ${diffModel.toFixed(4)}, pipeline ${diffPipe.toFixed(4)})`;
  modelCheck.className = `model-check ${ok ? 'ok' : 'fail'}`;
  return ok;
}

// ---- Features v2: espejo exacto de extract_features() en train_v2.py ----
// pose 33×3 + mano izq 21×3 + mano der 21×3 = 225, relativas al punto medio
// de los hombros y escaladas por el ancho de hombros.
const LH_START = 33 * 3;       // 99
const RH_START = LH_START + 21 * 3; // 162

function extractKeypoints(r) {
  if (!r.poseLandmarks) return null;
  if (!r.leftHandLandmarks && !r.rightHandLandmarks) return null;

  const pose = r.poseLandmarks;
  const cx = (pose[11].x + pose[12].x) / 2;
  const cy = (pose[11].y + pose[12].y) / 2;
  const s = Math.max(Math.hypot(pose[11].x - pose[12].x, pose[11].y - pose[12].y), 1e-3);

  const feats = new Float32Array(staticLength);
  pose.forEach((p, i) => {
    feats[i * 3] = (p.x - cx) / s;
    feats[i * 3 + 1] = (p.y - cy) / s;
    feats[i * 3 + 2] = p.z / s;
  });
  for (const [start, hand] of [[LH_START, r.leftHandLandmarks], [RH_START, r.rightHandLandmarks]]) {
    if (!hand) continue;
    hand.forEach((p, i) => {
      feats[start + i * 3] = (p.x - cx) / s;
      feats[start + i * 3 + 1] = (p.y - cy) / s;
      feats[start + i * 3 + 2] = p.z / s;
    });
  }
  return feats;
}

// ---- Normalización temporal: puerto de evaluate_model.normalize_keypoints ----
function normalizeKeypoints(seq, target) {
  const n = seq.length;
  const dim = seq[0].length;
  if (n === target) return seq;
  if (n > target) {
    const step = n / target;
    const out = [];
    for (let i = 0; i < target; i++) out.push(seq[Math.min(Math.floor(i * step), n - 1)]);
    return out;
  }
  // interpolación lineal cuando hay menos frames
  const out = [];
  for (let t = 0; t < target; t++) {
    const i = (t * (n - 1)) / (target - 1);
    const lo = Math.floor(i), hi = Math.ceil(i), w = i - lo;
    if (lo === hi) { out.push(seq[lo]); continue; }
    const interp = new Float32Array(dim);
    for (let k = 0; k < dim; k++) interp[k] = (1 - w) * seq[lo][k] + w * seq[hi][k];
    out.push(interp);
  }
  return out;
}

// ---- Features v3: espejo exacto de expand_static() y prepare() en train_v2.py ----
const N_HAND = 21;
const HAND_LOCAL = [[LH_START, 225], [RH_START, 225 + N_HAND * 3]];

function expandStatic(f) {
  // (225) → (351): cada mano relativa a su muñeca, escalada por el tamaño de la mano
  const out = new Float32Array(staticLength + N_HAND * 3 * 2);
  out.set(f);
  for (const [start, dst] of HAND_LOCAL) {
    const wx = f[start], wy = f[start + 1], wz = f[start + 2];
    const s = Math.max(Math.hypot(f[start + 27] - wx, f[start + 28] - wy, f[start + 29] - wz), 1e-3);
    for (let i = 0; i < N_HAND; i++) {
      out[dst + i * 3] = (f[start + i * 3] - wx) / s;
      out[dst + i * 3 + 1] = (f[start + i * 3 + 1] - wy) / s;
      out[dst + i * 3 + 2] = (f[start + i * 3 + 2] - wz) / s;
    }
  }
  return out;
}

function prepareSequence(seq15) {
  // (15×225) → (15×702): manos locales + velocidades entre frames
  const exp = seq15.map(expandStatic);
  const half = exp[0].length;
  return exp.map((e, t) => {
    const row = new Array(half * 2).fill(0);
    for (let k = 0; k < half; k++) {
      row[k] = e[k];
      if (t > 0) row[half + k] = e[k] - exp[t - 1][k];
    }
    return row;
  });
}

// ---- Predicción y UI comunicativa ----
// ---- Auto-aprendizaje con confirmación humana ----
// Cada predicción puede confirmarse o corregirse; la secuencia queda guardada
// etiquetada en localStorage y se envía al server (server.py:/contribute) que
// la deposita en dataset_contrib/. Si el server no responde, se conserva el
// archivo descargable como fallback.
const FEEDBACK_KEY = 'lsn_feedback_samples';
const FEEDBACK_MAX_SAMPLES = 200;
const SESSION_KEY = 'lsn_session_v1';
const CONTRIBUTE_URL = window.LSN_CONTRIBUTE_URL || `${location.origin}/contribute`;
const SESSION_URL = window.LSN_SESSION_URL || `${location.origin}/session`;
const FEEDBACK_DELETE_URL = window.LSN_DELETE_URL || `${location.origin}/feedback`;
const feedbackRow = document.getElementById('feedback-row');
const feedbackMeta = document.getElementById('feedback-meta');
const feedbackCount = document.getElementById('feedback-count');
let lastSeqForFeedback = null;
let sessionPromise = null;

function getCookie(name) {
  const m = document.cookie.match(new RegExp('(?:^|; )' + name.replace(/[.$?*|{}()[\]\\]/g, '\\$&') + '=([^;]*)'));
  return m ? decodeURIComponent(m[1]) : null;
}

async function ensureSession() {
  const cached = sessionStorage.getItem(SESSION_KEY);
  if (cached) {
    try {
      const obj = JSON.parse(cached);
      if (obj.expires && Date.now() < new Date(obj.expires).getTime() && obj.token) return obj;
    } catch {}
  }
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    try {
      const res = await fetch(SESSION_URL, { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`session ${res.status}`);
      const data = await res.json();
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(data));
      return data;
    } finally {
      sessionPromise = null;
    }
  })();
  return sessionPromise;
}

function safeName(raw) {
  return String(raw || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').slice(0, 30) || 'anonimo';
}

function escapeHtml(str) {
  return String(str || '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

async function deleteMyData() {
  const session = await ensureSession();
  const headers = {};
  if (session && session.token) headers['X-LSN-Token'] = session.token;
  const name = localStorage.getItem('lsn_last_contributor') || '';
  const url = `${FEEDBACK_DELETE_URL}/${encodeURIComponent(session?.token || '')}?contributor=${encodeURIComponent(name)}`;
  try {
    const res = await fetch(url, { method: 'DELETE', headers, credentials: 'same-origin' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    console.warn('[delete] error:', err);
  }
  try { localStorage.removeItem(FEEDBACK_KEY); } catch {}
  try { localStorage.removeItem('lsn_last_contributor'); } catch {}
  try { sessionStorage.removeItem(SESSION_KEY); } catch {}
  document.getElementById('feedback-meta').hidden = true;
  if (feedbackRow) {
    feedbackRow.hidden = false;
    feedbackRow.textContent = 'Tus muestras locales fueron eliminadas.';
  }
}

function textFromTemplate(strings, ...values) {
  return strings.reduce((acc, s, i) => acc + s + (i < values.length ? String(values[i]).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c])) : ''), '');
}

async function uploadContrib(payload) {
  const session = await ensureSession();
  const headers = { 'Content-Type': 'application/json' };
  if (session && session.token) headers['X-LSN-Token'] = session.token;
  try {
    const res = await fetch(CONTRIBUTE_URL, {
      method: 'POST',
      headers,
      credentials: 'same-origin',
      body: JSON.stringify(payload),
    });
    if (res.status === 401) {
      sessionStorage.removeItem(SESSION_KEY);
      throw new Error('unauthorized');
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[contrib] server no disponible:', err);
    return null;
  }
}

function downloadContrib(payload, filename) {
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

let uploadTimer = null;
function scheduleFeedbackUpload() {
  clearTimeout(uploadTimer);
  // pequeño debounce: si el voluntario confirma varias señas seguidas, suben juntas
  uploadTimer = setTimeout(flushFeedbackUpload, 1500);
}

async function flushFeedbackUpload() {
  const samples = feedbackSamples();
  if (!samples.length) return;
  const payload = {
    language: 'LSN',
    contributor: 'feedback_web',
    date: new Date().toISOString(),
    features: staticLength,
    samples,
  };
  const result = await uploadContrib(payload);
  if (result && result.ok) {
    localStorage.removeItem(FEEDBACK_KEY);
    updateFeedbackCounter();
    feedbackMeta.title = `Última subida: ${result.samples} muestras → ${result.file}`;
  } else {
    feedbackMeta.title = 'Server no disponible: usa el botón de descarga como respaldo';
  }
}

function feedbackSamples() {
  try { return JSON.parse(localStorage.getItem(FEEDBACK_KEY)) ?? []; }
  catch { return []; }
}

function updateFeedbackCounter() {
  const n = feedbackSamples().length;
  feedbackMeta.hidden = n === 0;
  feedbackCount.textContent = n;
}

function resolveFeedback(word) {
  if (lastSeqForFeedback) {
    const samples = feedbackSamples();
    samples.push({
      word,
      frames: lastSeqForFeedback.map((f) => [...f].map((v) => Math.round(v * 1e5) / 1e5)),
    });
    while (samples.length > FEEDBACK_MAX_SAMPLES) samples.shift();
    try {
      localStorage.setItem(FEEDBACK_KEY, JSON.stringify(samples));
    } catch {
      feedbackRow.innerHTML = '<span class="fb-q">Memoria llena: descarga las muestras acumuladas</span>';
      return;
    }
    lastSeqForFeedback = null;
    updateFeedbackCounter();
    scheduleFeedbackUpload();
  }
  feedbackRow.innerHTML = '<span class="fb-q"><i class="fa-solid fa-check"></i> Guardado para entrenamiento</span>';
  setTimeout(() => { feedbackRow.hidden = true; }, 1800);
}

function showFeedback(base) {
  feedbackRow.hidden = false;
  feedbackRow.innerHTML = `
    <span class="fb-q">¿Era <strong>${displayName(base)}</strong>?</span>
    <button class="btn-secondary fb-btn" id="fb-yes"><i class="fa-solid fa-check"></i> Sí</button>
    <button class="btn-secondary fb-btn" id="fb-other">Era otra</button>
    <button class="btn-secondary fb-btn" id="fb-none">No era seña</button>`;
  document.getElementById('fb-yes').onclick = () => resolveFeedback(base);
  document.getElementById('fb-none').onclick = () => resolveFeedback('_nada');
  document.getElementById('fb-other').onclick = () => {
    const options = captureBases()
      .map((b) => `<option value="${b}">${displayName(b)}</option>`).join('');
    feedbackRow.innerHTML = `
      <span class="fb-q">¿Cuál era?</span>
      <select id="fb-select" class="capture-input" style="margin: 0; flex: 1;">${options}</select>
      <button class="btn-primary fb-btn" id="fb-ok"><i class="fa-solid fa-check"></i></button>`;
    document.getElementById('fb-ok').onclick = () =>
      resolveFeedback(document.getElementById('fb-select').value);
  };
}

document.getElementById('feedback-download').addEventListener('click', () => {
  const samples = feedbackSamples();
  if (!samples.length) return;
  const payload = {
    language: 'LSN',
    contributor: 'feedback_web',
    date: new Date().toISOString(),
    features: staticLength,
    samples,
  };
  // Fallback: si las muestras siguen aquí es porque el server no respondió.
  // Igual dejamos descargar para no perderlas.
  downloadContrib(payload, `muestras_feedback_${Date.now()}.json`);
  if (confirm('¿Borrar las muestras locales? Solo si ya las enviaste por otro medio (WhatsApp, etc.).')) {
    localStorage.removeItem(FEEDBACK_KEY);
    updateFeedbackCounter();
  }
});

function predictSequence() {
  lastSeqForFeedback = kpSeq.slice();
  // votación: predecir sobre varios recortes temporales de la captura y promediar,
  // para no depender de cuándo exactamente subieron/bajaron las manos
  const n = kpSeq.length;
  const crops = [
    [0, n],
    [Math.floor(n * 0.1), n],
    [0, Math.ceil(n * 0.9)],
    [Math.floor(n * 0.05), Math.ceil(n * 0.95)],
    [Math.floor(n * 0.15), n],
  ]
    .map(([a, b]) => kpSeq.slice(a, Math.max(b, a + 5)))
    .filter((s) => s.length >= 5);

  const batch = crops.map((s) => prepareSequence(normalizeKeypoints(s, modelFrames)));
  const { probs, emb } = tf.tidy(() => {
    const input = tf.tensor3d(batch);
    return {
      probs: model.predict(input).mean(0).dataSync(),
      emb: embedModel.predict(input).mean(0).dataSync(),
    };
  });

  const ranked = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const best = ranked[0];
  const threshold = thresholdInput.value / 100;

  // open-set: si el embedding queda lejos del centroide de la clase predicha,
  // el gesto no es ninguna seña conocida (aunque el softmax esté "seguro")
  let dist = 0;
  for (let i = 0; i < centroids[best].length; i++) {
    const d = emb[i] - centroids[best][i];
    dist += d * d;
  }
  const ratio = Math.sqrt(dist) / radii[best];
  const factor = opensetInput.value / 10;
  const known = ratio <= factor;
  const accepted = known && probs[best] > threshold;

  renderConfidence(ranked.slice(0, 3), probs, accepted, known, ratio, factor);
  showFeedback(wordIds[best].split('-')[0]);

  if (accepted) {
    const text = displayName(wordIds[best]);
    sentence.unshift(text);
    renderSentence();
    showFlash(text);
    highlightSign(text);
    if (chkVoice.checked) speak(text);
  }
}

function renderConfidence(top, probs, accepted, known = true, ratio = null, factor = null) {
  const notice = known ? '' :
    '<p class="hint openset-notice"><i class="fa-solid fa-circle-question"></i> Movimiento no reconocido como una seña</p>';
  const readout = ratio === null ? '' :
    `<p class="dist-readout">distancia al patrón: ${ratio.toFixed(2)}× · límite del filtro: ${factor.toFixed(1)}×</p>`;
  confidenceBars.innerHTML = notice + top.map((idx, rank) => {
    const pct = (probs[idx] * 100).toFixed(0);
    const name = WORDS_TEXT[wordIds[idx].split('-')[0]] ?? wordIds[idx];
    const cls = rank === 0 ? (accepted ? 'best' : 'best rejected') : '';
    return `<div class="conf-row ${cls}">
      <span class="conf-label">${name}</span>
      <div class="conf-track"><div class="conf-fill" style="width:${pct}%"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>`;
  }).join('') + readout;
}

function renderSentence() {
  sentenceEl.innerHTML = sentence.length
    ? sentence.map((w, i) => `<span class="word-chip ${i === 0 ? 'newest' : ''}">${w}</span>`).join('')
    : '<p class="hint">Las palabras detectadas aparecerán aquí.</p>';
}

// ---- Lista de señas disponibles ----
function displayName(id) {
  const base = id.split('-')[0];
  return WORDS_TEXT[base] ?? base.toUpperCase();
}

const previewEl = document.getElementById('sign-preview');
const previewVideo = document.getElementById('preview-video');
const previewLabel = document.getElementById('preview-label');

function showPreview(chip) {
  const base = chip.dataset.base;
  previewVideo.src = `videos/${base}.mp4`;
  previewVideo.play().catch(() => {});
  previewLabel.textContent = chip.dataset.word;
  previewEl.hidden = false;

  // colocar sobre el chip, sin salirse de la pantalla
  const rect = chip.getBoundingClientRect();
  const w = 240, h = 230;
  let left = Math.min(Math.max(rect.left + rect.width / 2 - w / 2, 8), window.innerWidth - w - 8);
  let top = rect.top - h - 10;
  if (top < 8) top = rect.bottom + 10;
  previewEl.style.left = `${left}px`;
  previewEl.style.top = `${top}px`;
}

function hidePreview() {
  previewEl.hidden = true;
  previewVideo.pause();
  previewVideo.removeAttribute('src');
}

async function renderSignsList() {
  // únicos por id base: hola-der / hola-izq → hola
  const seen = new Map(); // base -> nombre
  for (const id of wordIds) {
    const base = id.split('-')[0];
    if (!seen.has(base)) seen.set(base, displayName(id));
  }
  const chipHtml = ([base, n]) => `<span class="sign-chip" data-base="${base}" data-word="${n}">${n}</span>`;
  const letters = [...seen.entries()].filter(([base]) => base.length <= 2);
  const words = [...seen.entries()].filter(([base]) => base.length > 2);

  signCount.textContent = `(${words.length})`;
  signsList.innerHTML = words.map(chipHtml).join('');
  abcCount.textContent = `(${letters.length})`;
  abcList.innerHTML = letters.map(chipHtml).join('');

  // detectar qué videos de ejemplo existen en web/videos/
  await Promise.all([...document.querySelectorAll('.sign-chip')].map(async (chip) => {
    try {
      const res = await fetch(`videos/${chip.dataset.base}.mp4`, { method: 'HEAD' });
      if (res.ok) {
        chip.classList.add('has-video');
        chip.addEventListener('mouseenter', () => showPreview(chip));
        chip.addEventListener('mouseleave', hidePreview);
        chip.addEventListener('click', () => (previewEl.hidden ? showPreview(chip) : hidePreview()));
      } else {
        chip.title = 'Aún no hay video de ejemplo para esta seña';
      }
    } catch { /* sin video: el chip queda informativo */ }
  }));
}

let signTimer = null;
function highlightSign(text) {
  document.querySelectorAll('.sign-chip.active').forEach((c) => c.classList.remove('active'));
  const chip = document.querySelector(`.sign-chip[data-word="${text}"]`);
  if (!chip) return;
  chip.classList.add('active');
  clearTimeout(signTimer);
  signTimer = setTimeout(() => chip.classList.remove('active'), 2500);
}

let flashTimer = null;
function showFlash(text) {
  flashWord.innerHTML = `<i class="fa-solid fa-check"></i> ${text}`;
  flashWord.hidden = false;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { flashWord.hidden = true; }, 2500);
}

// ---- Máquina de estados: puerto de evaluate_model.evaluate_model ----
function processResults(results) {
  const thereHand = results.leftHandLandmarks || results.rightHandLandmarks;

  if (thereHand || recording) {
    recording = false;
    countFrame++;
    if (countFrame > MARGIN_FRAME) {
      const feats = extractKeypoints(results);
      if (feats) kpSeq.push(feats);
    }
    setStatus('capturing', `Capturando seña… (${kpSeq.length})`);
  } else {
    if (countFrame >= MIN_LENGTH_FRAMES + MARGIN_FRAME) {
      fixFrames++;
      if (fixFrames < DELAY_FRAMES) { recording = true; return; }
      kpSeq = kpSeq.slice(0, -(MARGIN_FRAME + DELAY_FRAMES));
      if (kpSeq.length >= MIN_LENGTH_FRAMES) {
        if (captureMode) captureSequence(kpSeq);
        else predictSequence();
      }
    }
    recording = false;
    fixFrames = 0;
    countFrame = 0;
    kpSeq = [];
    setStatus('waiting', 'Esperando manos…');
  }
}

// ---- Dibujo del esqueleto ----
function drawResults(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  if (chkFace.checked && results.faceLandmarks) {
    drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, { color: 'rgba(255,255,255,0.15)', lineWidth: 0.5 });
  }
  drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#4f8cff', lineWidth: 2 });
  drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#34d399', lineWidth: 2 });
  drawLandmarks(ctx, results.leftHandLandmarks, { color: '#34d399', lineWidth: 1, radius: 2 });
  drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#f87171', lineWidth: 2 });
  drawLandmarks(ctx, results.rightHandLandmarks, { color: '#f87171', lineWidth: 1, radius: 2 });
  ctx.restore();
}

// ---- FPS ----
let lastTime = performance.now();
let fpsEMA = 0;
function trackFps() {
  const now = performance.now();
  const fps = 1000 / (now - lastTime);
  lastTime = now;
  fpsEMA = fpsEMA ? fpsEMA * 0.9 + fps * 0.1 : fps;
  fpsBadge.hidden = false;
  fpsBadge.textContent = `${fpsEMA.toFixed(0)} fps`;
}

// ---- Modo captura: recolectar muestras etiquetadas de voluntarios ----
let captureMode = false;
let capture = null; // { name, idx, samples: [{word, frames}], lastSeq }

const sidePanel = document.getElementById('side-panel');
const capturePanel = document.getElementById('capture-panel');
const captureIntro = document.getElementById('capture-intro');
const captureFlow = document.getElementById('capture-flow');
const captureDone = document.getElementById('capture-done');
const captureNameInput = document.getElementById('contributor-name');
const captureStartBtn = document.getElementById('capture-start');
const captureProgress = document.getElementById('capture-progress');
const captureExample = document.getElementById('capture-example');
const captureWordEl = document.getElementById('capture-word');
const captureStateEl = document.getElementById('capture-state');
const captureRedoBtn = document.getElementById('capture-redo');
const captureSkipBtn = document.getElementById('capture-skip');
const captureNextBtn = document.getElementById('capture-next');

document.getElementById('btn-contribute').addEventListener('click', () => {
  capturePanel.hidden = false;
  sidePanel.classList.add('capturing');
  captureIntro.hidden = false;
  captureFlow.hidden = true;
  captureDone.hidden = true;
});
document.getElementById('btn-delete-data').addEventListener('click', () => {
  if (!confirm('¿Borrar todas tus muestras locales y solicitar la eliminación de las del servidor?')) return;
  deleteMyData();
});
document.getElementById('capture-close').addEventListener('click', () => {
  captureMode = false;
  capture = null;
  capturePanel.hidden = true;
  sidePanel.classList.remove('capturing');
});
captureNameInput.addEventListener('input', () => {
  captureStartBtn.disabled = captureNameInput.value.trim().length < 2;
});
captureStartBtn.addEventListener('click', () => {
  const raw = captureNameInput.value.trim();
  if (!/^[A-Za-z0-9_\-]{2,30}$/.test(raw)) {
    alert('El alias sólo puede tener letras, números, guion y guion bajo (2-30 caracteres).');
    return;
  }
  capture = { name: raw, idx: 0, samples: [], lastSeq: null };
  try { localStorage.setItem('lsn_last_contributor', raw); } catch {}
  captureMode = true;
  captureIntro.hidden = true;
  captureFlow.hidden = false;
  showCaptureWord();
});

function captureBases() {
  return [...new Set(wordIds.map((id) => id.split('-')[0]))];
}

function showCaptureWord() {
  const bases = captureBases();
  const base = bases[capture.idx];
  captureProgress.textContent = `Seña ${capture.idx + 1} de ${bases.length} · ${capture.samples.length} guardadas`;
  captureWordEl.textContent = displayName(base);
  captureStateEl.textContent = 'Haz la seña y baja las manos…';
  captureExample.src = `videos/${base}.mp4`;
  captureExample.play().catch(() => {});
  capture.lastSeq = null;
  captureRedoBtn.disabled = true;
  captureNextBtn.disabled = true;
}

function captureSequence(seq) {
  // guardamos la secuencia cruda (sin normalizar a 15) para poder aumentarla al entrenar
  capture.lastSeq = seq.map((f) => [...f].map((v) => Math.round(v * 1e5) / 1e5));
  captureStateEl.innerHTML = `<i class="fa-solid fa-check"></i> Capturado (${seq.length} frames). ¿Se vio bien? Pasa a la siguiente o repite.`;
  captureRedoBtn.disabled = false;
  captureNextBtn.disabled = false;
}

captureRedoBtn.addEventListener('click', () => {
  capture.lastSeq = null;
  captureStateEl.textContent = 'Haz la seña y baja las manos…';
  captureRedoBtn.disabled = true;
  captureNextBtn.disabled = true;
});
captureSkipBtn.addEventListener('click', () => advanceCapture(false));
captureNextBtn.addEventListener('click', () => advanceCapture(true));

function advanceCapture(save) {
  const bases = captureBases();
  if (save && capture.lastSeq) {
    capture.samples.push({ word: bases[capture.idx], frames: capture.lastSeq });
  }
  capture.idx++;
  if (capture.idx < bases.length) {
    showCaptureWord();
  } else {
    captureMode = false;
    captureFlow.hidden = true;
    captureDone.hidden = false;
    document.getElementById('capture-summary').textContent =
      `¡Gracias, ${capture.name}! Guardaste ${capture.samples.length} de ${bases.length} señas. Enviando…`;
    uploadCaptureBatch();
  }
}

async function uploadCaptureBatch() {
  if (!capture || !capture.samples.length) return;
  const payload = {
    language: 'LSN',
    contributor: capture.name,
    date: new Date().toISOString(),
    features: staticLength,
    samples: capture.samples,
  };
  const summary = document.getElementById('capture-summary');
  const btn = document.getElementById('capture-download');
  btn.disabled = true;
  const result = await uploadContrib(payload);
  const safeNameEsc = escapeHtml(capture.name);
  if (result && result.ok) {
    summary.innerHTML = `¡Gracias, <strong>${safeNameEsc}</strong>! ` +
      `<i class="fa-solid fa-check"></i> ${result.samples} muestras enviadas al servidor ` +
      `(<code>${escapeHtml(result.file)}</code>).`;
    btn.innerHTML = '<i class="fa-solid fa-check"></i> Enviado';
  } else {
    summary.innerHTML = `¡Gracias, <strong>${safeNameEsc}</strong>! ` +
      `<i class="fa-solid fa-triangle-exclamation"></i> No se pudo contactar al servidor. ` +
      `Descarga el archivo y envíalo por WhatsApp a quien administra el proyecto.`;
    btn.innerHTML = '<i class="fa-solid fa-download"></i> Descargar mis muestras';
    btn.disabled = false;
  }
}

document.getElementById('capture-download').addEventListener('click', () => {
  if (!capture || !capture.samples.length) return;
  const payload = {
    language: 'LSN',
    contributor: capture.name,
    date: new Date().toISOString(),
    features: staticLength,
    samples: capture.samples,
  };
  const name = safeName(capture.name);
  downloadContrib(payload, `muestras_lsn_${name}_${Date.now()}.json`);
});

// ---- Traductor inverso: texto → señas animadas ----
// Reproduce los esqueletos digitalizados de la señante (web/signs_anim.json,
// mismas features de 225 dims que el reconocedor) sobre un canvas.
let anims = null;
const t2sInput = document.getElementById('t2s-input');
const t2sPlayBtn = document.getElementById('t2s-play');
const t2sTokens = document.getElementById('t2s-tokens');
const t2sStage = document.getElementById('t2s-stage');
const t2sCanvas = document.getElementById('t2s-canvas');
const t2sCtx = t2sCanvas.getContext('2d');
const t2sCaption = document.getElementById('t2s-caption');

const POSE_LINES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
const HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

function stripAccents(s) {
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

function tokenizeText(text) {
  const phrases = Object.keys(anims.words).filter((w) => w.length > 2);
  const words = stripAccents(text.toLowerCase()).replace(/[^a-zñ ]/g, ' ').split(/\s+/).filter(Boolean);
  const tokens = [];
  for (let i = 0; i < words.length;) {
    let match = null;
    for (const p of phrases) {
      const parts = p.split('_');
      if (parts.every((w, k) => words[i + k] === w) && (!match || parts.length > match.len)) {
        match = { id: p, len: parts.length };
      }
    }
    if (match) {
      tokens.push({ type: 'sign', id: match.id, label: displayName(match.id) });
      i += match.len;
    } else {
      const word = words[i];
      const letters = [];
      for (let c = 0; c < word.length;) {
        if (word.slice(c, c + 2) === 'ch') { letters.push('ch'); c += 2; }
        else { letters.push(word[c]); c += 1; }
      }
      tokens.push({ type: 'spell', word, letters });
      i += 1;
    }
  }
  return tokens;
}

const T2S_HAND_SCALE = 1.6;  // manos agrandadas: son lo importante en una seña
const T2S_POSE_UPPER = 17;   // el encuadre lo definen cabeza y brazos (0–16)

function t2sDisplayPoints(f) {
  // puntos en coordenadas de exhibición: cuerpo superior + manos agrandadas
  const pts = [];
  for (let i = 0; i < T2S_POSE_UPPER; i++) {
    const x = f[i * 3], y = f[i * 3 + 1];
    if (x !== 0 || y !== 0) pts.push([x, y]);
  }
  for (const base of [LH_START, RH_START]) {
    const wx = f[base], wy = f[base + 1];
    if (wx === 0 && wy === 0) continue;
    for (let i = 0; i < 21; i++) {
      pts.push([wx + (f[base + i * 3] - wx) * T2S_HAND_SCALE,
                wy + (f[base + i * 3 + 1] - wy) * T2S_HAND_SCALE]);
    }
  }
  return pts;
}

function animBounds(frames) {
  let minX = 9, maxX = -9, minY = 9, maxY = -9;
  for (const f of frames) {
    for (const [x, y] of t2sDisplayPoints(f)) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  return { minX, maxX, minY, maxY };
}

function drawSkeletonFrame(f, bounds) {
  const W = t2sCanvas.width, H = t2sCanvas.height, m = 40;
  const s = Math.min((W - m * 2) / (bounds.maxX - bounds.minX), (H - m * 2) / (bounds.maxY - bounds.minY));
  const ox = W / 2 - ((bounds.minX + bounds.maxX) / 2) * s;
  const oy = H / 2 - ((bounds.minY + bounds.maxY) / 2) * s;
  const px = (i, base = 0) => ox + f[base + i * 3] * s;
  const py = (i, base = 0) => oy + f[base + i * 3 + 1] * s;
  const zero = (i, base) => f[base + i * 3] === 0 && f[base + i * 3 + 1] === 0;

  const ctx = t2sCtx;
  ctx.clearRect(0, 0, W, H);
  ctx.lineCap = 'round';

  // torso y brazos
  ctx.strokeStyle = '#4f8cff';
  ctx.lineWidth = 7;
  for (const [a, b] of POSE_LINES) {
    if (zero(a, 0) || zero(b, 0)) continue;
    ctx.beginPath();
    ctx.moveTo(px(a), py(a));
    ctx.lineTo(px(b), py(b));
    ctx.stroke();
  }
  // cabeza: círculo en la nariz (landmark 0); hombros distan 1.0 por normalización
  if (!zero(0, 0)) {
    ctx.beginPath();
    ctx.arc(px(0), py(0) - 0.1 * s, 0.42 * s, 0, Math.PI * 2);
    ctx.stroke();
  }
  // manos detalladas con profundidad (2.5D): palma rellena para ver la
  // orientación, grosor/brillo por cercanía y oclusión lejos→cerca
  const PALM_IDX = [0, 1, 5, 9, 13, 17];
  for (const [base, color] of [[LH_START, '#34d399'], [RH_START, '#f87171']]) {
    if (zero(0, base) && zero(9, base)) continue;

    // manos agrandadas alrededor de su muñeca (misma expansión que animBounds)
    const wxr = f[base], wyr = f[base + 1];
    const hx = (i) => ox + (wxr + (f[base + i * 3] - wxr) * T2S_HAND_SCALE) * s;
    const hy = (i) => oy + (wyr + (f[base + i * 3 + 1] - wyr) * T2S_HAND_SCALE) * s;

    const zs = [];
    for (let i = 0; i < 21; i++) zs.push(f[base + i * 3 + 2]);
    const zmin = Math.min(...zs), zmax = Math.max(...zs);
    const zn = (i) => (zmax > zmin ? (zs[i] - zmin) / (zmax - zmin) : 0.5); // 0=cerca 1=lejos

    ctx.beginPath();
    PALM_IDX.forEach((i, k) => (k ? ctx.lineTo(hx(i), hy(i)) : ctx.moveTo(hx(i), hy(i))));
    ctx.closePath();
    ctx.fillStyle = color + '30';
    ctx.fill();

    const segs = HAND_LINES
      .filter(([a, b]) => !zero(a, base) && !zero(b, base))
      .map(([a, b]) => ({ a, b, z: (zn(a) + zn(b)) / 2 }))
      .sort((p, q) => q.z - p.z);
    ctx.strokeStyle = color;
    for (const { a, b, z } of segs) {
      ctx.globalAlpha = 1 - 0.5 * z;
      ctx.lineWidth = 6.5 - 3 * z;
      ctx.beginPath();
      ctx.moveTo(hx(a), hy(a));
      ctx.lineTo(hx(b), hy(b));
      ctx.stroke();
    }
    ctx.fillStyle = color;
    for (const tip of [4, 8, 12, 16, 20]) {
      if (zero(tip, base)) continue;
      ctx.globalAlpha = 1 - 0.4 * zn(tip);
      ctx.beginPath();
      ctx.arc(hx(tip), hy(tip), 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

const t2sVideo = document.getElementById('t2s-video');

function t2sView() {
  return document.querySelector('input[name="t2s-view"]:checked').value;
}

let t2sRun = 0; // token de cancelación: cada ▶ invalida la reproducción anterior

function playVideoClip(wordId, run) {
  return new Promise((resolve) => {
    const done = () => { clearInterval(watch); resolve(); };
    const watch = setInterval(() => {
      if (run !== t2sRun) { t2sVideo.pause(); done(); }
    }, 150);
    t2sVideo.onended = () => setTimeout(done, 200);
    t2sVideo.onerror = done;
    t2sVideo.src = `videos/${wordId}.mp4`;
    t2sVideo.play().catch(done);
  });
}

async function playAnimation(wordId, run) {
  const frames = anims.words[wordId];
  const bounds = animBounds(frames);
  const frameMs = 1000 / anims.fps;
  const t0 = performance.now();
  return new Promise((resolve) => {
    const render = (f) => drawSkeletonFrame(f, bounds);
    function tick(now) {
      if (run !== t2sRun) return resolve();
      // clamp: el timestamp de requestAnimationFrame puede ser anterior a t0
      const t = Math.max(0, (now - t0) / frameMs);
      const lo = Math.floor(t);
      if (lo >= frames.length - 1) {
        render(Float32Array.from(frames[frames.length - 1]));
        return setTimeout(resolve, 250);
      }
      const w = t - lo;
      const interp = frames[lo].map((v, k) => (1 - w) * v + w * frames[lo + 1][k]);
      render(Float32Array.from(interp));
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}

async function playText() {
  if (!anims) return;
  const tokens = tokenizeText(t2sInput.value);
  if (!tokens.length) return;
  const run = ++t2sRun;
  t2sStage.hidden = false;

  // fichas de progreso: señas enteras y letras deletreadas
  const chips = [];
  tokens.forEach((tok) => {
    if (tok.type === 'sign') chips.push({ label: tok.label, anim: tok.id });
    else tok.letters.forEach((l) => chips.push({
      label: l.toUpperCase(), anim: anims.words[l] ? l : null, spellOf: tok.word,
    }));
  });
  t2sTokens.innerHTML = chips.map((c, i) =>
    `<span class="t2s-token${c.anim ? '' : ' missing'}" id="t2s-tok-${i}" title="${c.anim ? '' : 'sin seña aún'}">${c.label}</span>`).join('');

  for (let i = 0; i < chips.length; i++) {
    if (run !== t2sRun) return;
    const chip = chips[i];
    const el = document.getElementById(`t2s-tok-${i}`);
    if (!chip.anim) continue; // letra sin seña: queda tachada
    el.classList.add('current');
    t2sCaption.textContent = chip.spellOf ? `${chip.label} · deletreando "${chip.spellOf}"` : chip.label;
    const view = t2sView();
    t2sVideo.hidden = view !== 'video';
    t2sCanvas.hidden = view !== '2d';
    if (view === 'video') await playVideoClip(chip.anim, run);
    else await playAnimation(chip.anim, run);
    el.classList.remove('current');
    el.classList.add('done');
  }
  if (run === t2sRun) t2sCaption.innerHTML = '<i class="fa-solid fa-check"></i>';
}

t2sPlayBtn.addEventListener('click', playText);
t2sInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') playText(); });

// ---- Arranque ----
async function main() {
  setStatus('loading', 'Cargando…');
  model = await loadModelFromBin();
  renderSignsList();
  fetch('signs_anim.json').then((r) => r.json()).then((a) => { anims = a; }); // no bloquea el arranque
  updateFeedbackCounter();
  tf.tidy(() => model.predict(tf.zeros([1, modelFrames, lengthKeypoints]))); // warmup de shaders
  await selfTest();

  setStatus('loading', 'Iniciando cámara…');
  const holistic = new Holistic({
    locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`,
  });
  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holistic.onResults((results) => {
    trackFps();
    drawResults(results);
    processResults(results);
  });

  const camera = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 640,
    height: 480,
  });
  await camera.start();
  setStatus('waiting', 'Esperando manos…');
}

main().catch((err) => {
  console.error(err);
  statusPill.className = 'status-pill loading';
  statusPill.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Error: ${err.message}`;
});
