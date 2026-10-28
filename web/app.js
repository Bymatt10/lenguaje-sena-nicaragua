/* Traductor LSN en el navegador.
 * Réplica exacta del pipeline Python (train_v2.py): MediaPipe Holistic → 225
 * features normalizadas → manos locales + velocidades → LSTM (TFJS) → texto y voz.
 */

// ---- Constantes (deben coincidir con constants.py) ----
const MIN_LENGTH_FRAMES = 5;
const MARGIN_FRAME = 1;
const DELAY_FRAMES = 3;

// nombres para mostrar; las lecciones (lessons.json) agregan los del curso
const WORDS_TEXT = {
  adios: 'ADIÓS',
  bienvenidos: 'BIENVENIDOS',
  saludar: 'SALUDAR',
  bien: 'BIEN',
  buenas_noches: 'BUENAS NOCHES',
  buenas_tardes: 'BUENAS TARDES',
  buenos_dias: 'BUENOS DÍAS',
  como_estas: 'CÓMO ESTÁS',
  disculpa: 'DISCULPA',
  gracias: 'GRACIAS',
  hola: 'HOLA',
  mal: 'MAL',
  mas_o_menos: 'MÁS O MENOS',
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
let modelVersion = 3;
let aspectCorrected = false; // modelos entrenados con x,z × ancho/alto (weights_spec.aspect)

let kpSeq = [];
let sentence = [];
let countFrame = 0;
let fixFrames = 0;
let recording = false;
let currentView = 'traducir';

// ---- Elementos de la UI ----
const $ = (id) => document.getElementById(id);
const videoEl = $('input-video');
const canvasEl = $('output-canvas');
const ctx = canvasEl.getContext('2d');
const stageEl = $('stage');
const statusPill = $('status-pill');
const fpsBadge = $('fps-badge');
const flashWord = $('flash-word');
const resultWord = $('result-word');
const confidenceBars = $('confidence-bars');
const sentenceEl = $('sentence');
const modelCheck = $('model-check');
const thresholdInput = $('threshold');
const thresholdValue = $('threshold-value');
const chkVoice = $('chk-voice');
const chkFace = $('chk-face');
const signsList = $('signs-list');
const signCount = $('sign-count');

thresholdInput.addEventListener('input', () => {
  thresholdValue.textContent = `${thresholdInput.value}%`;
});
const opensetInput = $('openset');
const opensetValue = $('openset-value');
opensetInput.addEventListener('input', () => {
  opensetValue.textContent = `${(opensetInput.value / 10).toFixed(1)}×`;
});
$('btn-clear').addEventListener('click', () => { sentence = []; renderSentence(); });
$('btn-undo').addEventListener('click', () => { sentence.shift(); renderSentence(); });
$('btn-copy').addEventListener('click', () => {
  navigator.clipboard.writeText([...sentence].reverse().join(' '));
});
$('btn-speak-sentence').addEventListener('click', () => {
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

function storageGet(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* modo privado */ }
}

// ---- Tema claro / oscuro (claro por defecto en proyectores) ----
const THEME_KEY = 'lsn_theme';
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
applyTheme(storageGet(THEME_KEY, null));
$('btn-theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  const next = dark ? 'light' : 'dark';
  applyTheme(next);
  storageSet(THEME_KEY, next);
});

// ---- Vistas (rutas por #hash) ----
const VIEWS = ['traducir', 'aprender', 'texto', 'proyecto'];

function showView(name) {
  if (!VIEWS.includes(name)) name = 'traducir';
  currentView = name;
  document.querySelectorAll('.view').forEach((v) => { v.hidden = v.dataset.view !== name; });
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.view === name));
  // una sola cámara: el escenario se mueve a la vista que lo usa
  const slot = $(`stage-slot-${name}`);
  if (slot && stageEl.parentElement !== slot) slot.appendChild(stageEl);
  if (name !== 'aprender') stopReference();
  if (name === 'aprender' && lessonIdx === null) openLesson(firstLessonIndex());
  window.scrollTo({ top: 0 });
}
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

// ---- Modelo: misma arquitectura que train_v2.py, pesos desde weights.bin ----
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
  modelVersion = spec.version ?? 3;
  aspectCorrected = spec.aspect === true;

  // recurrentActivation explícito: TFJS usa hardSigmoid por defecto, Keras usa sigmoid
  // recurrentInitializer 'zeros': el orthogonal por defecto es carísimo (congela Brave)
  // y de todos modos setWeights lo sobrescribe abajo
  const m = tf.sequential();
  m.add(tf.layers.lstm({ units: 64, returnSequences: true, recurrentActivation: 'sigmoid', recurrentInitializer: 'zeros', inputShape: [modelFrames, lengthKeypoints] }));
  m.add(tf.layers.dropout({ rate: 0.4 }));
  m.add(tf.layers.lstm({ units: 128, returnSequences: false, recurrentActivation: 'sigmoid', recurrentInitializer: 'zeros' }));
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
  modelCheck.textContent = ok ? 'Modelo y pipeline verificados contra Python ✓'
    : `⚠ difiere de Python (modelo ${diffModel.toFixed(4)}, pipeline ${diffPipe.toFixed(4)})`;
  modelCheck.className = `model-check ${ok ? 'ok' : 'fail'}`;
  return ok;
}

// ---- Features: espejo exacto de extract_features() en train_v2.py ----
// pose 33×3 + mano izq 21×3 + mano der 21×3 = 225, relativas al punto medio
// de los hombros y escaladas por el ancho de hombros. x y z se llevan a
// unidades del alto (× ancho/alto) para que 16:9 y 4:3 den lo mismo.
const LH_START = 33 * 3;       // 99
const RH_START = LH_START + 21 * 3; // 162

function extractKeypoints(r, aspect = 1) {
  if (!r.poseLandmarks) return null;
  if (!r.leftHandLandmarks && !r.rightHandLandmarks) return null;

  const pose = r.poseLandmarks;
  const cx = ((pose[11].x + pose[12].x) / 2) * aspect;
  const cy = (pose[11].y + pose[12].y) / 2;
  const s = Math.max(Math.hypot((pose[11].x - pose[12].x) * aspect, pose[11].y - pose[12].y), 1e-3);

  const feats = new Float32Array(staticLength);
  pose.forEach((p, i) => {
    feats[i * 3] = (p.x * aspect - cx) / s;
    feats[i * 3 + 1] = (p.y - cy) / s;
    feats[i * 3 + 2] = (p.z * aspect) / s;
  });
  for (const [start, hand] of [[LH_START, r.leftHandLandmarks], [RH_START, r.rightHandLandmarks]]) {
    if (!hand) continue;
    hand.forEach((p, i) => {
      feats[start + i * 3] = (p.x * aspect - cx) / s;
      feats[start + i * 3 + 1] = (p.y - cy) / s;
      feats[start + i * 3 + 2] = (p.z * aspect) / s;
    });
  }
  return feats;
}

// ---- Normalización temporal: puerto de train_v2.normalize_length ----
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

function prepareSequence(seqN) {
  // (N×225) → (N×702): manos locales + velocidades entre frames
  const exp = seqN.map(expandStatic);
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
const feedbackRow = $('feedback-row');
const feedbackMeta = $('feedback-meta');
const feedbackCount = $('feedback-count');
let lastSeqForFeedback = null;
let sessionPromise = null;

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
  let session = null;
  try { session = await ensureSession(); } catch (err) { console.warn('[delete] sin sesión:', err); }
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
  feedbackMeta.hidden = true;
  feedbackRow.hidden = false;
  feedbackRow.textContent = 'Tus muestras locales fueron eliminadas.';
}

async function uploadContrib(payload) {
  try {
    const session = await ensureSession();
    const headers = { 'Content-Type': 'application/json' };
    if (session && session.token) headers['X-LSN-Token'] = session.token;
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

function contribPayload(contributor, samples) {
  return { language: 'LSN', contributor, date: new Date().toISOString(), features: staticLength, samples };
}

async function flushFeedbackUpload() {
  const samples = feedbackSamples();
  if (!samples.length) return;
  const result = await uploadContrib(contribPayload('feedback_web', samples));
  if (result && result.ok) {
    localStorage.removeItem(FEEDBACK_KEY);
    updateFeedbackCounter();
    feedbackMeta.title = `Última subida: ${result.samples} muestras → ${result.file}`;
  } else {
    feedbackMeta.title = 'Server no disponible: usa el botón de descarga como respaldo';
  }
}

function feedbackSamples() {
  return storageGet(FEEDBACK_KEY, []);
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
    <span class="fb-q">¿Era <strong>${escapeHtml(displayName(base))}</strong>?</span>
    <button class="btn fb-btn" id="fb-yes"><i class="fa-solid fa-check"></i> Sí</button>
    <button class="btn fb-btn" id="fb-other">Era otra</button>
    <button class="btn fb-btn" id="fb-none">No era seña</button>`;
  $('fb-yes').onclick = () => resolveFeedback(base);
  $('fb-none').onclick = () => resolveFeedback('_nada');
  $('fb-other').onclick = () => {
    const options = captureBases()
      .map((b) => `<option value="${b}">${escapeHtml(displayName(b))}</option>`).join('');
    feedbackRow.innerHTML = `
      <span class="fb-q">¿Cuál era?</span>
      <select id="fb-select" class="select">${options}</select>
      <button class="btn btn-primary fb-btn" id="fb-ok"><i class="fa-solid fa-check"></i></button>`;
    $('fb-ok').onclick = () => resolveFeedback($('fb-select').value);
  };
}

$('feedback-download').addEventListener('click', () => {
  const samples = feedbackSamples();
  if (!samples.length) return;
  // Fallback: si las muestras siguen aquí es porque el server no respondió.
  downloadContrib(contribPayload('feedback_web', samples), `muestras_feedback_${Date.now()}.json`);
  if (confirm('¿Borrar las muestras locales? Solo si ya las enviaste por otro medio (WhatsApp, etc.).')) {
    localStorage.removeItem(FEEDBACK_KEY);
    updateFeedbackCounter();
  }
});

// ---- Predicción ----
function classify(seq) {
  // votación: predecir sobre varios recortes temporales de la captura y promediar,
  // para no depender de cuándo exactamente subieron/bajaron las manos
  const n = seq.length;
  const crops = [
    [0, n],
    [Math.floor(n * 0.1), n],
    [0, Math.ceil(n * 0.9)],
    [Math.floor(n * 0.05), Math.ceil(n * 0.95)],
    [Math.floor(n * 0.15), n],
  ]
    .map(([a, b]) => seq.slice(a, Math.max(b, a + 5)))
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
  const accepted = known && probs[best] > thresholdInput.value / 100;
  return { probs, ranked, best, ratio, factor, known, accepted };
}

function predictSequence() {
  lastSeqForFeedback = kpSeq.slice();
  const r = classify(kpSeq);

  if (currentView === 'aprender') {
    judgeLesson(r);
    return;
  }

  renderConfidence(r);
  showFeedback(wordIds[r.best].split('-')[0]);

  if (r.accepted) {
    const text = displayName(wordIds[r.best]);
    sentence.unshift(text);
    renderSentence();
    showFlash(text);
    highlightSign(wordIds[r.best].split('-')[0]);
    if (chkVoice.checked) speak(text);
  }
}

function renderConfidence({ ranked, probs, best, accepted, known, ratio, factor }) {
  resultWord.textContent = accepted ? displayName(wordIds[best]) : (known ? 'Poco seguro' : 'No reconocida');
  resultWord.className = `result-word ${accepted ? 'ok' : 'rejected'}`;

  const notice = known ? '' :
    '<p class="hint openset-notice"><i class="fa-solid fa-circle-question"></i> El movimiento no se parece a ninguna seña conocida</p>';
  confidenceBars.innerHTML = notice + ranked.slice(0, 3).map((idx, rank) => {
    const cls = rank === 0 ? (accepted ? 'best' : 'best rejected') : '';
    return `<div class="conf-row ${cls}">
      <span class="conf-label">${escapeHtml(displayName(wordIds[idx]))}</span>
      <div class="conf-track"><div class="conf-fill"></div></div>
      <span class="conf-pct">${(probs[idx] * 100).toFixed(0)}%</span>
    </div>`;
  }).join('') +
    `<p class="dist-readout">distancia al patrón: ${ratio.toFixed(2)}× · límite del filtro: ${factor.toFixed(1)}×</p>`;
  // el ancho va por CSSOM: la CSP bloquea atributos style en innerHTML
  const fills = confidenceBars.querySelectorAll('.conf-fill');
  requestAnimationFrame(() => ranked.slice(0, 3).forEach((idx, i) => {
    fills[i].style.width = `${(probs[idx] * 100).toFixed(1)}%`;
  }));
}

function renderSentence() {
  sentenceEl.innerHTML = sentence.length
    ? sentence.map((w, i) => `<span class="word-chip ${i === 0 ? 'newest' : ''}">${escapeHtml(w)}</span>`).join('')
    : '<p class="hint">Las palabras detectadas aparecerán aquí.</p>';
}

// ---- Nombres y lista de señas ----
let lessons = {};   // id → {text, signer, title, clip}

function displayName(id) {
  const base = id.split('-')[0];
  return WORDS_TEXT[base] ?? lessons[base]?.text ?? base.replace(/_/g, ' ').toUpperCase();
}

function captureBases() {
  return [...new Set(wordIds.map((id) => id.split('-')[0]))];
}

function renderSignsList() {
  const bases = captureBases();
  signCount.textContent = `(${bases.length})`;
  signsList.innerHTML = bases
    .map((b) => `<button class="sign-chip" data-base="${b}">${escapeHtml(displayName(b))}</button>`).join('');
  signsList.querySelectorAll('.sign-chip').forEach((chip) => chip.addEventListener('click', () => {
    const idx = lessonOrder.indexOf(chip.dataset.base);
    location.hash = '#aprender';
    if (idx >= 0) openLesson(idx);
  }));
}

let signTimer = null;
function highlightSign(base) {
  document.querySelectorAll('.sign-chip.active').forEach((c) => c.classList.remove('active'));
  const chip = signsList.querySelector(`.sign-chip[data-base="${base}"]`);
  if (!chip) return;
  chip.classList.add('active');
  clearTimeout(signTimer);
  signTimer = setTimeout(() => chip.classList.remove('active'), 2500);
}

let flashTimer = null;
function showFlash(text) {
  flashWord.innerHTML = `<i class="fa-solid fa-check"></i> ${escapeHtml(text)}`;
  flashWord.hidden = false;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { flashWord.hidden = true; }, 2500);
}

// ---- Máquina de estados: puerto de evaluate_model.evaluate_model ----
function processResults(results) {
  const thereHand = results.leftHandLandmarks || results.rightHandLandmarks;

  // video de ejemplo: el clip entero es una seña (se cierra al terminar, ver
  // useVideoUrl); un fotograma sin manos no la corta, igual que en train_v2
  if (!videoEl.srcObject) {
    const feats = extractKeypoints(results, frameAspect());
    if (feats) kpSeq.push(feats);
    if (kpSeq.length) setStatus('capturing', `Analizando el video… (${kpSeq.length})`);
    return;
  }

  if (thereHand || recording) {
    recording = false;
    countFrame++;
    if (countFrame > MARGIN_FRAME) {
      const feats = extractKeypoints(results, frameAspect());
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

function frameAspect() {
  if (!aspectCorrected) return 1;
  return videoEl.videoWidth && videoEl.videoHeight ? videoEl.videoWidth / videoEl.videoHeight : 4 / 3;
}

// ---- Dibujo del esqueleto ----
function drawResults(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  if (chkFace.checked && results.faceLandmarks) {
    drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, { color: 'rgba(255,255,255,0.15)', lineWidth: 0.5 });
  }
  drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#5b8cff', lineWidth: 2 });
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

// ---- Fuente de video: cámara o video de ejemplo ----
// El video de ejemplo pasa por el mismo pipeline que la cámara: sirve para
// demostrar el sistema aunque no haya cámara o la iluminación sea mala.
let camStream = null;
let holistic = null;
const filePick = $('file-pick');
const fileInput = $('file-input');
const demoSelect = $('demo-select');
const stageEmpty = $('stage-empty');

function resetCapture() {
  kpSeq = []; countFrame = 0; fixFrames = 0; recording = false;
}

function fitCanvas() {
  const w = videoEl.videoWidth || 640, h = videoEl.videoHeight || 480;
  canvasEl.width = 640;
  canvasEl.height = Math.round(640 * h / w);
  stageEl.style.aspectRatio = `${w} / ${h}`;
}
videoEl.addEventListener('loadedmetadata', fitCanvas);

function setSourceButtons(kind) {
  document.querySelectorAll('[data-source]').forEach((b) => b.classList.toggle('active', b.dataset.source === kind));
  filePick.hidden = kind !== 'file';
  demoSelect.hidden = kind !== 'file' || demoSelect.options.length === 0;
  stageEl.classList.toggle('mirror', kind === 'camera');
}

async function useCamera() {
  fileRun++; // cancela el recorrido de un video de ejemplo
  setSourceButtons('camera');
  resetCapture();
  if (holistic) await holistic.reset();
  stageEmpty.hidden = true;
  try {
    setStatus('loading', 'Iniciando cámara…');
    camStream = camStream ?? await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
    });
    videoEl.removeAttribute('src');
    videoEl.srcObject = camStream;
    videoEl.loop = false;
    await videoEl.play();
    setStatus('waiting', 'Esperando manos…');
  } catch (err) {
    console.warn('[cámara]', err);
    camStream = null;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    stageEmpty.hidden = false;
    $('stage-empty-text').textContent = 'No se pudo usar la cámara. Puedes probar el sistema con un video de ejemplo.';
    setStatus('error', 'Sin cámara');
  }
}

// Un video de ejemplo se recorre fotograma a fotograma a FILE_FPS fijos (no en
// tiempo real): el resultado no depende de la velocidad del equipo ni de si la
// pestaña está visible, y es reproducible en una presentación.
const FILE_FPS = 15;
let fileRun = 0; // token de cancelación del recorrido del video

function seekTo(t) {
  if (Math.abs(videoEl.currentTime - t) < 1e-3 && videoEl.readyState >= 2) return Promise.resolve();
  return new Promise((resolve) => {
    videoEl.addEventListener('seeked', resolve, { once: true });
    videoEl.currentTime = t;
  });
}

async function useVideoUrl(url) {
  const run = ++fileRun;
  setSourceButtons('file');
  resetCapture();
  stageEmpty.hidden = true;
  if (camStream) { camStream.getTracks().forEach((t) => t.stop()); camStream = null; }
  videoEl.srcObject = null;
  videoEl.loop = false;
  videoEl.src = url;
  try {
    await new Promise((resolve, reject) => {
      videoEl.addEventListener('loadeddata', resolve, { once: true });
      videoEl.addEventListener('error', reject, { once: true });
    });
  } catch (err) {
    console.warn('[video]', err);
    setStatus('error', 'No se pudo abrir el video');
    return;
  }
  videoEl.pause();
  while (run === fileRun) {
    resetCapture();
    // MediaPipe suaviza entre fotogramas: sin reiniciar, el primer fotograma
    // arrastra la pose del clip anterior y mete un salto falso en las velocidades
    await holistic.reset();
    for (let t = 0; t < videoEl.duration && run === fileRun; t += 1 / FILE_FPS) {
      await seekTo(t);
      await holistic.send({ image: videoEl });
    }
    if (run !== fileRun) return;
    // en un clip las manos pueden no salir nunca de cuadro: el final del
    // video cuenta como "bajar las manos" y cierra la seña capturada
    if (kpSeq.length >= MIN_LENGTH_FRAMES) {
      if (captureMode) captureSequence(kpSeq);
      else predictSequence();
    }
    resetCapture();
    setStatus('waiting', 'Repitiendo el video…');
    await new Promise((r) => setTimeout(r, 2500));
  }
}

function useFileSource() {
  setSourceButtons('file');
  if (demoSelect.options.length) useVideoUrl(demoSelect.value);
  else if (!videoEl.src) {
    stageEmpty.hidden = false;
    $('stage-empty-text').textContent = 'Elige un video tuyo haciendo señas (no se sube a ningún lado).';
  }
}

document.querySelectorAll('[data-source]').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.source === 'camera') useCamera(); else useFileSource();
}));
$('stage-use-file').addEventListener('click', () => {
  useFileSource();
  if (!demoSelect.options.length) fileInput.click();
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) useVideoUrl(URL.createObjectURL(file));
});
demoSelect.addEventListener('change', () => useVideoUrl(demoSelect.value));

async function loadDemos() {
  // web/demo/manifest.json es opcional y local (no se publica): [{file, label}]
  try {
    const res = await fetch('demo/manifest.json');
    if (!res.ok) return;
    const list = await res.json();
    demoSelect.innerHTML = list
      .map((d) => `<option value="demo/${encodeURIComponent(d.file)}">${escapeHtml(d.label)}</option>`).join('');
  } catch { /* sin demos: queda el selector de archivo */ }
}

let sending = false;
async function frameLoop() {
  if (!sending && holistic && videoEl.readyState >= 2 && !videoEl.paused && !videoEl.ended) {
    sending = true;
    try { await holistic.send({ image: videoEl }); } catch (err) { console.warn('[holistic]', err); }
    sending = false;
  }
  requestAnimationFrame(frameLoop);
}

// ---- Modo captura: recolectar muestras etiquetadas de voluntarios ----
let captureMode = false;
let capture = null; // { name, idx, samples: [{word, frames}], lastSeq }

const capturePanel = $('capture-panel');
const captureIntro = $('capture-intro');
const captureFlow = $('capture-flow');
const captureDone = $('capture-done');
const captureNameInput = $('contributor-name');
const captureStartBtn = $('capture-start');
const captureProgress = $('capture-progress');
const captureWordEl = $('capture-word');
const captureStateEl = $('capture-state');
const captureRedoBtn = $('capture-redo');
const captureSkipBtn = $('capture-skip');
const captureNextBtn = $('capture-next');

$('btn-contribute').addEventListener('click', () => {
  capturePanel.hidden = false;
  captureIntro.hidden = false;
  captureFlow.hidden = true;
  captureDone.hidden = true;
  capturePanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});
$('btn-delete-data').addEventListener('click', () => {
  if (!confirm('¿Borrar todas tus muestras locales y solicitar la eliminación de las del servidor?')) return;
  deleteMyData();
});
$('capture-close').addEventListener('click', () => {
  captureMode = false;
  capture = null;
  capturePanel.hidden = true;
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

function showCaptureWord() {
  const bases = captureBases();
  const base = bases[capture.idx];
  captureProgress.textContent = `Seña ${capture.idx + 1} de ${bases.length} · ${capture.samples.length} guardadas`;
  captureWordEl.textContent = displayName(base);
  captureStateEl.textContent = 'Haz la seña y baja las manos…';
  capture.lastSeq = null;
  captureRedoBtn.disabled = true;
  captureNextBtn.disabled = true;
}

function captureSequence(seq) {
  // guardamos la secuencia cruda (sin normalizar) para poder aumentarla al entrenar
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
    $('capture-summary').textContent =
      `¡Gracias, ${capture.name}! Guardaste ${capture.samples.length} de ${bases.length} señas. Enviando…`;
    uploadCaptureBatch();
  }
}

async function uploadCaptureBatch() {
  if (!capture || !capture.samples.length) return;
  const summary = $('capture-summary');
  const btn = $('capture-download');
  btn.hidden = false;
  btn.disabled = true;
  const result = await uploadContrib(contribPayload(capture.name, capture.samples));
  const safeNameEsc = escapeHtml(capture.name);
  if (result && result.ok) {
    summary.innerHTML = `¡Gracias, <strong>${safeNameEsc}</strong>! ` +
      `<i class="fa-solid fa-check"></i> ${result.samples} muestras enviadas al servidor.`;
    btn.innerHTML = '<i class="fa-solid fa-check"></i> Enviado';
  } else {
    summary.innerHTML = `¡Gracias, <strong>${safeNameEsc}</strong>! ` +
      `<i class="fa-solid fa-triangle-exclamation"></i> No se pudo contactar al servidor. ` +
      `Descarga el archivo y envíalo a quien administra el proyecto.`;
    btn.innerHTML = '<i class="fa-solid fa-download"></i> Descargar mis muestras';
    btn.disabled = false;
  }
}

$('capture-download').addEventListener('click', () => {
  if (!capture || !capture.samples.length) return;
  downloadContrib(contribPayload(capture.name, capture.samples),
    `muestras_lsn_${safeName(capture.name)}_${Date.now()}.json`);
});

// ---- Esqueleto animado (texto → señas y referencia del modo Aprender) ----
// Reproduce los esqueletos digitalizados (web/signs_anim.json, mismas
// features de 225 dims que el reconocedor) sobre un canvas.
let anims = null;

const POSE_LINES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
const HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

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

function drawSkeletonFrame(canvas, f, bounds) {
  const g = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, m = 40;
  const s = Math.min((W - m * 2) / (bounds.maxX - bounds.minX), (H - m * 2) / (bounds.maxY - bounds.minY));
  const ox = W / 2 - ((bounds.minX + bounds.maxX) / 2) * s;
  const oy = H / 2 - ((bounds.minY + bounds.maxY) / 2) * s;
  const px = (i, base = 0) => ox + f[base + i * 3] * s;
  const py = (i, base = 0) => oy + f[base + i * 3 + 1] * s;
  const zero = (i, base) => f[base + i * 3] === 0 && f[base + i * 3 + 1] === 0;

  g.clearRect(0, 0, W, H);
  g.lineCap = 'round';

  // torso y brazos
  g.strokeStyle = '#5b8cff';
  g.lineWidth = 7;
  for (const [a, b] of POSE_LINES) {
    if (zero(a, 0) || zero(b, 0)) continue;
    g.beginPath();
    g.moveTo(px(a), py(a));
    g.lineTo(px(b), py(b));
    g.stroke();
  }
  // cabeza: círculo en la nariz (landmark 0); hombros distan 1.0 por normalización
  if (!zero(0, 0)) {
    g.beginPath();
    g.arc(px(0), py(0) - 0.1 * s, 0.42 * s, 0, Math.PI * 2);
    g.stroke();
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

    g.beginPath();
    PALM_IDX.forEach((i, k) => (k ? g.lineTo(hx(i), hy(i)) : g.moveTo(hx(i), hy(i))));
    g.closePath();
    g.fillStyle = color + '30';
    g.fill();

    const segs = HAND_LINES
      .filter(([a, b]) => !zero(a, base) && !zero(b, base))
      .map(([a, b]) => ({ a, b, z: (zn(a) + zn(b)) / 2 }))
      .sort((p, q) => q.z - p.z);
    g.strokeStyle = color;
    for (const { a, b, z } of segs) {
      g.globalAlpha = 1 - 0.5 * z;
      g.lineWidth = 6.5 - 3 * z;
      g.beginPath();
      g.moveTo(hx(a), hy(a));
      g.lineTo(hx(b), hy(b));
      g.stroke();
    }
    g.fillStyle = color;
    for (const tip of [4, 8, 12, 16, 20]) {
      if (zero(tip, base)) continue;
      g.globalAlpha = 1 - 0.4 * zn(tip);
      g.beginPath();
      g.arc(hx(tip), hy(tip), 3.6, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;
  }
}

// devuelve una promesa que termina al acabar la animación o si `isCancelled()`
function playAnimation(canvas, wordId, isCancelled) {
  const frames = anims.words[wordId];
  const bounds = animBounds(frames);
  const frameMs = 1000 / anims.fps;
  const t0 = performance.now();
  return new Promise((resolve) => {
    const render = (f) => drawSkeletonFrame(canvas, f, bounds);
    function tick(now) {
      if (isCancelled()) return resolve();
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

// ---- Texto → señas ----
const t2sInput = $('t2s-input');
const t2sTokens = $('t2s-tokens');
const t2sStage = $('t2s-stage');
const t2sCanvas = $('t2s-canvas');
const t2sCaption = $('t2s-caption');

function stripAccents(s) {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '');
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

let t2sRun = 0; // token de cancelación: cada ▶ invalida la reproducción anterior

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
    `<span class="t2s-token${c.anim ? '' : ' missing'}" id="t2s-tok-${i}" title="${c.anim ? '' : 'sin seña aún'}">${escapeHtml(c.label)}</span>`).join('');

  for (let i = 0; i < chips.length; i++) {
    if (run !== t2sRun) return;
    const chip = chips[i];
    const el = $(`t2s-tok-${i}`);
    if (!chip.anim) continue; // letra sin seña: queda tachada
    el.classList.add('current');
    t2sCaption.textContent = chip.spellOf ? `${chip.label} · deletreando "${chip.spellOf}"` : chip.label;
    await playAnimation(t2sCanvas, chip.anim, () => run !== t2sRun);
    el.classList.remove('current');
    el.classList.add('done');
  }
  if (run === t2sRun) t2sCaption.innerHTML = '<i class="fa-solid fa-check"></i>';
}

$('t2s-play').addEventListener('click', playText);
t2sInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') playText(); });

// ---- Aprender: mira la seña del curso, imítala y el modelo te corrige ----
const LEARN_KEY = 'lsn_learned';
let lessonOrder = [];    // ids: primero las que el modelo reconoce
let lessonIdx = null;
let refMode = 'video';
let refRun = 0;
const lessonGrid = $('lesson-grid');
const lessonVerdict = $('lesson-verdict');
const refVideo = $('ref-video');
const refCanvas = $('ref-canvas');
const refMissing = $('ref-missing');

function learned() { return storageGet(LEARN_KEY, {}); }

function buildLessonOrder() {
  const inModel = new Set(captureBases());
  const all = new Set([...inModel, ...Object.keys(lessons)]);
  lessonOrder = [...all].sort((a, b) =>
    (inModel.has(b) - inModel.has(a)) || displayName(a).localeCompare(displayName(b), 'es'));
}

function firstLessonIndex() {
  const done = learned();
  const i = lessonOrder.findIndex((id) => captureBases().includes(id) && !done[id]);
  return i >= 0 ? i : 0;
}

function renderLessonGrid() {
  const inModel = new Set(captureBases());
  const done = learned();
  const q = stripAccents($('lesson-search').value.trim().toLowerCase());
  const items = lessonOrder
    .map((id, i) => ({ id, i }))
    .filter(({ id }) => !q || stripAccents(displayName(id).toLowerCase()).includes(q));
  $('lesson-count').textContent = `(${lessonOrder.length})`;
  lessonGrid.innerHTML = items.map(({ id, i }) => {
    const practice = inModel.has(id);
    const status = done[id] ? '<span class="done"><i class="fa-solid fa-circle-check"></i> aprendida</span>'
      : `<span><span class="dot ${practice ? 'dot-practice' : 'dot-watch'}"></span>${practice ? 'practicar' : 'ver'}</span>`;
    return `<button class="lesson-tile${i === lessonIdx ? ' current' : ''}" data-i="${i}">
      <strong>${escapeHtml(displayName(id))}</strong>${status}</button>`;
  }).join('');
  lessonGrid.querySelectorAll('.lesson-tile').forEach((t) =>
    t.addEventListener('click', () => { openLesson(Number(t.dataset.i)); $('lesson').scrollIntoView({ behavior: 'smooth' }); }));

  const practicable = lessonOrder.filter((id) => inModel.has(id));
  const nDone = practicable.filter((id) => done[id]).length;
  $('learn-progress').textContent = practicable.length ? `${nDone} / ${practicable.length} aprendidas` : '';
}
$('lesson-search').addEventListener('input', renderLessonGrid);

function stopReference() {
  refRun++;
  refVideo.innerHTML = '';
}

function playReference() {
  const id = lessonOrder[lessonIdx];
  const lesson = lessons[id];
  const hasAnim = anims && anims.words[id];
  const hasClip = Boolean(lesson?.clip);
  stopReference();
  const run = refRun;

  const mode = refMode === 'video' && hasClip ? 'video'
    : (hasAnim ? 'skeleton' : (hasClip ? 'video' : 'none'));
  document.querySelectorAll('[data-ref]').forEach((b) => b.classList.toggle('active', b.dataset.ref === mode));
  refVideo.hidden = mode !== 'video';
  refCanvas.hidden = mode !== 'skeleton';
  refMissing.hidden = mode !== 'none';
  $('lesson-credit').textContent = lesson
    ? `Fuente: ${lesson.title || 'Curso Básico de Lenguaje de Señas'} · CD-SINAPRED / ANSNIC`
    : '';

  if (mode === 'video') {
    // clip local recortado del curso (web/clips/, se genera con tools/export_lessons.py)
    const video = document.createElement('video');
    video.src = lesson.clip;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.autoplay = true;
    // si el clip no está en este equipo, se cae al esqueleto
    video.addEventListener('error', () => {
      if (run !== refRun) return;
      lesson.clip = null;
      playReference();
    });
    refVideo.appendChild(video);
  } else if (mode === 'skeleton') {
    (async () => {
      while (run === refRun) {
        await playAnimation(refCanvas, id, () => run !== refRun);
        await new Promise((r) => setTimeout(r, 600));
      }
    })();
  }
}

function openLesson(i) {
  if (!lessonOrder.length) return;
  lessonIdx = (i + lessonOrder.length) % lessonOrder.length;
  const id = lessonOrder[lessonIdx];
  $('lesson').hidden = false;
  $('lesson-word').textContent = displayName(id);
  const practice = captureBases().includes(id);
  lessonVerdict.className = 'verdict';
  lessonVerdict.innerHTML = practice
    ? '<p class="hint">Haz la seña frente a la cámara y baja las manos.</p>'
    : '<p class="hint">El modelo todavía no reconoce esta seña: por ahora solo puedes verla.</p>';
  if (currentView === 'aprender') playReference();
  renderLessonGrid();
}

document.querySelectorAll('[data-ref]').forEach((b) => b.addEventListener('click', () => {
  refMode = b.dataset.ref;
  playReference();
}));
$('ref-replay').addEventListener('click', playReference);
$('lesson-prev').addEventListener('click', () => openLesson(lessonIdx - 1));
$('lesson-next').addEventListener('click', () => openLesson(lessonIdx + 1));

function judgeLesson(r) {
  const target = lessonOrder[lessonIdx];
  const t = wordIds.findIndex((w) => w.split('-')[0] === target);
  if (t < 0) return;
  const p = r.probs[t];
  const pct = `${(p * 100).toFixed(0)}%`;
  const hit = r.best === t && r.known && p > thresholdInput.value / 100;
  if (hit) {
    const done = learned();
    done[target] = true;
    storageSet(LEARN_KEY, done);
    lessonVerdict.className = 'verdict ok';
    lessonVerdict.innerHTML = `<span class="score">${pct}</span><strong><i class="fa-solid fa-circle-check"></i> ¡Bien hecho!</strong>
      <p>El modelo reconoció <strong>${escapeHtml(displayName(target))}</strong>.</p>`;
    if (chkVoice.checked) speak(`bien hecho. ${displayName(target)}`);
    renderLessonGrid();
  } else {
    const seen = r.known ? `Se pareció más a <strong>${escapeHtml(displayName(wordIds[r.best]))}</strong>.`
      : 'El movimiento no se pareció a ninguna seña conocida.';
    lessonVerdict.className = 'verdict bad';
    lessonVerdict.innerHTML = `<span class="score">${pct}</span><strong>Casi. Inténtalo otra vez</strong>
      <p>${seen} Mira la referencia y fíjate en la forma de la mano y el movimiento.</p>`;
  }
}

// ---- El proyecto: métricas reales del último entrenamiento ----
async function renderProject() {
  $('m-classes').textContent = captureBases().length;
  $('arch-frames').textContent = modelFrames;
  let metrics = null;
  try {
    const res = await fetch('model/metrics.json');
    if (res.ok) metrics = await res.json();
  } catch {}
  if (!metrics) {
    $('data-text').textContent = 'Aún no hay métricas exportadas (tools/eval_reps.py --out o evaluate_loso.py).';
    return;
  }
  const pct = (v) => `${(v * 100).toFixed(1).replace('.0', '')}%`;
  $('m-signers').textContent = metrics.signers ?? '—';
  if (metrics.headline) {
    $('m-loso').textContent = pct(metrics.headline.value);
    $('m-loso-label').textContent = metrics.headline.label;
  } else if (metrics.loso_mean != null) {
    $('m-loso').textContent = pct(metrics.loso_mean);
    $('m-loso-label').textContent = 'precisión LOSO (persona nueva)';
  }
  $('data-text').textContent = metrics.data_summary ?? '';

  if (metrics.method === 'repeticiones') {
    $('eval-method').innerHTML = 'Cada seña del curso aparece <strong>dos veces</strong> seguidas: se entrena con una ' +
      'repetición y se evalúa con la otra, en ambos sentidos. Mide si el modelo aprende la seña y no la toma; ' +
      '<strong>no</strong> mide la generalización a otra persona, que se evalúa con voluntarios (LOSO).';
    $('loso-table').innerHTML = `<table class="table"><thead><tr><th>Prueba</th><th>Resultado</th></tr></thead><tbody>` +
      Object.entries(metrics.folds).map(([k, v]) => `<tr><td>top-1 · ${escapeHtml(k)}</td><td>${pct(v)}</td></tr>`).join('') +
      `<tr><td>top-3 (promedio)</td><td>${pct(metrics.top3)}</td></tr>` +
      `<tr><td>azar (1 / ${metrics.classes})</td><td>${pct(metrics.chance)}</td></tr>` +
      '</tbody></table>' +
      (metrics.confused?.length ? `<p class="hint legend">Se confunde en: ${metrics.confused.map((w) => escapeHtml(displayName(w))).join(', ')}.</p>` : '');
  } else if (metrics.loso) {
    $('loso-table').innerHTML = `<table class="table"><thead><tr><th>Persona evaluada (no vista)</th><th>Muestras</th><th>Precisión</th></tr></thead><tbody>` +
      Object.entries(metrics.loso).map(([who, v]) =>
        `<tr><td>${escapeHtml(who)}</td><td>${v.n ?? '—'}</td><td>${(v.acc * 100).toFixed(0)}%</td></tr>`).join('') +
      '</tbody></table>';
  }
}

// ---- Arranque ----
async function main() {
  showView(location.hash.slice(1));
  setStatus('loading', 'Cargando modelo…');
  const lessonsReady = fetch('lessons.json').then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  fetch('signs_anim.json').then((r) => r.json()).then((a) => { anims = a; }).catch(() => {});
  loadDemos();

  model = await loadModelFromBin();
  lessons = (await lessonsReady).words ?? {};
  buildLessonOrder();
  renderSignsList();
  renderLessonGrid();
  if (currentView === 'aprender') openLesson(firstLessonIndex());
  updateFeedbackCounter();
  tf.tidy(() => model.predict(tf.zeros([1, modelFrames, lengthKeypoints]))); // warmup de shaders
  await selfTest();
  renderProject();

  holistic = new Holistic({
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
  frameLoop();
  await useCamera();
}

main().catch((err) => {
  console.error(err);
  statusPill.className = 'status-pill loading';
  statusPill.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Error: ${escapeHtml(err.message)}`;
});
