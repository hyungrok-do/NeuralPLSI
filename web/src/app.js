// The page: reads the controls, runs fits and the bootstrap in the worker pool,
// and renders estimates, plots and downloads. Nothing leaves the browser.
import {simulate, parseCsv, describeColumns, prepareTable, prepareNhanes, LINKS, NHANES_SETTINGS} from './data.js';
import {NEURAL_DEFAULTS} from './engine.js';
import {Pool, Cancelled} from './pool.js';
import {linkChart, forestChart} from './charts.js';
import {fmt, duration, toCsv, linspace, mergeRows} from './stats.js';

const form = document.getElementById('controls');
const $ = id => document.getElementById(id);
const field = name => form.elements[name];
const pool = new Pool();
const state = {table: null, columns: null, fileName: '', nhanes: null, fit: null, boot: null, busy: false};
// Seeds must leave room for the bootstrap fit seeds seed+1337+b below 2**64.
const MAX_SEED = 2n**64n-1n-1337n-1000n;
const MODELS = [['neural', 'NeuralPLSI', 'var(--series-1)'], ['spline', 'SplinePLSI', 'var(--series-2)']];

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false)
    node.append(c instanceof Node ? c : String(c));
  return node;
}

// ---- Controls

const source = () => form.elements.source.value;

function integer(name, lo, hi, label) {
  const raw = field(name).value.trim(), v = Number(raw);
  if (!/^\d+$/.test(raw) || v < lo || v > hi) throw new Error(`${label} must be an integer from ${lo} to ${hi}`);
  return v;
}

function settings() {
  const family = field('family').value, decay = field('weightDecay').value.trim(), seed = field('seed').value.trim();
  if (decay === '' || !Number.isFinite(Number(decay)) || Number(decay) < 0)
    throw new Error('Weight decay must be finite and nonnegative');
  if (!/^\d+$/.test(seed) || BigInt(seed) > MAX_SEED) throw new Error(`The seed must be an integer from 0 to ${MAX_SEED}`);
  return {family, activation: field('activation').value, hiddenUnits: integer('hiddenUnits', 1, 1024, 'Hidden units'),
          layers: integer('layers', 1, 16, 'Hidden layers'), weightDecay: Number(decay),
          maxIter: integer('maxIter', 0, 1000000, 'Max iterations'), intercept: family !== 'cox' && field('intercept').checked,
          spline: field('spline').checked, seed: BigInt(seed)};
}

const checkedNames = list => [...$(list).querySelectorAll('input:checked')].map(input => input.value);

function readData() {
  if (source() === 'simulated')
    return simulate({n: integer('n', 20, 20000, 'Rows'), g: field('g').value, outcome: field('outcome').value,
                     seed: integer('simSeed', 0, Number.MAX_SAFE_INTEGER, 'The simulation seed')});
  if (source() === 'nhanes') {
    if (!state.nhanes) throw new Error('The NHANES file is still loading');
    return state.nhanes;
  }
  if (!state.table) throw new Error('Open a CSV file');
  return prepareTable(state.table, {family: field('family').value, outcome: field('outcomeColumn').value,
    event: field('eventColumn').value, exposures: checkedNames('exposure-list'), covariates: checkedNames('covariate-list'),
    log: field('logExposures').checked, standardize: field('standardize').checked});
}

function describe(d) {
  const parts = [d.rows ? `${d.n} of ${d.rows} rows used` : `${d.n} rows`];
  const dropped = Object.entries(d.dropped || {}).filter(([, k]) => k);
  if (dropped.length) parts[0] += ` (dropped: ${dropped.map(([why, k]) =>
    `${k} ${{missing: 'missing or text', nonfinite: 'non-finite', nonpositive: 'non-positive for log'}[why]}`).join(', ')})`;
  parts.push(`${d.p} exposure${d.p === 1 ? '' : 's'}`, `${d.q} covariate column${d.q === 1 ? '' : 's'}`);
  if (d.family === 'binary') parts.push(`${d.y.reduce((s, v) => s+v, 0)} cases`);
  if (d.family === 'cox') parts.push(`${d.y.filter((v, i) => i % 2).reduce((s, v) => s+v, 0)} events`);
  return parts.join(' · ');
}

function syncFamily() {
  const s = source(), family = field('family');
  if (s === 'simulated') family.value = field('outcome').value;
  if (s === 'nhanes') family.value = 'continuous';
  family.disabled = s !== 'csv';
  const cox = family.value === 'cox';
  field('intercept').disabled = cox;
  if (cox) field('intercept').checked = false;
  $('event-field').hidden = !cox;
  $('outcome-label').textContent = cox ? 'Time' : 'Outcome';
}

function updateSummary() {
  const out = $('data-summary');
  try {
    out.textContent = describe(readData());
    out.classList.remove('warning');
  } catch (error) {
    out.textContent = error.message;
    out.classList.add('warning');
  }
}

// Everything that defines a fit, to tell when results no longer match the controls.
function controlsKey() {
  const values = [];
  for (const e of form.elements) {
    if (!e.name || e.name === 'B' || e.type === 'file') continue;
    values.push(e.type === 'checkbox' || e.type === 'radio' ? `${e.name}=${e.value}:${e.checked}` : `${e.name}=${e.value}`);
  }
  return JSON.stringify([values, state.fileName, state.table?.rows.length]);
}

function onControls() {
  syncFamily();
  updateSummary();
  const stale = state.fit && controlsKey() !== state.fit.key;
  $('bootstrap').disabled = !state.fit || stale || state.busy;
  document.querySelector('.stale')?.toggleAttribute('hidden', !stale);
}

async function onSource() {
  for (const panel of document.querySelectorAll('[data-panel]')) panel.hidden = panel.dataset.panel !== source();
  // Presets: Python's defaults for the simulation, reproduce/nhanes.py for NHANES,
  // and an intercept for other data, since g is anchored at g(0) = 0.
  if (source() === 'simulated') {
    field('activation').value = NEURAL_DEFAULTS.activation; field('intercept').checked = NEURAL_DEFAULTS.intercept;
  } else if (source() === 'nhanes') {
    field('activation').value = NHANES_SETTINGS.activation; field('intercept').checked = NHANES_SETTINGS.intercept;
  } else field('intercept').checked = true;
  onControls();
  if (source() === 'nhanes' && !state.nhanes) {
    try {
      const response = await fetch('nhanes.csv');
      if (!response.ok) throw new Error(`Could not load the NHANES file (${response.status})`);
      state.nhanes = prepareNhanes(await response.text());
    } catch (error) {
      message(error.message);
    }
    onControls();
  }
}

function columnOption(c) {
  return h('option', {value: c.name}, c.name);
}

async function onFile() {
  const file = field('file').files[0];
  if (!file) return;
  try {
    state.table = parseCsv(await file.text());
    state.columns = describeColumns(state.table);
    state.fileName = file.name;
  } catch (error) {
    state.table = state.columns = null;
    $('csv-columns').hidden = true;
    message(`${file.name}: ${error.message}`);
    return onControls();
  }
  const numeric = state.columns.filter(c => c.numeric);
  if (!numeric.length) message(`${file.name} has no numeric columns`);
  field('outcomeColumn').replaceChildren(...numeric.map(columnOption));
  field('eventColumn').replaceChildren(...numeric.map(columnOption));
  const event = numeric.find(c => /^(event|status|dead|death|died|delta)$/i.test(c.name)) || numeric[1];
  if (event) field('eventColumn').value = event.name;
  const check = (c, name) => h('label', {title: c.name}, h('input', {type: 'checkbox', name, value: c.name}),
    h('span', {}, c.name, c.numeric ? null : h('span', {class: 'kind'}, ` (text, ${c.levels.length})`)));
  $('exposure-list').replaceChildren(...numeric.map(c => check(c, 'exposure')));
  $('covariate-list').replaceChildren(...state.columns.map(c => check(c, 'covariate')));
  $('csv-columns').hidden = false;
  message(null);
  onControls();
}

// ---- Running

function message(text, kind = 'error') {
  const box = $('message');
  box.hidden = !text;
  box.textContent = text || '';
  box.classList.toggle('info', kind === 'info');
}

let timer = null;
function busy(on, text = '') {
  state.busy = on;
  $('fit').disabled = on;
  $('progress').hidden = !on;
  const bar = $('progress').querySelector('progress');
  bar.removeAttribute('value');
  $('progress-text').textContent = text;
  clearInterval(timer);
  if (on) {
    const started = performance.now();
    if (text) timer = setInterval(() => {
      $('progress-text').textContent = `${text} ${duration((performance.now()-started)/1000)}`;
    }, 250);
  }
  onControls();
}

async function runFit() {
  let data, s;
  try { data = readData(); s = settings(); } catch (error) { return message(error.message); }
  message(null);
  const key = controlsKey();
  busy(true, s.spline ? 'Fitting NeuralPLSI and SplinePLSI…' : 'Fitting NeuralPLSI…');
  try {
    const workers = await pool.ensure(s.spline ? 2 : 1);
    const [neural, spline] = await Promise.all([pool.call(workers[0], 'fitNeural', [data, s, s.seed]),
      s.spline ? pool.call(workers[1], 'fitSpline', [data, s.seed]) : null]);
    state.fit = {data, settings: s, key, neural, spline};
    state.boot = null;
    render();
  } catch (error) {
    message(error instanceof Cancelled ? 'Fit cancelled.' : error.message, error instanceof Cancelled ? 'info' : 'error');
  } finally {
    busy(false);
  }
}

async function runBootstrap() {
  const f = state.fit;
  let B;
  try { B = integer('B', 2, 1000, 'B'); } catch (error) { return message(error.message); }
  message(null);
  const {data, settings: s} = f;
  const jobs = {neural: {state: f.neural.state, x: Float32Array.from(data.x), z: Float32Array.from(data.z),
                         y: Float32Array.from(data.y), n: data.n, grid: Float32Array.from(f.neural.grid), seed: s.seed}};
  if (f.spline) jobs.spline = {x: data.x, z: data.z, y: data.y, n: data.n, p: data.p, q: data.q, family: data.family,
                               grid: Float64Array.from(f.spline.grid), seed: s.seed};
  const kinds = Object.keys(jobs), tasks = [], rows = {neural: [], spline: []};
  for (let b = 0; b < B; ++b) for (const kind of kinds) tasks.push({kind, b});
  busy(true);
  const bar = $('progress').querySelector('progress'), started = performance.now();
  let done = 0, frame = 0;
  const show = () => {
    frame = 0;
    const elapsed = (performance.now()-started)/1000, left = done ? elapsed/done*(tasks.length-done) : NaN;
    bar.value = done/tasks.length;
    $('progress-text').textContent = `${done} of ${tasks.length} refits · ${duration(elapsed)} elapsed` +
      (done ? ` · about ${duration(left)} left` : '')+` · ${pool.workers.length} workers`;
  };
  show();
  try {
    await pool.run(jobs, tasks, (task, row) => {
      rows[task.kind][task.b] = row;
      ++done;
      frame ||= requestAnimationFrame(show);
    });
    const boot = {B, seconds: (performance.now()-started)/1000, workers: pool.workers.length};
    for (const kind of kinds) {
      const D = rows[kind][0].length, draws = mergeRows(rows[kind], D);
      boot[kind] = {draws, D, summary: await pool.call(pool.workers[0], 'summary', [draws.slice(), B, D])};
    }
    state.boot = boot;
    render();
  } catch (error) {
    // A failed replicate invalidates the whole bootstrap, as in the native engine.
    pool.cancel();
    message(error instanceof Cancelled ? 'Bootstrap cancelled.' : error.message, error instanceof Cancelled ? 'info' : 'error');
  } finally {
    if (frame) cancelAnimationFrame(frame);
    busy(false);
  }
}

// ---- Results

// Summary rows of a model's bootstrap: mean, SD, lower, upper over columns [from, from+count).
function interval(kind, from, count) {
  const b = state.boot?.[kind];
  if (!b) return null;
  const row = r => Array.from(b.summary.subarray(r*b.D+from, r*b.D+from+count));
  return {se: row(1), lo: row(2), hi: row(3)};
}

function gColumns(kind) {
  const f = state.fit[kind];
  return [state.boot?.[kind] ? state.boot[kind].D-f.grid.length : 0, f.grid.length];
}

function coefficientTable(title, names, block, truth) {
  const f = state.fit, models = MODELS.filter(([k]) => f[k]), p = f.data.p;
  const head1 = h('tr', {class: 'group'}, h('th'), truth ? h('th') : null,
    models.map(([, name]) => h('th', {colspan: 4, style: 'text-align:center'}, name)));
  const head2 = h('tr', {}, h('th', {}, block === 'beta' ? 'Exposure' : 'Covariate'), truth ? h('th', {}, 'Truth') : null,
    models.map(() => ['Estimate', 'SE', '2.5%', '97.5%'].map(t => h('th', {}, t))));
  const cells = (kind, i) => {
    const ci = interval(kind, (block === 'beta' ? 0 : p)+i, 1);
    return estimateCells((block === 'beta' ? f[kind].beta : f[kind].gamma)[i], ci);
  };
  const body = names.map((name, i) => h('tr', {}, h('td', {}, name), truth ? h('td', {}, fmt(truth[i])) : null,
    models.map(([kind]) => cells(kind, i))));
  if (block === 'gamma' && f.settings.intercept)
    body.push(h('tr', {}, h('td', {}, 'Intercept'), truth ? h('td', {}, '0') : null,
      models.map(([kind]) => kind === 'neural' ? estimateCells(f.neural.intercept, interval(kind, p+f.data.q, 1))
                                               : h('td', {class: 'na', colspan: 4}, 'part of g'))));
  return h('div', {class: 'card'}, h('h3', {}, title),
    h('div', {class: 'table-wrap'}, h('table', {}, h('thead', {}, head1, head2), h('tbody', {}, body))));
}

function estimateCells(estimate, ci) {
  return [h('td', {}, fmt(estimate)), ci ? [ci.se[0], ci.lo[0], ci.hi[0]].map(v => h('td', {}, fmt(v)))
                                         : h('td', {class: 'na', colspan: 3}, 'run the bootstrap')];
}

function diagnostics() {
  const f = state.fit, d = f.data, models = MODELS.filter(([k]) => f[k]), truth = d.truth;
  const rows = [['Fit time', m => duration(m.seconds)],
    ['Iterations', m => m.model === 'neural' ? `${m.diagnostics.iterations} L-BFGS` :
      `${m.diagnostics.iterations} BFGS${m.diagnostics.converged ? '' : ', not converged'}`],
    ['Objective', m => `${fmt(m.diagnostics.objective, 4)} ${m.model === 'neural' ? 'penalized' : 'profile'}`]];
  if (d.family === 'continuous') rows.push(['R² (training)', m => fmt(m.metrics.r2)], ['RMSE (training)', m => fmt(m.metrics.rmse)]);
  if (d.family === 'binary') rows.push(['AUC (training)', m => fmt(m.metrics.auc)], ['Log loss (training)', m => fmt(m.metrics.logLoss)]);
  if (d.family === 'cox') rows.push(['C-index (training)', m => fmt(m.metrics.concordance)]);
  if (truth) {
    const link = LINKS[truth.g];
    rows.push(['Angle to true β', m => {
      const c = Math.abs(m.beta.reduce((s, v, j) => s+v*truth.beta[j], 0));
      return `${fmt(Math.acos(Math.min(1, c))*180/Math.PI)}°`;
    }], ['RMSE of g vs truth', m => fmt(Math.sqrt(m.g.reduce((s, v, i) => s+(v-link(m.grid[i]))**2, 0)/m.g.length))]);
  }
  if (state.boot) rows.push(['Bootstrap', () => `B = ${state.boot.B}, ${duration(state.boot.seconds)}, ${state.boot.workers} workers`]);
  return h('div', {class: 'card'}, h('h3', {}, 'Fit diagnostics'), h('div', {class: 'table-wrap'}, h('table', {},
    h('thead', {}, h('tr', {}, h('th'), models.map(([, name]) => h('th', {}, name)))),
    h('tbody', {}, rows.map(([label, value]) => h('tr', {}, h('td', {}, label), models.map(([k]) => h('td', {}, value(f[k])))))))));
}

function download(name, text, type = 'text/csv') {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = h('a', {href: url, download: name});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function estimatesCsv() {
  const f = state.fit, d = f.data, header = ['model', 'parameter', 'estimate', 'se', 'ci_lower', 'ci_upper'];
  if (d.truth) header.push('truth');
  const rows = [];
  for (const [kind, name] of MODELS.filter(([k]) => f[k])) {
    const m = f[kind], params = [...d.names.x.map((n, j) => [`beta[${n}]`, m.beta[j], d.truth?.beta[j]]),
      ...d.names.z.map((n, j) => [`gamma[${n}]`, m.gamma[j], d.truth?.gamma[j]])];
    if (kind === 'neural' && m.intercept !== null) params.push(['intercept', m.intercept, 0]);
    const ci = interval(kind, 0, params.length);
    params.forEach(([label, estimate, truth], i) => {
      const row = [name, label, estimate, ci?.se[i], ci?.lo[i], ci?.hi[i]];
      if (d.truth) row.push(truth);
      rows.push(row);
    });
  }
  return toCsv(header, rows);
}

function drawsCsv(kind) {
  const f = state.fit, d = f.data, b = state.boot[kind], m = f[kind];
  const header = ['replicate', ...d.names.x.map(n => `beta[${n}]`), ...d.names.z.map(n => `gamma[${n}]`)];
  if (kind === 'neural' && m.intercept !== null) header.push('intercept');
  if (kind === 'spline') m.spline.forEach((_, k) => header.push(`spline[${k+1}]`));
  m.grid.forEach(u => header.push(`g(${Number(u.toPrecision(6))})`));
  const rows = [];
  for (let r = 0; r < b.draws.length/b.D; ++r) rows.push([r, ...b.draws.subarray(r*b.D, (r+1)*b.D)]);
  return toCsv(header, rows);
}

function render() {
  const results = $('results'), f = state.fit;
  if (!f) return;
  const d = f.data, models = MODELS.filter(([k]) => f[k]);
  const label = {simulated: `Simulated, g = ${d.truth?.g}`, nhanes: 'NHANES', csv: state.fileName}[d.source];
  const head = h('div', {class: 'card'}, h('div', {class: 'results-head'}, h('h2', {}, 'Results'),
      h('span', {class: 'facts'}, `${label} · ${d.family}`)),
    h('p', {class: 'facts'}, describe(d)),
    h('p', {class: 'stale', hidden: true}, 'The controls changed after this fit. Fit again to use them.'));
  // g(u) with bands, the truth and the index distribution.
  const gCard = h('div', {class: 'card'}, h('h3', {}, 'Link function g(u)'),
    h('div', {class: 'legend'}, models.map(([, name, color]) => h('span', {}, h('i', {class: 'key', style: `border-color:${color}`}), name)),
      state.boot ? h('span', {}, h('i', {class: 'key band', style: `background:${models[0][2]}`}), '95% bootstrap band') : null,
      d.truth ? h('span', {}, h('i', {class: 'key dashed'}), 'Truth') : null),
    h('div', {class: 'chart', id: 'link-chart'}));
  const fCard = h('div', {class: 'card'}, h('h3', {}, 'Exposure weights β'),
    h('div', {class: 'legend'}, models.map(([, name, color]) => h('span', {}, h('i', {class: 'swatch', style: `background:${color}`}), name)),
      state.boot ? h('span', {}, 'lines: 95% intervals') : null, d.truth ? h('span', {}, h('i', {class: 'key diamond'}), 'Truth') : null),
    h('div', {class: 'chart', id: 'forest-chart'}));
  const downloads = h('div', {class: 'card'}, h('h3', {}, 'Downloads'), h('div', {class: 'downloads'},
    h('button', {type: 'button', class: 'small', onclick: () => download('estimates.csv', estimatesCsv())}, 'Estimates (CSV)'),
    models.filter(([k]) => state.boot?.[k]).map(([k, name]) => h('button', {type: 'button', class: 'small',
      onclick: () => download(`${k === 'neural' ? 'neuralplsi' : 'splineplsi'}-bootstrap-draws.csv`, drawsCsv(k))}, `${name} draws (CSV)`)),
    h('button', {type: 'button', class: 'small', onclick: () => download('neuralplsi.nplsi', f.neural.state, 'text/plain')},
      'NeuralPLSI model (.nplsi)')),
    h('p', {class: 'help'}, 'The model file loads with NeuralPLSI.load(path) in the Python package and nplsi_load in C++.' +
      ' Estimates are the fit on all rows; SEs and intervals come from the bootstrap.'));
  results.replaceChildren(head, h('div', {class: 'figures'}, gCard, fCard),
    h('div', {class: 'tables'}, coefficientTable('Exposure weights β', d.names.x, 'beta', d.truth?.beta),
      coefficientTable('Covariate effects γ', d.names.z, 'gamma', d.truth?.gamma)),
    diagnostics(), downloads);
  drawCharts();
  onControls();
}

function drawCharts() {
  const f = state.fit;
  if (!f) return;
  const d = f.data, models = MODELS.filter(([k]) => f[k]);
  const series = models.map(([kind, name, color]) => {
    const m = f[kind], ci = state.boot?.[kind] ? interval(kind, ...gColumns(kind)) : null;
    return {name, color, x: m.grid, y: m.g, lo: ci?.lo, hi: ci?.hi};
  });
  let truth = null;
  if (d.truth) {
    const lo = Math.min(...series.map(s => s.x[0])), hi = Math.max(...series.map(s => s.x.at(-1)));
    const x = Array.from(linspace(lo, hi, 200));
    truth = {x, y: x.map(LINKS[d.truth.g])};
  }
  linkChart($('link-chart'), {series, truth, index: f.neural.index, indexLabel: 'u = Xβ̂; bars: distribution of the fitted NeuralPLSI index'});
  forestChart($('forest-chart'), {rows: d.names.x.map((name, j) => ({name, truth: d.truth?.beta[j]})),
    series: models.map(([kind, name, color]) => {
      const ci = interval(kind, 0, d.p);
      return {name, color, estimate: f[kind].beta, lo: ci?.lo, hi: ci?.hi};
    })});
}

// ---- Wiring

form.addEventListener('input', onControls);
form.addEventListener('change', event => {
  if (event.target.name === 'source') onSource();
  else if (event.target.name === 'file') onFile();
  else onControls();
});
$('fit').addEventListener('click', runFit);
$('bootstrap').addEventListener('click', runBootstrap);
$('cancel').addEventListener('click', () => pool.cancel());
let resize = 0;
addEventListener('resize', () => { clearTimeout(resize); resize = setTimeout(drawCharts, 150); });
onSource();
if ('serviceWorker' in navigator && location.protocol !== 'file:')
  navigator.serviceWorker.register('sw.js').catch(() => {});
// For tests and the browser console.
globalThis.nplsiDemo = {state, pool};
