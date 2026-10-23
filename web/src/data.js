// Data sources: the paper's simulation design, the NHANES example and CSV files.
// Every source returns {n, p, q, family, x, z, y, names, ...}, with row-major Float64Array
// matrices and y of width 2 ([time, event]) for Cox outcomes.

// xoshiro128** with 53-bit uniforms, seeded through splitmix64. Seeded and
// portable, but not NumPy's generator: simulate() follows the design of
// python/neuralplsi/simulation.py without reproducing its draws.
export class Random {
  constructor(seed) {
    let state = BigInt.asUintN(64, BigInt(seed));
    const words = [];
    for (let k = 0; k < 2; ++k) {
      state = BigInt.asUintN(64, state+0x9e3779b97f4a7c15n);
      let v = state;
      v = BigInt.asUintN(64, (v^(v>>30n))*0xbf58476d1ce4e5b9n);
      v = BigInt.asUintN(64, (v^(v>>27n))*0x94d049bb133111ebn);
      v ^= v>>31n;
      words.push(Number(v & 0xffffffffn), Number(v>>32n));
    }
    this.s = Uint32Array.from(words);
    if (this.s.every(w => w === 0)) this.s[0] = 1;
    this.spare = null;
  }

  next() {
    const s = this.s, rotl = (x, k) => (x << k) | (x >>> (32-k));
    const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0, t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 11);
    return result;
  }

  uniform() { return ((this.next() >>> 5)*67108864+(this.next() >>> 6))/9007199254740992; }

  normal() {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    const r = Math.sqrt(-2*Math.log(1-this.uniform())), angle = 2*Math.PI*this.uniform();
    this.spare = r*Math.sin(angle);
    return r*Math.cos(angle);
  }

  exponential(scale) { return -Math.log(1-this.uniform())*scale; }
}

const unit = v => { const s = Math.hypot(...v); return v.map(e => e/s); };
export const TRUE_BETA = unit([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0]);
export const TRUE_GAMMA = [1, -0.5, 0.5];
export const LINKS = {
  sigmoid: u => (1/(1+Math.exp(-2*u))-0.5)*5,
  sfun: u => (2/(1+Math.exp(-u))-0.2*u-1)*10,
  linear: u => u,
};

function median(values) {
  const s = Float64Array.from(values).sort(), h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h-1]+s[h])/2;
}

// Eight exposures, N(0, S) with unit variances and correlation 0.3; z1, z2 ~ N(0, 1),
// z3 = +-1 with probability one half; outcome g(x'beta) + z'gamma plus N(0, 1) noise,
// Bernoulli on the logit scale, or exponential times with hazard exp(g + z'gamma)
// and exponential censoring with scale median(1/hazard)/0.3.
export function simulate({n, g = 'sigmoid', outcome = 'continuous', seed = 0, censoringRate = 0.3}) {
  if (!Number.isInteger(n) || n < 2) throw new Error('n must be an integer of at least 2');
  if (!(g in LINKS)) throw new Error(`Unknown link ${g}`);
  const rng = new Random(seed), p = 8, q = 3, link = LINKS[g];
  const x = new Float64Array(n*p), z = new Float64Array(n*q), eta = new Float64Array(n);
  const shared = Math.sqrt(0.3), own = Math.sqrt(0.7);
  for (let i = 0; i < n; ++i) {
    const w = rng.normal();
    for (let j = 0; j < p; ++j) x[i*p+j] = shared*w+own*rng.normal();
  }
  for (let j = 0; j < 2; ++j) for (let i = 0; i < n; ++i) z[i*q+j] = rng.normal();
  for (let i = 0; i < n; ++i) z[i*q+2] = rng.uniform() < 0.5 ? 1 : -1;
  for (let i = 0; i < n; ++i) {
    let u = 0;
    for (let j = 0; j < p; ++j) u += x[i*p+j]*TRUE_BETA[j];
    eta[i] = link(u)+z[i*q]*TRUE_GAMMA[0]+z[i*q+1]*TRUE_GAMMA[1]+z[i*q+2]*TRUE_GAMMA[2];
  }
  let y;
  if (outcome === 'continuous') y = eta.map(v => v+rng.normal());
  else if (outcome === 'binary') y = eta.map(v => rng.uniform() < 1/(1+Math.exp(-v)) ? 1 : 0);
  else if (outcome === 'cox') {
    const scale = median(eta.map(v => Math.exp(-v)))/censoringRate;
    const event = eta.map(v => rng.exponential(Math.exp(-v)));
    y = new Float64Array(2*n);
    for (let i = 0; i < n; ++i) {
      const c = rng.exponential(scale);
      y[2*i] = Math.min(event[i], c); y[2*i+1] = event[i] <= c ? 1 : 0;
    }
  } else throw new Error(`Unknown outcome ${outcome}`);
  return {source: 'simulated', family: outcome, n, p, q, x, z, y,
          names: {x: Array.from({length: p}, (_, j) => `x${j+1}`), z: ['z1', 'z2', 'z3'],
                  y: outcome === 'cox' ? ['time', 'event'] : ['y']},
          truth: {beta: TRUE_BETA, gamma: TRUE_GAMMA, g}};
}

// RFC 4180 fields: quotes with doubled-quote escapes, delimiters and line breaks
// inside quotes, CRLF/LF/CR endings, a UTF-8 byte order mark. The delimiter is
// the most frequent of comma, semicolon and tab on the header line. Blank lines
// are skipped; short rows are padded with empty (missing) fields.
export function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const firstLine = text.slice(0, text.search(/\r|\n|$/)).replace(/"[^"]*"/g, '');
  const counts = [',', ';', '\t'].map(d => [d, firstLine.split(d).length-1]);
  const delimiter = counts.reduce((a, b) => b[1] > a[1] ? b : a)[0];
  const records = [];
  let row = [], field = '', quoted = false, wasQuoted = false, i = 0, line = 1;
  const endField = () => { row.push(wasQuoted ? field : field.trim()); field = ''; wasQuoted = false; };
  const endRow = () => {
    endField();
    if (row.length > 1 || row[0] !== '') records.push({fields: row, line});
    row = [];
  };
  while (i < text.length) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i+1] === '"') { field += '"'; i += 2; continue; }
        quoted = false; ++i; continue;
      }
      if (c === '\n') ++line;
      field += c; ++i; continue;
    }
    if (c === '"' && field.trim() === '') { quoted = wasQuoted = true; field = ''; ++i; continue; }
    if (c === delimiter) { endField(); ++i; continue; }
    if (c === '\r' || c === '\n') {
      endRow();
      i += c === '\r' && text[i+1] === '\n' ? 2 : 1; ++line; continue;
    }
    field += c; ++i;
  }
  if (quoted) throw new Error('The file ends inside a quoted field');
  if (field !== '' || wasQuoted || row.length) endRow();
  if (!records.length) throw new Error('The file is empty');
  // Duplicate names get .1, .2, ... as in pandas.
  const seen = new Set();
  const header = records[0].fields.map((name, j) => {
    const base = name || `column_${j+1}`;
    let out = base;
    for (let k = 1; seen.has(out); ++k) out = `${base}.${k}`;
    seen.add(out);
    return out;
  });
  const rows = records.slice(1).map(({fields, line: at}) => {
    if (fields.length > header.length)
      throw new Error(`Line ${at} has ${fields.length} fields but the header has ${header.length}`);
    return fields.concat(Array(header.length-fields.length).fill(''));
  });
  return {header, rows, delimiter};
}

const MISSING = /^(|na|n\/a|nan|null|none|\.|missing)$/i;
const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;
const INFINITE = /^[+-]?(inf|infinity)$/i;

// A number, NaN for a missing token, or undefined for text. Semicolon- and
// tab-separated files may use a decimal comma.
export function parseNumber(text, decimalComma = false) {
  const s = text.trim();
  if (MISSING.test(s)) return NaN;
  if (NUMBER.test(s)) return Number(s);
  if (decimalComma && /^[+-]?\d+,\d+(e[+-]?\d+)?$/i.test(s)) return Number(s.replace(',', '.'));
  if (INFINITE.test(s)) return s.startsWith('-') ? -Infinity : Infinity;
  return undefined;
}

// Numeric columns parse as numbers wherever a value is present; others are text.
export function describeColumns(table) {
  const decimalComma = table.delimiter !== ',';
  return table.header.map((name, j) => {
    const values = new Float64Array(table.rows.length);
    let text = 0, missing = 0;
    const levels = new Set();
    table.rows.forEach((row, i) => {
      const v = parseNumber(row[j], decimalComma);
      if (v === undefined) { ++text; levels.add(row[j].trim()); values[i] = NaN; }
      else { if (Number.isNaN(v)) ++missing; values[i] = v; }
    });
    return {name, values, numeric: text === 0, missing, levels: [...levels].sort()};
  });
}

function zscore(values, rows) {
  let s = 0;
  for (const i of rows) s += values[i];
  const m = s/rows.length;
  let ss = 0;
  for (const i of rows) ss += (values[i]-m)**2;
  const sd = Math.sqrt(ss/rows.length);
  return {mean: m, sd};
}

export const MAX_LEVELS = 20;

// A model matrix from chosen CSV columns. Rows with a missing or non-finite value
// in any chosen column (or a non-positive exposure when logging) are dropped
// and counted. Text covariates are dummy-coded against their first level.
// Standardization uses the mean and population SD of the retained rows.
export function prepareTable(table, {family, outcome, event, exposures, covariates = [], log = false, standardize = false}) {
  const columns = describeColumns(table), byName = new Map(columns.map(c => [c.name, c]));
  const get = name => {
    const c = byName.get(name);
    if (!c) throw new Error(`Unknown column ${name}`);
    return c;
  };
  if (!exposures.length) throw new Error('Choose at least one exposure');
  const used = [outcome, ...(family === 'cox' ? [event] : []), ...exposures, ...covariates];
  if (new Set(used).size !== used.length) throw new Error('A column can play only one role');
  for (const name of [outcome, ...(family === 'cox' ? [event] : []), ...exposures])
    if (!get(name).numeric) throw new Error(`Column ${name} contains text; it must be numeric`);
  const n0 = table.rows.length, dropped = {missing: 0, nonfinite: 0, nonpositive: 0};
  const keep = [];
  for (let i = 0; i < n0; ++i) {
    let reason = null;
    for (const name of used) {
      const c = get(name), v = c.values[i];
      if (c.numeric ? Number.isNaN(v) : MISSING.test(table.rows[i][table.header.indexOf(name)].trim())) reason = 'missing';
      else if (c.numeric && !Number.isFinite(v)) reason = 'nonfinite';
      if (reason) break;
    }
    if (!reason && log) for (const name of exposures) if (get(name).values[i] <= 0) { reason = 'nonpositive'; break; }
    if (reason) ++dropped[reason]; else keep.push(i);
  }
  const n = keep.length;
  if (n < 2) throw new Error(`Only ${n} complete row${n === 1 ? '' : 's'} remain`);
  const p = exposures.length, x = new Float64Array(n*p), scaling = [];
  exposures.forEach((name, j) => {
    const values = get(name).values.map(v => log ? Math.log(v) : v);
    const {mean, sd} = zscore(values, keep);
    if (!(sd > 0)) throw new Error(`Exposure ${name} is constant in the complete rows`);
    scaling.push(standardize ? {mean, sd} : null);
    keep.forEach((r, i) => { x[i*p+j] = standardize ? (values[r]-mean)/sd : values[r]; });
  });
  const zNames = [], zColumns = [];
  for (const name of covariates) {
    const c = get(name), at = table.header.indexOf(name);
    if (c.numeric) { zNames.push(name); zColumns.push(keep.map(r => c.values[r])); continue; }
    const levels = [...new Set(keep.map(r => table.rows[r][at].trim()))].sort();
    if (levels.length > MAX_LEVELS) throw new Error(`Covariate ${name} has ${levels.length} levels; at most ${MAX_LEVELS} are dummy-coded`);
    for (const level of levels.slice(1)) {
      zNames.push(`${name}=${level}`);
      zColumns.push(keep.map(r => table.rows[r][at].trim() === level ? 1 : 0));
    }
  }
  const q = zNames.length, z = new Float64Array(n*q);
  zColumns.forEach((col, j) => col.forEach((v, i) => { z[i*q+j] = v; }));
  const yd = family === 'cox' ? 2 : 1, y = new Float64Array(n*yd);
  keep.forEach((r, i) => {
    y[i*yd] = get(outcome).values[r];
    if (yd === 2) y[2*i+1] = get(event).values[r];
  });
  checkOutcome(family, y, n);
  return {source: 'csv', family, n, p, q, x, z, y, scaling,
          names: {x: exposures.map(name => log ? `log ${name}` : name), z: zNames,
                  y: yd === 2 ? [outcome, event] : [outcome]},
          rows: n0, dropped};
}

// The checks of python/neuralplsi/base.py validate_data, with its messages.
export function checkOutcome(family, y, n) {
  if (family === 'binary') {
    if (!y.every(v => v === 0 || v === 1)) throw new Error('Binary outcomes must be coded 0 or 1');
  } else if (family === 'cox') {
    let events = 0;
    for (let i = 0; i < n; ++i) {
      if (y[2*i] < 0) throw new Error('Cox y must contain nonnegative durations and event indicators');
      if (y[2*i+1] !== 0 && y[2*i+1] !== 1) throw new Error('Cox outcomes need binary event indicators and at least one event');
      events += y[2*i+1];
    }
    if (!events) throw new Error('Cox outcomes need binary event indicators and at least one event');
  }
}

// reproduce/nhanes.py: every exposure is logged (except retinol) and z-scored
// over its available values, triglyceride likewise, before complete cases are
// taken; covariates are age, sex (female = 1) and race dummies (reference
// Non-Hispanic White).
export const NHANES_EXPOSURES = [
  ['a7.a.Tocopherol', 'α-tocopherol'], ['a6.g.tocopherol', 'γ-tocopherol'],
  ['a4.Retinyl.stearate', 'retinyl stearate'], ['a5.Retinol', 'retinol'],
  ['a20.3.3.4.4.5.pncb', "3,3',4,4',5-pncb"], ['a17.PCB194', 'PCB194'],
  ['a22.2.3.4.6.7.8.hxcdf', '2,3,4,6,7,8-hxcdf'], ['a1.trans.b.carotene', 'trans-β-carotene'],
];
export const NHANES_COVARIATES = ['age', 'female', 'Non-Hispanic Black', 'Mexican American', 'Other Race', 'Other Hispanic'];
// The model settings of reproduce/nhanes.py.
export const NHANES_SETTINGS = {family: 'continuous', intercept: true, activation: 'ELU'};

export function prepareNhanes(text) {
  const table = parseCsv(text), columns = describeColumns(table), byName = new Map(columns.map(c => [c.name, c]));
  const all = Array.from(table.rows.keys());
  const standard = values => {
    const rows = all.filter(i => Number.isFinite(values[i])), {mean, sd} = zscore(values, rows);
    return values.map(v => Number.isFinite(v) ? (v-mean)/sd : NaN);
  };
  const logged = values => standard(values.map(v => v > 0 ? Math.log(v) : NaN));
  const exposures = NHANES_EXPOSURES.map(([name]) =>
    name === 'a5.Retinol' ? standard(byName.get(name).values) : logged(byName.get(name).values));
  const outcome = logged(byName.get('triglyceride').values);
  const age = byName.get('age').values, sex = byName.get('sex').values, race = byName.get('race').values;
  const keep = all.filter(i => exposures.every(e => Number.isFinite(e[i])) && Number.isFinite(outcome[i]) &&
    Number.isFinite(age[i]) && [1, 2].includes(sex[i]) && [1, 2, 3, 4, 5].includes(race[i]));
  const n = keep.length, p = exposures.length, q = NHANES_COVARIATES.length;
  const x = new Float64Array(n*p), z = new Float64Array(n*q), y = new Float64Array(n);
  keep.forEach((r, i) => {
    exposures.forEach((e, j) => { x[i*p+j] = e[r]; });
    z.set([age[r], sex[r]-1, +(race[r] === 2), +(race[r] === 3), +(race[r] === 4), +(race[r] === 5)], i*q);
    y[i] = outcome[r];
  });
  return {source: 'nhanes', family: 'continuous', n, p, q, x, z, y, rows: table.rows.length,
          dropped: {missing: table.rows.length-n, nonfinite: 0, nonpositive: 0},
          names: {x: NHANES_EXPOSURES.map(([, label]) => label), z: NHANES_COVARIATES, y: ['log triglyceride (z-score)']}};
}
