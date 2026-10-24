// Data sources: the paper's simulation design, the NHANES example and CSV files.
// Each becomes a table that goes through the same column setup (roles.js);
// prepareTable then returns {n, p, q, family, x, z, y, names, ...}, with row-major
// Float64Array matrices and y of width 2 ([time, event]) for Cox outcomes.

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
  threshold: u => 2*Math.tanh(1.5*Math.max(u, 0))+0.1*u,
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

// The strings pandas.read_csv reads as missing by default, matched exactly (NA,
// not na), after the field is trimmed; '' is an empty field. They are missing in
// numeric columns. Text columns keep None as a category (a level like Former or
// Current); '.' and 'missing' are never missing values.
export const NA_TOKENS = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                          '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'];
const NUMERIC_NA = new Set(NA_TOKENS), TEXT_NA = new Set(NA_TOKENS.filter(token => token !== 'None'));
// Linear-time patterns: no nested or adjacent unbounded repeats over the same digits.
const NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i;
const DECIMAL_COMMA = /^[+-]?\d+,\d+(e[+-]?\d+)?$/i;
const THOUSANDS = /^[+-]?\d{1,3}(,\d{3})+$/;
const GROUPS = /^[+-]?\d{1,3}(,\d{3}){2,}$/;       // 1,250,000: only thousands separators
const AMBIGUOUS = /^[+-]?[1-9]\d{0,2},\d{3}$/;      // 1,250: thousands or three decimals
const INFINITE = /^[+-]?(inf|infinity)$/i;

// A number, NaN for a missing token, or undefined for text. With decimalComma,
// 1,5 reads as 1.5 (describeColumns decides this per column).
export function parseNumber(text, decimalComma = false) {
  const s = text.trim();
  if (NUMERIC_NA.has(s)) return NaN;
  if (NUMBER.test(s)) return Number(s);
  if (decimalComma && DECIMAL_COMMA.test(s)) return Number(s.replace(',', '.'));
  if (INFINITE.test(s)) return s.startsWith('-') ? -Infinity : Infinity;
  return undefined;
}

// A comma is a decimal point in a column only when the file is not
// comma-separated and no value in the column has a point; values are trimmed
// and not missing. The column's comma values decide the rest: one with two or
// more groups (1,250,000) can only have thousands separators; one that cannot
// (0,125, 1,5, 1234,5, 2,5e3) shows decimal commas, which then holds for the
// whole column. When every comma value could be either (1,250), the column is
// text. Otherwise values with commas are text.
export function commaRule(delimiter, values) {
  if (delimiter === ',') return {decimalComma: false, why: 'delimiter'};
  if (values.some(s => s.includes('.'))) return {decimalComma: false, why: 'point'};
  if (values.some(s => GROUPS.test(s))) return {decimalComma: false, why: 'thousands'};
  const commas = values.filter(s => DECIMAL_COMMA.test(s));
  if (commas.length && commas.every(s => AMBIGUOUS.test(s))) return {decimalComma: false, why: 'ambiguous'};
  return {decimalComma: true, why: null};
}

// More than this share of missing or non-finite values marks a column as mostly missing.
export const MOSTLY_MISSING = 0.5;
// NeuralPLSI computes in float32: larger magnitudes would become infinite there,
// so they count as non-finite.
export const FLOAT32_MAX = 3.4028234663852886e38;
// Text confounders and numeric confounders marked categorical are dummy-coded
// up to this many levels; numeric columns with at most CODE_LEVELS distinct
// integer values (other than 0/1) are suggested as categorical.
export const MAX_LEVELS = 20;
export const CODE_LEVELS = 10;

// Numeric columns parse as numbers wherever a value is present; others are text.
// Per column: values (NaN for missing and text, +-Infinity beyond FLOAT32_MAX),
// numeric, missing (missing values: NA_TOKENS in numeric columns, the same but
// None in text columns), tokens ([token, count] of the missing values, most
// frequent first), nonfinite (+-Inf and values beyond the float32 range, in
// numeric columns), beyond (those beyond the float32 range), decimalComma
// (numeric with a value written with a decimal comma), levels (the distinct
// non-missing values of a text column, as they are dummy-coded), distinct
// (distinct finite values of a numeric column, counted up to MAX_LEVELS+1), codes
// (numeric, not 0/1, with 2 to CODE_LEVELS distinct integer values: suggested as
// categorical), binary (numeric with every finite value 0 or 1), present (rows
// with a usable value), up to three distinct examples with exampleValues (the
// number read, where it is not what the text says as written, else null), and
// kind: 'numeric', '0/1', 'text' or 'mostly missing'. A text column whose values
// are at least half numbers also has notNumbers (up to three values that are not)
// and, when one of them is a number written with a comma, commas: why commas are
// not read here ('delimiter', 'point', 'thousands' or 'ambiguous', as commaRule).
export function describeColumns(table) {
  const n = table.rows.length;
  return table.header.map((name, j) => {
    const raw = table.rows.map(row => row[j].trim()), filled = raw.filter(s => !NUMERIC_NA.has(s));
    const rule = commaRule(table.delimiter, filled), parsed = raw.map(s => parseNumber(s, rule.decimalComma));
    const numeric = parsed.every(v => v !== undefined), absent = numeric ? NUMERIC_NA : TEXT_NA;
    const values = new Float64Array(n), seen = new Set(), tokens = new Map();
    let missing = 0, nonfinite = 0, beyond = 0, binary = true, integers = true, decimalComma = false;
    parsed.forEach((v, i) => {
      values[i] = NaN;
      if (absent.has(raw[i])) { ++missing; tokens.set(raw[i], (tokens.get(raw[i]) ?? 0)+1); return; }
      if (!numeric) return;
      if (!Number.isFinite(v) || Math.abs(v) > FLOAT32_MAX) {
        ++nonfinite; values[i] = Math.sign(v)*Infinity;
        if (Number.isFinite(v)) ++beyond;
        return;
      }
      values[i] = v;
      if (raw[i].includes(',')) decimalComma = true;
      if (v !== 0 && v !== 1) binary = false;
      if (!Number.isInteger(v)) integers = false;
      if (seen.size <= MAX_LEVELS) seen.add(v);
    });
    const examples = [], exampleValues = [];
    for (let i = 0; i < n && examples.length < 3; ++i) {
      if (absent.has(raw[i]) || examples.includes(raw[i])) continue;
      const v = parsed[i];
      examples.push(raw[i]);
      exampleValues.push(numeric && Number.isFinite(v) && Number(raw[i]) !== v ? v : null);
    }
    const levels = new Set(numeric ? [] : raw.filter(s => !TEXT_NA.has(s)));
    const present = n-missing-nonfinite;
    binary = numeric && binary && present > 0;
    const distinct = numeric ? seen.size : levels.size;
    const kind = n && (missing+nonfinite)/n > MOSTLY_MISSING ? 'mostly missing' : !numeric ? 'text' : binary ? '0/1' : 'numeric';
    const column = {name, index: j, values, numeric, missing, tokens: [...tokens].sort((a, b) => b[1]-a[1]), nonfinite,
                    beyond, present, binary, kind, decimalComma, examples, exampleValues, levels: [...levels].sort(), distinct,
                    codes: numeric && !binary && integers && distinct >= 2 && distinct <= CODE_LEVELS};
    if (!numeric) {
      // Values that keep the column from being numeric, when most of it is numbers.
      const blocking = filled.filter(s => parseNumber(s, rule.decimalComma) === undefined);
      const numberLike = filled.length-blocking.length+blocking.filter(s => DECIMAL_COMMA.test(s) || THOUSANDS.test(s)).length;
      if (2*numberLike >= filled.length) {
        column.notNumbers = [...new Set(blocking)].slice(0, 3);
        if (blocking.some(s => DECIMAL_COMMA.test(s) || THOUSANDS.test(s))) column.commas = rule.why;
      }
    }
    return column;
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

// The rows of a table with a usable value in every listed column, and the count
// of the others by the first reason found: a missing value, a non-finite
// number, or (for the columns in `logged`) a value that is not positive.
export function completeRows(table, columns, used, logged = []) {
  const dropped = {missing: 0, nonfinite: 0, nonpositive: 0}, keep = [];
  for (let i = 0; i < table.rows.length; ++i) {
    let reason = null;
    for (const c of used) {
      const v = c.values[i];
      if (c.numeric ? Number.isNaN(v) : TEXT_NA.has(table.rows[i][c.index].trim())) reason = 'missing';
      else if (c.numeric && !Number.isFinite(v)) reason = 'nonfinite';
      if (reason) break;
    }
    if (!reason) for (const c of logged) if (c.values[i] <= 0) { reason = 'nonpositive'; break; }
    if (reason) ++dropped[reason]; else keep.push(i);
  }
  return {keep, dropped};
}

// The levels of a text column in the given rows, sorted; the first is the
// reference and each other level becomes one indicator column.
export function textLevels(table, column, rows) {
  return [...new Set(rows.map(r => table.rows[r][column.index].trim()))].sort();
}

// The levels of a numeric column treated as categorical, in the given rows,
// sorted by value: the lowest is the reference.
export function numericLevels(column, rows) {
  return [...new Set(rows.map(r => column.values[r]))].sort((a, b) => a-b);
}

// The levels of a confounder dummy-coded in the given rows: text columns, and
// numeric ones listed as categorical.
export const confounderLevels = (table, column, rows) =>
  column.numeric ? numericLevels(column, rows) : textLevels(table, column, rows);

// A model matrix from chosen CSV columns: the chemical exposures X, the
// confounders Z (`confounders`, or `covariates` as before) and the outcome, with
// X and Z columns in the order given. Rows with a missing or non-finite value in
// any chosen column (or a value of zero or below in a logged column) are dropped
// and counted. `log` is true (every exposure), false, or the exposures to log;
// `standardize` z-scores every exposure after its log. Text confounders, and
// numeric ones listed in `categorical`, are dummy-coded against their first
// level (the lowest number). Numeric confounders that enter as they are can be
// logged (`zLog`) and standardized (`zStandardize`), each its own list. A
// continuous outcome can be logged (`logOutcome`)
// and standardized (`standardizeOutcome`). Standardization uses the mean and
// population SD of the retained rows. The result's `transforms` say how each
// model column was made from the file. `columns` may pass describeColumns(table).
export function prepareTable(table, {family, outcome, event, exposures, confounders, covariates = [], log = false,
                                     standardize = false, categorical = [], zLog = [], zStandardize = [],
                                     logOutcome = false, standardizeOutcome = false}, columns = describeColumns(table)) {
  confounders ??= covariates;
  const byName = new Map(columns.map(c => [c.name, c]));
  const get = name => {
    const c = byName.get(name);
    if (!c) throw new Error(`Unknown column ${name}`);
    return c;
  };
  if (!exposures.length) throw new Error('Choose at least one chemical exposure (X)');
  const used = [outcome, ...(family === 'cox' ? [event] : []), ...exposures, ...confounders];
  if (new Set(used).size !== used.length) throw new Error('A column can play only one role');
  for (const name of [outcome, ...(family === 'cox' ? [event] : []), ...exposures])
    if (!get(name).numeric) throw new Error(`Column ${name} contains text; it must be numeric`);
  const logged = new Set(log === true ? exposures : log || []), asLevels = new Set(categorical);
  // Only numeric confounders that enter as they are have a log or a z-score.
  const plain = name => get(name).numeric && !asLevels.has(name);
  const zLogged = new Set(zLog.filter(plain)), zScaled = new Set(zStandardize.filter(plain));
  const continuous = family === 'continuous', logY = continuous && logOutcome, scaleY = continuous && standardizeOutcome;
  const n0 = table.rows.length;
  const {keep, dropped} = completeRows(table, columns, used.map(get),
    [...exposures.filter(name => logged.has(name)), ...confounders.filter(name => zLogged.has(name)),
     ...(logY ? [outcome] : [])].map(get));
  const n = keep.length;
  if (n < 2) throw new Error(`Only ${n} complete ${n === 1 ? 'row remains' : 'rows remain'}`);
  const p = exposures.length, x = new Float64Array(n*p), scaling = [], transforms = {x: [], z: [], y: []};
  exposures.forEach((name, j) => {
    const takeLog = logged.has(name), values = get(name).values.map(v => takeLog ? Math.log(v) : v);
    const {mean, sd} = zscore(values, keep);
    if (!(sd > 0)) throw new Error(`Chemical exposure ${name} is constant in the complete rows`);
    scaling.push(standardize ? {mean, sd} : null);
    keep.forEach((r, i) => { x[i*p+j] = standardize ? (values[r]-mean)/sd : values[r]; });
    transforms.x.push({name: takeLog ? `log ${name}` : name, column: name, log: takeLog,
                       mean: standardize ? mean : null, sd: standardize ? sd : null});
  });
  const zColumns = [];
  for (const name of confounders) {
    const c = get(name), at = c.index;
    if (c.numeric && !asLevels.has(name)) {
      const takeLog = zLogged.has(name), scale = zScaled.has(name), values = c.values.map(v => takeLog ? Math.log(v) : v);
      const {mean, sd} = zscore(values, keep);
      if (scale && !(sd > 0)) throw new Error(`Confounder ${name} is constant in the complete rows`);
      transforms.z.push(takeLog || scale ? {name: takeLog ? `log ${name}` : name, column: name, log: takeLog,
                                            mean: scale ? mean : null, sd: scale ? sd : null} : {name, column: name});
      zColumns.push(keep.map(r => scale ? (values[r]-mean)/sd : values[r]));
      continue;
    }
    const levels = confounderLevels(table, c, keep);
    if (levels.length > MAX_LEVELS) throw new Error(`Confounder ${name} has ${levels.length} levels; at most ${MAX_LEVELS} are dummy-coded`);
    for (const level of levels.slice(1)) {
      transforms.z.push({name: `${name}=${level}`, column: name, equals: level, reference: levels[0]});
      zColumns.push(keep.map(r => (c.numeric ? c.values[r] : table.rows[r][at].trim()) === level ? 1 : 0));
    }
  }
  const q = zColumns.length, z = new Float64Array(n*q);
  zColumns.forEach((col, j) => col.forEach((v, i) => { z[i*q+j] = v; }));
  const yd = family === 'cox' ? 2 : 1, y = new Float64Array(n*yd);
  const time = get(outcome).values.map(v => logY ? Math.log(v) : v), {mean, sd} = zscore(time, keep);
  if (scaleY && !(sd > 0)) throw new Error(`The outcome ${outcome} is constant in the complete rows`);
  keep.forEach((r, i) => {
    y[i*yd] = scaleY ? (time[r]-mean)/sd : time[r];
    if (yd === 2) y[2*i+1] = get(event).values[r];
  });
  checkOutcome(family, y, n);
  transforms.y.push({name: `${logY ? 'log ' : ''}${outcome}${scaleY ? ' (z-score)' : ''}`, column: outcome, log: logY,
                     mean: scaleY ? mean : null, sd: scaleY ? sd : null});
  if (yd === 2) transforms.y.push({name: event, column: event});
  return {source: 'csv', family, n, p, q, x, z, y, scaling, transforms,
          names: {x: transforms.x.map(t => t.name), z: transforms.z.map(t => t.name), y: transforms.y.map(t => t.name)},
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

// The NHANES example as reproduce/nhanes.py sets it up, as a column setup of
// its file: log triglyceride, z-scored; eight exposures in the script's order,
// logged except retinol, z-scored; age, sex and race as confounders, with sex
// and race dummy-coded against their lowest code (male, Non-Hispanic White).
// The file has no missing or nonpositive values, so z-scores over the rows used
// equal the script's z-scores over each column's values.
export const NHANES_EXPOSURES = ['a7.a.Tocopherol', 'a6.g.tocopherol', 'a4.Retinyl.stearate', 'a5.Retinol',
                                 'a20.3.3.4.4.5.pncb', 'a17.PCB194', 'a22.2.3.4.6.7.8.hxcdf', 'a1.trans.b.carotene'];
export const NHANES_SETUP = {family: 'continuous', outcome: 'triglyceride', event: null, exposures: NHANES_EXPOSURES,
                             confounders: ['age', 'sex', 'race'], log: NHANES_EXPOSURES.filter(name => name !== 'a5.Retinol'),
                             standardize: true, categorical: ['sex', 'race'], logOutcome: true, standardizeOutcome: true};
// The model settings of reproduce/nhanes.py.
export const NHANES_SETTINGS = {family: 'continuous', intercept: true, activation: 'ELU'};

// The simulation as a table, so it goes through the same column setup as a
// file: columns x1..x8, z1..z3 and y (time and event for Cox). Numbers are
// written with String(), which reads back as the same doubles. The setup is the
// design itself: no transforms, columns in the order generated.
export function simulatedTable(options) {
  const d = simulate(options), yd = d.names.y.length;
  const rows = Array.from({length: d.n}, (_, i) => [...d.x.subarray(i*d.p, (i+1)*d.p), ...d.z.subarray(i*d.q, (i+1)*d.q),
                                                   ...d.y.subarray(i*yd, (i+1)*yd)].map(String));
  return {table: {header: [...d.names.x, ...d.names.z, ...d.names.y], rows, delimiter: ','}, truth: d.truth,
          setup: {family: d.family, outcome: d.names.y[0], event: d.names.y[1] ?? null, exposures: d.names.x,
                  confounders: d.names.z, log: [], standardize: false, categorical: [], logOutcome: false,
                  standardizeOutcome: false}};
}
