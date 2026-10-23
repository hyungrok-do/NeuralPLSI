// Small numerical helpers: grids, fit metrics, CSV output and number formatting.

// numpy.linspace: start + i*step, with the last point exactly stop.
export function linspace(start, stop, count) {
  const step = (stop-start)/(count-1), out = new Float64Array(count);
  for (let i = 0; i < count; ++i) out[i] = start+i*step;
  out[count-1] = stop;
  return out;
}

// Rows in replicate order, as nplsi_bootstrap writes them.
export function mergeRows(rows, D) {
  const draws = new Float64Array(rows.length*D);
  rows.forEach((row, b) => draws.set(row, b*D));
  return draws;
}

export function mean(values) {
  let s = 0;
  for (const v of values) s += v;
  return s/values.length;
}

// Average ranks (1-based) with ties sharing their mean rank.
function ranks(values) {
  const order = Array.from(values.keys()).sort((a, b) => values[a]-values[b]), out = new Float64Array(values.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j+1 < order.length && values[order[j+1]] === values[order[i]]) ++j;
    for (let k = i; k <= j; ++k) out[order[k]] = (i+j)/2+1;
    i = j+1;
  }
  return out;
}

export function auc(score, label) {
  const r = ranks(score);
  let positives = 0, sum = 0;
  for (let i = 0; i < label.length; ++i) if (label[i] === 1) { ++positives; sum += r[i]; }
  const negatives = label.length-positives;
  return positives && negatives ? (sum-positives*(positives+1)/2)/(positives*negatives) : NaN;
}

// Harrell's C: among pairs with an event at the earlier time, the share in which
// that subject has the higher risk (risk ties count one half). O(n log n), with
// doubled average ranks as integer positions in a Fenwick tree.
export function concordance(risk, time, event) {
  const n = risk.length, r = ranks(risk), size = 2*n, tree = new Float64Array(size+1);
  const put = i => { for (; i <= size; i += i & -i) tree[i] += 1; };
  const count = i => { let s = 0; for (; i > 0; i -= i & -i) s += tree[i]; return s; };
  const order = Array.from(time.keys()).sort((a, b) => time[b]-time[a]);
  let concordant = 0, comparable = 0, inside = 0;
  for (let i = 0; i < n;) {
    let j = i;
    while (j+1 < n && time[order[j+1]] === time[order[i]]) ++j;
    for (let k = i; k <= j; ++k) {
      const a = order[k];
      if (event[a] !== 1 || !inside) continue;
      const rank = 2*r[a], lower = count(rank-1);
      concordant += lower+(count(rank)-lower)/2; comparable += inside;
    }
    for (let k = i; k <= j; ++k) { put(2*r[order[k]]); ++inside; }
    i = j+1;
  }
  return comparable ? concordant/comparable : NaN;
}

// Goodness of fit of a linear predictor on the training rows.
export function fitMetrics(family, predictor, y) {
  const n = predictor.length;
  if (family === 'continuous') {
    const m = mean(y);
    let sse = 0, sst = 0;
    for (let i = 0; i < n; ++i) { sse += (y[i]-predictor[i])**2; sst += (y[i]-m)**2; }
    return {r2: 1-sse/sst, rmse: Math.sqrt(sse/n)};
  }
  if (family === 'binary') {
    let loss = 0;
    for (let i = 0; i < n; ++i) {
      const v = predictor[i];
      loss += Math.max(v, 0)-y[i]*v+Math.log1p(Math.exp(-Math.abs(v)));
    }
    return {auc: auc(predictor, y), logLoss: loss/n};
  }
  const time = new Float64Array(n), event = new Float64Array(n);
  for (let i = 0; i < n; ++i) { time[i] = y[2*i]; event[i] = y[2*i+1]; }
  return {concordance: concordance(predictor, time, event)};
}

export function csvCell(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '';
  const text = String(value);
  return /[",\n\r]/.test(text) ? '"'+text.replace(/"/g, '""')+'"' : text;
}

export function toCsv(header, rows) {
  return [header, ...rows].map(row => row.map(csvCell).join(',')).join('\n')+'\n';
}

// Fixed significant digits without exponent noise for ordinary magnitudes.
export function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  if (value === 0) return '0';
  const a = Math.abs(value);
  if (a >= 1e5 || a < 1e-4) return value.toExponential(digits-1).replace('-', '−');
  const decimals = Math.max(0, digits-1-Math.floor(Math.log10(a)));
  return value.toFixed(Math.min(decimals, 8)).replace('-', '−');
}

export function duration(seconds) {
  if (!Number.isFinite(seconds)) return '–';
  if (seconds < 10) return seconds.toFixed(1)+' s';
  const s = Math.round(seconds), m = Math.floor(s/60);
  return m ? `${m} min ${String(s%60).padStart(2, '0')} s` : `${s} s`;
}

// Text for the page from a file's names and values: without the Unicode bidi
// controls (embeddings, overrides, isolates and marks), so a header such as
// "a\u202ebc" cannot reorder itself or the text around it. Data keep the names.
const BIDI_CONTROLS = /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/g;
export const displayText = text => String(text).replace(BIDI_CONTROLS, '');
