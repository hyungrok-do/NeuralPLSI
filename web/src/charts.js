// Inline SVG charts: the link function with its band, and a forest plot of beta.
// Colors come from CSS custom properties, so both follow the light/dark theme.
import {fmt} from './stats.js';

const NS = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

export function niceTicks(lo, hi, count = 5) {
  const span = hi-lo || Math.abs(hi) || 1, rough = span/count, mag = 10**Math.floor(Math.log10(rough)), r = rough/mag;
  const step = (r >= 7.5 ? 10 : r >= 3.5 ? 5 : r >= 1.5 ? 2 : 1)*mag, out = [];
  for (let v = Math.ceil(lo/step-1e-9)*step; v <= hi+step*1e-9; v += step) out.push(Math.abs(v) < step*1e-9 ? 0 : v);
  return {values: out, decimals: Math.max(0, -Math.floor(Math.log10(step)+1e-9))};
}

const label = (v, decimals) => v.toFixed(decimals).replace('-', '−');
const scale = (d0, d1, r0, r1) => v => r0+(v-d0)/(d1-d0 || 1)*(r1-r0);

function pad([lo, hi], fraction = 0.05) {
  const span = hi-lo || Math.abs(hi) || 1;
  return [lo-span*fraction, hi+span*fraction];
}

function extent(arrays) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) for (const v of a ?? []) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  return [lo, hi];
}

// A tooltip in the chart's container, kept inside its bounds.
function tooltip(host) {
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  tip.hidden = true;
  host.appendChild(tip);
  return {
    show(html, x, y) {
      tip.innerHTML = html; tip.hidden = false;
      const w = tip.offsetWidth, h = tip.offsetHeight, W = host.clientWidth;
      tip.style.left = `${Math.max(0, Math.min(W-w, x+12 > W-w ? x-w-12 : x+12))}px`;
      tip.style.top = `${Math.max(0, y-h-8)}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function xAxis(svg, ticks, x, y0, height, decimals) {
  for (const v of ticks) {
    el('line', {x1: x(v), x2: x(v), y1: y0, y2: y0+height, class: 'grid'}, svg);
    el('text', {x: x(v), y: y0+height+16, class: 'tick', 'text-anchor': 'middle'}, svg).textContent = label(v, decimals);
  }
}

// series: [{name, color, x, y, lo?, hi?}], truth: {x, y} | null, index: values for the histogram.
// The vertical range covers the curves and the truth, and the bands up to half
// that range again; wider bands (a spline near its boundary) are clipped.
export function linkChart(host, {series, truth, index, indexLabel}) {
  host.replaceChildren();
  const W = Math.max(280, host.clientWidth), plot = 260, hist = 40, H = plot+hist+52;
  const m = {left: 44, right: 12, top: 10};
  const xs = pad(extent(series.map(s => s.x)), 0.02);
  const core = extent([...series.map(s => s.y), truth?.y]), wide = extent(series.flatMap(s => [s.y, s.lo, s.hi]));
  const reach = (core[1]-core[0] || 1)/2;
  const ys = pad([Math.max(wide[0], core[0]-reach), Math.min(wide[1], core[1]+reach)]);
  const bottom = m.top+plot, top = bottom+8;
  const x = scale(xs[0], xs[1], m.left, W-m.right), y = scale(ys[0], ys[1], bottom, m.top);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
                         'aria-label': 'Estimated link function g(u) with pointwise 95% bootstrap bands'}, host);
  const clip = el('clipPath', {id: 'link-clip'}, el('defs', {}, svg));
  el('rect', {x: m.left, y: m.top, width: W-m.left-m.right, height: plot}, clip);
  const yt = niceTicks(ys[0], ys[1]), xt = niceTicks(xs[0], xs[1], Math.max(3, Math.round(W/90)));
  for (const v of yt.values) {
    el('line', {x1: m.left, x2: W-m.right, y1: y(v), y2: y(v), class: v === 0 ? 'axis' : 'grid'}, svg);
    el('text', {x: m.left-6, y: y(v)+4, class: 'tick', 'text-anchor': 'end'}, svg).textContent = label(v, yt.decimals);
  }
  for (const v of xt.values)
    el('text', {x: x(v), y: top+hist+16, class: 'tick', 'text-anchor': 'middle'}, svg).textContent = label(v, xt.decimals);
  el('text', {x: (m.left+W-m.right)/2, y: H-4, class: 'axis-title', 'text-anchor': 'middle'}, svg).textContent = indexLabel;
  el('text', {x: 12, y: m.top+plot/2, class: 'axis-title', 'text-anchor': 'middle',
              transform: `rotate(-90 12 ${m.top+plot/2})`}, svg).textContent = 'g(u)';
  const layer = el('g', {'clip-path': 'url(#link-clip)'}, svg);
  const path = (xv, yv) => xv.map((v, i) => `${i ? 'L' : 'M'}${x(v).toFixed(1)},${y(yv[i]).toFixed(1)}`).join('');
  for (const s of series) if (s.lo) {
    const back = s.x.map((_, k) => s.x.length-1-k).map(i => `L${x(s.x[i]).toFixed(1)},${y(s.lo[i]).toFixed(1)}`);
    el('path', {d: path(s.x, s.hi)+back.join('')+'Z', style: `fill:${s.color}`, class: 'band'}, layer);
  }
  if (truth) el('path', {d: path(truth.x, truth.y), class: 'truth'}, layer);
  for (const s of series) el('path', {d: path(s.x, s.y), style: `stroke:${s.color}`, class: 'curve'}, layer);

  // Histogram of the fitted index on the same horizontal axis.
  const bins = 40, counts = new Array(bins).fill(0), width = (xs[1]-xs[0])/bins;
  for (const v of index) counts[Math.min(bins-1, Math.max(0, Math.floor((v-xs[0])/width)))]++;
  const peak = Math.max(...counts), barW = Math.max(1, x(xs[0]+width)-x(xs[0])-1);
  counts.forEach((c, i) => {
    if (!c) return;
    const h = Math.max(1, c/peak*hist);
    el('rect', {x: x(xs[0]+i*width)+0.5, y: top+hist-h, width: barW, height: h, rx: 1, class: 'hist'}, svg);
  });
  el('line', {x1: m.left, x2: W-m.right, y1: top+hist, y2: top+hist, class: 'axis'}, svg);

  // Crosshair and values at the nearest grid point of each curve.
  const cross = el('line', {y1: m.top, y2: top+hist, class: 'crosshair', visibility: 'hidden'}, svg);
  const tip = tooltip(host), hit = el('rect', {x: m.left, y: m.top, width: W-m.left-m.right, height: top+hist-m.top,
                                               fill: 'transparent'}, svg);
  const nearest = (xv, u) => {
    let best = 0;
    xv.forEach((v, i) => { if (Math.abs(v-u) < Math.abs(xv[best]-u)) best = i; });
    return best;
  };
  hit.addEventListener('pointermove', event => {
    const box = svg.getBoundingClientRect(), px = (event.clientX-box.left)*W/box.width;
    const u = xs[0]+(px-x(xs[0]))/(x(xs[1])-x(xs[0]))*(xs[1]-xs[0]);
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
    const rows = series.map(s => {
      const i = nearest(s.x, u), inside = u >= s.x[0] && u <= s.x[s.x.length-1];
      return inside ? `<div><span class="swatch" style="background:${s.color}"></span>${s.name}: <b>${fmt(s.y[i])}</b>` +
        (s.lo ? ` <span class="muted">(${fmt(s.lo[i])}, ${fmt(s.hi[i])})</span>` : '')+'</div>' : '';
    });
    if (truth) rows.push(`<div><span class="swatch truth-swatch"></span>Truth: <b>${fmt(truth.y[nearest(truth.x, u)])}</b></div>`);
    tip.show(`<div class="muted">u = ${fmt(u)}</div>${rows.join('')}`, event.clientX-host.getBoundingClientRect().left,
             event.clientY-host.getBoundingClientRect().top);
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hide(); });
}

// rows: [{name, truth?}], series: [{name, color, estimate[], lo?[], hi?[]}].
export function forestChart(host, {rows, series}) {
  host.replaceChildren();
  const W = Math.max(280, host.clientWidth), rowH = series.length > 1 ? 34 : 28;
  const longest = Math.max(...rows.map(r => r.name.length));
  const m = {left: Math.min(150, 14+longest*6.6), right: 14, top: 6, bottom: 30};
  const H = m.top+rows.length*rowH+m.bottom;
  const xs = pad(extent([[0], ...series.flatMap(s => [s.estimate, s.lo, s.hi]), rows.map(r => r.truth)]));
  const x = scale(xs[0], xs[1], m.left, W-m.right);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
                         'aria-label': 'Exposure weights beta with 95% bootstrap intervals'}, host);
  const xt = niceTicks(xs[0], xs[1], Math.max(3, Math.round((W-m.left)/80)));
  xAxis(svg, xt.values, x, m.top, rows.length*rowH, xt.decimals);
  el('line', {x1: x(0), x2: x(0), y1: m.top, y2: m.top+rows.length*rowH, class: 'axis'}, svg);
  const tip = tooltip(host);
  rows.forEach((r, i) => {
    const cy = m.top+(i+0.5)*rowH;
    const name = el('text', {x: m.left-8, y: cy+4, class: 'row-label', 'text-anchor': 'end'}, svg);
    name.textContent = r.name.length > 22 ? r.name.slice(0, 21)+'…' : r.name;
    if (r.truth !== undefined) {
      const t = x(r.truth);
      el('path', {d: `M${t},${cy-6}L${t+5},${cy}L${t},${cy+6}L${t-5},${cy}Z`, class: 'truth-mark'}, svg);
    }
    series.forEach((s, k) => {
      const yy = cy+(series.length > 1 ? (k-0.5)*10 : 0);
      if (s.lo) el('line', {x1: x(s.lo[i]), x2: x(s.hi[i]), y1: yy, y2: yy, style: `stroke:${s.color}`, class: 'whisker'}, svg);
      el('circle', {cx: x(s.estimate[i]), cy: yy, r: 4, style: `fill:${s.color}`, class: 'dot'}, svg);
    });
    const hit = el('rect', {x: 0, y: cy-rowH/2, width: W, height: rowH, fill: 'transparent'}, svg);
    hit.addEventListener('pointerenter', event => {
      const lines = series.map((s, k) => `<div><span class="swatch" style="background:${s.color}"></span>${s.name}: ` +
        `<b>${fmt(s.estimate[i])}</b>${s.lo ? ` <span class="muted">(${fmt(s.lo[i])}, ${fmt(s.hi[i])})</span>` : ''}</div>`);
      if (r.truth !== undefined) lines.push(`<div><span class="swatch truth-swatch"></span>Truth: <b>${fmt(r.truth)}</b></div>`);
      const box = host.getBoundingClientRect();
      tip.show(`<div class="muted">${r.name}</div>${lines.join('')}`, event.clientX-box.left, cy*box.height/H);
    });
    hit.addEventListener('pointerleave', () => tip.hide());
  });
  el('text', {x: (m.left+W-m.right)/2, y: H-2, class: 'axis-title', 'text-anchor': 'middle'}, svg).textContent = 'β';
}
