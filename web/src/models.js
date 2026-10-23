// Point fits and single bootstrap replicates, shared by the Web Workers and the tests.
import {neuralConfig, splineConfig} from './engine.js';
import {fitMetrics, linspace} from './stats.js';
export {mergeRows} from './stats.js';

export const GRID_SIZE = 200;
const BOOTSTRAP_SEED_OFFSET = 1337n;  // splitmix64_bootstrap_v1: replicate b refits with seed+1337+b

// Engine inputs: NeuralPLSI takes float32 (as Python converts), SplinePLSI float64.
export function neuralArrays(data) {
  return {x: Float32Array.from(data.x), z: Float32Array.from(data.z), y: Float32Array.from(data.y)};
}

function indexOf(x, beta, n, p) {
  const index = new Float64Array(n);
  for (let i = 0; i < n; ++i) {
    let v = 0;
    for (let j = 0; j < p; ++j) v += x[i*p+j]*beta[j];
    index[i] = v;
  }
  return index;
}

function range(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

export function fitNeural(engine, data, settings, randomState) {
  const {n, p, q, family} = data, {x, z, y} = neuralArrays(data);
  const {config, hyper} = neuralConfig(p, q, {...settings, family});
  const started = performance.now();
  const handle = engine.create(config, hyper, randomState);
  try {
    engine.fit(handle, x, z, y, n, randomState);
    const seconds = (performance.now()-started)/1000;
    const {beta, gamma, intercept} = engine.coefficients(handle);
    const {diagnostics} = engine.getConfig(handle);
    const index = indexOf(x, beta, n, p);
    const grid = Float32Array.from(linspace(...range(index), GRID_SIZE));
    const predictor = engine.predict(handle, x, z, n, 0);
    return {model: 'neural', beta: Array.from(beta), gamma: Array.from(gamma), intercept,
            grid: Array.from(grid), g: Array.from(engine.link(handle, grid)), index: Array.from(index),
            state: engine.exportState(handle), seconds,
            diagnostics: {iterations: diagnostics[0], gammaIterations: diagnostics[1], objective: diagnostics[2]},
            metrics: fitMetrics(family, predictor, data.y)};
  } finally {
    engine.destroy(handle);
  }
}

function splineLink(basis, coefficients, offset, rows) {
  const nb = coefficients.length-offset, g = new Float64Array(rows);
  for (let i = 0; i < rows; ++i) {
    let v = 0;
    for (let k = 0; k < nb; ++k) v += basis[i*nb+k]*coefficients[offset+k];
    g[i] = v;
  }
  return g;
}

export function fitSpline(engine, data, randomState) {
  const {n, p, q, family} = data, spline = splineConfig(p, q, family);
  const started = performance.now();
  const f = engine.splineFit(spline, data.x, data.z, data.y, n, randomState);
  const seconds = (performance.now()-started)/1000;
  const index = indexOf(data.x, f.beta, n, p);
  const grid = linspace(...range(index), GRID_SIZE);
  const g = splineLink(engine.splineBasis(f.knots, spline.degree, grid), f.coefficients, 0, grid.length);
  const gi = splineLink(engine.splineBasis(f.knots, spline.degree, index), f.coefficients, 0, n);
  const predictor = new Float64Array(n);
  for (let i = 0; i < n; ++i) {
    let v = gi[i];
    for (let j = 0; j < q; ++j) v += data.z[i*q+j]*f.gamma[j];
    predictor[i] = v;
  }
  return {model: 'spline', beta: Array.from(f.beta), gamma: Array.from(f.gamma), spline: Array.from(f.coefficients),
          grid: Array.from(grid), g: Array.from(g), index: Array.from(index), seconds,
          diagnostics: {iterations: f.iterations, converged: f.converged, objective: f.loss},
          metrics: fitMetrics(family, predictor, data.y)};
}

// Replicate b of nplsi_bootstrap: resample with nplsi_bootstrap_indices, refit a
// model imported from the source state from the seed seed+1337+b, and return
// [beta, gamma, intercept, g(grid)] as one row of draws.
export class NeuralReplicate {
  constructor(engine, {state, x, z, y, n, grid, seed}) {
    const e = this.engine = engine;
    this.handle = e.importState(state);
    const {config} = e.getConfig(this.handle);
    [this.p, this.q] = config; this.intercept = config[4] === 1; this.yd = config[5] === 2 ? 2 : 1;
    Object.assign(this, {x, z, y, n, seed: BigInt(seed), ng: grid.length});
    this.D = this.p+this.q+(this.intercept ? 1 : 0)+this.ng;
    this.buffers = {x: e.alloc(Float32Array, n*this.p), z: e.alloc(Float32Array, n*this.q),
                    y: e.alloc(Float32Array, n*this.yd), rows: e.alloc(Int32Array, n), grid: e.put(grid),
                    beta: e.alloc(Float32Array, this.p), gamma: e.alloc(Float32Array, this.q),
                    intercept: e.alloc(Float32Array, 1), link: e.alloc(Float32Array, this.ng)};
  }

  run(b) {
    const e = this.engine, m = e.m, B = this.buffers, {n, p, q, yd} = this;
    e.check(m._nplsi_bootstrap_indices(n, this.seed, b, B.rows));
    gather(e, Float32Array, e.read(Int32Array, B.rows, n), [[this.x, B.x, p], [this.z, B.z, q], [this.y, B.y, yd]]);
    e.check(m._nplsi_fit(this.handle, B.x, B.z, B.y, n, this.seed+BOOTSTRAP_SEED_OFFSET+BigInt(b)));
    e.check(m._nplsi_coefficients(this.handle, B.beta, B.gamma, B.intercept));
    e.check(m._nplsi_link(this.handle, B.grid, this.ng, B.link));
    const row = new Float64Array(this.D);
    row.set(e.read(Float32Array, B.beta, p));
    row.set(e.read(Float32Array, B.gamma, q), p);
    if (this.intercept) row[p+q] = e.read(Float32Array, B.intercept, 1)[0];
    row.set(e.read(Float32Array, B.link, this.ng), this.D-this.ng);
    return row;
  }

  close() {
    for (const pointer of Object.values(this.buffers)) this.engine.m._free(pointer);
    this.engine.destroy(this.handle);
  }
}

// Replicate b of splsi_bootstrap: the same resample, splsi_fit from the direction
// seeded by seed+1337+b, and [beta, gamma, spline coefficients, g(grid)], with g
// summed over the basis in the order the engine sums it.
export class SplineReplicate {
  constructor(engine, {x, z, y, n, p, q, family, grid, seed}) {
    const e = this.engine = engine, s = this.spline = splineConfig(p, q, family);
    Object.assign(this, {x, z, y, n, p, q, yd: family === 'cox' ? 2 : 1, seed: BigInt(seed), grid: Float64Array.from(grid)});
    this.knots = s.config[3]+2*s.degree;
    this.D = p+q+s.bases+grid.length;
    this.buffers = {config: e.put(s.config), hyper: e.put(s.hyper), x: e.alloc(Float64Array, n*p),
                    z: e.alloc(Float64Array, n*q), y: e.alloc(Float64Array, n*this.yd), rows: e.alloc(Int32Array, n),
                    beta: e.alloc(Float64Array, p), gamma: e.alloc(Float64Array, q),
                    coef: e.alloc(Float64Array, s.bases), t: e.alloc(Float64Array, this.knots),
                    diag: e.alloc(Float64Array, 3), grid: e.put(this.grid),
                    basis: e.alloc(Float64Array, grid.length*s.bases)};
  }

  run(b) {
    const e = this.engine, m = e.m, B = this.buffers, {n, p, q, yd} = this, nb = this.spline.bases;
    const ng = this.grid.length;
    e.check(m._nplsi_bootstrap_indices(n, this.seed, b, B.rows));
    gather(e, Float64Array, e.read(Int32Array, B.rows, n), [[this.x, B.x, p], [this.z, B.z, q], [this.y, B.y, yd]]);
    e.check(m._splsi_fit(B.config, B.hyper, B.x, B.z, B.y, n, this.seed+BOOTSTRAP_SEED_OFFSET+BigInt(b), 0,
                         B.beta, B.gamma, B.coef, B.t, B.diag), true);
    e.check(m._splsi_basis(B.t, this.knots, this.spline.degree, B.grid, ng, B.basis), true);
    const coef = e.read(Float64Array, B.coef, nb), row = new Float64Array(this.D);
    row.set(e.read(Float64Array, B.beta, p));
    row.set(e.read(Float64Array, B.gamma, q), p);
    row.set(coef, p+q);
    row.set(splineLink(e.read(Float64Array, B.basis, ng*nb), coef, 0, ng), p+q+nb);
    return row;
  }

  close() {
    for (const pointer of Object.values(this.buffers)) this.engine.m._free(pointer);
  }
}

// Copies rows of each source matrix into its engine buffer.
function gather(engine, Type, rows, blocks) {
  const heap = engine.heap(Type);
  for (const [source, pointer, width] of blocks) {
    if (!width) continue;
    const base = pointer/Type.BYTES_PER_ELEMENT;
    for (let i = 0; i < rows.length; ++i)
      heap.set(source.subarray(rows[i]*width, rows[i]*width+width), base+i*width);
  }
}

