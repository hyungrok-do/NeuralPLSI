// The C API of core/include/neuralplsi.h on the WebAssembly build. Mirrors the
// argument construction of python/neuralplsi/native.py.
import createNeuralPLSI from './neuralplsi.js';

export const FAMILIES = {continuous: 0, binary: 1, cox: 2};
export const ACTIVATIONS = {ELU: 0, LeakyReLU: 1, ReLU: 2, Tanh: 3};
// NeuralPLSI() and SplinePLSI() defaults.
export const NEURAL_DEFAULTS = {family: 'continuous', maxIter: 5000, weightDecay: 1e-3, hiddenUnits: 32,
                                layers: 2, activation: 'ELU', intercept: false};
export const SPLINE_DEFAULTS = {knots: 5, degree: 3, alpha: 1e-6, maxIter: 200, tol: 1e-6};

export function parameterCount(p, q, width, depth, intercept) {
  return p+q+1+2*width+(depth-1)*(width*width+width)+width+1+(intercept ? 1 : 0);
}

// The network sizes the engine accepts (the Model checks in core/src/neuralplsi.cpp).
export const LIMITS = {p: 100000, q: 100000, hiddenUnits: 10000, layers: 128, parameters: 10000000};

// Why the engine would reject a network for p exposures and q confounders, or null.
export function networkProblem(p, q, {hiddenUnits, layers, intercept}) {
  const count = parameterCount(p, q, hiddenUnits, layers, intercept), number = v => v.toLocaleString('en-US');
  if (p > LIMITS.p) return `${number(p)} chemical exposures (X): the engine takes at most ${number(LIMITS.p)}`;
  if (q > LIMITS.q) return `${number(q)} confounder columns (Z): the engine takes at most ${number(LIMITS.q)}`;
  if (count > LIMITS.parameters)
    return `The network would have ${number(count)} parameters, more than the engine's ${number(LIMITS.parameters)}; ` +
      'use fewer hidden units or layers';
  return null;
}

export const OUT_OF_MEMORY = 'Out of memory: this browser could not allocate the memory the fit needs. ' +
  'Use fewer hidden units or layers, or fewer rows';

// An error as the page shows it: std::bad_alloc from the engine and failed
// allocations in WebAssembly or JavaScript become OUT_OF_MEMORY.
export function errorMessage(error) {
  const text = error?.message ?? String(error);
  return /bad_alloc|Out of memory|Aborted\(OOM\)|allocation failed|Cannot enlarge memory/i.test(text) ? OUT_OF_MEMORY : text;
}

// [p, q, width, depth, intercept, family, activation, parameter_count, max_iter,
//  linear_init, optimizer (L-BFGS), batch_size, patience]; hyper [penalty, sign, learning rate].
export function neuralConfig(p, q, settings) {
  const s = {...NEURAL_DEFAULTS, ...settings};
  if (s.family === 'cox' && s.intercept) throw new Error('A Cox partial likelihood cannot identify an intercept');
  return {config: Int32Array.of(p, q, s.hiddenUnits, s.layers, s.intercept ? 1 : 0, FAMILIES[s.family],
                                ACTIVATIONS[s.activation], parameterCount(p, q, s.hiddenUnits, s.layers, s.intercept),
                                s.maxIter, 0, 0, 64, 10),
          hyper: Float32Array.of(s.weightDecay, 1, 1e-3)};
}

export function splineConfig(p, q, family, settings = {}) {
  const s = {...SPLINE_DEFAULTS, ...settings};
  return {config: Int32Array.of(p, q, FAMILIES[family], s.knots, s.degree, s.maxIter),
          hyper: Float64Array.of(s.alpha, s.tol), bases: s.knots+s.degree-1, degree: s.degree};
}

export async function loadEngine(options = {}) {
  return new Engine(await createNeuralPLSI(options));
}

// Options that instantiate an already compiled module, so workers need not fetch it.
export function fromCompiled(module) {
  return {instantiateWasm(imports, done) {
    WebAssembly.instantiate(module, imports).then(instance => done(instance, module));
    return {};
  }};
}

const seed = value => BigInt.asUintN(64, BigInt(value));

export class Engine {
  constructor(module) { this.m = module; }

  version() { return this.m.UTF8ToString(this.m._nplsi_version()); }

  check(status, spline = false) {
    if (status) throw new Error(errorMessage(this.m.UTF8ToString(spline ? this.m._splsi_error() : this.m._nplsi_error())));
  }

  // Heap views are re-read on every use: any allocation may grow memory and detach them.
  heap(Type) {
    return Type === Float32Array ? this.m.HEAPF32 : Type === Float64Array ? this.m.HEAPF64 :
           Type === Int32Array ? this.m.HEAP32 : this.m.HEAPU8;
  }

  alloc(Type, length) {
    const pointer = this.m._malloc(Math.max(length, 1)*Type.BYTES_PER_ELEMENT);
    if (!pointer) throw new Error(OUT_OF_MEMORY);
    return pointer;
  }

  put(array) {
    const pointer = this.alloc(array.constructor, array.length);
    this.heap(array.constructor).set(array, pointer/array.BYTES_PER_ELEMENT);
    return pointer;
  }

  read(Type, pointer, length) {
    const start = pointer/Type.BYTES_PER_ELEMENT;
    return this.heap(Type).slice(start, start+length);
  }

  // Runs body(put, alloc) and frees every buffer it allocated.
  scope(body) {
    const pointers = [];
    const put = array => { const p = this.put(array); pointers.push(p); return p; };
    const alloc = (Type, length) => { const p = this.alloc(Type, length); pointers.push(p); return p; };
    try { return body(put, alloc); } finally { for (const p of pointers) this.m._free(p); }
  }

  create(config, hyper, randomState = 0) {
    const handle = this.scope(put => this.m._nplsi_create(put(config), put(hyper), seed(randomState)));
    if (!handle) this.check(1);
    return handle;
  }

  destroy(handle) { if (handle) this.m._nplsi_destroy(handle); }

  fit(handle, x, z, y, n, randomState = 0) {
    this.scope(put => this.check(this.m._nplsi_fit(handle, put(x), put(z), put(y), n, seed(randomState))));
  }

  getConfig(handle) {
    return this.scope((put, alloc) => {
      const c = alloc(Int32Array, 13), h = alloc(Float32Array, 3), d = alloc(Float64Array, 4);
      this.check(this.m._nplsi_get_config(handle, c, h, d));
      return {config: this.read(Int32Array, c, 13), hyper: this.read(Float32Array, h, 3),
              diagnostics: this.read(Float64Array, d, 4)};
    });
  }

  coefficients(handle) {
    const {config} = this.getConfig(handle), [p, q] = config;
    return this.scope((put, alloc) => {
      const b = alloc(Float32Array, p), g = alloc(Float32Array, q), i = alloc(Float32Array, 1);
      this.check(this.m._nplsi_coefficients(handle, b, g, i));
      return {beta: this.read(Float32Array, b, p), gamma: this.read(Float32Array, g, q),
              intercept: config[4] ? this.read(Float32Array, i, 1)[0] : null};
    });
  }

  // mode: 0 linear predictor, 1 probability, 2 partial hazard, 3 g(X beta).
  predict(handle, x, z, n, mode = 0) {
    return this.scope((put, alloc) => {
      const out = alloc(Float32Array, n);
      this.check(this.m._nplsi_predict(handle, put(x), put(z), n, 128, mode, out));
      return this.read(Float32Array, out, n);
    });
  }

  link(handle, grid) {
    return this.scope((put, alloc) => {
      const out = alloc(Float32Array, grid.length);
      this.check(this.m._nplsi_link(handle, put(grid), grid.length, out));
      return this.read(Float32Array, out, grid.length);
    });
  }

  exportState(handle) {
    const size = this.m._nplsi_state_size(handle);
    if (!size) this.check(1);
    return this.scope((put, alloc) => {
      const buffer = alloc(Uint8Array, size);
      this.check(this.m._nplsi_export_state(handle, buffer, size));
      return new TextDecoder().decode(this.read(Uint8Array, buffer, size));
    });
  }

  importState(text) {
    const bytes = new TextEncoder().encode(text);
    const handle = this.scope(put => this.m._nplsi_import_state(put(bytes), bytes.length));
    if (!handle) this.check(1);
    return handle;
  }

  bootstrapIndices(n, randomState, replicate) {
    return this.scope((put, alloc) => {
      const out = alloc(Int32Array, n);
      this.check(this.m._nplsi_bootstrap_indices(n, seed(randomState), replicate, out));
      return this.read(Int32Array, out, n);
    });
  }

  // draws: B x D row-major; returns 4 x D (mean, SD, lower, upper).
  bootstrapSummary(draws, B, D, ci = 0.95, quantileMethod = 0) {
    return this.scope((put, alloc) => {
      const out = alloc(Float64Array, 4*D);
      this.check(this.m._nplsi_bootstrap_summary(put(draws), B, D, ci, quantileMethod, out));
      return this.read(Float64Array, out, 4*D);
    });
  }

  // The whole native bootstrap in this one instance (n_jobs = 1, no threads).
  bootstrap(handle, x, z, y, n, B, randomState, grid, ci = 0.95, quantileMethod = 0) {
    const {config} = this.getConfig(handle);
    const D = config[0]+config[1]+config[4]+grid.length;
    return this.scope((put, alloc) => {
      const draws = alloc(Float64Array, B*D), summary = alloc(Float64Array, 4*D), workers = alloc(Int32Array, 1);
      this.check(this.m._nplsi_bootstrap(handle, put(x), put(z), put(y), n, B, seed(randomState), 1,
                                         put(grid), grid.length, ci, quantileMethod, draws, summary, workers));
      return {draws: this.read(Float64Array, draws, B*D), summary: this.read(Float64Array, summary, 4*D), D};
    });
  }

  weights(handle) {
    const {config} = this.getConfig(handle), size = config[7];
    return this.scope((put, alloc) => {
      const w = alloc(Float32Array, size);
      this.check(this.m._nplsi_get_weights(handle, w));
      return this.read(Float32Array, w, size);
    });
  }

  // Sandwich inference in double precision (nplsi_sandwich_* in the header):
  // the penalized objective's value and gradient; the sum of the rows' loss
  // gradients and of their outer products over rows [begin, end); and the
  // Hessian's columns [begin, end), (end-begin) x d.
  sandwichGradient(handle, x, z, y, n) {
    const d = this.getConfig(handle).config[7];
    return this.scope((put, alloc) => {
      const value = alloc(Float64Array, 1), gradient = alloc(Float64Array, d);
      this.check(this.m._nplsi_sandwich_gradient(handle, put(x), put(z), put(y), n, value, gradient));
      return {objective: this.read(Float64Array, value, 1)[0], gradient: this.read(Float64Array, gradient, d)};
    });
  }

  sandwichScores(handle, x, z, y, n, begin, end) {
    const d = this.getConfig(handle).config[7];
    return this.scope((put, alloc) => {
      const sum = alloc(Float64Array, d), outer = alloc(Float64Array, d*d);
      this.check(this.m._nplsi_sandwich_scores(handle, put(x), put(z), put(y), n, begin, end, sum, outer));
      return {sum: this.read(Float64Array, sum, d), outer: this.read(Float64Array, outer, d*d)};
    });
  }

  sandwichHessian(handle, x, z, y, n, begin, end) {
    const d = this.getConfig(handle).config[7];
    return this.scope((put, alloc) => {
      const columns = alloc(Float64Array, (end-begin)*d);
      this.check(this.m._nplsi_sandwich_hessian(handle, put(x), put(z), put(y), n, begin, end, columns));
      return this.read(Float64Array, columns, (end-begin)*d);
    });
  }

  // Lowest-level loss for tests: the objective at the current weights.
  lossGrad(handle, x, z, y, n, anchor = true) {
    const {config, hyper} = this.getConfig(handle), size = config[7];
    return this.scope((put, alloc) => {
      const w = alloc(Float32Array, size), loss = alloc(Float64Array, 1), grad = alloc(Float32Array, size);
      this.check(this.m._nplsi_get_weights(handle, w));
      this.check(this.m._nplsi_loss_grad(put(config), put(hyper), w, put(x), put(z), put(y), n,
                                         anchor ? 1 : 0, loss, grad));
      return this.read(Float64Array, loss, 1)[0];
    });
  }

  splineFit(spline, x, z, y, n, randomState = 0) {
    const [p, q] = spline.config, m = spline.config[3]+2*spline.degree;
    return this.scope((put, alloc) => {
      const beta = alloc(Float64Array, p), gamma = alloc(Float64Array, q), coef = alloc(Float64Array, spline.bases);
      const knots = alloc(Float64Array, m), diag = alloc(Float64Array, 3);
      this.check(this.m._splsi_fit(put(spline.config), put(spline.hyper), put(x), put(z), put(y), n,
                                   seed(randomState), 0, beta, gamma, coef, knots, diag), true);
      const d = this.read(Float64Array, diag, 3);
      return {beta: this.read(Float64Array, beta, p), gamma: this.read(Float64Array, gamma, q),
              coefficients: this.read(Float64Array, coef, spline.bases), knots: this.read(Float64Array, knots, m),
              loss: d[0], iterations: d[1], converged: d[2] === 1};
    });
  }

  // n x (knots.length-degree-1) basis, clipped to the boundary knots.
  splineBasis(knots, degree, x) {
    const nb = knots.length-degree-1;
    return this.scope((put, alloc) => {
      const out = alloc(Float64Array, x.length*nb);
      this.check(this.m._splsi_basis(put(knots), knots.length, degree, put(x), x.length, out), true);
      return this.read(Float64Array, out, x.length*nb);
    });
  }

  splineBootstrap(spline, x, z, y, n, B, randomState, grid, ci = 0.95, quantileMethod = 0) {
    const [p, q] = spline.config, D = p+q+spline.bases+grid.length;
    return this.scope((put, alloc) => {
      const draws = alloc(Float64Array, B*D), summary = alloc(Float64Array, 4*D), workers = alloc(Int32Array, 1);
      this.check(this.m._splsi_bootstrap(put(spline.config), put(spline.hyper), put(x), put(z), put(y), n, B,
                                         seed(randomState), 1, put(grid), grid.length, ci, quantileMethod,
                                         draws, summary, workers), true);
      return {draws: this.read(Float64Array, draws, B*D), summary: this.read(Float64Array, summary, 4*D), D};
    });
  }
}
