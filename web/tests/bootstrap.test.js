// The page's bootstrap runs one replicate per worker message. These tests show
// that merging those rows reproduces the single-call native bootstrap exactly,
// whichever instance (worker) fits which replicate, in any order.
import test from 'node:test';
import assert from 'node:assert/strict';
import {loadBuiltEngine, sameBytes} from './helpers.js';
import {simulate} from '../src/data.js';

const {fitNeural, neuralArrays, NeuralReplicate, SplineReplicate, mergeRows} = await import('../dist/models.js');
const {splineConfig} = await import('../dist/engine.js');
const [first, second] = [await loadBuiltEngine(), await loadBuiltEngine()];
const B = 6, seed = 9;

// Replicates split over two instances and run out of order, one handle each.
function replicate(Kind, job) {
  const workers = [new Kind(first, job), new Kind(second, job)], rows = [];
  for (const b of [4, 1, 5, 0, 3, 2]) rows[b] = workers[b % 2].run(b);
  workers.forEach(w => w.close());
  return rows;
}

for (const family of ['continuous', 'binary', 'cox']) {
  test(`NeuralPLSI ${family}: worker replicates equal nplsi_bootstrap draws and summary`, () => {
    const data = simulate({n: 120, outcome: family, seed: 5});
    const settings = {hiddenUnits: 8, layers: 2, maxIter: 300, intercept: family !== 'cox', activation: 'ELU'};
    const fit = fitNeural(first, data, settings, 4);
    const {x, z, y} = neuralArrays(data), grid = Float32Array.from(fit.grid.filter((_, i) => i % 10 === 0));
    const source = first.importState(fit.state);
    const native = first.bootstrap(source, x, z, y, data.n, B, seed, grid);
    first.destroy(source);
    const rows = replicate(NeuralReplicate, {state: fit.state, x, z, y, n: data.n, grid, seed});
    const draws = mergeRows(rows, native.D);
    assert.equal(native.D, data.p+data.q+(settings.intercept ? 1 : 0)+grid.length);
    assert.ok(sameBytes(draws, native.draws), 'draws differ');
    assert.ok(sameBytes(second.bootstrapSummary(draws, B, native.D), native.summary), 'summaries differ');
    // Not a degenerate pass: replicates differ from each other.
    assert.notDeepEqual(rows[0], rows[1]);
  });

  test(`SplinePLSI ${family}: worker replicates equal splsi_bootstrap draws and summary`, () => {
    const data = simulate({n: 150, outcome: family, seed: 6});
    const spline = splineConfig(data.p, data.q, family);
    const grid = new Float64Array(Array.from({length: 21}, (_, i) => -2+i/5));
    const native = first.splineBootstrap(spline, data.x, data.z, data.y, data.n, B, seed, grid);
    const rows = replicate(SplineReplicate, {x: data.x, z: data.z, y: data.y, n: data.n, p: data.p, q: data.q,
                                             family, grid, seed});
    const draws = mergeRows(rows, native.D);
    assert.equal(native.D, data.p+data.q+spline.bases+grid.length);
    assert.ok(sameBytes(draws, native.draws), 'draws differ');
    assert.ok(sameBytes(second.bootstrapSummary(draws, B, native.D), native.summary), 'summaries differ');
    assert.notDeepEqual(rows[0], rows[1]);
  });
}

test('a failing replicate raises the engine error', () => {
  // A Cox resample of two rows with a single event loses it in some replicate.
  const x = Float32Array.of(0.1, -0.3), z = new Float32Array(0), y = Float32Array.of(1, 1, 2, 0);
  const data = {n: 2, p: 1, q: 0, family: 'cox', x, z, y};
  const fit = fitNeural(first, data, {hiddenUnits: 2, layers: 1, maxIter: 5}, 0);
  const worker = new NeuralReplicate(first, {state: fit.state, x, z, y, n: 2, grid: Float32Array.of(0), seed: 0});
  assert.throws(() => { for (let b = 0; b < 50; ++b) worker.run(b); }, /Cox data need at least one event/);
  worker.close();
});

// An engine whose k-th buffer allocation fails, recording what is allocated,
// freed, created and destroyed.
function failingEngine(engine, k) {
  const log = {allocated: [], freed: [], handles: [], destroyed: []};
  const e = Object.create(engine);
  e.m = Object.create(engine.m, {_free: {value: pointer => { log.freed.push(pointer); engine.m._free(pointer); }}});
  e.alloc = (Type, length) => {
    if (log.allocated.length+1 === k) throw new Error('Out of memory: test');
    const pointer = engine.alloc.call(e, Type, length);
    log.allocated.push(pointer);
    return pointer;
  };
  e.importState = text => { const h = engine.importState.call(e, text); log.handles.push(h); return h; };
  e.destroy = handle => { log.destroyed.push(handle); engine.destroy(handle); };
  return {e, log};
}

test('a replicate whose allocation fails frees every buffer and the model it made', () => {
  const data = simulate({n: 40, outcome: 'continuous', seed: 3}), {x, z, y} = neuralArrays(data);
  const fit = fitNeural(first, data, {hiddenUnits: 4, layers: 1, maxIter: 20}, 0);
  const jobs = [[NeuralReplicate, {state: fit.state, x, z, y, n: data.n, grid: Float32Array.of(-1, 0, 1), seed: 0}],
                [SplineReplicate, {x: data.x, z: data.z, y: data.y, n: data.n, p: data.p, q: data.q, family: 'continuous',
                                   grid: Float64Array.of(-1, 0, 1), seed: 0}]];
  for (const [Kind, job] of jobs) {
    let failures = 0;
    for (let k = 1; ; ++k) {
      const {e, log} = failingEngine(first, k);
      let replicate = null;
      try { replicate = new Kind(e, job); } catch (error) { assert.match(error.message, /Out of memory/); ++failures; }
      if (replicate) replicate.close();
      assert.deepEqual([...log.freed].sort(), [...log.allocated].sort(), `${Kind.name}, allocation ${k} fails`);
      assert.deepEqual(log.destroyed, log.handles, `${Kind.name}, allocation ${k} fails`);
      if (replicate) break;
    }
    assert.ok(failures >= 9, `${Kind.name}: ${failures} failing allocations tried`);
  }
});
