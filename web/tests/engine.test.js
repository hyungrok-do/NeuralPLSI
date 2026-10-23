import test from 'node:test';
import assert from 'node:assert/strict';
import {loadBuiltEngine, sameBytes} from './helpers.js';
import {simulate, TRUE_BETA} from '../src/data.js';

const {neuralConfig, splineConfig, NEURAL_DEFAULTS} = await import('../dist/engine.js');
const {fitNeural, fitSpline, neuralArrays} = await import('../dist/models.js');
const engine = await loadBuiltEngine();
const cosine = beta => beta.reduce((s, v, j) => s+v*TRUE_BETA[j], 0);

test('the WebAssembly engine loads and reports its version', () => {
  assert.equal(engine.version(), '0.1.0');
});

for (const family of ['continuous', 'binary', 'cox']) {
  test(`NeuralPLSI ${family}: a fit lowers the objective of its own initialization`, () => {
    const data = simulate({n: 500, outcome: family, seed: 1}), {x, z, y} = neuralArrays(data);
    const objective = maxIter => {
      const {config, hyper} = neuralConfig(data.p, data.q, {family, maxIter});
      const handle = engine.create(config, hyper, 3);
      engine.fit(handle, x, z, y, data.n, 3);
      const {diagnostics} = engine.getConfig(handle);
      engine.destroy(handle);
      return diagnostics;
    };
    const [initial, fitted] = [objective(0), objective(NEURAL_DEFAULTS.maxIter)];
    assert.equal(initial[0], 0);
    assert.ok(fitted[0] > 0 && fitted[2] < initial[2], `${fitted[2]} vs ${initial[2]}`);
    const fit = fitNeural(engine, data, {}, 3);
    assert.equal(fit.diagnostics.objective, fitted[2]);
    assert.ok(cosine(fit.beta) > 0.95, `cosine ${cosine(fit.beta)}`);
    assert.ok(fit.beta[0] >= 0 && fit.g.every(Number.isFinite));
  });

  test(`SplinePLSI ${family}: the fit recovers the index`, () => {
    const data = simulate({n: 500, outcome: family, seed: 1});
    const fit = fitSpline(engine, data, 3);
    assert.ok(fit.diagnostics.converged && cosine(fit.beta) > 0.95, `cosine ${cosine(fit.beta)}`);
    assert.equal(fit.spline.length, splineConfig(data.p, data.q, family).bases);
  });
}

test('an exported state imports to the same model', () => {
  const data = simulate({n: 200, outcome: 'binary', seed: 2}), {x, z, y} = neuralArrays(data);
  const {config, hyper} = neuralConfig(data.p, data.q, {family: 'binary', intercept: true, maxIter: 200, activation: 'Tanh'});
  const handle = engine.create(config, hyper, 7);
  engine.fit(handle, x, z, y, data.n, 7);
  const text = engine.exportState(handle), copy = engine.importState(text);
  assert.match(text, /^NEURALPLSI_NATIVE 5\n/);
  assert.equal(engine.exportState(copy), text);
  assert.ok(sameBytes(engine.predict(copy, x, z, data.n, 1), engine.predict(handle, x, z, data.n, 1)));
  engine.destroy(handle); engine.destroy(copy);
});

test('engine errors surface as exceptions with the native message', () => {
  const data = simulate({n: 50, outcome: 'continuous', seed: 2}), {x, z} = neuralArrays(data);
  const {config, hyper} = neuralConfig(data.p, data.q, {family: 'binary'});
  const handle = engine.create(config, hyper, 0);
  assert.throws(() => engine.fit(handle, x, z, new Float32Array(50).fill(2), 50, 0), /Invalid outcome/);
  engine.destroy(handle);
  assert.throws(() => engine.importState('NEURALPLSI_NATIVE 4\n'), /Unsupported model state format/);
  assert.throws(() => neuralConfig(8, 3, {family: 'cox', intercept: true}), /cannot identify an intercept/);
  // The engine still works after a caught C++ exception.
  assert.equal(engine.bootstrapIndices(5, 1, 0).length, 5);
});
