// Cross-checks against the Python package and scripts of this checkout.
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, symlinkSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {loadBuiltEngine, noPython, root, runPython} from './helpers.js';
import {prepareNhanes, simulate, NHANES_SETTINGS} from '../src/data.js';

const {fitNeural, neuralArrays} = await import('../dist/models.js');
const {NEURAL_DEFAULTS} = await import('../dist/engine.js');
const engine = await loadBuiltEngine();

// Runs reproduce/nhanes.py up to its model fit, with NeuralPLSI replaced by a
// stub that captures the preprocessed arrays. The script writes into output/
// of its working directory, so it runs in a temporary one.
const nhanesCapture = `
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('nhanes', sys.argv[1])
nhanes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nhanes)
class Captured(Exception):
    pass
class Stub:
    def __init__(self, **settings):
        self.settings = settings
    def fit(self, x, z, y, random_state=0):
        raise Captured(x, z, y, self.settings)
nhanes.NeuralPLSI = Stub
try:
    nhanes.main()
except Captured as captured:
    x, z, y, settings = captured.args
print(json.dumps({'x': x.tolist(), 'z': z.tolist(), 'y': y.tolist(), 'settings': settings}))`;

test('NHANES preprocessing matches reproduce/nhanes.py', {skip: noPython}, () => {
  const work = mkdtempSync(join(tmpdir(), 'nplsi-nhanes-'));
  try {
    symlinkSync(join(root, 'NHANES'), join(work, 'NHANES'));
    const out = runPython(nhanesCapture, [join(root, 'reproduce', 'nhanes.py')], {cwd: work}).trim().split('\n');
    const py = JSON.parse(out.at(-1));
    const d = prepareNhanes(readFileSync(join(root, 'NHANES', '12940_2020_644_MOESM2_ESM.csv'), 'utf8'));
    assert.equal(d.n, py.y.length);
    assert.match(out[0], new RegExp(`Using ${d.n} complete observations; excluded ${d.rows-d.n} rows`));
    assert.deepEqual([d.p, d.q], [py.x[0].length, py.z[0].length]);
    let largest = 0;
    py.x.forEach((row, i) => row.forEach((v, j) => { largest = Math.max(largest, Math.abs(d.x[i*d.p+j]-v)); }));
    py.y.forEach((v, i) => { largest = Math.max(largest, Math.abs(d.y[i]-v)); });
    assert.ok(largest < 1e-12, `largest difference ${largest}`);
    console.log(`# NHANES: ${d.n} of ${d.rows} rows complete; largest difference ${largest.toExponential(2)}`);
    assert.deepEqual(Array.from(d.z), py.z.flat());
    assert.deepEqual({family: py.settings.family, intercept: py.settings.add_intercept, activation: py.settings.activation},
                     NHANES_SETTINGS);
  } finally {
    rmSync(work, {recursive: true, force: true});
  }
});

test('bootstrap resamples match the Python package', {skip: noPython}, () => {
  const seed = 2n**40n+5n;
  const py = JSON.parse(runPython(`
import json
from neuralplsi.native import bootstrap_indices
print(json.dumps([bootstrap_indices(37, ${seed}, b).tolist() for b in (0, 1, 999)]))`));
  [0, 1, 999].forEach((b, k) => assert.deepEqual(Array.from(engine.bootstrapIndices(37, seed, b)), py[k]));
});

// A .nplsi file from the page loads in NeuralPLSI.load and predicts what the
// WebAssembly model predicts, up to libm differences between the builds.
test('a model file exported by the WebAssembly build loads in Python', {skip: noPython}, () => {
  const work = mkdtempSync(join(tmpdir(), 'nplsi-model-'));
  try {
    const cases = [['continuous', {intercept: true, activation: 'ELU'}, 0], ['binary', {activation: 'Tanh'}, 1],
                   ['cox', {}, 2]];
    for (const [family, settings, mode] of cases) {
      const data = simulate({n: 300, outcome: family, seed: 4}), {x, z} = neuralArrays(data);
      const fit = fitNeural(engine, data, {...settings, maxIter: 500}, 4);
      const handle = engine.importState(fit.state), wasm = engine.predict(handle, x, z, data.n, mode);
      engine.destroy(handle);
      const path = join(work, `${family}.nplsi`);
      writeFileSync(path, fit.state);
      writeFileSync(join(work, 'x.json'), JSON.stringify([Array.from(x), Array.from(z)]));
      const py = JSON.parse(runPython(`
import json, sys, numpy as np
from neuralplsi import NeuralPLSI
model = NeuralPLSI.load(sys.argv[1])
x, z = (np.array(a, dtype=np.float32).reshape(${data.n}, -1) for a in json.load(open(sys.argv[2])))
predict = [model.predict, model.predict_proba, model.predict_partial_hazard][${mode}]
print(json.dumps({'params': model.get_params(), 'beta': model.beta.tolist(),
                  'prediction': predict(x, z).astype(float).tolist()}))`, [path, join(work, 'x.json')]));
      assert.equal(py.params.family, family);
      assert.equal(py.params.add_intercept, Boolean(settings.intercept));
      assert.equal(py.params.activation, settings.activation || NEURAL_DEFAULTS.activation);
      assert.equal(py.params.hidden_units, NEURAL_DEFAULTS.hiddenUnits);
      assert.deepEqual(py.beta, fit.beta);
      const scale = Math.max(...wasm.map(Math.abs));
      const largest = Math.max(...py.prediction.map((v, i) => Math.abs(v-wasm[i])));
      assert.ok(largest <= 1e-5*scale, `${family}: largest difference ${largest} at scale ${scale}`);
      console.log(`# ${family}: largest prediction difference ${largest.toExponential(2)} (scale ${scale.toFixed(2)})`);
    }
  } finally {
    rmSync(work, {recursive: true, force: true});
  }
});
