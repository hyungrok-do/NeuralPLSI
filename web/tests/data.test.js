// The JavaScript generator follows python/neuralplsi/simulation.py in design, not in draws.
import test from 'node:test';
import assert from 'node:assert/strict';
import {noPython, runPython} from './helpers.js';
import {Random, simulate, LINKS, TRUE_BETA, TRUE_GAMMA} from '../src/data.js';

const column = (m, width, j) => Array.from({length: m.length/width}, (_, i) => m[i*width+j]);
const mean = v => v.reduce((s, e) => s+e, 0)/v.length;
const cov = (a, b) => { const ma = mean(a), mb = mean(b); return mean(a.map((e, i) => (e-ma)*(b[i]-mb))); };

test('the generator is seeded and its uniforms and normals are standard', () => {
  const draws = seed => { const r = new Random(seed); return Array.from({length: 5}, () => r.uniform()); };
  assert.deepEqual(draws(3), draws(3));
  assert.notDeepEqual(draws(3), draws(4));
  const r = new Random(11), u = Array.from({length: 200000}, () => r.uniform());
  const z = Array.from({length: 200000}, () => r.normal());
  assert.ok(u.every(v => v >= 0 && v < 1));
  assert.ok(Math.abs(mean(u)-0.5) < 0.003 && Math.abs(cov(u, u)-1/12) < 0.001);
  assert.ok(Math.abs(mean(z)) < 0.01 && Math.abs(cov(z, z)-1) < 0.01);
});

test('exposures and covariates follow the design', () => {
  const d = simulate({n: 20000, seed: 1});
  assert.deepEqual([d.n, d.p, d.q, d.x.length, d.z.length, d.y.length], [20000, 8, 3, 160000, 60000, 20000]);
  const xs = Array.from({length: 8}, (_, j) => column(d.x, 8, j));
  for (let j = 0; j < 8; ++j) {
    assert.ok(Math.abs(mean(xs[j])) < 0.03, `mean x${j+1}`);
    assert.ok(Math.abs(cov(xs[j], xs[j])-1) < 0.04, `variance x${j+1}`);
    for (let k = 0; k < j; ++k) {
      const r = cov(xs[j], xs[k])/Math.sqrt(cov(xs[j], xs[j])*cov(xs[k], xs[k]));
      assert.ok(Math.abs(r-0.3) < 0.03, `correlation x${j+1}, x${k+1}: ${r}`);
    }
  }
  for (const j of [0, 1]) {
    const zj = column(d.z, 3, j);
    assert.ok(Math.abs(mean(zj)) < 0.03 && Math.abs(cov(zj, zj)-1) < 0.04);
  }
  const z3 = column(d.z, 3, 2);
  assert.ok(z3.every(v => v === 1 || v === -1) && Math.abs(mean(z3)) < 0.03);
  // Continuous noise is N(0, 1) around g(x'beta) + z'gamma.
  const noise = Array.from({length: d.n}, (_, i) => {
    const u = TRUE_BETA.reduce((s, b, j) => s+b*d.x[i*8+j], 0);
    return d.y[i]-LINKS.sigmoid(u)-TRUE_GAMMA.reduce((s, g, j) => s+g*d.z[i*3+j], 0);
  });
  assert.ok(Math.abs(mean(noise)) < 0.03 && Math.abs(cov(noise, noise)-1) < 0.04);
  assert.deepEqual(simulate({n: 50, seed: 1}).x.slice(0, 8), simulate({n: 50, seed: 1}).x.slice(0, 8));
});

test('binary and Cox outcomes are coded as the engine expects', () => {
  const b = simulate({n: 2000, outcome: 'binary', g: 'nlrelu', seed: 2});
  assert.ok(b.y.every(v => v === 0 || v === 1));
  const c = simulate({n: 2000, outcome: 'cox', g: 'linear', seed: 2});
  assert.equal(c.y.length, 4000);
  assert.deepEqual(c.names.y, ['time', 'event']);
  for (let i = 0; i < c.n; ++i) assert.ok(c.y[2*i] > 0 && (c.y[2*i+1] === 0 || c.y[2*i+1] === 1));
});

test('truth, links and outcome rates agree with the Python design', {skip: noPython}, () => {
  const code = `
import json, numpy as np
from neuralplsi.simulation import simulate_data, beta, gamma
grid = np.linspace(-3, 3, 13)
out = {'beta': beta.tolist(), 'gamma': gamma.tolist(), 'links': {}, 'rates': {}}
for g in ('sigmoid', 'nlrelu', 'linear'):
    *_, g_fn = simulate_data(2, g_type=g)
    out['links'][g] = g_fn(grid).tolist()
    out['rates'][g] = {'binary': float(simulate_data(100000, 'binary', g, seed=1)[2].mean()),
                       'cox': float(simulate_data(100000, 'cox', g, seed=1)[2][:, 1].mean())}
print(json.dumps(out))`;
  const py = JSON.parse(runPython(code));
  TRUE_BETA.forEach((v, j) => assert.ok(Math.abs(v-py.beta[j]) < 1e-15));
  assert.deepEqual(TRUE_GAMMA, py.gamma);
  for (const [g, values] of Object.entries(py.links))
    values.forEach((v, i) => assert.ok(Math.abs(LINKS[g](-3+i/2)-v) < 1e-12, `${g}(${-3+i/2})`));
  // With 100000 rows, sampling noise in these rates is below 0.002.
  for (const [g, rates] of Object.entries(py.rates)) {
    const binary = mean(Array.from(simulate({n: 100000, outcome: 'binary', g, seed: 1}).y));
    const events = mean(column(simulate({n: 100000, outcome: 'cox', g, seed: 1}).y, 2, 1));
    assert.ok(Math.abs(binary-rates.binary) < 0.01, `${g} prevalence ${binary} vs ${rates.binary}`);
    assert.ok(Math.abs(events-rates.cox) < 0.01, `${g} event rate ${events} vs ${rates.cox}`);
  }
});
