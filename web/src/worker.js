// One single-threaded engine instance per Web Worker, driven by numbered requests.
import {loadEngine, fromCompiled, errorMessage} from './engine.js';
import {fitNeural, fitSpline, NeuralReplicate, SplineReplicate} from './models.js';
import {neuralSandwich, splineSandwich} from './sandwich.js';

let engine = null, replicates = {};

const operations = {
  async init(compiled) {
    engine = await loadEngine(compiled ? fromCompiled(compiled) : {});
    return engine.version();
  },
  fitNeural: (data, settings, seed) => fitNeural(engine, data, settings, seed),
  fitSpline: (data, seed, settings) => fitSpline(engine, data, seed, settings),
  setup(jobs) {
    for (const r of Object.values(replicates)) r.close();
    replicates = {};
    if (jobs.neural) replicates.neural = new NeuralReplicate(engine, jobs.neural);
    if (jobs.spline) replicates.spline = new SplineReplicate(engine, jobs.spline);
  },
  replicate: (kind, b) => replicates[kind].run(b),
  summary: (draws, B, D) => engine.bootstrapSummary(draws, B, D),
  // A part of a NeuralPLSI sandwich, from the fitted model's state: its weights, the
  // gradient, the score sums of rows [begin, end) or the Hessian's columns [begin, end).
  sandwichPart(state, {x, z, y, n}, part, begin, end) {
    const handle = engine.importState(state);
    try {
      if (part === 'weights') return engine.weights(handle);
      if (part === 'gradient') return engine.sandwichGradient(handle, x, z, y, n);
      if (part === 'scores') return engine.sandwichScores(handle, x, z, y, n, begin, end);
      return engine.sandwichHessian(handle, x, z, y, n, begin, end);
    } finally {
      engine.destroy(handle);
    }
  },
  neuralSandwich: parts => neuralSandwich(parts),
  splineSandwich: (data, fit, settings) => splineSandwich(data, fit, settings),
};

self.onmessage = async ({data: {id, op, args}}) => {
  try {
    const result = await operations[op](...args);
    self.postMessage({id, result}, result?.buffer instanceof ArrayBuffer ? [result.buffer] : []);
  } catch (error) {
    self.postMessage({id, error: errorMessage(error)});
  }
};
