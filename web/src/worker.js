// One single-threaded engine instance per Web Worker, driven by numbered requests.
import {loadEngine, fromCompiled, errorMessage} from './engine.js';
import {fitNeural, fitSpline, NeuralReplicate, SplineReplicate} from './models.js';

let engine = null, replicates = {};

const operations = {
  async init(compiled) {
    engine = await loadEngine(compiled ? fromCompiled(compiled) : {});
    return engine.version();
  },
  fitNeural: (data, settings, seed) => fitNeural(engine, data, settings, seed),
  fitSpline: (data, seed) => fitSpline(engine, data, seed),
  setup(jobs) {
    for (const r of Object.values(replicates)) r.close();
    replicates = {};
    if (jobs.neural) replicates.neural = new NeuralReplicate(engine, jobs.neural);
    if (jobs.spline) replicates.spline = new SplineReplicate(engine, jobs.spline);
  },
  replicate: (kind, b) => replicates[kind].run(b),
  summary: (draws, B, D) => engine.bootstrapSummary(draws, B, D),
};

self.onmessage = async ({data: {id, op, args}}) => {
  try {
    const result = await operations[op](...args);
    self.postMessage({id, result}, result?.buffer instanceof ArrayBuffer ? [result.buffer] : []);
  } catch (error) {
    self.postMessage({id, error: errorMessage(error)});
  }
};
