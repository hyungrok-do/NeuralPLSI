// Shared test setup: the built engine in web/dist and the checkout's Python.
import {existsSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

export const web = fileURLToPath(new URL('..', import.meta.url));
export const root = fileURLToPath(new URL('../..', import.meta.url));
export const dist = fileURLToPath(new URL('../dist/', import.meta.url));

export async function loadBuiltEngine() {
  if (!existsSync(dist+'neuralplsi.wasm')) throw new Error('web/dist is missing: run web/build.sh first');
  const {loadEngine} = await import('../dist/engine.js');
  return loadEngine();
}

export const sameBytes = (a, b) =>
  a.length === b.length && Buffer.compare(Buffer.from(a.buffer, a.byteOffset, a.byteLength),
                                          Buffer.from(b.buffer, b.byteOffset, b.byteLength)) === 0;

// The first interpreter with the reference package's dependencies: $PYTHON, then
// python3, /usr/bin/python3 and python. Tests that need it are skipped without one.
const candidates = [process.env.PYTHON, 'python3', '/usr/bin/python3', 'python'].filter(Boolean);
export const python = candidates.find(exe => spawnSync(exe, ['-c', 'import numpy, pandas, scipy, sklearn, matplotlib'],
                                                      {stdio: 'ignore'}).status === 0) || null;
export const noPython = python ? false : 'needs a Python with numpy, pandas, scipy, scikit-learn and matplotlib';

export function runPython(code, args = [], options = {}) {
  const result = spawnSync(python, ['-c', code, ...args], {
    cwd: options.cwd || root, encoding: 'utf8', maxBuffer: 1 << 28,
    env: {...process.env, PYTHONPATH: root+'python', MPLBACKEND: 'Agg', ...options.env}});
  if (result.status !== 0) throw new Error(`Python failed:\n${result.stderr}`);
  return result.stdout;
}
