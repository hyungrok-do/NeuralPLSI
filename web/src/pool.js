// Web Workers, each with its own WebAssembly instance. The engine runs single
// threaded (n_jobs = 1) inside each worker; parallelism comes from the pool.
// Terminating the workers is the only way to stop a running fit, so cancel()
// discards the pool and the next request starts a new one.

export class Cancelled extends Error {
  constructor() { super('Cancelled'); this.name = 'Cancelled'; }
}

export const defaultSize = () => Math.max(1, Math.min((navigator.hardwareConcurrency || 2)-1, 16));

export class Pool {
  constructor(size = defaultSize()) {
    this.size = size;
    this.workers = [];
    this.compiled = null;
  }

  // The module is compiled once and shared, so new workers fetch nothing.
  async module() {
    if (!this.compiled) {
      const url = new URL('./neuralplsi.wasm', import.meta.url);
      this.compiled = WebAssembly.compileStreaming(fetch(url))
        .catch(() => fetch(url).then(r => r.arrayBuffer()).then(b => WebAssembly.compile(b)));
    }
    return this.compiled;
  }

  async ensure(count) {
    const compiled = await this.module();
    while (this.workers.length < Math.min(count, this.size)) {
      const w = {worker: new Worker(new URL('./worker.js', import.meta.url), {type: 'module'}), pending: new Map(), next: 0};
      w.worker.onmessage = ({data: {id, result, error}}) => {
        const call = w.pending.get(id);
        w.pending.delete(id);
        if (error === undefined) call.resolve(result); else call.reject(new Error(error));
      };
      w.worker.onerror = event => {
        event.preventDefault();
        for (const call of w.pending.values()) call.reject(new Error(event.message || 'A worker failed to start'));
        w.pending.clear();
      };
      // A browser that cannot pass the compiled module lets the worker fetch it.
      w.ready = this.call(w, 'init', [compiled]).catch(error => {
        if (error?.name !== 'DataCloneError') throw error;
        return this.call(w, 'init', [null]);
      });
      this.workers.push(w);
    }
    await Promise.all(this.workers.map(w => w.ready));
    return this.workers.slice(0, count);
  }

  call(w, op, args, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = w.next++;
      w.pending.set(id, {resolve, reject});
      try {
        w.worker.postMessage({id, op, args}, transfer);
      } catch (error) {
        w.pending.delete(id);
        reject(error);
      }
    });
  }

  cancel() {
    for (const w of this.workers) {
      w.worker.terminate();
      for (const call of w.pending.values()) call.reject(new Cancelled());
    }
    this.workers = [];
  }

  // Each worker takes the next task as soon as it finishes one, so the order in
  // which replicates complete varies; onRow receives each row with its task.
  async run(jobs, tasks, onRow) {
    const workers = await this.ensure(tasks.length);
    await Promise.all(workers.map(w => this.call(w, 'setup', [jobs])));
    let next = 0;
    await Promise.all(workers.map(async w => {
      while (next < tasks.length) {
        const task = tasks[next++];
        onRow(task, await this.call(w, 'replicate', [task.kind, task.b]));
      }
    }));
  }
}
