// Drives the built page in headless Chrome over the DevTools protocol (no npm
// packages): a simulated fit and bootstrap through the real Web Worker pool, a
// CSV file, the NHANES example, Cancel, a model download and offline use after
// the first load. Screenshots in light and dark mode at desktop and phone widths
// go to NPLSI_SCREENSHOTS when it names a directory. Skipped without Chrome (set
// CHROME to its path) or with NPLSI_BROWSER=0.
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn, spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {extname, join} from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';
import {dist, noPython, runPython} from './helpers.js';

const chrome = [process.env.CHROME, '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ...['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'].map(name =>
    spawnSync('which', [name], {encoding: 'utf8'}).stdout.trim())].find(path => path && existsSync(path));
const skip = process.env.NPLSI_BROWSER === '0' ? 'NPLSI_BROWSER=0' : chrome ? false : 'Chrome not found (set CHROME)';
const shots = process.env.NPLSI_SCREENSHOTS;

const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm',
               '.csv': 'text/csv', '.svg': 'image/svg+xml'};
function serve() {
  const server = createServer((request, response) => {
    const path = join(dist, decodeURIComponent(new URL(request.url, 'http://x').pathname).replace(/^\/$/, '/index.html'));
    if (!path.startsWith(dist) || !existsSync(path)) { response.writeHead(404).end(); return; }
    response.writeHead(200, {'content-type': TYPES[extname(path)] || 'application/octet-stream'}).end(readFileSync(path));
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

class DevTools {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.next = 0; this.pending = new Map(); this.listeners = [];
    this.socket.onmessage = ({data}) => {
      const message = JSON.parse(data);
      if (message.id !== undefined) {
        const {resolve, reject} = this.pending.get(message.id);
        this.pending.delete(message.id);
        message.error ? reject(new Error(message.error.message)) : resolve(message.result);
      } else for (const listener of this.listeners) listener(message);
    };
    this.opened = new Promise(resolve => { this.socket.onopen = resolve; });
  }
  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.next++;
      this.pending.set(id, {resolve, reject});
      this.socket.send(JSON.stringify({id, method, params}));
    });
  }
  async evaluate(expression) {
    const {result, exceptionDetails} = await this.send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
    return result.value;
  }
  // Polls a page expression until it is truthy.
  async until(expression, seconds = 120) {
    for (const end = Date.now()+seconds*1000; Date.now() < end; await sleep(100))
      if (await this.evaluate(expression)) return;
    throw new Error(`Timed out waiting for ${expression}`);
  }
}

async function screenshot(page, name) {
  if (!shots) return;
  mkdirSync(shots, {recursive: true});
  const {cssContentSize: size} = await page.send('Page.getLayoutMetrics');
  const {data} = await page.send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true,
    clip: {x: 0, y: 0, width: size.width, height: size.height, scale: 1}});
  writeFileSync(join(shots, `${name}.png`), Buffer.from(data, 'base64'));
}

const setValue = (name, value) => `(() => { const e = document.forms.controls.elements[${JSON.stringify(name)}];
  e.value = ${JSON.stringify(value)}; e.dispatchEvent(new Event('input', {bubbles: true})); })()`;
const click = selector => `document.querySelector(${JSON.stringify(selector)}).click()`;
const choose = value => `(() => { const e = document.querySelector('input[name=source][value=${value}]');
  e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true})); })()`;
// The pool's draws against one-call nplsi_bootstrap and splsi_bootstrap in the page.
const COMPARE = `(async () => {
  const {loadEngine, splineConfig} = await import('./engine.js');
  const e = await loadEngine(), {fit, boot} = nplsiDemo.state, d = fit.data, s = fit.settings;
  const x = Float32Array.from(d.x), z = Float32Array.from(d.z), y = Float32Array.from(d.y);
  const h = e.importState(fit.neural.state);
  const neural = e.bootstrap(h, x, z, y, d.n, boot.B, s.seed, Float32Array.from(fit.neural.grid));
  const spline = e.splineBootstrap(splineConfig(d.p, d.q, d.family), d.x, d.z, d.y, d.n, boot.B, s.seed,
                                   Float64Array.from(fit.spline.grid));
  const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  return {neural: same(neural.draws, boot.neural.draws) && same(neural.summary, boot.neural.summary),
          spline: same(spline.draws, boot.spline.draws) && same(spline.summary, boot.spline.summary),
          workers: boot.workers};
})()`;

test('the page fits, bootstraps in parallel, cancels and downloads in headless Chrome', {skip, timeout: 600000}, async t => {
  const server = await serve(), profile = mkdtempSync(join(tmpdir(), 'nplsi-chrome-'));
  const downloads = join(profile, 'downloads');
  const browser = spawn(chrome, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', ...(process.env.CI ? ['--no-sandbox'] : []), 'about:blank'],
    {stdio: 'ignore'});
  try {
    const portFile = join(profile, 'DevToolsActivePort');
    for (let k = 0; k < 200 && !existsSync(portFile); ++k) await sleep(50);
    const port = readFileSync(portFile, 'utf8').split('\n')[0];
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = new DevTools(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await page.opened;
    // Errors raised by the page's own scripts; managed browser extensions may add others.
    const origin = `http://127.0.0.1:${server.address().port}`, errors = [];
    page.listeners.push(({method, params}) => {
      if (method === 'Runtime.exceptionThrown' && JSON.stringify(params).includes(origin))
        errors.push(params.exceptionDetails.exception?.description);
      if (method === 'Runtime.consoleAPICalled' && params.type === 'error' && JSON.stringify(params).includes(origin))
        errors.push(params.args.map(a => a.value).join(' '));
    });
    await page.send('Runtime.enable');
    await page.send('Page.enable');
    await page.send('Browser.setDownloadBehavior', {behavior: 'allow', downloadPath: downloads});
    await page.send('Emulation.setDeviceMetricsOverride', {width: 1360, height: 900, deviceScaleFactor: 1, mobile: false});
    await page.send('Page.navigate', {url: `${origin}/`});
    await page.until('document.readyState === "complete" && globalThis.nplsiDemo');

    await t.test('simulated fit and pooled bootstrap equal the one-call bootstrap', async () => {
      await page.evaluate(setValue('n', '400'));
      await page.evaluate(setValue('outcome', 'cox'));
      await page.evaluate(`document.forms.controls.elements.spline.checked = true`);
      await page.evaluate(setValue('B', '48'));
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
      await page.evaluate(click('#bootstrap'));
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      const result = await page.evaluate(COMPARE);
      assert.deepEqual({neural: result.neural, spline: result.spline}, {neural: true, spline: true});
      assert.ok(result.workers >= 1);
      console.log(`# pooled bootstrap on ${result.workers} workers equals nplsi_bootstrap and splsi_bootstrap`);
      assert.ok(await page.evaluate('document.querySelectorAll("#link-chart path.band").length === 2'));
      assert.ok(await page.evaluate('document.querySelectorAll("#forest-chart line.whisker").length === 16'));
    });

    await t.test('screenshots and no horizontal scrolling', async () => {
      for (const scheme of ['light', 'dark']) {
        await page.send('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: scheme}]});
        await sleep(200);
        await screenshot(page, `desktop-${scheme}`);
      }
      await page.send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 2, mobile: true});
      await page.evaluate('dispatchEvent(new Event("resize"))');
      await sleep(400);
      assert.ok(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'the page scrolls sideways');
      for (const scheme of ['light', 'dark']) {
        await page.send('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value: scheme}]});
        await sleep(200);
        await screenshot(page, `phone-${scheme}`);
      }
      await page.send('Emulation.setDeviceMetricsOverride', {width: 1360, height: 900, deviceScaleFactor: 1, mobile: false});
    });

    await t.test('a CSV file with missing values and a text covariate fits', async () => {
      const rows = ['id,y,x1,x2,x3,age,group'];
      for (let i = 0; i < 120; ++i) {
        const x = [Math.sin(i), Math.cos(3*i), ((i*7) % 11)/5-1], u = 0.8*x[0]+0.6*x[1];
        rows.push([i, (Math.tanh(2*u)+0.3*x[2]+0.1*Math.sin(5*i)).toFixed(4), ...x.map(v => v.toFixed(4)),
                   30+(i % 40), 'abc'[i % 3]].join(','));
      }
      rows[5] = rows[5].replace(/,[^,]*$/, ',NA'); rows[9] = rows[9].replace(/^(\d+),[^,]*/, '$1,');
      const file = join(profile, 'example.csv');
      writeFileSync(file, rows.join('\n'));
      await page.evaluate(choose('csv'));
      const {root} = await page.send('DOM.getDocument');
      const {nodeId} = await page.send('DOM.querySelector', {nodeId: root.nodeId, selector: 'input[name=file]'});
      await page.send('DOM.setFileInputFiles', {nodeId, files: [file]});
      await page.until('!document.getElementById("csv-columns").hidden');
      await page.evaluate(setValue('family', 'continuous'));
      await page.evaluate(setValue('outcomeColumn', 'y'));
      await page.evaluate(`for (const box of document.querySelectorAll('#exposure-list input, #covariate-list input'))
        box.checked = ['x1', 'x2', 'x3'].includes(box.value) ? box.closest('#exposure-list') !== null
                    : ['age', 'group'].includes(box.value) && box.closest('#covariate-list') !== null;
        document.forms.controls.dispatchEvent(new Event('input'))`);
      assert.match(await page.evaluate('document.getElementById("data-summary").textContent'),
                   /^118 of 120 rows used \(dropped: 2 missing or text\) · 3 exposures · 3 covariate columns/);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "csv" && !nplsiDemo.state.busy');
      assert.deepEqual(await page.evaluate('nplsiDemo.state.fit.data.names.z'), ['age', 'group=b', 'group=c']);
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), true);
      await page.evaluate(setValue('hiddenUnits', '16'));
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), false);
      assert.equal(await page.evaluate('document.getElementById("bootstrap").disabled'), true);
    });

    await t.test('NHANES loads, fits and a bootstrap can be cancelled', async () => {
      await page.evaluate(choose('nhanes'));
      await page.until('nplsiDemo.state.nhanes');
      assert.match(await page.evaluate('document.getElementById("data-summary").textContent'), /^800 of 800 rows used/);
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.activation.value,
        document.forms.controls.elements.intercept.checked]`), ['ELU', true]);
      await page.evaluate(`document.forms.controls.elements.spline.checked = false`);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "nhanes" && !nplsiDemo.state.busy');
      await page.evaluate(setValue('B', '1000'));
      await page.evaluate(click('#bootstrap'));
      await page.until('nplsiDemo.state.busy && /^[1-9]/.test(document.getElementById("progress-text").textContent)');
      await page.evaluate(click('#cancel'));
      await page.until('!nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").textContent'), 'Bootstrap cancelled.');
      assert.equal(await page.evaluate('nplsiDemo.state.boot'), null);
      await page.evaluate(setValue('B', '8'));
      await page.evaluate(click('#bootstrap'));
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      await screenshot(page, 'desktop-nhanes');
    });

    await t.test('the downloaded model file loads in Python', {skip: noPython}, async () => {
      await page.evaluate(`[...document.querySelectorAll('.downloads button')].find(b => /nplsi/.test(b.textContent)).click()`);
      const file = join(downloads, 'neuralplsi.nplsi');
      for (let k = 0; k < 100 && !existsSync(file); ++k) await sleep(100);
      assert.ok(existsSync(file), `no download in ${readdirSync(profile)}`);
      const expected = await page.evaluate('nplsiDemo.state.fit.neural.beta');
      const py = JSON.parse(runPython(`
import json, sys
from neuralplsi import NeuralPLSI
model = NeuralPLSI.load(sys.argv[1])
print(json.dumps({'beta': model.beta.tolist(), 'params': model.get_params()}))`, [file]));
      assert.deepEqual(py.beta, expected);
      assert.equal(py.params.activation, 'ELU');
      assert.equal(py.params.add_intercept, true);
    });
    await t.test('workers load the module themselves when it cannot be passed to them', async () => {
      const version = await page.evaluate(`(async () => {
        const {Pool} = await import('./pool.js'), pool = new Pool(1);
        pool.compiled = Promise.resolve({notCloneable() {}});
        const [worker] = await pool.ensure(1), version = await worker.ready;
        const fit = await pool.call(worker, 'fitSpline', [nplsiDemo.state.fit.data, 0n]);
        pool.cancel();
        return version+' '+fit.beta.length;
      })()`);
      assert.equal(version, '0.1.0 8');
    });

    await t.test('the page works offline once loaded', async () => {
      await page.until('navigator.serviceWorker.controller !== null');
      await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
      await page.send('Page.reload');
      await page.until('document.readyState === "complete" && globalThis.nplsiDemo');
      await page.evaluate(choose('nhanes'));
      await page.until('nplsiDemo.state.nhanes');
      await page.evaluate(choose('simulated'));
      await page.evaluate(setValue('n', '100'));
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
    });
    assert.deepEqual(errors, []);
    page.socket.close();
  } finally {
    browser.kill();
    server.close();
    await sleep(300);
    rmSync(profile, {recursive: true, force: true});
  }
});
