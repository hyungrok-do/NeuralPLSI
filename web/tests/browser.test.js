// Drives the built page in headless Chrome over the DevTools protocol (no npm
// packages): Cancel while the WebAssembly module loads, a simulated fit with
// bootstrap through the real Web Worker pool, focus and announcements, text
// contrast, the result tables on a phone, the simulated data in the column setup
// (its tabs, the log checkboxes and the truth), CSV files (column roles, arrow keys,
// missing-value strings and decimal commas, the summary and the reasons Fit is
// disabled, column names that look like HTML or carry bidi controls, long names
// on a phone, a file chosen while another is read, the NHANES file as a user file
// giving the NHANES example exactly and its role table on a phone in Cox mode),
// the NHANES example button under Your own data (a single fit, then a bootstrap of it) and a failed download of it, Cancel, model and preprocessing
// downloads and offline use after the first load. Screenshots in light and dark
// mode at desktop and phone widths go to NPLSI_SCREENSHOTS when it names a directory.
// Skipped without Chrome (set CHROME to its path) or with NPLSI_BROWSER=0.
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn, spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {extname, join} from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';
import {dist, root, noPython, runPython} from './helpers.js';
import {NHANES_EXPOSURES, parseCsv, describeColumns} from '../src/data.js';

const chrome = [process.env.CHROME, '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ...['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'].map(name =>
    spawnSync('which', [name], {encoding: 'utf8'}).stdout.trim())].find(path => path && existsSync(path));
const skip = process.env.NPLSI_BROWSER === '0' ? 'NPLSI_BROWSER=0' : chrome ? false : 'Chrome not found (set CHROME)';
const shots = process.env.NPLSI_SCREENSHOTS;

const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm',
               '.csv': 'text/csv', '.svg': 'image/svg+xml'};
// Paths the server answers with 503, to test failed downloads.
const failing = new Set();
function serve() {
  const server = createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    if (failing.has(pathname)) { response.writeHead(503).end(); return; }
    const path = join(dist, pathname.replace(/^\/$/, '/index.html'));
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

// The whole page, or the box around the elements a selector matches.
async function screenshot(page, name, selector = null) {
  if (!shots) return;
  mkdirSync(shots, {recursive: true});
  const {cssContentSize: size} = await page.send('Page.getLayoutMetrics');
  const clip = selector ? await page.evaluate(`(() => {
    const boxes = [...document.querySelectorAll(${JSON.stringify(selector)})].map(e => e.getBoundingClientRect());
    const top = Math.min(...boxes.map(b => b.top))+scrollY, bottom = Math.max(...boxes.map(b => b.bottom))+scrollY;
    return {x: 0, y: Math.max(0, top-8), width: document.documentElement.clientWidth, height: bottom-top+16, scale: 1};
  })()`) : {x: 0, y: 0, width: size.width, height: size.height, scale: 1};
  const {data} = await page.send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true, clip});
  writeFileSync(join(shots, `${name}.png`), Buffer.from(data, 'base64'));
}

const setValue = (name, value) => `(() => { const e = document.forms.controls.elements[${JSON.stringify(name)}];
  e.value = ${JSON.stringify(value)}; e.dispatchEvent(new Event('input', {bubbles: true})); })()`;
const click = selector => `document.querySelector(${JSON.stringify(selector)}).click()`;
const choose = value => `(() => { const e = document.querySelector('input[name=source][value=${value}]');
  e.checked = true; e.dispatchEvent(new Event('change', {bubbles: true})); })()`;
// The CSV step: a column's role radio, its checked role, and the summary above Fit.
const roleInput = (column, role) => `document.querySelector(${JSON.stringify(
  `#column-table tr[data-column=${JSON.stringify(column)}] input[value=${role}]`)})`;
const assign = (column, role) => `${roleInput(column, role)}.click()`;
const roleOf = column => `document.querySelector(${JSON.stringify(
  `#column-table tr[data-column=${JSON.stringify(column)}] input[type=radio]:checked`)}).value`;
// An X's log box ('log', in the X tab) or a Z's categorical box ('cat', in the Z tab).
const option = (column, kind) => `document.querySelector(${JSON.stringify(
  `#${kind === 'log' ? 'x' : 'z'}-list li[data-column=${JSON.stringify(column)}] input[name^=${kind}-]`)})`;
// The small histogram of each column of the role table.
const HISTOGRAMS = `[...document.querySelectorAll('#column-table tbody tr')].map(row =>
  [row.dataset.column, row.querySelectorAll('.col-detail svg.hist-mini rect').length > 0])`;
const tab = name => `document.getElementById('tab-${name}').click()`;
// The steps of the analysis, each a page under its tab along the top.
const step = name => `document.getElementById('step-${name}').click()`;
const STEP_SHOWN = `[document.querySelector('.steps [aria-selected=true]').id, [...document.querySelectorAll('.page')]
  .filter(e => !e.hidden).map(e => e.id).join()]`;
const check = name => `document.forms.controls.elements[${JSON.stringify(name)}].click()`;
const checked = name => `document.forms.controls.elements[${JSON.stringify(name)}].checked`;
// The position shown in each X row.
const X_ORDER = `Object.fromEntries([...document.querySelectorAll('#column-table tbody tr[data-role=exposure]')].map(row =>
  [row.dataset.column, row.querySelector('.col-order').textContent]))`;
const RESULTS_LABEL = `document.querySelector('.results-head .facts').textContent`;
// The summary of the NHANES file set up as reproduce/nhanes.py.
const NHANES_SUMMARY = {'Outcome': 'log triglyceride (z-score) · continuous',
  'Chemical exposures (X)': '8, logged except a5.Retinol, then standardized',
  'Confounders (Z)': '6 (sex: 1 indicator vs 1; race: 4 indicators vs 1)', 'Rows': '800 of 800 used'};
const SUMMARY = `Object.fromEntries([...document.querySelectorAll('#data-summary div')].map(row =>
  [row.querySelector('dt').textContent, row.querySelector('dd').textContent]))`;
const FIT_STATE = `[document.getElementById('run').disabled, document.getElementById('fit-status').textContent]`;
const filter = text => `(() => { const e = document.getElementById('column-filter'); e.value = ${JSON.stringify(text)};
  e.dispatchEvent(new Event('input', {bubbles: true})); })()`;
// Every table in the results scrolls sideways inside its card when it is wider
// than the card: [wraps, wraps that scroll, problems].
const TABLES = `(() => {
  const problems = [], wraps = [...document.querySelectorAll('.results table')].map(t => t.parentElement);
  let scrolling = 0;
  for (const wrap of wraps) {
    const name = wrap.closest('.card').querySelector('h3').textContent, box = wrap.getBoundingClientRect();
    if (!wrap.classList.contains('table-wrap') || getComputedStyle(wrap).overflowX !== 'auto') problems.push(name+': no scroll container');
    if (box.right > document.documentElement.clientWidth) problems.push(name+': wider than the page');
    if (wrap.scrollWidth <= wrap.clientWidth) continue;
    ++scrolling;
    if (wrap.tabIndex !== 0) problems.push(name+': not reachable with the keyboard');
    if (getComputedStyle(wrap.querySelector('td')).position !== 'sticky') problems.push(name+': first column not sticky');
    wrap.scrollLeft = 60;
    if (wrap.scrollLeft === 0) problems.push(name+': does not scroll');
    wrap.scrollLeft = 0;
  }
  return [wraps.length, scrolling, problems];
})()`;

// Each result table that scrolls on a phone keeps most of its width for the
// numbers: [width of its sticky first column, width of the table's box].
const FIRST_COLUMNS = `[...document.querySelectorAll('.results .table-wrap.scrolls')].map(wrap =>
  [Math.max(...[...wrap.querySelectorAll('tr > :first-child')].map(cell => cell.getBoundingClientRect().width)), wrap.clientWidth])`;

async function chooseFile(page, file) {
  const {root: doc} = await page.send('DOM.getDocument');
  const {nodeId} = await page.send('DOM.querySelector', {nodeId: doc.nodeId, selector: 'input[name=file]'});
  await page.send('DOM.setFileInputFiles', {nodeId, files: [file]});
}

async function upload(page, file) {
  await chooseFile(page, file);
  await page.until(`nplsiDemo.setup().fileName === ${JSON.stringify(file.split('/').at(-1))} &&
    !document.getElementById('setup').hidden`);
}

// A key pressed on the focused element (Enter activates a button).
const KEYS = {Enter: 13, End: 35, Home: 36, ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40};
async function press(page, key) {
  const text = key === 'Enter' ? {text: String.fromCharCode(13)} : {};
  for (const type of ['keyDown', 'keyUp'])
    await page.send('Input.dispatchKeyEvent', {type, key, code: key, windowsVirtualKeyCode: KEYS[key], ...(type === 'keyDown' ? text : {})});
}
const ACTIVE = `(e => e.id || (e.closest('tr') ? e.closest('tr').dataset.column+':'+e.value : e.tagName))(document.activeElement)`;
const ANNOUNCED = `document.getElementById('announce').textContent.trim()`;
// Every column's role, to tell that a key changed none.
const ROLES = `[...document.querySelectorAll('#column-table input[type=radio]:checked')].map(e => e.value).join()`;

// WCAG contrast ratios of text against its background (the nearest opaque one).
const CONTRAST = `(() => {
  const rgb = s => s.match(/[0-9.]+/g).map(Number);
  const lum = c => c.slice(0, 3).map(v => { v /= 255; return v <= 0.04045 ? v/12.92 : ((v+0.055)/1.055)**2.4; })
    .reduce((s, v, k) => s+v*[0.2126, 0.7152, 0.0722][k], 0);
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q-p); return (x+0.05)/(y+0.05); };
  const background = e => {
    for (; e; e = e.parentElement) { const c = rgb(getComputedStyle(e).backgroundColor); if (c.length < 4 || c[3] === 1) return c; }
    return [255, 255, 255];
  };
  return Object.fromEntries([['Fit with bootstrap', '#run'], ['Single fit', '#fit'], ['Cancel', '#cancel'], ['selected tab', '[role=tab][aria-selected=true]'],
                             ['link', '.intro a'], ['footer link', '.footer a']].map(([name, at]) => {
    const e = document.querySelector(at);
    return [name, Math.round(100*ratio(rgb(getComputedStyle(e).color), background(e)))/100];
  }));
})()`;

// The CSV step's role table: name parts (between <wbr>s) broken across lines, row
// heights, sideways scrolling, notes past the table's edge and the role headers shown.
const ROLE_TABLE = `(() => {
  const broken = [];
  for (const name of document.querySelectorAll('#column-table .col-name')) for (const node of name.childNodes) {
    if (node.nodeType !== Node.TEXT_NODE) continue;
    const range = document.createRange();
    range.selectNodeContents(node);
    if (new Set([...range.getClientRects()].map(r => Math.round(r.top))).size > 1) broken.push(node.textContent);
  }
  const table = document.getElementById('column-table'), wrap = table.parentElement, right = table.getBoundingClientRect().right;
  return {broken, tallest: Math.max(...[...table.tBodies[0].rows].map(row => row.getBoundingClientRect().height)),
          scrolls: wrap.scrollWidth > wrap.clientWidth || document.documentElement.scrollWidth > innerWidth,
          outside: [...table.querySelectorAll('.col-detail')].filter(e => e.getBoundingClientRect().right > right+0.5).length,
          heads: [...table.tHead.rows[0].cells].filter(th => getComputedStyle(th).display !== 'none')
            .map(th => [...th.childNodes].filter(n => n.nodeType === Node.TEXT_NODE ||
              (getComputedStyle(n).position !== 'absolute' && getComputedStyle(n).display !== 'none'))
            .map(n => n.textContent).join(''))};
})()`;

async function scheme(page, value) {
  await page.send('Emulation.setEmulatedMedia', {features: [{name: 'prefers-color-scheme', value}]});
  await sleep(200);
}
const PHONE = {width: 390, height: 844, deviceScaleFactor: 2, mobile: true};
const DESKTOP = {width: 1360, height: 900, deviceScaleFactor: 1, mobile: false};

// A messy file: an ID, a text race column with missing values, a survey weight,
// three metals with a missing value, a zero and an infinite value, missing ages,
// 0/1 sex, status and case columns, and a mostly empty notes column. The rows
// with a problem are disjoint, so each is dropped for one reason.
function messyCsv() {
  const races = ['White', 'Black', 'Mexican American', 'Other'], rows = [], drops = {missing: 0, nonfinite: 0, nonpositive: 0};
  let cases = 0;
  for (let i = 0; i < 200; ++i) {
    const u = Math.sin(i)*0.8+Math.cos(2.3*i)*0.5, risk = 1/(1+Math.exp(-2*u));
    const pb = i % 23 === 3 ? 'NA' : Math.exp(u+0.3*Math.sin(5*i)).toFixed(3);
    const cd = i % 41 === 7 ? '0' : Math.exp(0.5*u+0.4*Math.cos(7*i)).toFixed(3);
    const hg = i === 50 ? 'inf' : Math.exp(0.2*Math.sin(3*i)).toFixed(3);
    const age = i % 53 === 9 ? '' : String(20+(i*7) % 60), race = i % 37 === 5 ? '' : races[i % 4];
    const c = (Math.sin(11.7*i)+1)/2 < risk ? 1 : 0;
    const reason = pb === 'NA' || age === '' || race === '' ? 'missing' : hg === 'inf' ? 'nonfinite' : cd === '0' ? 'nonpositive' : null;
    if (reason) ++drops[reason]; else cases += c;
    rows.push([1001+i, race, (1000+(i*7919) % 49000).toFixed(2), pb, cd, hg, age, i % 2, (i*3) % 5 > 1 ? 1 : 0, c,
               i % 10 === 0 ? 'x' : ''].join(','));
  }
  return {text: ['id,race,wt,metal_pb,metal_cd,metal_hg,age,sex,status,case,notes', ...rows].join('\n'), drops, cases};
}

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

// Starts headless Chrome and returns it with its DevTools port. A cold start on
// a CI runner can take more than 10 s, so this waits up to 60 s and tries twice;
// if Chrome exits or never opens the port, the error carries its stderr.
async function launchChrome(profile, attempts = 2) {
  const portFile = join(profile, 'DevToolsActivePort');
  for (let attempt = 1; ; ++attempt) {
    rmSync(portFile, {force: true});
    const browser = spawn(chrome, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
      '--no-first-run', '--no-default-browser-check',
      ...(process.env.CI ? ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'] : []), 'about:blank'],
      {stdio: ['ignore', 'ignore', 'pipe']});
    let stderr = '', exited = null;
    browser.stderr.on('data', chunk => { stderr = (stderr+chunk).slice(-4000); });
    browser.on('exit', (code, signal) => { exited = signal ?? code; });
    browser.on('error', error => { exited = error.code; });
    const ready = () => existsSync(portFile) && readFileSync(portFile, 'utf8').includes('\n');
    for (let k = 0; k < 1200 && exited === null && !ready(); ++k) await sleep(50);
    if (ready()) return {browser, port: readFileSync(portFile, 'utf8').split('\n')[0]};
    browser.kill();
    if (attempt >= attempts)
      throw new Error(`Chrome (${chrome}) did not start: ${exited === null ? 'no DevTools port after 60 s'
                      : `it exited with ${exited}`}. Its last output:\n${stderr}`);
  }
}

test('the page fits, bootstraps in parallel, cancels and downloads in headless Chrome', {skip, timeout: 600000}, async t => {
  const server = await serve(), profile = mkdtempSync(join(tmpdir(), 'nplsi-chrome-'));
  const downloads = join(profile, 'downloads');
  let browser;
  try {
    const launched = await launchChrome(profile);
    browser = launched.browser;
    const port = launched.port;
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
    await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
    await page.send('Page.navigate', {url: `${origin}/`});
    await page.until('document.readyState === "complete" && globalThis.nplsiDemo');

    await t.test('the page opens on your own data; Cancel stops a fit while the WebAssembly module is still loading', async () => {
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.source.value, document.getElementById('setup').hidden,
        [...document.querySelectorAll('.segmented:not(.inference *) span')].map(e => e.textContent)]`),
        ['csv', true, ['Your own data', 'Simulated']]);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Open a CSV file.']);
      // The steps are tabs along the top; Data shows first.
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('.steps [role=tab]')].map(e => e.textContent)`),
                       ['1Data', '2Model', '3Run', '4Results']);
      assert.deepEqual(await page.evaluate(STEP_SHOWN), ['step-data', 'page-data']);
      await page.evaluate(choose('simulated'));
      await page.evaluate(step('run'));
      // The module arrives 4 s after it is first asked for; nothing has asked yet.
      assert.equal(await page.evaluate('nplsiDemo.pool.compiled'), null);
      await page.evaluate(`void (nplsiDemo.pool.compiled = new Promise(resolve => setTimeout(() =>
        resolve(WebAssembly.compileStreaming(fetch('neuralplsi.wasm'))), 4000)))`);
      const started = Date.now();
      // With the keyboard: Enter on Fit with bootstrap moves focus to Cancel, Enter on Cancel stops the run.
      await page.evaluate(`document.getElementById('run').focus()`);
      await press(page, 'Enter');
      assert.equal(await page.evaluate('nplsiDemo.state.busy'), true);
      assert.equal(await page.evaluate(ACTIVE), 'cancel');
      await sleep(500);
      await press(page, 'Enter');
      await page.until('!nplsiDemo.state.busy', 1);
      assert.ok(Date.now()-started < 2500, `the run ended ${Date.now()-started} ms after it started`);
      assert.deepEqual(await page.evaluate(`[document.getElementById('message').textContent, nplsiDemo.state.fit,
        nplsiDemo.pool.workers.length]`), ['Fit cancelled.', null, 0]);
      // Focus is back on Fit with bootstrap, as no results were drawn.
      assert.equal(await page.evaluate(ACTIVE), 'run');
      // Nothing starts when the module arrives.
      await sleep(Math.max(0, 4500-(Date.now()-started)));
      assert.deepEqual(await page.evaluate('[nplsiDemo.state.busy, nplsiDemo.state.fit, nplsiDemo.pool.workers.length]'),
                       [false, null, 0]);
      // A request made directly to the pool is cancelled the same way.
      assert.equal(await page.evaluate(`(async () => {
        const {Pool} = await import('./pool.js'), pool = new Pool(1);
        pool.compiled = new Promise(resolve => setTimeout(resolve, 3000));
        const asked = pool.ensure(1).then(() => 'ready', e => e.name), when = performance.now();
        setTimeout(() => pool.cancel(), 100);
        const answer = await asked;
        return answer+' after '+(performance.now()-when < 1000 ? 'less' : 'more')+' than 1 s';
      })()`), 'Cancelled after less than 1 s');
    });

    await t.test('Fit with bootstrap on simulated data: the pooled bootstrap equals the one-call bootstrap', async () => {
      await page.evaluate(setValue('n', '400'));
      await page.evaluate(setValue('outcome', 'cox'));
      await page.evaluate(`document.forms.controls.elements.runSpline.checked = true`);
      await page.evaluate(setValue('B', '48'));
      // Results are not a live region and the progress text is not announced; one status line is.
      assert.deepEqual(await page.evaluate(`['results', 'progress-text'].map(id => document.getElementById(id).getAttribute('aria-live'))
        .concat(document.getElementById('announce').getAttribute('role'))`), [null, null, 'status']);
      // The simulated columns come set up as generated: Cox, with time and event.
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.outcomeType.value,
        [...document.querySelectorAll('#column-table tbody tr')].map(row => row.dataset.column+':'+row.dataset.role).join()]`),
        ['cox', 'x1:exposure,x2:exposure,x3:exposure,x4:exposure,x5:exposure,x6:exposure,x7:exposure,x8:exposure,' +
                'z1:confounder,z2:confounder,z3:confounder,time:outcome,event:event']);
      // Each announcement: the fit, its end with the bootstrap's start, 25, 50 and 75% and the end.
      await page.evaluate(`nplsiDemo.said = []; new MutationObserver(() => nplsiDemo.said.push(document.getElementById('announce')
        .textContent.trim())).observe(document.getElementById('announce'), {childList: true, characterData: true, subtree: true})`);
      await page.evaluate(`document.getElementById('run').focus()`);
      await press(page, 'Enter');
      assert.equal(await page.evaluate(ACTIVE), 'cancel');
      // The fit's results show first, on their page, with the bootstrap running.
      await page.until('nplsiDemo.state.fit && nplsiDemo.state.booting');
      assert.equal(await page.evaluate(`document.querySelector('.results .tables td.na').textContent`), 'bootstrap running…');
      assert.deepEqual(await page.evaluate(STEP_SHOWN), ['step-results', 'page-results']);
      // The progress and Cancel stay in view under the tabs.
      assert.equal(await page.evaluate(`document.getElementById('progress').closest('.topbar') !== null &&
        getComputedStyle(document.querySelector('.topbar')).position`), 'sticky');
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
      // Focus moves to the Results heading.
      assert.equal(await page.evaluate(ACTIVE), 'results-title');
      const said = await page.evaluate('nplsiDemo.said');
      assert.equal(said[0], 'Fitting NeuralPLSI and SplinePLSI…');
      assert.match(said[1], /^Fit done in [0-9.]+ s\. Bootstrap started: 96 refits\.$/);
      assert.deepEqual(said.slice(2, -1), ['Bootstrap 25% done.', 'Bootstrap 50% done.', 'Bootstrap 75% done.']);
      assert.match(said.at(-1), /^Bootstrap done: B = 48 in [0-9.]+ s\. Standard errors and intervals are in the results\.$/);
      const result = await page.evaluate(COMPARE);
      assert.deepEqual({neural: result.neural, spline: result.spline}, {neural: true, spline: true});
      assert.ok(result.workers >= 1);
      console.log(`# pooled bootstrap on ${result.workers} workers equals nplsi_bootstrap and splsi_bootstrap`);
      assert.ok(await page.evaluate('document.querySelectorAll("#link-chart path.band").length === 2'));
      assert.ok(await page.evaluate('document.querySelectorAll("#forest-chart line.whisker").length === 16'));
      // γ too: three confounders, two models, each with its estimate, interval and truth.
      assert.deepEqual(await page.evaluate(`['line.whisker', 'circle.dot', 'path.truth-mark', 'text.row-label'].map(s =>
        document.querySelectorAll('#gamma-chart '+s).length).concat(document.querySelector('#gamma-chart svg').getAttribute('aria-label'))`),
        [6, 6, 3, 3, 'Confounder effects gamma with 95% intervals']);
      // The setup is the design, so the results show the truth.
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.fit.data.truth.g, nplsiDemo.state.fit.data.names.x.join(),
        document.querySelector('.results .tables thead tr:not(.group) th:nth-child(2)').textContent]`), ['sigmoid', 'x1,x2,x3,x4,x5,x6,x7,x8', 'Truth']);
    });

    await t.test('sandwich standard errors: not for Cox; with the bootstrap, a toggle between the two', async () => {
      const choose = value => `(() => { const e = document.querySelector('input[name=inference][value=${value}]');
        e.click(); })()`;
      // The simulated setup is still Cox: only the bootstrap.
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('input[name=inference]')].map(e => e.disabled)`),
                       [false, true, true]);
      await page.evaluate(setValue('outcome', 'continuous'));
      await page.evaluate(choose('sandwich'));
      assert.deepEqual(await page.evaluate(`[document.getElementById('bootstrap-size').hidden, document.getElementById('run').textContent]`),
                       [true, 'Fit with sandwich SEs']);
      await page.evaluate(choose('both'));
      await page.evaluate(setValue('B', '8'));
      assert.deepEqual(await page.evaluate(`[document.getElementById('bootstrap-size').hidden, document.getElementById('run').textContent]`),
                       [false, 'Fit with both']);
      await page.evaluate(click('#run'));
      await page.until('nplsiDemo.state.sandwich && nplsiDemo.state.boot && !nplsiDemo.state.busy', 300);
      const sw = await page.evaluate(`(() => { const s = nplsiDemo.state.sandwich;
        return ['neural', 'spline'].map(k => [s[k].se.length, Array.from(s[k].se).every(v => v > 0 && v < 1)]); })()`);
      assert.deepEqual(sw, [[11, true], [11, true]]);
      // Bootstrap first; the toggle shows the sandwich SEs and Wald intervals in the tables.
      assert.match(await page.evaluate(`document.querySelector('.se-line').textContent`), /^Standard errors: bootstrap, B = 8/);
      await page.evaluate(`[...document.querySelectorAll('.view-toggle button')].find(b => b.textContent === 'Sandwich').click()`);
      assert.match(await page.evaluate(`document.querySelector('.se-line').textContent`), /^Standard errors: sandwich \(estimate ± 1\.96 SE\)/);
      const cells = await page.evaluate(`(() => { const s = nplsiDemo.state.sandwich.neural.se[0], b = nplsiDemo.state.fit.neural.beta[0];
        const td = [...document.querySelector('.results .tables tbody tr').children].map(c => c.textContent.replace('\u2212', '-'));
        return {se: +td[3], lower: +td[4], expect: s, lowerExpect: b-1.959964*s}; })()`);
      assert.ok(Math.abs(cells.se/cells.expect-1) < 0.01 && Math.abs(cells.lower-cells.lowerExpect) < 0.01, JSON.stringify(cells));
      assert.match(await page.evaluate(`[...document.querySelectorAll('tr')].find(r => /^Sandwich check/.test(r.textContent)).textContent`),
                   /Newton step [0-9.]+ SE \(β\)/);
      await page.evaluate(`[...document.querySelectorAll('.view-toggle button')].find(b => b.textContent === 'Bootstrap').click()`);
      await page.evaluate(choose('bootstrap'));
      await page.evaluate(setValue('outcome', 'cox'));
    });

    await t.test('buttons and links have AA contrast in light and dark', async () => {
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        const ratios = await page.evaluate(CONTRAST);
        for (const [name, ratio] of Object.entries(ratios)) assert.ok(ratio >= 4.5, `${name} in ${value}: ${ratio}`);
        console.log(`# contrast in ${value}: ${Object.entries(ratios).map(([name, ratio]) => `${name} ${ratio}`).join(', ')}`);
        // The Run card with its Cancel button shown.
        await page.evaluate(step('run'));
        await page.evaluate(`document.getElementById('progress').hidden = false`);
        await screenshot(page, `run-buttons-${value}`, '.topbar, #page-run');
        await page.evaluate(`document.getElementById('progress').hidden = true`);
      }
    });

    await t.test('screenshots, no horizontal scrolling and result tables that scroll on a phone', async () => {
      await page.evaluate(step('results'));
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `desktop-${value}`);
      }
      await page.send('Emulation.setDeviceMetricsOverride', PHONE);
      await page.evaluate('dispatchEvent(new Event("resize"))');
      await sleep(400);
      assert.ok(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'the page scrolls sideways');
      const [tables, scrolling, problems] = await page.evaluate(TABLES);
      assert.deepEqual(problems, []);
      assert.equal(tables, 3);
      assert.ok(scrolling >= 2, `${scrolling} of ${tables} tables scroll on a phone`);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `phone-${value}`);
      }
      await scheme(page, 'light');
      await screenshot(page, 'phone-results-tables', '.results .tables, .results .card:has(> h3 + .table-wrap)');
      await page.evaluate(`for (const wrap of document.querySelectorAll('.results .table-wrap')) wrap.scrollLeft = 10000`);
      await sleep(100);
      await screenshot(page, 'phone-results-tables-scrolled', '.results .tables, .results .card:has(> h3 + .table-wrap)');
      await page.evaluate(`for (const wrap of document.querySelectorAll('.results .table-wrap')) wrap.scrollLeft = 0`);
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
    });

    await t.test('the step tabs and Next buttons', async () => {
      await page.evaluate(`document.getElementById('step-results').focus()`);
      const visited = [];
      for (const key of ['Home', 'ArrowRight', 'ArrowRight', 'End', 'ArrowRight']) {
        await press(page, key);
        visited.push(await page.evaluate(`[document.activeElement.id, ...${STEP_SHOWN}]`));
      }
      assert.deepEqual(visited, [['step-data', 'step-data', 'page-data'], ['step-model', 'step-model', 'page-model'],
        ['step-run', 'step-run', 'page-run'], ['step-results', 'step-results', 'page-results'],
        ['step-data', 'step-data', 'page-data']]);
      await page.evaluate(click('[data-goto=model]'));
      assert.deepEqual(await page.evaluate(`[document.activeElement.id, ...${STEP_SHOWN}]`), ['step-model', 'step-model', 'page-model']);
      await page.evaluate(click('[data-goto=run]'));
      assert.deepEqual(await page.evaluate(STEP_SHOWN), ['step-run', 'page-run']);
      // On a 360 px phone the four tabs fit without scrolling.
      await page.send('Emulation.setDeviceMetricsOverride', {...PHONE, width: 360});
      await sleep(300);
      assert.ok(await page.evaluate(`(e => e.scrollWidth <= e.clientWidth)(document.querySelector('.steps'))`), 'the step tabs scroll on a phone');
      await screenshot(page, 'phone-steps', '.topbar');
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await sleep(200);
    });

    await t.test('the simulated data in the column setup: tabs, log boxes, a new draw and the truth', async () => {
      await page.evaluate(step('data'));
      // The tabs, with the keyboard: arrows, Home and End move along them and show their panel.
      await page.evaluate(`document.getElementById('tab-columns').focus()`);
      const visited = [];
      for (const key of ['ArrowRight', 'ArrowRight', 'End', 'ArrowRight', 'ArrowLeft', 'Home']) {
        await press(page, key);
        visited.push(await page.evaluate(`[document.activeElement.id, [...document.querySelectorAll('#setup [role=tabpanel]')]
          .filter(e => !e.hidden).map(e => e.id).join()]`));
      }
      assert.deepEqual(visited, [['tab-x', 'panel-x'], ['tab-z', 'panel-z'], ['tab-outcome', 'panel-outcome'],
        ['tab-columns', 'panel-columns'], ['tab-outcome', 'panel-outcome'], ['tab-columns', 'panel-columns']]);
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#setup [role=tab]')].map(e => [e.tabIndex, e.getAttribute('aria-label')])`),
        [[0, null], [-1, 'Chemical exposures (X): 8 columns'], [-1, 'Confounders (Z): 3 columns'], [-1, null]]);
      assert.equal(await page.evaluate(`document.getElementById('outcome-chosen').textContent`), 'Time: time · event: event');
      // Log-transform: off, mixed after one log box, then every X; simulated exposures can be negative.
      await page.evaluate(tab('x'));
      const LOG_ALL = `(e => [e.checked, e.indeterminate])(document.getElementById('log-all'))`;
      assert.deepEqual(await page.evaluate(LOG_ALL), [false, false]);
      await page.evaluate(`${option('x1', 'log')}.click()`);
      assert.deepEqual(await page.evaluate(LOG_ALL), [false, true]);
      await page.evaluate(click('#log-all'));
      assert.deepEqual(await page.evaluate(LOG_ALL), [true, false]);
      assert.equal(await page.evaluate(`document.querySelectorAll('#x-list input:checked').length`), 8);
      const logged = await page.evaluate(SUMMARY);
      assert.equal(logged['Chemical exposures (X)'], '8, logged');
      assert.match(logged['Rows'], /^[0-9]+ of 400 used · dropped: [0-9]+ not positive \(log\)$/);
      await page.evaluate(click('#log-all'));
      assert.deepEqual(await page.evaluate(LOG_ALL), [false, false]);
      assert.equal((await page.evaluate(SUMMARY))['Chemical exposures (X)'], '8, untransformed');
      // A new draw keeps the setup; without x8 the setup is not the design, so a fit has no truth.
      await page.evaluate(tab('columns'));
      await page.evaluate(assign('x8', 'ignore'));
      await page.evaluate(setValue('simSeed', '3'));
      assert.deepEqual(await page.evaluate(`[document.getElementById('count-x').textContent, ${roleOf('x8')}]`), ['7', 'ignore']);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.p === 7 && !nplsiDemo.state.busy');
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.fit.data.truth ?? null, document.querySelector('.results-head .facts').textContent,
        document.querySelector('.results .tables thead tr:not(.group) th:nth-child(2)').textContent]`),
        [null, 'Simulated, g = sigmoid · cox', 'Estimate']);
      // Back to the design, at seed 0.
      await page.evaluate(assign('x8', 'exposure'));
      await page.evaluate(setValue('simSeed', '0'));
      assert.deepEqual(Object.values(await page.evaluate(X_ORDER)), ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']);
      // A new outcome starts again from the design.
      await page.evaluate(setValue('outcome', 'binary'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.outcomeType.value, document.forms.controls.elements.family.value,
        ${roleOf('y')}, document.querySelectorAll('#column-table tbody tr').length]`), ['binary', 'binary', 'outcome', 12]);
      await page.evaluate(setValue('outcome', 'cox'));
      assert.equal(await page.evaluate('document.forms.controls.elements.family.value'), 'cox');
    });

    await t.test('the Model page: either model runs, each with its own settings', async () => {
      assert.equal(await page.evaluate(`document.querySelector('h1').textContent`), 'Neural Partially Linear Single-Index Model Demo');
      await page.evaluate(step('model'));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('.model-card')].map(card =>
        [card.querySelector('h2').textContent, card.querySelector('.run-check input').name])`),
        [['NeuralPLSI', 'runNeural'], ['SplinePLSI', 'runSpline']]);
      const NONE = `(() => { const e = document.forms.controls.elements; e.runNeural.checked = false; e.runSpline.checked = false;
        e.runSpline.dispatchEvent(new Event('input', {bubbles: true})); })()`;
      await page.evaluate(NONE);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Choose a model to run under Model.']);
      assert.deepEqual(await page.evaluate(`['neural', 'spline'].map(k => document.getElementById('fields-'+k).disabled)`), [true, true]);
      // A setting of a model that does not run is not checked.
      await page.evaluate(setValue('hiddenUnits', '0'));
      await page.evaluate(check('runSpline'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      await page.evaluate(setValue('hiddenUnits', '32'));
      await page.evaluate(setValue('knots', '1'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Knots must be an integer from 2 to 50.']);
      await page.evaluate(setValue('knots', '7'));
      // SplinePLSI alone, with 7 knots: a cubic spline with 9 coefficients, and its bootstrap.
      await page.evaluate(setValue('B', '8'));
      await page.evaluate(click('#run'));
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      assert.deepEqual(await page.evaluate(`(() => { const f = nplsiDemo.state.fit;
        return [f.neural, f.spline.spline.length, Object.keys(nplsiDemo.state.boot).filter(k => ['neural', 'spline'].includes(k)),
                [...document.querySelectorAll('.downloads button')].map(b => b.textContent),
                document.querySelector('.results .tables thead tr.group').textContent.trim()]; })()`),
        [null, 9, ['spline'], ['Estimates (CSV)', 'SplinePLSI draws (CSV)'], 'SplinePLSI']);
      assert.match(await page.evaluate(`document.querySelector('#link-chart').textContent`), /fitted SplinePLSI index/);
      // Back to NeuralPLSI with the spline, as before.
      await page.evaluate(check('runNeural'));
      await page.evaluate(setValue('knots', '5'));
      assert.deepEqual(await page.evaluate(`['neural', 'spline'].map(k => document.getElementById('fields-'+k).disabled)`), [false, false]);
    });

    await t.test('switching to a CSV file keeps nothing of the simulated Cox setup', async () => {
      assert.equal(await page.evaluate('document.forms.controls.elements.family.value'), 'cox');
      await page.evaluate(choose('csv'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.family.value,
        document.forms.controls.elements.outcomeType.value, document.forms.controls.elements.intercept.checked,
        document.forms.controls.elements.intercept.disabled, document.getElementById('setup').hidden]`),
        ['continuous', 'continuous', true, false, true]);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Open a CSV file.']);
    });

    await t.test('a CSV file with missing values and a text confounder fits', async () => {
      const rows = ['id,y,x1,x2,x3,age,group'];
      for (let i = 0; i < 120; ++i) {
        const x = [Math.sin(i), Math.cos(3*i), ((i*7) % 11)/5-1], u = 0.8*x[0]+0.6*x[1];
        rows.push([i, (Math.tanh(2*u)+0.3*x[2]+0.1*Math.sin(5*i)).toFixed(4), ...x.map(v => v.toFixed(4)),
                   30+(i % 40), 'abc'[i % 3]].join(','));
      }
      rows[5] = rows[5].replace(/,[^,]*$/, ',NA'); rows[9] = rows[9].replace(/^(\d+),[^,]*/, '$1,');
      const file = join(profile, 'example.csv');
      writeFileSync(file, rows.join('\n'));
      await upload(page, file);
      // Defaults: every column ignored, no outcome, Fit disabled with the reason.
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#column-table input:checked')].map(e => e.value)`),
                       Array(7).fill('ignore'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Choose the outcome column.']);
      await page.evaluate(assign('y', 'outcome'));
      for (const name of ['x1', 'x2', 'x3']) await page.evaluate(assign(name, 'exposure'));
      for (const name of ['age', 'group']) await page.evaluate(assign(name, 'confounder'));
      assert.deepEqual(await page.evaluate(SUMMARY), {'Outcome': 'y · continuous', 'Chemical exposures (X)': '3, standardized',
        'Confounders (Z)': '3 (group: 2 indicators vs a)', 'Rows': '118 of 120 used · dropped: 2 missing'});
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      // A network over the engine's 10,000,000 parameters cannot be fitted.
      await page.evaluate(setValue('layers', '16'));
      await page.evaluate(setValue('hiddenUnits', '1024'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. The network would have 15,747,081 ' +
        "parameters, more than the engine's 10,000,000; use fewer hidden units or layers."]);
      await page.evaluate(setValue('hiddenUnits', '32'));
      await page.evaluate(setValue('layers', '2'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      // Single fit: the estimates, without a bootstrap.
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "csv" && !nplsiDemo.state.busy');
      assert.deepEqual(await page.evaluate('nplsiDemo.state.fit.data.names.z'), ['age', 'group=b', 'group=c']);
      assert.match(await page.evaluate(ANNOUNCED), /^Fit done in [0-9.]+ s\. The results are below\.$/);
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.boot, document.querySelector('.results .tables td.na').textContent]`),
                       [null, 'no standard errors']);
      assert.match(await page.evaluate('document.querySelector(".results-head + .facts").textContent'),
                   /^118 of 120 rows used · dropped: 2 missing · 3 chemical exposures \(X\) · 3 confounders \(Z\)$/);
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('.results thead tr:not(.group) th:first-child')]
        .map(th => th.textContent)`), ['Chemical exposure (X)', 'Confounder (Z)', '']);
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), true);
      await page.evaluate(setValue('hiddenUnits', '16'));
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), false);
      await page.evaluate(setValue('hiddenUnits', '32'));
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), true);
    });

    await t.test('a messy CSV: column types, exclusive roles, select all, Cox event and the summary', async () => {
      await page.evaluate(step('data'));
      const messy = messyCsv(), file = join(profile, 'messy.csv');
      writeFileSync(file, messy.text);
      await upload(page, file);
      const notes = await page.evaluate(`Object.fromEntries([...document.querySelectorAll('#column-table tbody tr')].map(row =>
        [row.dataset.column, row.querySelector('.col-meta').textContent]))`);
      // The strings counted as missing, blank for an empty field.
      assert.deepEqual(notes, {id: 'numeric · none missing', race: 'text, 4 levels · 6 missing (blank)', wt: 'numeric · none missing',
        metal_pb: 'numeric · 9 missing (NA)', metal_cd: 'numeric · none missing', metal_hg: 'numeric · 1 non-finite',
        age: 'numeric · 4 missing (blank)', sex: '0/1 · none missing', status: '0/1 · none missing', case: '0/1 · none missing',
        notes: 'mostly missing (text) · 180 of 200 missing (blank)'});
      // Numeric columns show a small histogram; text columns do not.
      assert.deepEqual(Object.fromEntries(await page.evaluate(HISTOGRAMS)), {id: true, race: false, wt: true, metal_pb: true,
        metal_cd: true, metal_hg: true, age: true, sex: true, status: true, case: true, notes: false});
      assert.match(await page.evaluate(`document.querySelector('tr[data-column=race] .col-text').textContent`),
                   /^Text: Z only, as 3 indicators vs “Black”$/);
      assert.equal(await page.evaluate(`document.querySelector('tr[data-column=wt] .col-examples').textContent`),
                   'e.g. 1000.00, 8919.00, 16838.00');
      // Text columns cannot be exposures: the radio is disabled and says why.
      assert.deepEqual(await page.evaluate(`[${roleInput('race', 'exposure')}.disabled, ${roleInput('race', 'exposure')}.parentElement.title]`),
                       [true, 'Text columns cannot be chemical exposures (X)']);
      await page.evaluate(assign('race', 'exposure'));
      assert.equal(await page.evaluate(roleOf('race')), 'ignore');
      // Arrow keys skip the Event radio, hidden unless the outcome is a survival time.
      await page.evaluate(`${roleInput('wt', 'ignore')}.focus()`);
      for (const type of ['keyDown', 'keyUp'])
        await page.send('Input.dispatchKeyEvent', {type, key: 'ArrowLeft', code: 'ArrowLeft', windowsVirtualKeyCode: 37});
      assert.equal(await page.evaluate(roleOf('wt')), 'outcome');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
      assert.equal(await page.evaluate(ANNOUNCED), 'wt: outcome.');
      // A 0/1 outcome suggests the binary family; the outcome it replaces is said to be ignored.
      await page.evaluate(assign('case', 'outcome'));
      assert.equal(await page.evaluate(ANNOUNCED), 'case: outcome. wt, the outcome before, is now ignored. Outcome type: binary (0/1).');
      assert.equal(await page.evaluate(roleOf('wt')), 'ignore');
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.outcomeType.value, document.forms.controls.elements.family.value,
        document.getElementById('family-note').textContent]`), ['binary', 'binary', 'Binary (logistic), because case is coded 0/1.']);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Choose at least one chemical exposure (X).']);
      // The name filter, then All for X applies to the columns shown.
      await page.evaluate(filter('metal'));
      assert.equal(await page.evaluate(`document.querySelectorAll('#column-table tbody tr:not([hidden])').length`), 3);
      await page.evaluate(click('[data-bulk=exposure][data-on="1"]'));
      await page.evaluate(filter(''));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#column-table tbody tr')]
        .filter(row => row.querySelector('input:checked').value === 'exposure').map(row => row.dataset.column)`),
        ['metal_pb', 'metal_cd', 'metal_hg']);
      // One role per row; arrow keys move along a row's roles.
      await page.evaluate(assign('age', 'exposure'));
      assert.equal(await page.evaluate(roleOf('age')), 'exposure');
      assert.equal(await page.evaluate(ANNOUNCED), 'age: chemical exposure X4.');
      await page.evaluate(`${roleInput('age', 'exposure')}.focus()`);
      for (const type of ['keyDown', 'keyUp'])
        await page.send('Input.dispatchKeyEvent', {type, key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39});
      assert.equal(await page.evaluate(roleOf('age')), 'confounder');
      assert.equal(await page.evaluate(`document.querySelectorAll('tr[data-column=age] input:checked').length`), 1);
      assert.equal(await page.evaluate(ANNOUNCED), 'age: confounder (Z).');
      // Up and down move to the same role in the next row shown and change nothing, even
      // past the outcome; a row that cannot take the role gets focus on its own.
      const before = await page.evaluate(ROLES);
      await page.evaluate(`${roleInput('metal_hg', 'confounder')}.focus()`);
      const path = [];
      for (const key of ['ArrowDown', 'ArrowDown', 'ArrowDown', 'ArrowDown', 'ArrowUp']) {
        await press(page, key);
        path.push(await page.evaluate(ACTIVE));
      }
      assert.deepEqual(path, ['age:confounder', 'sex:confounder', 'status:confounder', 'case:confounder', 'status:confounder']);
      await page.evaluate(`${roleInput('wt', 'exposure')}.focus()`);
      await press(page, 'ArrowUp');
      assert.equal(await page.evaluate(ACTIVE), 'race:ignore');
      await press(page, 'ArrowUp');
      assert.equal(await page.evaluate(ACTIVE), 'id:ignore');
      await press(page, 'ArrowUp');
      assert.equal(await page.evaluate(ACTIVE), 'id:ignore');
      // Rows hidden by the filter are skipped.
      await page.evaluate(filter('metal'));
      await page.evaluate(`${roleInput('metal_pb', 'exposure')}.focus()`);
      await press(page, 'ArrowDown');
      await press(page, 'ArrowDown');
      await press(page, 'ArrowDown');
      assert.equal(await page.evaluate(ACTIVE), 'metal_hg:exposure');
      await page.evaluate(filter(''));
      assert.equal(await page.evaluate(ROLES), before);
      for (const name of ['race', 'sex']) await page.evaluate(assign(name, 'confounder'));
      // The X tab lists the exposures in model order; Log-transform ticks the log box of each.
      await page.evaluate(tab('x'));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#x-list li')].map(e => e.textContent)`),
        ['X1metal_pblog', 'X2metal_cd5 values ≤ 0, dropped when logged' + 'log', 'X3metal_hglog']);
      await page.evaluate(click('#log-all'));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#x-list input[name^=log-]:checked')]
        .map(e => e.closest('li').dataset.column)`), ['metal_pb', 'metal_cd', 'metal_hg']);
      await page.evaluate(tab('columns'));
      const kept = 200-messy.drops.missing-messy.drops.nonfinite-messy.drops.nonpositive;
      assert.deepEqual(messy.drops, {missing: 19, nonfinite: 1, nonpositive: 5});
      assert.deepEqual(await page.evaluate(SUMMARY), {
        'Outcome': `case · binary (logistic), ${messy.cases} cases`,
        'Chemical exposures (X)': '3, logged, then standardized',
        'Confounders (Z)': '5 (race: 3 indicators vs Black)',
        'Rows': `${kept} of 200 used · dropped: 19 missing, 1 non-finite, 5 not positive (log)`});
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);

      // Cox: the Event column appears and guesses status; the event can never be the outcome.
      await page.evaluate(setValue('outcomeType', 'cox'));
      assert.deepEqual(await page.evaluate(`[getComputedStyle(document.querySelector('#column-table .h-event')).display,
        document.querySelector('#head-outcome .h-full').textContent, document.forms.controls.elements.intercept.disabled]`),
        ['table-cell', 'Time', true]);
      assert.deepEqual([await page.evaluate(roleOf('case')), await page.evaluate(roleOf('status'))], ['outcome', 'event']);
      await page.evaluate(assign('case', 'event'));
      assert.deepEqual([await page.evaluate(roleOf('case')), await page.evaluate(roleOf('status'))], ['event', 'ignore']);
      assert.equal(await page.evaluate(`document.querySelectorAll('#column-table input[value=outcome]:checked').length`), 0);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Choose the survival time column (Time).']);
      assert.equal((await page.evaluate(SUMMARY))['Outcome'], 'not chosen');
      // Back to binary: the event role is dropped and the outcome type stays as chosen.
      await page.evaluate(setValue('outcomeType', 'binary'));
      assert.equal(await page.evaluate(roleOf('case')), 'ignore');
      assert.equal(await page.evaluate(`getComputedStyle(document.querySelector('#column-table .h-event')).display`), 'none');
      await page.evaluate(assign('case', 'outcome'));
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      assert.equal(await page.evaluate('document.forms.controls.elements.intercept.checked'), true);

      await page.evaluate(`document.activeElement.blur(); document.querySelector('.role-wrap').scrollTop = 0`);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-desktop-${value}`);
      }
      await scheme(page, 'light');
      await screenshot(page, 'csv-step-desktop-light', '#page-data');
      await page.send('Emulation.setDeviceMetricsOverride', PHONE);
      await sleep(300);
      assert.ok(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'the page scrolls sideways');
      assert.ok(await page.evaluate(`(() => { const wrap = document.querySelector('.role-wrap');
        return wrap.scrollWidth <= wrap.clientWidth; })()`), 'the column table scrolls sideways on a phone');
      await screenshot(page, 'csv-phone-light');
      await screenshot(page, 'csv-step-phone-light', '#page-data');
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await sleep(200);

      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "csv" && nplsiDemo.state.fit.fileName === "messy.csv" && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
      assert.deepEqual(await page.evaluate(`(() => { const d = nplsiDemo.state.fit.data;
        return [d.family, d.n, d.p, d.q, d.names.x, d.names.z, nplsiDemo.state.fit.neural.beta.length]; })()`),
        ['binary', kept, 3, 5, ['log metal_pb', 'log metal_cd', 'log metal_hg'],
         ['age', 'race=Mexican American', 'race=Other', 'race=White', 'sex'], 3]);
    });

    await t.test('missing-value strings, decimal commas, bidi controls in a name, and a file chosen while another is read', async () => {
      await page.evaluate(step('data'));
      // Copied from a spreadsheet: tab-separated, thousands separators in income and decimal commas in rate.
      const rlo = String.fromCharCode(0x202e), evil = `evil${rlo}SCV.exe`, rows = [['id', 'smoke', 'edu', 'income', 'rate', 'pb', evil, 'y'].join('\t')];
      for (let i = 0; i < 60; ++i)
        rows.push([i+1, ['None', 'Former', 'Current'][i % 3], ['missing', 'HS', 'College'][i % 3], ['1,250', '2,500', '12,000'][i % 3],
                   (0.5+i/10).toFixed(2).replace('.', ','), i % 10 === 0 ? 'NA' : i % 15 === 4 ? '#N/A' : i % 20 === 13 ? '' : (1+i/7).toFixed(3),
                   'ab'[i % 2], (Math.sin(i)+i/30).toFixed(3)].join('\t'));
      const file = join(profile, 'excel-copy.tsv');
      writeFileSync(file, rows.join('\n'));
      await upload(page, file);
      const notes = await page.evaluate(`Object.fromEntries([...document.querySelectorAll('#column-table tbody tr')].map((row, j) =>
        [j, [...row.querySelectorAll('.col-detail > span:not(.col-order)')].map(e => e.textContent)]))`);
      assert.deepEqual(Object.values(notes), [
        ['numeric · none missing', 'e.g. 1, 2, 3'],
        ['text, 3 levels · none missing', 'e.g. None, Former, Current', 'Text: Z only, as 2 indicators vs “Current”'],
        ['text, 3 levels · none missing', 'e.g. missing, HS, College', 'Text: Z only, as 2 indicators vs “College”'],
        ['text, 3 levels · none missing', 'e.g. 1,250, 2,500, 12,000',
         'Not numbers: “1,250”, “2,500”, “12,000” (commas that may be thousands separators or decimal commas)',
         'Text: Z only, as 2 indicators vs “1,250”'],
        ['numeric, decimal commas · none missing', 'e.g. 0,50 → 0.5, 0,60 → 0.6, 0,70 → 0.7'],
        ['numeric · 13 missing (NA, #N/A, blank)', 'e.g. 1.143, 1.286, 1.429'],
        ['text, 2 levels · none missing', 'e.g. a, b', 'Text: Z only, as 1 indicator vs “a”'],
        ['numeric · none missing', 'e.g. 0.000, 0.875, 0.976']]);
      // The values that make income text are flagged in the critical color.
      assert.ok(await page.evaluate(`getComputedStyle(document.querySelector('tr[data-column=income] .col-text.warn')).color ===
        getComputedStyle(document.getElementById('cancel')).color`));
      assert.deepEqual(await page.evaluate(`[${roleInput('income', 'exposure')}.disabled, ${roleInput('income', 'outcome')}.disabled]`), [true, true]);
      await page.evaluate(assign('y', 'outcome'));
      for (const name of ['rate', 'pb']) await page.evaluate(assign(name, 'exposure'));
      for (const name of ['smoke', 'edu', evil]) await page.evaluate(assign(name, 'confounder'));
      assert.deepEqual(await page.evaluate(SUMMARY), {'Outcome': 'y · continuous', 'Chemical exposures (X)': '2, standardized',
        'Confounders (Z)': '5 (smoke: 2 indicators vs Current; edu: 2 indicators vs College; evilSCV.exe: 1 indicator vs a)',
        'Rows': '47 of 60 used · dropped: 13 missing'});
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      // The name keeps its control character in the data and loses it on the page.
      const bidi = `new RegExp('[' + String.fromCharCode(0x61c, 0x200e, 0x200f, 0x202a) + '-' + String.fromCharCode(0x202e, 0x2066) + '-' +
        String.fromCharCode(0x2069) + ']')`;
      assert.deepEqual(await page.evaluate(`(() => { const bidi = ${bidi};
        return [nplsiDemo.setup().roles.get(${JSON.stringify(evil)}), document.querySelector('#column-table tr[data-index="6"] .col-name').textContent,
                bidi.test(document.body.innerText), [...document.querySelectorAll('[title], [aria-label]')].some(e =>
                  bidi.test(e.title) || bidi.test(e.getAttribute('aria-label') || ''))]; })()`),
        ['confounder', 'evilSCV.exe', false, false]);
      await page.evaluate(`document.activeElement.blur(); document.querySelector('.role-wrap').style.maxHeight = 'none'`);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-missing-tokens-${value}`, '#column-table thead, #column-table tr[data-column=pb]');
      }
      await scheme(page, 'light');
      await page.evaluate(`document.querySelector('.role-wrap').style.maxHeight = ''`);

      // A second file chosen while the first is still being read replaces it; the first is dropped when it arrives.
      writeFileSync(join(profile, 'slow.csv'), 'a,b\n1,2\n3,4\n');
      writeFileSync(join(profile, 'quick.csv'), 'q1,q2,q3\n1,2,3\n4,5,6\n');
      await page.evaluate(`(() => { const text = File.prototype.text; nplsiDemo.restoreText = () => { File.prototype.text = text; };
        File.prototype.text = function () {
          const read = text.call(this);
          return this.name === 'slow.csv' ? read.then(v => new Promise(resolve => setTimeout(() => resolve(v), 1500))) : read;
        }; })()`);
      try {
        await chooseFile(page, join(profile, 'slow.csv'));
        await chooseFile(page, join(profile, 'quick.csv'));
        await page.until('nplsiDemo.setup().fileName === "quick.csv"');
        await sleep(2000);
        assert.deepEqual(await page.evaluate(`[nplsiDemo.setup().fileName, nplsiDemo.setup().table.header.join(),
          [...document.querySelectorAll('#column-table tbody tr')].map(row => row.dataset.column).join()]`),
          ['quick.csv', 'q1,q2,q3', 'q1,q2,q3']);
      } finally {
        await page.evaluate('nplsiDemo.restoreText()');
      }
    });

    await t.test('column names are shown as text, long ones wrap on a phone, results keep the fitted file name', async () => {
      // A header that would run as HTML, and long names.
      const names = ['<img src=x onerror=globalThis.xss=1>', 'urinary_mono_2_ethyl_5_carboxypentyl_phthalate_creatinine_adjusted',
                     'serum_perfluorooctane_sulfonic_acid_linear_and_branched_isomers_ng_per_ml',
                     'household_income_to_poverty_ratio_category_imputed'];
      const rows = [['y', ...names].join(',')];
      for (let i = 0; i < 150; ++i) {
        const x = [Math.sin(i), Math.cos(2*i), Math.sin(3*i+1)], u = 0.7*x[0]+0.5*x[1]+0.5*x[2];
        rows.push([(Math.tanh(2*u)+0.2*Math.sin(7*i)).toFixed(4), ...x.map(v => v.toFixed(4)), i % 5].join(','));
      }
      const file = join(profile, 'long-names.csv');
      writeFileSync(file, rows.join('\n'));
      await upload(page, file);
      assert.equal(await page.evaluate(`document.querySelector('#column-table tr[data-index="1"] .col-name').textContent`), names[0]);
      await page.evaluate(assign('y', 'outcome'));
      for (const name of names.slice(0, 3)) await page.evaluate(assign(name, 'exposure'));
      await page.evaluate(assign(names[3], 'confounder'));
      await page.evaluate(`document.forms.controls.elements.runSpline.checked = true`);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.fileName === "long-names.csv" && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate(RESULTS_LABEL), 'long-names.csv · continuous');
      // Fit with bootstrap, with the setup unchanged, bootstraps this fit; another file
      // opened during the bootstrap does not rename its results.
      await page.evaluate(setValue('B', '100'));
      await page.evaluate('void (nplsiDemo.before = nplsiDemo.state.fit)');
      await page.evaluate(click('#run'));
      await page.until('nplsiDemo.state.busy && /^[1-9]/.test(document.getElementById("progress-text").textContent)');
      await upload(page, join(profile, 'example.csv'));
      assert.equal(await page.evaluate('nplsiDemo.state.busy'), true, 'the bootstrap ended before the second file opened');
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('nplsiDemo.state.fit === nplsiDemo.before'), true, 'the fit was redone');
      assert.equal(await page.evaluate(RESULTS_LABEL), 'long-names.csv · continuous');
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), false);
      // The forest plot tooltip shows the name as text and runs nothing.
      const tip = await page.evaluate(`(async () => {
        const hit = document.querySelector('#forest-chart svg rect[fill=transparent]'), box = hit.getBoundingClientRect();
        hit.dispatchEvent(new PointerEvent('pointerenter', {clientX: box.left+40, clientY: box.top+5}));
        await new Promise(resolve => setTimeout(resolve, 500));
        const tip = document.querySelector('#forest-chart .tooltip');
        return {title: tip.firstChild.textContent, hidden: tip.hidden, images: document.querySelectorAll('#results img').length,
                ran: globalThis.xss !== undefined};
      })()`);
      assert.deepEqual(tip, {title: names[0], hidden: false, images: 0, ran: false});
      // On phones of three widths, long names wrap and the numbers stay in view.
      for (const width of [360, 390, 430]) {
        await page.send('Emulation.setDeviceMetricsOverride', {...PHONE, width});
        await page.evaluate('dispatchEvent(new Event("resize"))');
        await sleep(400);
        assert.ok(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), `the page scrolls sideways at ${width}px`);
        const [, scrolling, problems] = await page.evaluate(TABLES);
        assert.deepEqual(problems, []);
        assert.ok(scrolling >= 2, `${scrolling} tables scroll at ${width}px`);
        for (const [first, wrap] of await page.evaluate(FIRST_COLUMNS))
          assert.ok(first <= 0.45*width+16+1 && first <= 0.55*wrap, `first column ${first}px of ${wrap}px at ${width}px`);
        if (width === 390) {
          await screenshot(page, 'phone-results-long-names', '.results .tables, .results .card:has(> h3 + .table-wrap)');
          await page.evaluate(`for (const wrap of document.querySelectorAll('.results .table-wrap')) wrap.scrollLeft = 10000`);
          await sleep(100);
          await screenshot(page, 'phone-results-long-names-scrolled', '.results .tables, .results .card:has(> h3 + .table-wrap)');
          await page.evaluate(`for (const wrap of document.querySelectorAll('.results .table-wrap')) wrap.scrollLeft = 0`);
        }
      }
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await page.evaluate('dispatchEvent(new Event("resize"))');
      await sleep(200);
    });

    await t.test('the NHANES file as a user CSV, set up as reproduce/nhanes.py, gives the NHANES example', async () => {
      await page.evaluate(step('data'));
      await upload(page, join(root, 'NHANES', '12940_2020_644_MOESM2_ESM.csv'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.outcomeType.value,
        document.querySelectorAll('#column-table input[value=ignore]:checked').length,
        document.querySelectorAll('#x-list li, #z-list li').length]`), ['continuous', 28, 0]);
      // reproduce/nhanes.py: log triglyceride, z-scored; eight exposures in its order, logged except
      // retinol, z-scored; age, sex (female = 1) and race as indicators against White (1).
      await page.evaluate(assign('triglyceride', 'outcome'));
      for (const name of NHANES_EXPOSURES) await page.evaluate(assign(name, 'exposure'));
      for (const name of ['age', 'sex', 'race']) await page.evaluate(assign(name, 'confounder'));
      assert.deepEqual(await page.evaluate(X_ORDER), Object.fromEntries(NHANES_EXPOSURES.map((name, k) => [name, `X${k+1}`])));
      assert.deepEqual(await page.evaluate(`['x', 'z'].map(t => document.getElementById('count-'+t).textContent)`), ['8', '3']);
      // The Z tab: codes are suggested as categorical, not set; age has too many values for the box.
      await page.evaluate(tab('z'));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#z-list li')].map(li =>
        [li.dataset.column, li.querySelector('.col-meta').textContent, li.querySelector('.suggest')?.textContent ?? null,
         [...li.querySelectorAll('input')].map(e => e.name.split('-')[0]+(e.checked ? '+' : '')).join()])`),
        [['age', 'numeric, enters as it is', null, 'zlog,zstd'], ['sex', 'numeric, enters as it is', '(suggested)', 'zlog,zstd,cat'],
         ['race', 'numeric, enters as it is', '(suggested)', 'zlog,zstd,cat']]);
      assert.equal((await page.evaluate(SUMMARY))['Confounders (Z)'], '3');
      // Z has its own log and standardize, for each column and for all; the X boxes do not touch them.
      await page.evaluate(click('#zstd-all'));
      assert.equal((await page.evaluate(SUMMARY))['Confounders (Z)'], '3 (standardized: age, sex, race)');
      await page.evaluate(`${option('age', 'zstd')}.click()`);
      assert.deepEqual(await page.evaluate(`(e => [e.checked, e.indeterminate])(document.getElementById('zstd-all'))`), [false, true]);
      await page.evaluate(click('#zstd-all'));
      await page.evaluate(click('#zstd-all'));
      assert.equal((await page.evaluate(SUMMARY))['Confounders (Z)'], '3');
      for (const name of ['sex', 'race']) await page.evaluate(`${option(name, 'cat')}.click()`);
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#z-list .col-meta')].map(e => e.textContent)`),
                       ['numeric, enters as it is', '1 indicator vs 1', '4 indicators vs 1']);
      // Categorical confounders have no log, z-score or histogram.
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#z-list li')].map(li =>
        [li.querySelectorAll('input[name^=zlog]:disabled').length, li.querySelectorAll('svg').length])`), [[0, 1], [1, 0], [1, 0]]);
      // The X tab: Log-transform logs every X, then retinol is left unlogged and the box shows a mix.
      await page.evaluate(tab('x'));
      await page.evaluate(click('#log-all'));
      await page.evaluate(`${option('a5.Retinol', 'log')}.click()`);
      assert.deepEqual(await page.evaluate(`(e => [e.checked, e.indeterminate])(document.getElementById('log-all'))`), [false, true]);
      await page.evaluate(tab('outcome'));
      assert.deepEqual(await page.evaluate(`[${checked('standardize')}, ${checked('logOutcome')}, ${checked('standardizeOutcome')}]`),
                       [true, false, false]);
      for (const name of ['logOutcome', 'standardizeOutcome']) await page.evaluate(check(name));
      await page.evaluate(setValue('activation', 'ELU'));
      await page.evaluate(`document.forms.controls.elements.runSpline.checked = false`);
      assert.equal(await page.evaluate(checked('intercept')), true);
      assert.deepEqual(await page.evaluate(SUMMARY), NHANES_SUMMARY);
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      for (const name of ['x', 'z', 'outcome']) {
        await page.evaluate(tab(name));
        await page.evaluate('document.activeElement.blur()');
        await screenshot(page, `csv-step-nhanes-tab-${name}-desktop-light`, '#page-data');
      }
      await page.evaluate(tab('columns'));

      // The column table: the last X rows.
      const showLastRows = `document.activeElement.blur(); (wrap => { wrap.scrollTop =
        wrap.querySelector('tr[data-column="a22.2.3.4.6.7.8.hxcdf"]').offsetTop-wrap.querySelector('thead').offsetHeight; })(
        document.querySelector('.role-wrap'))`;
      await page.evaluate(showLastRows);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-step-nhanes-desktop-${value}`, '#page-data');
      }
      for (const width of [360, 390]) {
        await page.send('Emulation.setDeviceMetricsOverride', {...PHONE, width});
        await sleep(300);
        assert.ok(await page.evaluate(`(() => { const wrap = document.querySelector('.role-wrap');
          return document.documentElement.scrollWidth <= innerWidth && wrap.scrollWidth <= wrap.clientWidth; })()`),
          `the CSV step scrolls sideways at ${width}px`);
      }
      await page.evaluate(showLastRows);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-step-nhanes-phone-${value}`, '#page-data');
      }
      await scheme(page, 'light');
      // Cox on a 360 px phone: short role headers, names broken only at their <wbr>s, notes
      // across the row under the role buttons, nothing scrolling sideways.
      await page.send('Emulation.setDeviceMetricsOverride', {...PHONE, width: 360});
      await page.evaluate(setValue('outcomeType', 'cox'));
      await sleep(300);
      const cox = await page.evaluate(ROLE_TABLE);
      assert.deepEqual([cox.broken, cox.scrolls, cox.outside, cox.heads], [[], false, 0, ['Column', 'X', 'Z', 'Time', 'Evt', 'Ign']]);
      assert.ok(cox.tallest <= 130, `a row ${cox.tallest} px tall`);
      console.log(`# role table at 360 px in Cox mode: rows at most ${cox.tallest} px tall`);
      await page.evaluate(`document.activeElement.blur(); document.querySelector('.role-wrap').scrollTop = 0`);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-step-nhanes-phone-cox-360-${value}`, '#page-data');
      }
      await scheme(page, 'light');
      await page.evaluate(setValue('outcomeType', 'continuous'));
      // Back to continuous: the setup is as before.
      assert.deepEqual(await page.evaluate(SUMMARY), NHANES_SUMMARY);
      assert.equal(await page.evaluate(checked('intercept')), true);
      // The desktop's full headers and no broken names there either.
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await sleep(200);
      const desktop = await page.evaluate(ROLE_TABLE);
      assert.deepEqual([desktop.broken, desktop.scrolls, desktop.heads], [[], false, ['Column', 'X', 'Z', 'Outcome', 'Ignore']]);

      await page.evaluate(click('#fit'));
      await page.until(`nplsiDemo.state.fit?.data.source === "csv" && nplsiDemo.state.fit.fileName.startsWith("12940") &&
        !nplsiDemo.state.busy`);
      const fit = await page.evaluate(`(() => { const {data, neural} = nplsiDemo.state.fit;
        nplsiDemo.csvFit = nplsiDemo.state.fit;
        return {n: data.n, p: data.p, q: data.q, x: data.names.x, z: data.names.z, beta: neural.beta}; })()`);
      assert.deepEqual([fit.n, fit.p, fit.q, fit.beta.length], [800, 8, 6, 8]);
      assert.deepEqual(fit.x, NHANES_EXPOSURES.map(name => name === 'a5.Retinol' ? name : `log ${name}`));
      assert.deepEqual(fit.z, ['age', 'sex=2', 'race=2', 'race=3', 'race=4', 'race=5']);
      console.log(`# NHANES as a CSV: beta = ${fit.beta.map(v => v.toFixed(3)).join(', ')}`);
    });

    await t.test('a failed download of the NHANES example is reported until Retry loads it', async () => {
      // The example is a button under Your own data; the file opened by hand stays open when it fails.
      failing.add('/nhanes.csv');
      await page.evaluate(click('#nhanes-load'));
      await page.until('nplsiDemo.state.nhanesError !== null');
      assert.deepEqual(await page.evaluate(`[document.getElementById('nhanes-error').hidden,
        document.getElementById('nhanes-error-text').textContent, nplsiDemo.setup().fileName, nplsiDemo.setup().example,
        document.getElementById('nhanes-load').disabled]`),
        [false, 'Could not load the NHANES example (HTTP 503).', '12940_2020_644_MOESM2_ESM.csv', false, false]);
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
      failing.delete('/nhanes.csv');
      await page.evaluate(click('#nhanes-retry'));
      await page.until('nplsiDemo.setup().example && !nplsiDemo.state.nhanesLoading');
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.nhanesError, document.getElementById('nhanes-error').hidden,
        document.forms.controls.elements.file.value, document.forms.controls.elements.source.value]`), [null, true, '', 'csv']);
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
    });

    await t.test('the NHANES example comes set up as reproduce/nhanes.py, fits, and its bootstrap can be cancelled', async () => {
      // A setup changed by hand goes back to the script's when the example is loaded again.
      await page.evaluate(assign('a5.Retinol', 'ignore'));
      await page.evaluate(click('#nhanes-load'));
      await page.until('nplsiDemo.setup().example && !nplsiDemo.state.nhanesLoading');
      // The same column setup the file got by hand above.
      assert.deepEqual(await page.evaluate(SUMMARY), NHANES_SUMMARY);
      assert.deepEqual(await page.evaluate(X_ORDER), Object.fromEntries(NHANES_EXPOSURES.map((name, k) => [name, `X${k+1}`])));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.activation.value,
        document.forms.controls.elements.intercept.checked]`), ['ELU', true]);
      await page.evaluate(`document.forms.controls.elements.runSpline.checked = false`);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "nhanes" && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate(RESULTS_LABEL), 'NHANES example · continuous');
      // The CSV set up as reproduce/nhanes.py gave the same data and, with the same seed, the same fit.
      const same = await page.evaluate(`(() => {
        const a = nplsiDemo.csvFit, b = nplsiDemo.state.fit;
        const eq = (u, v) => u.length === v.length && Array.prototype.every.call(u, (e, i) => Object.is(e, v[i]));
        const keys = ['family', 'activation', 'hiddenUnits', 'layers', 'weightDecay', 'maxIter', 'intercept', 'seed'];
        return {x: eq(a.data.x, b.data.x), z: eq(a.data.z, b.data.z), y: eq(a.data.y, b.data.y),
                settings: keys.every(k => a.settings[k] === b.settings[k]), beta: eq(a.neural.beta, b.neural.beta),
                gamma: eq(a.neural.gamma, b.neural.gamma), intercept: Object.is(a.neural.intercept, b.neural.intercept),
                g: eq(a.neural.g, b.neural.g), state: a.neural.state === b.neural.state};
      })()`);
      assert.deepEqual(same, {x: true, z: true, y: true, settings: true, beta: true, gamma: true, intercept: true, g: true,
                              state: true});
      // Fit with bootstrap bootstraps this fit; Cancel keeps the fit.
      await page.evaluate('void (nplsiDemo.before = nplsiDemo.state.fit)');
      await page.evaluate(setValue('B', '1000'));
      await page.evaluate(click('#run'));
      await page.until('nplsiDemo.state.busy && /^[1-9]/.test(document.getElementById("progress-text").textContent)');
      await page.evaluate(click('#cancel'));
      await page.until('!nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").textContent'), 'Bootstrap cancelled.');
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.boot, nplsiDemo.state.fit === nplsiDemo.before,
        document.querySelector('.results .tables td.na').textContent]`), [null, true, 'no standard errors']);
      await page.evaluate(setValue('B', '8'));
      await page.evaluate(click('#run'));
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('nplsiDemo.state.fit === nplsiDemo.before'), true);
      await screenshot(page, 'desktop-nhanes');
    });

    await t.test('the preprocessing file rebuilds the fitted data from the raw file', async () => {
      await page.evaluate(`[...document.querySelectorAll('.downloads button')].find(b => /Preprocessing/.test(b.textContent)).click()`);
      const file = join(downloads, 'neuralplsi-preprocessing.json');
      for (let k = 0; k < 100 && !existsSync(file); ++k) await sleep(100);
      const json = JSON.parse(readFileSync(file, 'utf8'));
      assert.deepEqual([json.source, json.file, json.family, json.intercept, json.rows.used, json.x.length, json.z.length],
                       ['nhanes', '12940_2020_644_MOESM2_ESM.csv', 'continuous', true, 800, 8, 6]);
      assert.deepEqual(json.z[1], {name: 'sex=2', column: 'sex', equals: 2, reference: 1});
      assert.match(await page.evaluate(`document.querySelector('.downloads + .help').textContent`),
                   /^The model file loads with NeuralPLSI\.load\(path\) in the Python package, load_model\(\) in R and nplsi_load in C\+\+\. .*preprocessing file says/);
      // Every row of the NHANES file is used, so the rebuilt matrices are the fitted ones.
      const table = parseCsv(readFileSync(join(root, 'NHANES', '12940_2020_644_MOESM2_ESM.csv'), 'utf8'));
      const values = new Map(describeColumns(table).map(c => [c.name, c.values]));
      const build = specs => Array.from(table.rows.keys()).flatMap(i => specs.map(t => {
        const v = values.get(t.column)[i];
        if (t.equals !== undefined) return v === t.equals ? 1 : 0;
        const u = t.log ? Math.log(v) : v;
        return t.mean === undefined || t.mean === null ? u : (u-t.mean)/t.sd;
      }));
      const fitted = await page.evaluate(`(() => { const d = nplsiDemo.state.fit.data;
        return {x: Array.from(d.x), z: Array.from(d.z), y: Array.from(d.y)}; })()`);
      // Node's Math.log and Chrome's can differ in the last bit, so x and y agree to float64 rounding.
      assert.deepEqual(build(json.z), fitted.z);
      for (const k of ['x', 'y']) {
        const rebuilt = build(json[k]), largest = Math.max(...rebuilt.map((v, i) => Math.abs(v-fitted[k][i])));
        assert.equal(rebuilt.length, fitted[k].length);
        assert.ok(largest < 1e-13, `${k}: largest difference ${largest}`);
        console.log(`# preprocessing file: ${k} rebuilt to within ${largest.toExponential(1)}`);
      }
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
        // A call to a worker of a cancelled pool fails as cancelled.
        const after = await pool.call(worker, 'summary', [new Float64Array(4), 2, 2]).then(() => 'ran', e => e.name);
        return version+' '+fit.beta.length+' '+after;
      })()`);
      assert.equal(version, '0.1.0 8 Cancelled');
    });

    await t.test('the page works offline once loaded', async () => {
      await page.until('navigator.serviceWorker.controller !== null');
      await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
      await page.send('Page.reload');
      await page.until('document.readyState === "complete" && globalThis.nplsiDemo');
      await page.evaluate(click('#nhanes-load'));
      await page.until('nplsiDemo.setup().example && !nplsiDemo.state.nhanesLoading');
      await page.evaluate(choose('simulated'));
      await page.evaluate(setValue('n', '100'));
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
    });
    assert.deepEqual(errors, []);
    page.socket.close();
  } finally {
    browser?.kill();
    server.close();
    await sleep(300);
    rmSync(profile, {recursive: true, force: true});
  }
});
