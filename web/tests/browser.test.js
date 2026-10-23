// Drives the built page in headless Chrome over the DevTools protocol (no npm
// packages): a simulated fit and bootstrap through the real Web Worker pool, the
// result tables on a phone, CSV files (column roles, the summary and the reasons
// Fit is disabled, column names that look like HTML, long names on a phone, the
// NHANES file as a user file giving the NHANES example exactly), the NHANES
// example and a failed download of it, Cancel, model and preprocessing downloads
// and offline use after the first load. Screenshots in light and dark mode at
// desktop and phone widths go to NPLSI_SCREENSHOTS when it names a directory.
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
// A row's log ('log') or categorical ('cat') box.
const option = (column, kind) => `document.querySelector(${JSON.stringify(
  `#column-table tr[data-column=${JSON.stringify(column)}] input[name^=${kind}-]`)})`;
const check = name => `document.forms.controls.elements[${JSON.stringify(name)}].click()`;
const checked = name => `document.forms.controls.elements[${JSON.stringify(name)}].checked`;
// The position shown in each X row.
const X_ORDER = `Object.fromEntries([...document.querySelectorAll('#column-table tbody tr[data-role=exposure]')].map(row =>
  [row.dataset.column, row.querySelector('.col-order').textContent]))`;
const RESULTS_LABEL = `document.querySelector('.results-head .facts').textContent`;
const SUMMARY = `Object.fromEntries([...document.querySelectorAll('#data-summary div')].map(row =>
  [row.querySelector('dt').textContent, row.querySelector('dd').textContent]))`;
const FIT_STATE = `[document.getElementById('fit').disabled, document.getElementById('fit-status').textContent]`;
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

async function upload(page, file) {
  const {root: doc} = await page.send('DOM.getDocument');
  const {nodeId} = await page.send('DOM.querySelector', {nodeId: doc.nodeId, selector: 'input[name=file]'});
  await page.send('DOM.setFileInputFiles', {nodeId, files: [file]});
  await page.until(`nplsiDemo.state.fileName === ${JSON.stringify(file.split('/').at(-1))} &&
    !document.getElementById('csv-columns').hidden`);
}

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
    await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
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

    await t.test('screenshots, no horizontal scrolling and result tables that scroll on a phone', async () => {
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

    await t.test('switching to a CSV file keeps nothing of the simulated Cox setup', async () => {
      assert.equal(await page.evaluate('document.forms.controls.elements.family.value'), 'cox');
      await page.evaluate(choose('csv'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.family.value,
        document.forms.controls.elements.csvFamily.value, document.forms.controls.elements.intercept.checked,
        document.forms.controls.elements.intercept.disabled]`), ['continuous', 'continuous', true, false]);
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
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "csv" && !nplsiDemo.state.busy');
      assert.deepEqual(await page.evaluate('nplsiDemo.state.fit.data.names.z'), ['age', 'group=b', 'group=c']);
      assert.match(await page.evaluate('document.querySelector(".results-head + .facts").textContent'),
                   /^118 of 120 rows used · dropped: 2 missing · 3 chemical exposures \(X\) · 3 confounders \(Z\)$/);
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('.results thead tr:not(.group) th:first-child')]
        .map(th => th.textContent)`), ['Chemical exposure (X)', 'Confounder (Z)', '']);
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), true);
      await page.evaluate(setValue('hiddenUnits', '16'));
      assert.equal(await page.evaluate('document.querySelector(".stale").hidden'), false);
      assert.equal(await page.evaluate('document.getElementById("bootstrap").disabled'), true);
      await page.evaluate(setValue('hiddenUnits', '32'));
    });

    await t.test('a messy CSV: column types, exclusive roles, select all, Cox event and the summary', async () => {
      const messy = messyCsv(), file = join(profile, 'messy.csv');
      writeFileSync(file, messy.text);
      await upload(page, file);
      const notes = await page.evaluate(`Object.fromEntries([...document.querySelectorAll('#column-table tbody tr')].map(row =>
        [row.dataset.column, row.querySelector('.col-meta').textContent]))`);
      assert.deepEqual(notes, {id: 'numeric · none missing', race: 'text, 4 levels · 6 missing', wt: 'numeric · none missing',
        metal_pb: 'numeric · 9 missing', metal_cd: 'numeric · none missing', metal_hg: 'numeric · 1 non-finite',
        age: 'numeric · 4 missing', sex: '0/1 · none missing', status: '0/1 · none missing', case: '0/1 · none missing',
        notes: 'mostly missing (text) · 180 of 200 missing'});
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
      await page.evaluate(assign('wt', 'ignore'));
      // A 0/1 outcome suggests the binary family.
      await page.evaluate(assign('case', 'outcome'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.csvFamily.value, document.forms.controls.elements.family.value,
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
      await page.evaluate(`${roleInput('age', 'exposure')}.focus()`);
      for (const type of ['keyDown', 'keyUp'])
        await page.send('Input.dispatchKeyEvent', {type, key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39});
      assert.equal(await page.evaluate(roleOf('age')), 'confounder');
      assert.equal(await page.evaluate(`document.querySelectorAll('tr[data-column=age] input:checked').length`), 1);
      for (const name of ['race', 'sex']) await page.evaluate(assign(name, 'confounder'));
      // Log-transform X: All ticks the log box of every X row.
      await page.evaluate(click('[data-log="1"]'));
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#column-table input[name^=log-]:checked')]
        .map(e => e.closest('tr').dataset.column)`), ['metal_pb', 'metal_cd', 'metal_hg']);
      const kept = 200-messy.drops.missing-messy.drops.nonfinite-messy.drops.nonpositive;
      assert.deepEqual(messy.drops, {missing: 19, nonfinite: 1, nonpositive: 5});
      assert.deepEqual(await page.evaluate(SUMMARY), {
        'Outcome': `case · binary (logistic), ${messy.cases} cases`,
        'Chemical exposures (X)': '3, logged, then standardized',
        'Confounders (Z)': '5 (race: 3 indicators vs Black)',
        'Rows': `${kept} of 200 used · dropped: 19 missing, 1 non-finite, 5 not positive (log)`});
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);

      // Cox: the Event column appears and guesses status; the event can never be the outcome.
      await page.evaluate(setValue('csvFamily', 'cox'));
      assert.deepEqual(await page.evaluate(`[getComputedStyle(document.querySelector('#column-table .h-event')).display,
        document.getElementById('head-outcome').textContent, document.forms.controls.elements.intercept.disabled]`),
        ['table-cell', 'Time', true]);
      assert.deepEqual([await page.evaluate(roleOf('case')), await page.evaluate(roleOf('status'))], ['outcome', 'event']);
      await page.evaluate(assign('case', 'event'));
      assert.deepEqual([await page.evaluate(roleOf('case')), await page.evaluate(roleOf('status'))], ['event', 'ignore']);
      assert.equal(await page.evaluate(`document.querySelectorAll('#column-table input[value=outcome]:checked').length`), 0);
      assert.deepEqual(await page.evaluate(FIT_STATE), [true, 'Cannot fit yet. Choose the survival time column (Time).']);
      assert.equal((await page.evaluate(SUMMARY))['Outcome'], 'not chosen');
      // Back to binary: the event role is dropped and the outcome type stays as chosen.
      await page.evaluate(setValue('csvFamily', 'binary'));
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
      await screenshot(page, 'csv-step-desktop-light', '#controls');
      await page.send('Emulation.setDeviceMetricsOverride', PHONE);
      await sleep(300);
      assert.ok(await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'the page scrolls sideways');
      assert.ok(await page.evaluate(`(() => { const wrap = document.querySelector('.role-wrap');
        return wrap.scrollWidth <= wrap.clientWidth; })()`), 'the column table scrolls sideways on a phone');
      await screenshot(page, 'csv-phone-light');
      await screenshot(page, 'csv-step-phone-light', '#controls > section:first-child, #controls > section:last-child');
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await sleep(200);

      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "csv" && nplsiDemo.state.fileName === "messy.csv" && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate('document.getElementById("message").hidden'), true);
      assert.deepEqual(await page.evaluate(`(() => { const d = nplsiDemo.state.fit.data;
        return [d.family, d.n, d.p, d.q, d.names.x, d.names.z, nplsiDemo.state.fit.neural.beta.length]; })()`),
        ['binary', kept, 3, 5, ['log metal_pb', 'log metal_cd', 'log metal_hg'],
         ['age', 'race=Mexican American', 'race=Other', 'race=White', 'sex'], 3]);
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
      await page.evaluate(`document.forms.controls.elements.spline.checked = true`);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.fileName === "long-names.csv" && !nplsiDemo.state.busy');
      assert.equal(await page.evaluate(RESULTS_LABEL), 'long-names.csv · continuous');
      // Another file opened during a bootstrap of this fit does not rename its results.
      await page.evaluate(setValue('B', '100'));
      await page.evaluate(click('#bootstrap'));
      await page.until('nplsiDemo.state.busy && /^[1-9]/.test(document.getElementById("progress-text").textContent)');
      await upload(page, join(profile, 'example.csv'));
      assert.equal(await page.evaluate('nplsiDemo.state.busy'), true, 'the bootstrap ended before the second file opened');
      await page.until('nplsiDemo.state.boot && !nplsiDemo.state.busy');
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
      await upload(page, join(root, 'NHANES', '12940_2020_644_MOESM2_ESM.csv'));
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.csvFamily.value,
        document.querySelectorAll('#column-table input[value=ignore]:checked').length,
        document.querySelectorAll('#column-table input[type=checkbox]:checked').length]`), ['continuous', 28, 0]);
      // reproduce/nhanes.py: log triglyceride, z-scored; eight exposures in its order, logged except
      // retinol, z-scored; age, sex (female = 1) and race as indicators against White (1).
      await page.evaluate(assign('triglyceride', 'outcome'));
      for (const [name] of NHANES_EXPOSURES) await page.evaluate(assign(name, 'exposure'));
      for (const name of ['age', 'sex', 'race']) await page.evaluate(assign(name, 'confounder'));
      assert.deepEqual(await page.evaluate(X_ORDER), Object.fromEntries(NHANES_EXPOSURES.map(([name], k) => [name, `X${k+1}`])));
      // Codes are suggested as categorical, not set.
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#column-table .suggest')].map(e =>
        [e.closest('tr').dataset.column, e.textContent, e.closest('label').querySelector('input').checked])`),
        [['sex', '(suggested)', false], ['race', '(suggested)', false]]);
      assert.deepEqual(await page.evaluate(`[...document.querySelectorAll('#column-table tr[data-role=confounder]')].map(row =>
        [row.dataset.column, row.querySelector('.col-meta').textContent, getComputedStyle(row.querySelector('.opt-categorical')).display])`),
        [['age', 'numeric · none missing', 'none'], ['sex', 'numeric, 2 integer codes · none missing', 'flex'],
         ['race', 'numeric, 5 integer codes · none missing', 'flex']]);
      assert.equal((await page.evaluate(SUMMARY))['Confounders (Z)'], '3');
      await page.evaluate(click('[data-log="1"]'));
      await page.evaluate(`${option('a5.Retinol', 'log')}.click()`);
      for (const name of ['sex', 'race']) await page.evaluate(`${option(name, 'cat')}.click()`);
      assert.equal(await page.evaluate(`${option('age', 'cat')}.checked`), false);
      assert.deepEqual(await page.evaluate(`[${checked('standardize')}, ${checked('logOutcome')}, ${checked('standardizeOutcome')}]`),
                       [true, false, false]);
      for (const name of ['logOutcome', 'standardizeOutcome']) await page.evaluate(check(name));
      await page.evaluate(setValue('activation', 'ELU'));
      await page.evaluate(`document.forms.controls.elements.spline.checked = false`);
      assert.equal(await page.evaluate(checked('intercept')), true);
      assert.deepEqual(await page.evaluate(SUMMARY), {'Outcome': 'log triglyceride (z-score) · continuous',
        'Chemical exposures (X)': '8, logged except a5.Retinol, then standardized',
        'Confounders (Z)': '6 (sex: 1 indicator vs 1; race: 4 indicators vs 1)', 'Rows': '800 of 800 used'});
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);

      // The CSV step with its new controls: the last X rows with their log boxes and the Z rows of codes.
      const showLastRows = `document.activeElement.blur(); (wrap => { wrap.scrollTop =
        wrap.querySelector('tr[data-column="a22.2.3.4.6.7.8.hxcdf"]').offsetTop-wrap.querySelector('thead').offsetHeight; })(
        document.querySelector('.role-wrap'))`;
      await page.evaluate(showLastRows);
      for (const value of ['light', 'dark']) {
        await scheme(page, value);
        await screenshot(page, `csv-step-nhanes-desktop-${value}`, '#controls > section:first-child');
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
        await screenshot(page, `csv-step-nhanes-phone-${value}`, '#controls > section:first-child');
      }
      await scheme(page, 'light');
      await page.send('Emulation.setDeviceMetricsOverride', DESKTOP);
      await sleep(200);

      await page.evaluate(click('#fit'));
      await page.until(`nplsiDemo.state.fit?.data.source === "csv" && nplsiDemo.state.fit.fileName.startsWith("12940") &&
        !nplsiDemo.state.busy`);
      const fit = await page.evaluate(`(() => { const {data, neural} = nplsiDemo.state.fit;
        nplsiDemo.csvFit = nplsiDemo.state.fit;
        return {n: data.n, p: data.p, q: data.q, x: data.names.x, z: data.names.z, beta: neural.beta}; })()`);
      assert.deepEqual([fit.n, fit.p, fit.q, fit.beta.length], [800, 8, 6, 8]);
      assert.deepEqual(fit.x, NHANES_EXPOSURES.map(([name]) => name === 'a5.Retinol' ? name : `log ${name}`));
      assert.deepEqual(fit.z, ['age', 'sex=2', 'race=2', 'race=3', 'race=4', 'race=5']);
      console.log(`# NHANES as a CSV: beta = ${fit.beta.map(v => v.toFixed(3)).join(', ')}`);
    });

    await t.test('a failed NHANES download is reported until Retry loads it', async () => {
      failing.add('/nhanes.csv');
      await page.evaluate(choose('nhanes'));
      await page.until('nplsiDemo.state.nhanesError !== null');
      assert.deepEqual(await page.evaluate(FIT_STATE), [true,
        'Cannot fit yet. Could not load the NHANES file (HTTP 503). Press Retry under Data to load it again.']);
      assert.deepEqual(await page.evaluate(`[document.getElementById('nhanes-error').hidden,
        document.getElementById('nhanes-error-text').textContent]`), [false, 'Could not load the NHANES file (HTTP 503).']);
      failing.delete('/nhanes.csv');
      await page.evaluate(click('#nhanes-retry'));
      await page.until('nplsiDemo.state.nhanes && !nplsiDemo.state.nhanesLoading');
      assert.deepEqual(await page.evaluate(`[nplsiDemo.state.nhanesError, document.getElementById('nhanes-error').hidden]`),
                       [null, true]);
      assert.deepEqual(await page.evaluate(FIT_STATE), [false, '']);
    });

    await t.test('NHANES loads, fits and a bootstrap can be cancelled', async () => {
      await page.evaluate(choose('nhanes'));
      await page.until('nplsiDemo.state.nhanes');
      assert.deepEqual(await page.evaluate(SUMMARY), {'Outcome': 'log triglyceride (z-score) · continuous',
        'Chemical exposures (X)': '8, logged (except retinol), then standardized',
        'Confounders (Z)': '6 (age, female and 4 race indicators)', 'Rows': '800 of 800 used'});
      assert.deepEqual(await page.evaluate(`[document.forms.controls.elements.activation.value,
        document.forms.controls.elements.intercept.checked]`), ['ELU', true]);
      await page.evaluate(`document.forms.controls.elements.spline.checked = false`);
      await page.evaluate(click('#fit'));
      await page.until('nplsiDemo.state.fit?.data.source === "nhanes" && !nplsiDemo.state.busy');
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

    await t.test('the preprocessing file rebuilds the fitted data from the raw file', async () => {
      await page.evaluate(`[...document.querySelectorAll('.downloads button')].find(b => /Preprocessing/.test(b.textContent)).click()`);
      const file = join(downloads, 'neuralplsi-preprocessing.json');
      for (let k = 0; k < 100 && !existsSync(file); ++k) await sleep(100);
      const json = JSON.parse(readFileSync(file, 'utf8'));
      assert.deepEqual([json.source, json.file, json.family, json.intercept, json.rows.used, json.x.length, json.z.length],
                       ['nhanes', '12940_2020_644_MOESM2_ESM.csv', 'continuous', true, 800, 8, 6]);
      assert.deepEqual(json.z[1], {name: 'female', column: 'sex', equals: 2, reference: 1});
      assert.match(await page.evaluate(`document.querySelector('.downloads + .help').textContent`), /preprocessing file says/);
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
