// The page: reads the controls, runs fits and the bootstrap in the worker pool,
// and renders estimates, plots and downloads. Nothing leaves the browser.
import {simulatedTable, parseCsv, describeColumns, LINKS, NHANES_SETUP, NHANES_SETTINGS, MAX_LEVELS} from './data.js';
import {ROLES, blockedRole, blockedCategorical, initialRoles, presetRoles, roleOf, assignRole, setRoleForMany, selection,
        suggestFamily, changeFamily, planCsv, planFacts, summaryRows} from './roles.js';
import {NEURAL_DEFAULTS, networkProblem} from './engine.js';
import {Pool, Cancelled} from './pool.js';
import {linkChart, forestChart} from './charts.js';
import {fmt, duration, toCsv, linspace, mergeRows, displayText} from './stats.js';

const form = document.getElementById('controls');
const $ = id => document.getElementById(id);
const field = name => form.elements[name];
const pool = new Pool();
// The controls every source's column setup shares; each setup keeps its own values.
const OPTIONS = ['outcomeType', 'standardize', 'logOutcome', 'standardizeOutcome'];
// One column setup per data source, so switching sources keeps each one's
// choices: the table and its columns, the roles, logged (exposures to log) and
// categorical (numeric confounders to dummy-code), familyAuto (true until the
// visitor picks the outcome type, which until then follows the outcome column)
// and the values of the shared controls (options). fileName names an opened CSV
// file. A simulated setup also keeps the truth, the setup of the design itself,
// the simulation's settings and why the simulation could not run (error).
const blankSetup = () => ({table: null, columns: null, roles: null, logged: new Set(), categorical: new Set(), familyAuto: true,
  fileName: '', options: {outcomeType: 'continuous', standardize: true, logOutcome: false, standardizeOutcome: false},
  truth: null, design: null, simulation: null, error: null});
// nhanesError keeps why the NHANES file did not load. booting is true while a
// bootstrap of the fit shown runs. tab is the column setup's open tab.
const state = {setups: {csv: blankSetup(), nhanes: blankSetup(), simulated: blankSetup()}, nhanesError: null,
               nhanesLoading: null, fit: null, boot: null, busy: false, booting: false, family: null, tab: 'columns'};
// Seeds must leave room for the bootstrap fit seeds seed+1337+b below 2**64.
const MAX_SEED = 2n**64n-1n-1337n-1000n;
const MODELS = [['neural', 'NeuralPLSI', 'var(--series-1)'], ['spline', 'SplinePLSI', 'var(--series-2)']];
// The intercept each source starts with: Python's default for the simulation,
// reproduce/nhanes.py for NHANES, and one for other data, since g(0) = 0.
const INTERCEPT = {simulated: NEURAL_DEFAULTS.intercept, nhanes: NHANES_SETTINGS.intercept, csv: true};

// Elements are built from text nodes, never parsed as HTML. Text, titles and
// labels lose their bidi controls (displayText), so a column name cannot reorder
// the text around it; data attributes keep names as they are.
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined)
      node.setAttribute(k, v === true ? '' : k === 'title' || k === 'aria-label' ? displayText(v) : v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false)
    node.append(c instanceof Node ? c : displayText(c));
  return node;
}

const setText = (node, text) => { node.textContent = displayText(text); };

// One polite status line for screen readers: finished runs, coarse progress and
// role changes. The same text twice still counts as a change.
function announce(text) {
  const line = $('announce'), next = displayText(text);
  line.textContent = line.textContent === next ? `${next} ` : next;
}

// ---- Controls

const source = () => form.elements.source.value;
const setup = () => state.setups[source()];
const outcomeType = () => setup().options.outcomeType;
const plural = (k, word, many = `${word}s`) => `${k} ${k === 1 ? word : many}`;

function integer(name, lo, hi, label) {
  const raw = field(name).value.trim(), v = Number(raw);
  if (!/^\d+$/.test(raw) || v < lo || v > hi) throw new Error(`${label} must be an integer from ${lo} to ${hi}`);
  return v;
}

function settings() {
  const family = field('family').value, decay = field('weightDecay').value.trim(), seed = field('seed').value.trim();
  if (decay === '' || !Number.isFinite(Number(decay)) || Number(decay) < 0)
    throw new Error('Weight decay must be finite and nonnegative');
  if (!/^\d+$/.test(seed) || BigInt(seed) > MAX_SEED) throw new Error(`The seed must be an integer from 0 to ${MAX_SEED}`);
  return {family, activation: field('activation').value, hiddenUnits: integer('hiddenUnits', 1, 1024, 'Hidden units'),
          layers: integer('layers', 1, 16, 'Hidden layers'), weightDecay: Number(decay),
          maxIter: integer('maxIter', 0, 1000000, 'Max iterations'), intercept: family !== 'cox' && field('intercept').checked,
          spline: field('spline').checked, seed: BigInt(seed)};
}

// A table becomes a setup: from a preset ({family, outcome, event, exposures,
// confounders, log, standardize, categorical, logOutcome, standardizeOutcome},
// as NHANES_SETUP), or with every column ignored, nothing logged or categorical
// and an outcome type that follows the outcome.
function loadTable(s, table, preset = null) {
  s.table = table;
  s.columns = describeColumns(table);
  s.roles = preset ? presetRoles(s.columns, preset) : initialRoles(s.columns);
  s.logged = new Set(preset?.log ?? []);
  s.categorical = new Set(preset?.categorical ?? []);
  s.familyAuto = !preset;
  s.options.outcomeType = preset?.family ?? 'continuous';
  if (preset) Object.assign(s.options, {standardize: preset.standardize, logOutcome: preset.logOutcome,
                                        standardizeOutcome: preset.standardizeOutcome});
}

function setupPlan(s = setup()) {
  return planCsv(s.table, s.columns, {family: s.options.outcomeType, roles: s.roles, log: [...s.logged],
    standardize: s.options.standardize, categorical: [...s.categorical], logOutcome: s.options.logOutcome,
    standardizeOutcome: s.options.standardizeOutcome});
}

// The simulated table follows its controls. A new draw of the same outcome keeps
// the setup, since its columns are the same; a new outcome starts from the design.
function simulateTable() {
  const s = state.setups.simulated;
  let options;
  try {
    options = {n: integer('n', 20, 20000, 'Rows'), g: field('g').value, outcome: field('outcome').value,
               seed: integer('simSeed', 0, Number.MAX_SAFE_INTEGER, 'The simulation seed')};
  } catch (error) {
    s.error = error.message;
    return;
  }
  const {table, truth, setup: design} = simulatedTable(options);
  if (s.table && s.design.family === design.family) {
    s.table = table;
    s.columns = describeColumns(table);
  } else loadTable(s, table, design);
  Object.assign(s, {truth, design, simulation: options, error: null});
}

// Whether a simulated setup is still the design, so the truth applies.
function matchesDesign(s, plan) {
  const d = s.design, same = (a, b) => a.length === b.length && a.every((v, k) => v === b[k]);
  return plan.family === d.family && plan.outcome === d.outcome && plan.event === d.event &&
    same(plan.exposures, d.exposures) && same(plan.confounders, d.confounders) && !plan.logged.length &&
    !plan.standardize && !plan.logOutcome && !plan.standardizeOutcome && plan.terms.every(t => t.levels === undefined);
}

// Why the source has no table to set up, or null.
function missingTable() {
  const s = setup();
  if (source() === 'nhanes') {
    if (state.nhanesError) return `${state.nhanesError}. Press Retry under Data to load it again`;
    if (!s.table) return 'The NHANES file is still loading';
  }
  if (source() === 'simulated' && s.error) return s.error;
  return s.table ? null : 'Open a CSV file';
}

function readData() {
  const missing = missingTable();
  if (missing) throw new Error(missing);
  const s = setup(), plan = setupPlan(s);
  if (!plan.ready) throw new Error(plan.reason);
  const data = {...plan.data, source: source()};
  if (source() === 'simulated') {
    data.simulation = s.simulation;
    if (matchesDesign(s, plan)) data.truth = s.truth;
  }
  return data;
}

// summaryRows input for prepared data, for the line above the results.
function dataFacts(d) {
  const total = (from, step) => { let s = 0; for (let i = from; i < d.y.length; i += step) s += d.y[i]; return s; };
  return {family: d.family, outcomeLabel: d.names.y.join(', '), p: d.p, q: d.q, rows: d.rows, n: d.n, dropped: d.dropped,
          cases: d.family === 'binary' ? total(0, 1) : null, events: d.family === 'cox' ? total(1, 2) : null};
}

// One line for the results: rows, X, Z and cases or events.
function describe(d) {
  const facts = dataFacts(d), [, rows] = summaryRows(facts).at(-1);
  const parts = [d.rows ? rows.replace(' used', ' rows used') : `${d.n} rows`,
                 `${plural(d.p, 'chemical exposure')} (X)`, `${plural(d.q, 'confounder')} (Z)`];
  if (facts.cases !== null) parts.push(plural(facts.cases, 'case'));
  if (facts.events !== null) parts.push(plural(facts.events, 'event'));
  return parts.join(' · ');
}

// The Model card's Family follows the outcome type of the setup. Leaving Cox
// restores the source's intercept.
function syncFamily() {
  const family = field('family'), intercept = field('intercept'), type = outcomeType();
  family.value = type;
  const cox = type === 'cox', head = $('head-outcome');
  if (state.family === 'cox' && !cox) intercept.checked = INTERCEPT[source()];
  state.family = type;
  intercept.disabled = cox;
  if (cox) intercept.checked = false;
  $('column-table').classList.toggle('cox', cox);
  $('outcome-options').hidden = type !== 'continuous';
  // The full header, and the short one shown on phones.
  head.querySelector('.h-full').textContent = cox ? 'Time' : 'Outcome';
  head.querySelector('.h-short').textContent = cox ? 'Time' : 'Out';
  head.title = cox ? 'Survival time (the outcome)' : 'Outcome';
}

// What the data step would fit, or why it cannot: summary facts and a reason.
function dataStatus() {
  const missing = missingTable();
  if (missing) return {facts: null, reason: missing};
  const plan = setupPlan();
  return {facts: planFacts(plan), reason: plan.reason};
}

function renderSummary(facts) {
  $('data-summary').replaceChildren(...(facts ? summaryRows(facts) : []).map(([label, text]) =>
    h('div', {}, h('dt', {}, label), h('dd', {}, text))));
}

function familyNote() {
  const note = $('family-note'), s = setup();
  let text = '';
  if (s.columns) {
    const {outcome} = selection(s.roles, s.columns), column = s.columns.find(c => c.name === outcome);
    if (column?.binary && outcomeType() === 'binary' && s.familyAuto)
      text = `Binary (logistic), because ${outcome} is coded 0/1.`;
    else if (column?.binary && outcomeType() === 'continuous')
      text = `${outcome} is coded 0/1: the binary (logistic) outcome type usually fits it.`;
  }
  setText(note, text);
  note.hidden = !text;
}

// Everything that defines a fit, to tell when results no longer match the controls.
function controlsKey() {
  const values = [];
  for (const e of form.elements) {
    if (!e.name || e.name === 'B' || e.type === 'file') continue;
    values.push(e.type === 'checkbox' || e.type === 'radio' ? `${e.name}=${e.value}:${e.checked}` : `${e.name}=${e.value}`);
  }
  return JSON.stringify([values, setup().fileName, setup().table?.rows.length]);
}

function onControls() {
  syncFamily();
  const status = dataStatus();
  let reason = status.reason;
  if (!reason) try { reason = networkProblem(status.facts.p, status.facts.q, settings()); } catch (error) { reason = error.message; }
  renderSummary(status.facts);
  familyNote();
  setText($('fit-status'), reason ? `Cannot fit yet. ${reason}${/[.!?]$/.test(reason) ? '' : '.'}` : '');
  for (const id of ['run', 'fit']) $(id).disabled = state.busy || Boolean(reason);
  const stale = state.fit && controlsKey() !== state.fit.key;
  document.querySelector('.stale')?.toggleAttribute('hidden', !stale);
}

// The shared controls show the values of the source's setup.
function loadOptions() {
  const {options} = setup();
  for (const name of OPTIONS) {
    const e = field(name);
    if (e.type === 'checkbox') e.checked = options[name]; else e.value = options[name];
  }
}

async function onSource() {
  for (const panel of document.querySelectorAll('[data-panel]')) panel.hidden = panel.dataset.panel !== source();
  // Presets: Python's defaults for the simulation, reproduce/nhanes.py for NHANES,
  // and an intercept for other data, since g is anchored at g(0) = 0.
  if (source() === 'simulated') field('activation').value = NEURAL_DEFAULTS.activation;
  else if (source() === 'nhanes') field('activation').value = NHANES_SETTINGS.activation;
  field('intercept').checked = INTERCEPT[source()];
  if (source() === 'simulated' && !setup().table) simulateTable();
  renderSetup();
  onControls();
  if (source() === 'nhanes') await loadNhanes();
}

// The NHANES file loads when NHANES is first chosen, set up as reproduce/nhanes.py.
// A failure is kept and shown with a Retry button until a retry starts.
function loadNhanes() {
  const s = state.setups.nhanes;
  if (s.table || state.nhanesLoading) return state.nhanesLoading;
  state.nhanesError = null;
  state.nhanesLoading = (async () => {
    try {
      const response = await fetch('nhanes.csv');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      loadTable(s, parseCsv(await response.text()), NHANES_SETUP);
    } catch (error) {
      state.nhanesError = `Could not load the NHANES file (${error.message})`;
    } finally {
      state.nhanesLoading = null;
    }
    showNhanesError();
    if (source() === 'nhanes') renderSetup();
    onControls();
  })();
  showNhanesError();
  onControls();
  return state.nhanesLoading;
}

function showNhanesError() {
  $('nhanes-error').hidden = !state.nhanesError;
  $('nhanes-error-text').textContent = state.nhanesError ? `${state.nhanesError}.` : '';
}

function onSimulation() {
  simulateTable();
  renderSetup();
  onControls();
}

// ---- The column setup: the outcome type, then tabs for the column roles, the
// chemical exposures (X), the confounders (Z) and the outcome.

const TABS = ['columns', 'x', 'z', 'outcome'];
function selectTab(name, focus = false) {
  state.tab = name;
  for (const t of TABS) {
    const tab = $(`tab-${t}`), on = t === name;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    $(`panel-${t}`).hidden = !on;
  }
  if (focus) $(`tab-${name}`).focus();
}

// Left and right arrows, Home and End move between the tabs, as in any tab list.
function onTabKey(event) {
  const at = TABS.indexOf(event.target.id.slice(4)), last = TABS.length-1;
  const next = {ArrowLeft: at ? at-1 : last, ArrowRight: at === last ? 0 : at+1, Home: 0, End: last}[event.key];
  if (next === undefined || event.altKey || event.ctrlKey || event.metaKey) return;
  event.preventDefault();
  selectTab(TABS[next], true);
}

// The setup of the source shown: hidden until it has a table.
function renderSetup() {
  const s = setup();
  $('setup').hidden = !s.table;
  loadOptions();
  if (!s.table) return;
  renderColumns();
  syncRoles();
  selectTab(state.tab);
}

const short = text => text.length > 16 ? `${text.slice(0, 15)}…` : text;
// A column name with line break opportunities after its punctuation (a3.Retinyl.|palmitate).
const breakable = name => name.split(/(?<=[._\-/])/).flatMap((part, k) => k ? [h('wbr'), part] : [part]);
// File values in notes, each isolated from the text around it.
const quoted = values => values.map((v, k) => [k ? ', ' : '', '“', h('bdi', {}, short(v)), '”']);
// Why values with commas made a column text (data.js commaRule).
const COMMAS = {delimiter: 'a comma-separated file has no decimal commas', point: 'decimal commas and points mixed',
                thousands: 'thousands separators',
                ambiguous: 'commas that may be thousands separators or decimal commas'};

// The type, the missing values with the tokens counted as missing (blank for an
// empty field), examples (with the number read where it differs from the text),
// and for text columns why they are text and how they would be coded.
function columnNotes(c) {
  const absent = c.missing+c.nonfinite, beyond = c.beyond ? ` (${c.beyond} beyond ±3.4e38)` : '';
  const tokens = c.tokens.length ? ` (${c.tokens.map(([token]) => token === '' ? 'blank' : token).join(', ')})` : '';
  const missing = !absent ? 'none missing' : c.kind === 'mostly missing' ? `${absent} of ${setup().table.rows.length} missing${tokens}` :
    [c.missing ? `${c.missing} missing${tokens}` : '', c.nonfinite ? `${c.nonfinite} non-finite${beyond}` : ''].filter(Boolean).join(', ');
  const kind = c.kind === 'text' ? `text, ${plural(c.levels.length, 'level')}` :
    c.kind === 'mostly missing' ? `mostly missing (${c.numeric ? 'numeric' : 'text'})` :
    c.codes ? `numeric, ${c.distinct} integer codes` : c.decimalComma ? `${c.kind}, decimal commas` : c.kind;
  const notes = [h('span', {class: c.kind === 'mostly missing' ? 'col-meta warn' : 'col-meta'}, `${kind} · ${missing}`)];
  if (c.examples.length) notes.push(h('span', {class: 'col-examples'}, 'e.g. ', c.examples.map((e, k) =>
    [k ? ', ' : '', h('bdi', {}, short(e)), c.exampleValues[k] === null ? '' : ` → ${c.exampleValues[k]}`])));
  if (c.notNumbers?.length) notes.push(h('span', {class: 'col-text warn'}, 'Not numbers: ', quoted(c.notNumbers),
    c.commas ? ` (${COMMAS[c.commas]})` : ''));
  if (!c.numeric && c.present) notes.push(h('span', {class: 'col-text'}, c.levels.length > MAX_LEVELS ?
    `Too many levels to use; text cannot be X` :
    ['Text: Z only, as ', plural(c.levels.length-1, 'indicator'), ' vs ', quoted([c.levels[0]])]));
  return notes;
}

function renderColumns() {
  const rows = setup().columns.map((c, j) => h('tr', {'data-column': c.name, 'data-index': j},
    h('th', {scope: 'row'}, h('span', {class: 'col-name', id: `col-${j}`}, breakable(c.name)),
      h('div', {class: 'col-detail'}, columnNotes(c),
        h('span', {class: 'col-order', title: 'Position in the index X′β; the first X has a positive weight'}))),
    ROLES.map(role => h('td', {class: `role role-${role}`},
      h('label', {}, h('input', {type: 'radio', name: `role-${j}`, value: role, 'aria-labelledby': `col-${j} head-${role}`}))))));
  $('column-table').tBodies[0].replaceChildren(...rows);
  $('column-filter').value = '';
  applyFilter();
}

// Checks each row's role and disables the roles its column cannot take; shows
// each X's position; then the X, Z and outcome tabs.
function syncRoles() {
  const s = setup(), family = outcomeType(), {exposures} = selection(s.roles, s.columns);
  for (const row of $('column-table').tBodies[0].rows) {
    const c = s.columns[Number(row.dataset.index)], role = roleOf(s.roles, c.name);
    row.dataset.role = role;
    for (const input of row.querySelectorAll('input[type=radio]')) {
      const blocked = blockedRole(c, input.value, family);
      input.checked = input.value === role;
      input.disabled = Boolean(blocked);
      input.title = input.parentElement.title = displayText(blocked || '');
    }
    const at = exposures.indexOf(c.name);
    row.querySelector('.col-order').textContent = at < 0 ? '' : `X${at+1}`;
  }
  renderTerms();
  renderOutcome();
}

// Values of zero or below in a column, which a log drops.
const nonpositive = c => c.values.reduce((k, v) => k+(v <= 0), 0);

// The X tab lists the exposures in model order, each with its log box; the Z
// tab lists the confounders, each numeric one with its categorical box. Focus
// stays on the box it was on.
function renderTerms() {
  const s = setup(), {exposures, confounders} = selection(s.roles, s.columns);
  const byName = new Map(s.columns.map(c => [c.name, c])), focused = document.activeElement?.name;
  const levels = new Map(setupPlan(s).terms.filter(t => t.levels !== undefined).map(t => [t.name, t]));
  $('x-list').replaceChildren(...exposures.map((name, k) => {
    const c = byName.get(name), low = nonpositive(c);
    return h('li', {class: 'term', 'data-column': name}, h('span', {class: 'term-order'}, `X${k+1}`),
      h('span', {class: 'term-name'}, h('span', {class: 'col-name'}, breakable(name)),
        low ? h('span', {class: 'col-meta warn'}, `${plural(low, 'value')} ≤ 0, dropped when logged`) : null),
      h('label', {class: 'opt'}, h('input', {type: 'checkbox', name: `log-${c.index}`, checked: s.logged.has(name),
                                             'aria-label': `Log-transform ${name}`}), 'log'));
  }));
  // Columns with too many values to dummy-code do not offer the categorical box.
  $('z-list').replaceChildren(...confounders.map((name, k) => {
    const c = byName.get(name), blocked = c.numeric && blockedCategorical(c), coded = levels.get(name);
    const note = coded ? `${plural(coded.count, 'indicator')} vs ${coded.reference}` : c.numeric ? 'numeric, enters as it is' : '';
    return h('li', {class: 'term', 'data-column': name}, h('span', {class: 'term-order'}, `Z${k+1}`),
      h('span', {class: 'term-name'}, h('span', {class: 'col-name'}, breakable(name)),
        h('span', {class: 'col-meta'}, c.numeric ? note : `text: ${note}`)),
      c.numeric && !blocked ? h('label', {class: 'opt'}, h('input', {type: 'checkbox', name: `cat-${c.index}`,
          checked: s.categorical.has(name),
          'aria-label': `Treat ${name} as categorical${c.codes ? ` (suggested: ${c.distinct} integer codes)` : ''}`}),
        'categorical', c.codes ? h('span', {class: 'suggest'}, '(suggested)') : null) : null);
  }));
  if (/^(log|cat)-\d+$/.test(focused)) form.elements[focused]?.focus();
  $('x-empty').hidden = exposures.length > 0;
  $('z-empty').hidden = confounders.length > 0;
  for (const [id, label, k] of [['x', 'Chemical exposures (X)', exposures.length], ['z', 'Confounders (Z)', confounders.length]]) {
    $(`count-${id}`).textContent = k;
    $(`tab-${id}`).setAttribute('aria-label', `${label}: ${plural(k, 'column')}`);
  }
  syncLogAll();
}

// Log-transform is checked when every X is logged, mixed when some are.
function syncLogAll() {
  const s = setup(), {exposures} = selection(s.roles, s.columns), logged = exposures.filter(name => s.logged.has(name)).length;
  const box = $('log-all');
  box.disabled = !exposures.length;
  box.checked = exposures.length > 0 && logged === exposures.length;
  box.indeterminate = logged > 0 && logged < exposures.length;
}

// The outcome tab names the outcome (and event) columns; only a continuous
// outcome has transforms.
function renderOutcome() {
  const s = setup(), {outcome, event} = selection(s.roles, s.columns), type = outcomeType();
  setText($('outcome-title'), type === 'cox' ? 'Survival time and event (Cox)' : `Outcome (${type})`);
  const chosen = type === 'cox' ? `Time: ${outcome ?? 'not chosen'} · event: ${event ?? 'not chosen'}` :
    outcome ?? 'Not chosen: give a column the Outcome role under Columns.';
  setText($('outcome-chosen'), chosen);
  const fixed = type === 'binary' ? 'A binary outcome is used as it is coded, 0 or 1.' :
    type === 'cox' ? 'Survival times are used as recorded; the event column is 1 for an event and 0 for censoring.' : '';
  setText($('outcome-fixed'), fixed);
  $('outcome-fixed').hidden = !fixed;
}

const shownColumns = () => [...$('column-table').tBodies[0].rows].filter(row => !row.hidden).map(row => row.dataset.column);

function applyFilter() {
  const query = $('column-filter').value.trim().toLowerCase(), rows = [...$('column-table').tBodies[0].rows];
  for (const row of rows) row.hidden = Boolean(query) && !row.dataset.column.toLowerCase().includes(query);
  const shown = rows.filter(row => !row.hidden).length;
  setText($('filter-count'), !query ? `${plural(rows.length, 'column')} in the ${source() === 'csv' ? 'file' : 'data'}.` :
    shown ? `Showing ${shown} of ${plural(rows.length, 'column')}; All and None apply to these.` :
    `No column name contains “${query}”.`);
}

// Until the visitor picks the outcome type, it follows the outcome: binary for 0/1.
function followOutcome() {
  const s = setup();
  if (!s.familyAuto || outcomeType() === 'cox') return;
  const {outcome} = selection(s.roles, s.columns);
  s.options.outcomeType = field('outcomeType').value = suggestFamily(s.columns.find(c => c.name === outcome));
}

const ROLE_NAMES = {exposure: 'chemical exposure', confounder: 'confounder (Z)', outcome: 'outcome', event: 'event (0/1)',
                    ignore: 'ignored'};

function onRole(input) {
  const s = setup(), c = s.columns[Number(input.name.slice(5))], before = s.roles, family = outcomeType();
  try {
    s.roles = assignRole(s.roles, s.columns, c.name, input.value, family);
  } catch (error) {
    message(error.message);
  }
  followOutcome();
  syncRoles();
  onControls();
  // Said aloud: the new role, a column that lost the outcome or event role to it,
  // and an outcome type that followed the outcome.
  const {exposures} = selection(s.roles, s.columns), cox = outcomeType() === 'cox';
  const label = (name, role) => role === 'exposure' ? `chemical exposure X${exposures.indexOf(name)+1}` :
    role === 'outcome' && cox ? 'survival time' : ROLE_NAMES[role];
  const lost = [...before].filter(([name, role]) => name !== c.name && role !== 'ignore' && roleOf(s.roles, name) === 'ignore')
    .map(([name, role]) => `${name}, the ${role === 'outcome' && cox ? 'survival time' : ROLE_NAMES[role]} before, is now ignored`);
  const type = outcomeType() !== family ? [`Outcome type: ${field('outcomeType').selectedOptions[0].textContent}`] : [];
  announce([`${c.name}: ${label(c.name, roleOf(s.roles, c.name))}`, ...lost, ...type].join('. ')+'.');
}

// Up and down arrows move to the same role in the row above or below, changing
// nothing (or to that row's role, when it cannot take this one); left and right
// arrows and Space change the role, as in any radio group.
function onRoleKey(event) {
  const input = event.target;
  if (input.type !== 'radio' || !input.name.startsWith('role-') || (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') ||
      event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  event.preventDefault();
  const rows = [...$('column-table').tBodies[0].rows].filter(row => !row.hidden);
  const row = rows[rows.indexOf(input.closest('tr'))+(event.key === 'ArrowDown' ? 1 : -1)];
  if (!row) return;
  const radios = [...row.querySelectorAll('input[type=radio]')];
  (radios.find(e => e.value === input.value && !e.disabled) ?? radios.find(e => e.checked)).focus();
}

function onOutcomeType() {
  const s = setup();
  s.familyAuto = false;
  s.options.outcomeType = field('outcomeType').value;
  s.roles = changeFamily(s.roles, s.columns, outcomeType());
  syncRoles();
  onControls();
}

// Standardize X and the outcome's log and standardize boxes.
function onOptionField(input) {
  setup().options[input.name] = input.checked;
  onControls();
}

function onBulk(button) {
  const s = setup();
  s.roles = setRoleForMany(s.roles, s.columns, shownColumns(), button.dataset.bulk, button.dataset.on === '1', outcomeType());
  syncRoles();
  onControls();
}

// Log-transform: every chemical exposure, or none.
function onLogAll(box) {
  const s = setup();
  for (const name of selection(s.roles, s.columns).exposures) s.logged[box.checked ? 'add' : 'delete'](name);
  renderTerms();
  onControls();
}

// An X's log box or a Z's categorical box.
function onOption(input) {
  const [kind, j] = input.name.split('-'), s = setup(), name = s.columns[Number(j)].name;
  const chosen = kind === 'log' ? s.logged : s.categorical;
  chosen[input.checked ? 'add' : 'delete'](name);
  if (kind === 'cat') renderTerms(); else syncLogAll();
  onControls();
}

// Only the latest choice counts: a file still being read when another is chosen
// (or the choice is cleared) is dropped.
let fileToken = 0;
async function onFile() {
  const file = field('file').files[0], token = ++fileToken, s = state.setups.csv;
  if (!file) return;
  try {
    const text = await file.text();
    if (token !== fileToken) return;
    const table = parseCsv(text);
    if (!table.rows.length) throw new Error('the file has a header but no data rows');
    loadTable(s, table);
    s.fileName = file.name;
  } catch (error) {
    if (token !== fileToken) return;
    Object.assign(s, {table: null, columns: null, roles: null, fileName: ''});
    message(`${file.name}: ${error.message}`);
    if (source() === 'csv') renderSetup();
    return onControls();
  }
  if (source() === 'csv') renderSetup();
  message(s.columns.some(c => c.numeric && c.present) ? null : `${file.name} has no numeric columns`);
  onControls();
}

// ---- Running

function message(text, kind = 'error') {
  const box = $('message');
  box.hidden = !text;
  setText(box, text || '');
  box.classList.toggle('info', kind === 'info');
}

// The progress text of a step without a count: the text and the time it has run.
let timer = null;
function clock(text) {
  clearInterval(timer);
  $('progress').querySelector('progress').removeAttribute('value');
  $('progress-text').textContent = text;
  if (!text) return;
  const started = performance.now();
  timer = setInterval(() => {
    $('progress-text').textContent = `${text} ${duration((performance.now()-started)/1000)}`;
  }, 250);
}

// While a run goes, the progress row shows with Cancel. Focus moves from the
// button that started the run (`from`) to Cancel, and when the run ends, from
// Cancel to the Results heading if results were drawn, else back to `from`. The
// progress text is not a live region: announce() says the coarse steps.
function busy(on, {from = null, rendered = false} = {}) {
  const cancel = $('cancel'), active = document.activeElement;
  state.busy = on;
  if (on) {
    $('progress').hidden = false;
    if (!active || active === document.body || active === from) cancel.focus();
  } else clock('');
  onControls();
  if (!on) {
    // Before Cancel hides, which would drop focus to the page.
    if (active === cancel) [rendered && $('results-title'), from, $('results-title')].find(e => e && !e.disabled)?.focus();
    $('progress').hidden = true;
  }
}

// Fit with bootstrap (`bootstrap` true) fits, then bootstraps the fit; when the
// controls still match the fit shown, it bootstraps that fit. Single fit only fits.
async function run(bootstrap) {
  let data, s, B;
  try {
    data = readData(); s = settings();
    const problem = networkProblem(data.p, data.q, s);
    if (problem) throw new Error(problem);
    if (bootstrap) B = integer('B', 2, 1000, 'B');
  } catch (error) {
    return message(error.message);
  }
  message(null);
  const key = controlsKey(), from = $(bootstrap ? 'run' : 'fit');
  let step = bootstrap && state.fit?.key === key ? 'bootstrap' : 'fit', rendered = false;
  busy(true, {from});
  try {
    let done = '';
    if (step === 'fit') {
      const seconds = await fitStep(data, s, key, bootstrap);
      rendered = true;
      done = `Fit done in ${duration(seconds)}.`;
    }
    if (bootstrap) {
      step = 'bootstrap';
      await bootstrapStep(B, done);
      rendered = true;
    } else announce(`${done} The results are below.`);
  } catch (error) {
    // A failed replicate invalidates the whole bootstrap, as in the native engine.
    if (step === 'bootstrap') pool.cancel();
    const cancelled = error instanceof Cancelled;
    message(cancelled ? `${step === 'fit' ? 'Fit' : 'Bootstrap'} cancelled.` : error.message, cancelled ? 'info' : 'error');
  } finally {
    if (state.booting) {
      state.booting = false;
      render();
    }
    busy(false, {from, rendered});
  }
}

// Fits NeuralPLSI (and SplinePLSI) on all rows and draws the results; returns
// the seconds it took. With a bootstrap to follow, the tables say it is running.
async function fitStep(data, s, key, bootstrap) {
  // The file is named as it was when the fit started: another may be open by the time it ends.
  const fileName = data.source === 'csv' ? setup().fileName : null;
  const text = s.spline ? 'Fitting NeuralPLSI and SplinePLSI…' : 'Fitting NeuralPLSI…', started = performance.now();
  clock(text);
  announce(text);
  const workers = await pool.ensure(s.spline ? 2 : 1);
  const [neural, spline] = await Promise.all([pool.call(workers[0], 'fitNeural', [data, s, s.seed]),
    s.spline ? pool.call(workers[1], 'fitSpline', [data, s.seed]) : null]);
  state.fit = {data, settings: s, key, fileName, neural, spline};
  state.boot = null;
  state.booting = bootstrap;
  render();
  return (performance.now()-started)/1000;
}

// B bootstrap refits of the fit shown, in parallel on the pool's workers. `done`
// (the fit's announcement) comes before the bootstrap's.
async function bootstrapStep(B, done) {
  const f = state.fit, {data, settings: s} = f;
  const jobs = {neural: {state: f.neural.state, x: Float32Array.from(data.x), z: Float32Array.from(data.z),
                         y: Float32Array.from(data.y), n: data.n, grid: Float32Array.from(f.neural.grid), seed: s.seed}};
  if (f.spline) jobs.spline = {x: data.x, z: data.z, y: data.y, n: data.n, p: data.p, q: data.q, family: data.family,
                               grid: Float64Array.from(f.spline.grid), seed: s.seed};
  const kinds = Object.keys(jobs), tasks = [], rows = {neural: [], spline: []};
  for (let b = 0; b < B; ++b) for (const kind of kinds) tasks.push({kind, b});
  // The tables say the bootstrap is running, unless they show an earlier one.
  state.booting = true;
  if (!state.boot) render();
  clock('');
  announce(`${done ? `${done} ` : ''}Bootstrap started: ${tasks.length} refits.`);
  const bar = $('progress').querySelector('progress'), started = performance.now();
  let count = 0, frame = 0, quarter = 0;
  const show = () => {
    frame = 0;
    const elapsed = (performance.now()-started)/1000, left = count ? elapsed/count*(tasks.length-count) : NaN;
    bar.value = count/tasks.length;
    $('progress-text').textContent = `${count} of ${tasks.length} refits · ${duration(elapsed)} elapsed` +
      (count ? ` · about ${duration(left)} left` : '')+` · ${pool.workers.length} workers`;
  };
  show();
  try {
    await pool.run(jobs, tasks, (task, row) => {
      rows[task.kind][task.b] = row;
      ++count;
      frame ||= requestAnimationFrame(show);
      // Said aloud at 25, 50 and 75%.
      if (Math.floor(4*count/tasks.length) > quarter && count < tasks.length)
        announce(`Bootstrap ${25*(quarter = Math.floor(4*count/tasks.length))}% done.`);
    });
    const boot = {B, seconds: (performance.now()-started)/1000, workers: pool.workers.length};
    for (const kind of kinds) {
      const D = rows[kind][0].length, draws = mergeRows(rows[kind], D);
      // Cancel empties the pool; it may have done so since the last replicate.
      if (!pool.workers.length) throw new Cancelled();
      boot[kind] = {draws, D, summary: await pool.call(pool.workers[0], 'summary', [draws.slice(), B, D])};
    }
    state.boot = boot;
    state.booting = false;
    render();
    announce(`Bootstrap done: B = ${B} in ${duration(boot.seconds)}. Standard errors and intervals are in the results.`);
  } finally {
    if (frame) cancelAnimationFrame(frame);
  }
}

// ---- Results

// Summary rows of a model's bootstrap: mean, SD, lower, upper over columns [from, from+count).
function interval(kind, from, count) {
  const b = state.boot?.[kind];
  if (!b) return null;
  const row = r => Array.from(b.summary.subarray(r*b.D+from, r*b.D+from+count));
  return {se: row(1), lo: row(2), hi: row(3)};
}

function gColumns(kind) {
  const f = state.fit[kind];
  return [state.boot?.[kind] ? state.boot[kind].D-f.grid.length : 0, f.grid.length];
}

function coefficientTable(title, names, block, truth) {
  const f = state.fit, models = MODELS.filter(([k]) => f[k]), p = f.data.p;
  const head1 = h('tr', {class: 'group'}, h('th'), truth ? h('th') : null,
    models.map(([, name]) => h('th', {colspan: 4, style: 'text-align:center'}, name)));
  const head2 = h('tr', {}, h('th', {}, cellName(block === 'beta' ? 'Chemical exposure (X)' : 'Confounder (Z)')),
    truth ? h('th', {}, 'Truth') : null,
    models.map(() => ['Estimate', 'SE', '2.5%', '97.5%'].map(t => h('th', {}, t))));
  const cells = (kind, i) => {
    const ci = interval(kind, (block === 'beta' ? 0 : p)+i, 1);
    return estimateCells((block === 'beta' ? f[kind].beta : f[kind].gamma)[i], ci);
  };
  const body = names.map((label, i) => h('tr', {}, h('td', {}, cellName(label)), truth ? h('td', {}, fmt(truth[i])) : null,
    models.map(([kind]) => cells(kind, i))));
  if (block === 'gamma' && f.settings.intercept)
    body.push(h('tr', {}, h('td', {}, cellName('Intercept')), truth ? h('td', {}, '0') : null,
      models.map(([kind]) => kind === 'neural' ? estimateCells(f.neural.intercept, interval(kind, p+f.data.q, 1))
                                               : h('td', {class: 'na', colspan: 4}, 'part of g'))));
  return h('div', {class: 'card'}, h('h3', {}, title),
    h('div', {class: 'table-wrap', role: 'region', 'aria-label': title}, h('table', {}, h('thead', {}, head1, head2), h('tbody', {}, body))));
}

// The first cell of a result row: long names wrap inside a bounded width, so a
// sticky first column never covers the numbers on a phone.
const cellName = text => h('span', {class: 'cell-name'}, breakable(text));

function estimateCells(estimate, ci) {
  return [h('td', {}, fmt(estimate)), ci ? [ci.se[0], ci.lo[0], ci.hi[0]].map(v => h('td', {}, fmt(v)))
                                         : h('td', {class: 'na', colspan: 3}, state.booting ? 'bootstrap running…' : 'no bootstrap')];
}

function diagnostics() {
  const f = state.fit, d = f.data, models = MODELS.filter(([k]) => f[k]), truth = d.truth;
  const rows = [['Fit time', m => duration(m.seconds)],
    ['Iterations', m => m.model === 'neural' ? `${m.diagnostics.iterations} L-BFGS` :
      `${m.diagnostics.iterations} BFGS${m.diagnostics.converged ? '' : ', not converged'}`],
    ['Objective', m => `${fmt(m.diagnostics.objective, 4)} ${m.model === 'neural' ? 'penalized' : 'profile'}`]];
  if (d.family === 'continuous') rows.push(['R² (training)', m => fmt(m.metrics.r2)], ['RMSE (training)', m => fmt(m.metrics.rmse)]);
  if (d.family === 'binary') rows.push(['AUC (training)', m => fmt(m.metrics.auc)], ['Log loss (training)', m => fmt(m.metrics.logLoss)]);
  if (d.family === 'cox') rows.push(['C-index (training)', m => fmt(m.metrics.concordance)]);
  if (truth) {
    const link = LINKS[truth.g];
    rows.push(['Angle to true β', m => {
      const c = Math.abs(m.beta.reduce((s, v, j) => s+v*truth.beta[j], 0));
      return `${fmt(Math.acos(Math.min(1, c))*180/Math.PI)}°`;
    }], ['RMSE of g vs truth', m => fmt(Math.sqrt(m.g.reduce((s, v, i) => s+(v-link(m.grid[i]))**2, 0)/m.g.length))]);
  }
  rows.push(['Bootstrap', () => state.boot ? `B = ${state.boot.B}, ${duration(state.boot.seconds)}, ${state.boot.workers} workers` :
    state.booting ? 'running…' : 'none (single fit)']);
  return h('div', {class: 'card'}, h('h3', {}, 'Fit diagnostics'), h('div', {class: 'table-wrap', role: 'region', 'aria-label': 'Fit diagnostics'}, h('table', {},
    h('thead', {}, h('tr', {}, h('th'), models.map(([, name]) => h('th', {}, name)))),
    h('tbody', {}, rows.map(([label, value]) => h('tr', {}, h('td', {}, cellName(label)), models.map(([k]) => h('td', {}, value(f[k])))))))));
}

function download(name, text, type = 'text/csv') {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = h('a', {href: url, download: name});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function estimatesCsv() {
  const f = state.fit, d = f.data, header = ['model', 'parameter', 'estimate', 'se', 'ci_lower', 'ci_upper'];
  if (d.truth) header.push('truth');
  const rows = [];
  for (const [kind, name] of MODELS.filter(([k]) => f[k])) {
    const m = f[kind], params = [...d.names.x.map((n, j) => [`beta[${n}]`, m.beta[j], d.truth?.beta[j]]),
      ...d.names.z.map((n, j) => [`gamma[${n}]`, m.gamma[j], d.truth?.gamma[j]])];
    if (kind === 'neural' && m.intercept !== null) params.push(['intercept', m.intercept, 0]);
    const ci = interval(kind, 0, params.length);
    params.forEach(([label, estimate, truth], i) => {
      const row = [name, label, estimate, ci?.se[i], ci?.lo[i], ci?.hi[i]];
      if (d.truth) row.push(truth);
      rows.push(row);
    });
  }
  return toCsv(header, rows);
}

function drawsCsv(kind) {
  const f = state.fit, d = f.data, b = state.boot[kind], m = f[kind];
  const header = ['replicate', ...d.names.x.map(n => `beta[${n}]`), ...d.names.z.map(n => `gamma[${n}]`)];
  if (kind === 'neural' && m.intercept !== null) header.push('intercept');
  if (kind === 'spline') m.spline.forEach((_, k) => header.push(`spline[${k+1}]`));
  m.grid.forEach(u => header.push(`g(${Number(u.toPrecision(6))})`));
  const rows = [];
  for (let r = 0; r < b.draws.length/b.D; ++r) rows.push([r, ...b.draws.subarray(r*b.D, (r+1)*b.D)]);
  return toCsv(header, rows);
}

// How each column of the fitted data was made from the file, for using the
// model on new data.
function preprocessingJson() {
  const f = state.fit, d = f.data;
  return JSON.stringify({
    description: 'How the columns of the NeuralPLSI fit were made from the data file, in model order. The model ' +
      'takes inputs in this form. x: the column, its natural log when log is true, then (value - mean) / sd when ' +
      'mean and sd are given. z: the column, or an indicator that is 1 where the column equals `equals` and 0 ' +
      'elsewhere (reference: the level without one). y likewise. Means and SDs (population SD) are over the rows ' +
      'used. Rows with a missing or non-finite value, or a value of zero or below in a logged column, were dropped.',
    source: d.source, file: d.source === 'nhanes' ? '12940_2020_644_MOESM2_ESM.csv' : f.fileName, family: d.family,
    intercept: f.settings.intercept, rows: {file: d.rows, used: d.n, dropped: d.dropped},
    x: d.transforms.x, z: d.transforms.z, y: d.transforms.y}, null, 2)+'\n';
}

function render() {
  const results = $('results'), f = state.fit;
  if (!f) return;
  const d = f.data, models = MODELS.filter(([k]) => f[k]);
  const label = {simulated: `Simulated, g = ${d.simulation?.g}`, nhanes: 'NHANES', csv: f.fileName}[d.source];
  // The heading takes focus when a run that drew results ends (busy).
  const head = h('div', {class: 'card'}, h('div', {class: 'results-head'}, h('h2', {id: 'results-title', tabindex: '-1'}, 'Results'),
      h('span', {class: 'facts'}, h('bdi', {}, label), ` · ${d.family}`)),
    h('p', {class: 'facts'}, describe(d)),
    h('p', {class: 'stale', hidden: true}, 'The controls changed after this fit. Fit again to use them.'));
  // g(u) with bands, the truth and the index distribution.
  const gCard = h('div', {class: 'card'}, h('h3', {}, 'Link function g(u)'),
    h('div', {class: 'legend'}, models.map(([, name, color]) => h('span', {}, h('i', {class: 'key', style: `border-color:${color}`}), name)),
      state.boot ? h('span', {}, h('i', {class: 'key band', style: `background:${models[0][2]}`}), '95% bootstrap band') : null,
      d.truth ? h('span', {}, h('i', {class: 'key dashed'}), 'Truth') : null),
    h('div', {class: 'chart', id: 'link-chart'}));
  const fCard = h('div', {class: 'card'}, h('h3', {}, 'Chemical exposure weights β'),
    h('div', {class: 'legend'}, models.map(([, name, color]) => h('span', {}, h('i', {class: 'swatch', style: `background:${color}`}), name)),
      state.boot ? h('span', {}, 'lines: 95% intervals') : null, d.truth ? h('span', {}, h('i', {class: 'key diamond'}), 'Truth') : null),
    h('div', {class: 'chart', id: 'forest-chart'}));
  const downloads = h('div', {class: 'card'}, h('h3', {}, 'Downloads'), h('div', {class: 'downloads'},
    h('button', {type: 'button', class: 'small', onclick: () => download('estimates.csv', estimatesCsv())}, 'Estimates (CSV)'),
    models.filter(([k]) => state.boot?.[k]).map(([k, name]) => h('button', {type: 'button', class: 'small',
      onclick: () => download(`${k === 'neural' ? 'neuralplsi' : 'splineplsi'}-bootstrap-draws.csv`, drawsCsv(k))}, `${name} draws (CSV)`)),
    h('button', {type: 'button', class: 'small', onclick: () => download('neuralplsi.nplsi', f.neural.state, 'text/plain')},
      'NeuralPLSI model (.nplsi)'),
    d.source !== 'simulated' ? h('button', {type: 'button', class: 'small',
      onclick: () => download('neuralplsi-preprocessing.json', preprocessingJson(), 'application/json')}, 'Preprocessing (JSON)') : null),
    h('p', {class: 'help'}, 'The model file loads with NeuralPLSI.load(path) in the Python package, load_model() in R ' +
      'and nplsi_load in C++.' +
      (d.source !== 'simulated' ? ' It takes inputs as they were fitted: transform new data as the preprocessing file says ' +
        '(column order, logs, means and SDs, indicators) before predicting.' : '') +
      ' Estimates are the fit on all rows; SEs and intervals come from the bootstrap.'));
  results.replaceChildren(head, h('div', {class: 'figures'}, gCard, fCard),
    h('div', {class: 'tables'}, coefficientTable('Chemical exposure weights β', d.names.x, 'beta', d.truth?.beta),
      coefficientTable('Confounder effects γ', d.names.z, 'gamma', d.truth?.gamma)),
    diagnostics(), downloads);
  drawCharts();
  markScrollable();
  onControls();
}

function drawCharts() {
  const f = state.fit;
  if (!f) return;
  const d = f.data, models = MODELS.filter(([k]) => f[k]);
  const series = models.map(([kind, name, color]) => {
    const m = f[kind], ci = state.boot?.[kind] ? interval(kind, ...gColumns(kind)) : null;
    return {name, color, x: m.grid, y: m.g, lo: ci?.lo, hi: ci?.hi};
  });
  let truth = null;
  if (d.truth) {
    const lo = Math.min(...series.map(s => s.x[0])), hi = Math.max(...series.map(s => s.x.at(-1)));
    const x = Array.from(linspace(lo, hi, 200));
    truth = {x, y: x.map(LINKS[d.truth.g])};
  }
  linkChart($('link-chart'), {series, truth, index: f.neural.index, indexLabel: 'u = Xβ̂; bars: distribution of the fitted NeuralPLSI index'});
  forestChart($('forest-chart'), {rows: d.names.x.map((name, j) => ({name, truth: d.truth?.beta[j]})),
    series: models.map(([kind, name, color]) => {
      const ci = interval(kind, 0, d.p);
      return {name, color, estimate: f[kind].beta, lo: ci?.lo, hi: ci?.hi};
    })});
}

// Tables wider than their card scroll sideways inside it. Those that do get a
// tab stop (to scroll with the keyboard), a sticky first column and edge shadows.
function markScrollable() {
  for (const wrap of document.querySelectorAll('.results .table-wrap')) {
    const scrolls = wrap.scrollWidth > wrap.clientWidth+1;
    wrap.classList.toggle('scrolls', scrolls);
    if (scrolls) wrap.tabIndex = 0; else wrap.removeAttribute('tabindex');
    wrap.onscroll ??= () => wrap.classList.toggle('scrolled', wrap.scrollLeft > 0);
  }
}

// ---- Wiring

const ROW_OPTION = /^(log|cat)-\d+$/, SIMULATION = new Set(['n', 'simSeed', 'g', 'outcome']);
const OPTION_BOXES = new Set(['standardize', 'logOutcome', 'standardizeOutcome']);
// Controls with their own handler on input; change events of the others refresh the page.
const handled = target => target.id === 'column-filter' || target.id === 'log-all' || target.name === 'outcomeType' ||
  target.name?.startsWith('role-') || ROW_OPTION.test(target.name) || SIMULATION.has(target.name) || OPTION_BOXES.has(target.name);
form.addEventListener('input', event => {
  const target = event.target;
  if (target.id === 'column-filter') applyFilter();
  else if (target.id === 'log-all') onLogAll(target);
  else if (target.name === 'outcomeType') onOutcomeType();
  else if (target.name?.startsWith('role-')) onRole(target);
  else if (ROW_OPTION.test(target.name)) onOption(target);
  else if (SIMULATION.has(target.name)) onSimulation();
  else if (OPTION_BOXES.has(target.name)) onOptionField(target);
  else if (target.name !== 'source' && target.name !== 'file') onControls();
});
form.addEventListener('change', event => {
  const target = event.target;
  if (target.name === 'source') onSource();
  else if (target.name === 'file') onFile();
  else if (!handled(target)) onControls();
});
form.addEventListener('submit', event => event.preventDefault());
form.addEventListener('click', event => {
  const bulk = event.target.closest('[data-bulk]'), tab = event.target.closest('[role=tab]');
  if (bulk) onBulk(bulk);
  if (tab) selectTab(tab.id.slice(4));
});
document.querySelector('[role=tablist]').addEventListener('keydown', onTabKey);
$('nhanes-retry').addEventListener('click', loadNhanes);
$('run').addEventListener('click', () => run(true));
$('fit').addEventListener('click', () => run(false));
$('cancel').addEventListener('click', () => pool.cancel());
$('column-table').addEventListener('keydown', onRoleKey);
let resize = 0;
addEventListener('resize', () => { clearTimeout(resize); resize = setTimeout(() => { drawCharts(); markScrollable(); }, 150); });
onSource();
if ('serviceWorker' in navigator && location.protocol !== 'file:')
  navigator.serviceWorker.register('sw.js').catch(() => {});
// For tests and the browser console.
globalThis.nplsiDemo = {state, pool, setup};
