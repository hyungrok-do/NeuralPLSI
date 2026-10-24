// Column roles for a CSV file: which columns are the chemical exposures (X), the
// confounders (Z), the outcome and the Cox event indicator, and whether that
// choice can be fitted. Pure functions over describeColumns() output; roles are
// a Map from column name to role, and a column without an entry is ignored. The
// Map keeps the order in which columns got their role, which is the order of the
// X and Z columns in the model: β has a positive weight on the first X.
import {MAX_LEVELS, completeRows, confounderLevels, prepareTable} from './data.js';

export const ROLES = ['exposure', 'confounder', 'outcome', 'event', 'ignore'];
export const FAMILY_LABELS = {continuous: 'continuous', binary: 'binary (logistic)', cox: 'survival (Cox)'};
const EVENT_NAME = /^(event|status|dead|death|died|delta)$/i;

// Why a column cannot take a role under a family, or null when it can.
export function blockedRole(column, role, family) {
  if (role === 'ignore') return null;
  if (!ROLES.includes(role)) return `Unknown role ${role}`;
  if (role === 'event' && family !== 'cox') return 'Only a survival (Cox) outcome has an event column';
  if (!column.present) return `${column.name} has no values`;
  if (role === 'exposure' && !column.numeric) return 'Text columns cannot be chemical exposures (X)';
  if (role === 'outcome' && !column.numeric) return 'The outcome must be numeric';
  if (role === 'event' && !column.binary) return 'The event column must be coded 0/1';
  if (role === 'confounder' && !column.numeric && column.levels.length > MAX_LEVELS)
    return `Text columns with more than ${MAX_LEVELS} levels cannot be dummy-coded`;
  return null;
}

export const allowedRoles = (column, family) => ROLES.filter(role => !blockedRole(column, role, family));

// Why a confounder cannot be marked categorical, or null when it can. Text
// confounders are always categorical.
export function blockedCategorical(column) {
  if (!column.numeric) return 'Text confounders are always categorical';
  if (column.distinct > MAX_LEVELS) return `More than ${MAX_LEVELS} distinct values: too many levels to dummy-code`;
  return null;
}

// Every column starts ignored.
export const initialRoles = columns => new Map(columns.map(c => [c.name, 'ignore']));
export const roleOf = (roles, name) => roles.get(name) ?? 'ignore';

function find(columns, name) {
  const column = columns.find(c => c.name === name);
  if (!column) throw new Error(`Unknown column ${name}`);
  return column;
}

// Gives one column one role, after the columns that have a role already. There
// is one outcome and one event column: giving either role to a column returns
// the column that had it to ignore, so the event can never be the outcome.
export function assignRole(roles, columns, name, role, family) {
  const blocked = blockedRole(find(columns, name), role, family);
  if (blocked) throw new Error(blocked);
  const next = new Map(roles);
  if (roleOf(next, name) === role) return next;
  if (role === 'outcome' || role === 'event') for (const [k, r] of next) if (r === role) next.set(k, 'ignore');
  next.delete(name);
  next.set(name, role);
  return next;
}

// "Select all" (on) gives the role to every listed column that is ignored and
// can take it, never taking the outcome, the event or the other list's columns;
// "select none" (off) returns the listed columns with that role to ignore.
export function setRoleForMany(roles, columns, names, role, on, family) {
  if (role !== 'exposure' && role !== 'confounder') throw new Error(`Cannot set ${role} for many columns`);
  const next = new Map(roles);
  for (const name of names) {
    const current = roleOf(next, name);
    if (on && current === 'ignore' && !blockedRole(find(columns, name), role, family)) {
      next.delete(name);
      next.set(name, role);
    }
    if (!on && current === role) next.set(name, 'ignore');
  }
  return next;
}

// The roles of a preset setup ({family, outcome, event, exposures, confounders},
// as NHANES_SETUP): the outcome, the Cox event, then the X and Z columns in order.
export function presetRoles(columns, {family, outcome, event, exposures, confounders}) {
  const chosen = [[outcome, 'outcome'], ...(family === 'cox' ? [[event, 'event']] : []),
                  ...exposures.map(name => [name, 'exposure']), ...confounders.map(name => [name, 'confounder'])];
  return chosen.reduce((roles, [name, role]) => assignRole(roles, columns, name, role, family), initialRoles(columns));
}

// The chosen columns, in the order they got their role.
export function selection(roles, columns) {
  const known = new Set(columns.map(c => c.name));
  const pick = role => [...roles].filter(([name, r]) => r === role && known.has(name)).map(([name]) => name);
  return {outcome: pick('outcome')[0] ?? null, event: pick('event')[0] ?? null,
          exposures: pick('exposure'), confounders: pick('confounder')};
}

// The family a column suggests as the outcome: binary for a 0/1 column.
export const suggestFamily = column => column?.binary ? 'binary' : 'continuous';

// Roles after the family changes: only Cox keeps an event column, and switching
// to Cox picks one by its name (event, status, death, ...) among the ignored
// 0/1 columns, so the guess is never the outcome.
export function changeFamily(roles, columns, family) {
  const next = new Map(roles);
  if (family !== 'cox') for (const [k, r] of next) if (r === 'event') next.set(k, 'ignore');
  if (family === 'cox' && ![...next.values()].includes('event')) {
    const guess = columns.find(c => EVENT_NAME.test(c.name) && !blockedRole(c, 'event', 'cox') && roleOf(next, c.name) === 'ignore');
    if (guess) next.set(guess.name, 'event');
  }
  return next;
}

// Everything the page shows about a CSV setup before fitting: the chosen
// columns, p and q (q after dummy coding), the rows kept and dropped by reason
// for the columns chosen so far, and either the prepared data (ready) or the
// first reason the fit cannot run. Options as prepareTable: `log` (true, false
// or the exposures to log), `standardize`, `categorical` (numeric confounders to
// dummy-code), `zLog` and `zStandardize` (numeric confounders to log and to
// z-score) and, for a continuous outcome, `logOutcome` and `standardizeOutcome`.
export function planCsv(table, columns, {family, roles, log = false, standardize = false, categorical = [],
                                         zLog = [], zStandardize = [], logOutcome = false, standardizeOutcome = false}) {
  const byName = new Map(columns.map(c => [c.name, c])), cox = family === 'cox', continuous = family === 'continuous';
  const {outcome, event: chosenEvent, exposures, confounders} = selection(roles, columns);
  const event = cox ? chosenEvent : null, asLevels = new Set(categorical), logSet = new Set(log === true ? exposures : log || []);
  const logged = exposures.filter(name => logSet.has(name)), logY = continuous && logOutcome && outcome !== null;
  // Logs and z-scores apply to numeric confounders that enter as they are.
  const plain = name => byName.get(name).numeric && !asLevels.has(name);
  const zLogged = confounders.filter(name => plain(name) && zLog.includes(name));
  const zStandardized = confounders.filter(name => plain(name) && zStandardize.includes(name));
  const used = [outcome, event, ...exposures, ...confounders].filter(name => name !== null).map(name => byName.get(name));
  const {keep, dropped} = completeRows(table, columns, used,
    [...logged, ...zLogged, ...(logY ? [outcome] : [])].map(name => byName.get(name)));
  const terms = confounders.map(name => {
    const c = byName.get(name);
    if (c.numeric && !asLevels.has(name)) return {name, count: 1};
    const levels = confounderLevels(table, c, keep);
    return {name, count: Math.max(levels.length-1, 0), levels: levels.length, reference: levels[0] ?? null};
  });
  const sum = (name, rows) => rows.reduce((s, r) => s+byName.get(name).values[r], 0);
  const plan = {family, outcome, event, exposures, confounders, p: exposures.length, terms,
                q: terms.reduce((s, t) => s+t.count, 0), rows: table.rows.length, n: keep.length, dropped, logged, standardize,
                zLogged, zStandardized,
                logOutcome: logY, standardizeOutcome: continuous && standardizeOutcome && outcome !== null,
                cases: null, events: null, reason: null, ready: false, data: null};
  const y = outcome === null ? null : byName.get(outcome).values;
  const binaryOutcome = y !== null && keep.every(r => y[r] === 0 || y[r] === 1);
  if (family === 'binary' && binaryOutcome) plan.cases = sum(outcome, keep);
  if (event !== null) plan.events = sum(event, keep);
  if (outcome === null) plan.reason = cox ? 'Choose the survival time column (Time)' : 'Choose the outcome column';
  else if (cox && event === null) plan.reason = 'Choose the event column (0/1)';
  else if (!exposures.length) plan.reason = 'Choose at least one chemical exposure (X)';
  else if (family === 'binary' && !binaryOutcome)
    plan.reason = `The outcome ${outcome} is not coded 0/1; choose another outcome or outcome type`;
  // A logistic fit needs both outcomes; with fewer than two rows prepareTable says so.
  else if (family === 'binary' && keep.length >= 2 && (plan.cases === 0 || plan.cases === keep.length))
    plan.reason = `The outcome ${outcome} has ${plan.cases ? 'no non-cases (all 1)' : 'no cases (all 0)'}` +
      (keep.length < table.rows.length ? ' in the rows used' : '');
  else {
    try {
      plan.data = prepareTable(table, {family, outcome, event, exposures, confounders, log: logged, standardize,
                                       categorical: confounders.filter(name => asLevels.has(name)), zLog: zLogged,
                                       zStandardize: zStandardized,
                                       logOutcome: plan.logOutcome, standardizeOutcome: plan.standardizeOutcome}, columns);
      plan.ready = true;
    } catch (error) {
      plan.reason = error.message;
    }
  }
  return plan;
}

const plural = (k, word, many = `${word}s`) => `${k} ${k === 1 ? word : many}`;
const DROP_LABELS = {missing: 'missing', nonfinite: 'non-finite', nonpositive: 'not positive (log)'};

// The rows of the summary above the Fit button, as [label, text] pairs, for a
// plan or prepared data with {family, outcomeLabel, cases, events, p, xNote, q,
// zNote, rows, n, dropped}.
export function summaryRows(s) {
  const outcome = s.outcomeLabel === null ? 'not chosen' :
    `${s.outcomeLabel} · ${FAMILY_LABELS[s.family]}` +
    (s.cases !== null && s.cases !== undefined ? `, ${plural(s.cases, 'case')}` : '') +
    (s.events !== null && s.events !== undefined ? `, ${plural(s.events, 'event')}` : '');
  const dropped = Object.entries(s.dropped || {}).filter(([, k]) => k).map(([why, k]) => `${k} ${DROP_LABELS[why]}`);
  const rows = (s.rows ? `${s.n} of ${s.rows} used` : `${s.n}`)+(dropped.length ? ` · dropped: ${dropped.join(', ')}` : '');
  return [['Outcome', outcome], ['Chemical exposures (X)', `${s.p}${s.xNote ? `, ${s.xNote}` : ''}`],
          ['Confounders (Z)', `${s.q}${s.zNote ? ` (${s.zNote})` : ''}`], ['Rows', rows]];
}

// summaryRows input for a CSV plan.
export function planFacts(plan) {
  const k = plan.logged.length, unlogged = plan.exposures.filter(name => !plan.logged.includes(name));
  const logs = !k ? '' : k === plan.p ? 'logged' : unlogged.length <= 2 ? `logged except ${unlogged.join(' and ')}` :
    `${k} of ${plan.p} logged`;
  const transform = logs && plan.standardize ? `${logs}, then standardized` : logs || (plan.standardize ? 'standardized' : 'untransformed');
  const text = plan.terms.filter(t => t.levels !== undefined);
  const outcome = plan.outcome === null ? null :
    `${plan.logOutcome ? 'log ' : ''}${plan.outcome}${plan.standardizeOutcome ? ' (z-score)' : ''}`;
  return {family: plan.family, cases: plan.cases, events: plan.events, p: plan.p, q: plan.q, rows: plan.rows, n: plan.n,
          dropped: plan.dropped, outcomeLabel: outcome === null ? null : plan.event === null ? outcome : `${outcome}, ${plan.event}`,
          xNote: plan.p ? transform : '',
          zNote: [...text.map(t => `${t.name}: ${plural(t.count, 'indicator')}${t.reference !== null ? ` vs ${t.reference}` : ''}`),
                  ...(plan.zLogged.length ? [`logged: ${plan.zLogged.join(', ')}`] : []),
                  ...(plan.zStandardized.length ? [`standardized: ${plan.zStandardized.join(', ')}`] : [])].join('; ')};
}
