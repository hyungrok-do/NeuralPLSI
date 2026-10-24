// Column classification and the role and validation logic of the CSV step.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {root} from './helpers.js';
import {parseCsv, describeColumns, prepareTable, simulate, simulatedTable, MAX_LEVELS, FLOAT32_MAX, NHANES_EXPOSURES,
        NHANES_SETUP} from '../src/data.js';
import {ROLES, blockedRole, blockedCategorical, allowedRoles, initialRoles, presetRoles, roleOf, assignRole, setRoleForMany, selection,
        suggestFamily, changeFamily, planCsv, planFacts, summaryRows} from '../src/roles.js';

// An ID, a survey weight, a text column, missing values, a 0/1 outcome, a 0/1
// column named "status", a mostly missing numeric column and an infinite value.
const messy = parseCsv([
  'id,race,wt,pb,cd,hg,age,status,case,notes',
  '1,White,1.5,2.1,0.4,1.0,30,1,0,',
  '2,Black,2.5,3.3,0.2,2.0,41,0,1,',
  '3,Asian,0.7,NA,0.9,1.5,52,1,0,5',
  '4,White,1.1,1.2,0,0.5,63,0,1,',
  '5,Hispanic,3.2,4.8,1.1,inf,35,1,1,',
  '6,,1.9,2.2,0.7,0.8,47,0,0,',
  '7,Black,1.0,5.0,0.3,1.2,,1,0,7',
  '8,White,2.2,0.9,0.6,0.9,58,0,1,'].join('\n'));
const columns = describeColumns(messy);
const col = name => columns.find(c => c.name === name);
const plan = (roles, options = {}) => planCsv(messy, columns, {family: 'continuous', roles, ...options});
const withRoles = (pairs, family = 'continuous') =>
  pairs.reduce((roles, [name, role]) => assignRole(roles, columns, name, role, family), initialRoles(columns));

test('columns are classified with missing counts and examples', () => {
  const kinds = Object.fromEntries(columns.map(c => [c.name, c.kind]));
  assert.deepEqual(kinds, {id: 'numeric', race: 'text', wt: 'numeric', pb: 'numeric', cd: 'numeric', hg: 'numeric',
                           age: 'numeric', status: '0/1', case: '0/1', notes: 'mostly missing'});
  assert.deepEqual([col('pb').missing, col('hg').nonfinite, col('age').missing, col('race').missing], [1, 1, 1, 1]);
  assert.deepEqual([col('pb').present, col('hg').present, col('notes').present], [7, 7, 2]);
  assert.deepEqual(col('race').examples, ['White', 'Black', 'Asian']);
  assert.deepEqual(col('race').levels, ['Asian', 'Black', 'Hispanic', 'White']);
  assert.deepEqual(col('pb').examples, ['2.1', '3.3', '1.2']);
  // Numeric-looking values of a text column are levels, as they are dummy-coded.
  assert.deepEqual(describeColumns(parseCsv('g\na\n1\n\nb\n'))[0].levels, ['1', 'a', 'b']);
  const empty = describeColumns(parseCsv('a,b\n1,\n2,NA\n'))[1];
  assert.deepEqual([empty.kind, empty.present, empty.binary], ['mostly missing', 0, false]);
  assert.deepEqual(allowedRoles(empty, 'cox'), ['ignore']);
});

test('defaults are valid: every column ignored, no outcome or event, Fit waits for an outcome', () => {
  const roles = initialRoles(columns);
  assert.ok(columns.every(c => roleOf(roles, c.name) === 'ignore'));
  assert.deepEqual(selection(roles, columns), {outcome: null, event: null, exposures: [], confounders: []});
  const p = plan(roles);
  assert.equal(p.ready, false);
  assert.equal(p.reason, 'Choose the outcome column');
  assert.deepEqual([p.p, p.q, p.n, p.rows], [0, 0, 8, 8]);
  assert.equal(plan(roles, {family: 'cox'}).reason, 'Choose the survival time column (Time)');
  // Switching to Cox guesses an event column only among ignored 0/1 columns.
  assert.equal(selection(changeFamily(roles, columns, 'cox'), columns).event, 'status');
  const statusIsOutcome = withRoles([['status', 'outcome']]);
  const cox = changeFamily(statusIsOutcome, columns, 'cox'), chosen = selection(cox, columns);
  assert.equal(chosen.outcome, 'status');
  assert.equal(chosen.event, null);
  // Leaving Cox drops the event role.
  assert.equal(selection(changeFamily(changeFamily(roles, columns, 'cox'), columns, 'binary'), columns).event, null);
});

test('roles are exclusive and there is one outcome and one event', () => {
  let roles = withRoles([['pb', 'exposure'], ['cd', 'exposure']]);
  roles = assignRole(roles, columns, 'pb', 'confounder', 'continuous');
  assert.deepEqual(selection(roles, columns), {outcome: null, event: null, exposures: ['cd'], confounders: ['pb']});
  roles = assignRole(roles, columns, 'case', 'outcome', 'continuous');
  roles = assignRole(roles, columns, 'age', 'outcome', 'continuous');
  assert.equal(selection(roles, columns).outcome, 'age');
  assert.equal(roleOf(roles, 'case'), 'ignore');
  assert.equal([...roles.values()].filter(r => r === 'outcome').length, 1);
  // The input map is not changed.
  const before = new Map(roles);
  assignRole(roles, columns, 'cd', 'ignore', 'continuous');
  assert.deepEqual(roles, before);
  assert.throws(() => assignRole(roles, columns, 'nope', 'exposure', 'continuous'), /Unknown column nope/);
  assert.throws(() => assignRole(roles, columns, 'cd', 'weight', 'continuous'), /Unknown role/);
});

test('text columns cannot be exposures, outcomes or events; they can be confounders', () => {
  const race = col('race');
  assert.deepEqual(allowedRoles(race, 'cox'), ['confounder', 'ignore']);
  assert.match(blockedRole(race, 'exposure', 'continuous'), /Text columns cannot be chemical exposures \(X\)/);
  assert.throws(() => withRoles([['race', 'exposure']]), /Text columns cannot be chemical exposures/);
  assert.throws(() => withRoles([['race', 'outcome']]), /must be numeric/);
  const wide = describeColumns(parseCsv(['g', ...Array.from({length: MAX_LEVELS+1}, (_, k) => `level${k}`)].join('\n')))[0];
  assert.match(blockedRole(wide, 'confounder', 'continuous'), /more than 20 levels/);
  assert.deepEqual(allowedRoles(wide, 'continuous'), ['ignore']);
  assert.deepEqual(allowedRoles(col('wt'), 'continuous'), ['exposure', 'confounder', 'outcome', 'ignore']);
  assert.deepEqual(allowedRoles(col('case'), 'cox'), ROLES);
});

test('select all and none skip text columns, the outcome and the other list', () => {
  const names = columns.map(c => c.name);
  let roles = withRoles([['case', 'outcome'], ['age', 'confounder']]);
  roles = setRoleForMany(roles, columns, names, 'exposure', true, 'continuous');
  const chosen = selection(roles, columns);
  assert.deepEqual(chosen.exposures, ['id', 'wt', 'pb', 'cd', 'hg', 'status', 'notes']);
  assert.deepEqual([chosen.outcome, chosen.confounders], ['case', ['age']]);
  roles = setRoleForMany(roles, columns, ['id', 'wt', 'notes', 'race'], 'exposure', false, 'continuous');
  assert.deepEqual(selection(roles, columns).exposures, ['pb', 'cd', 'hg', 'status']);
  // In the order they got the role: age first, then the others in the order listed.
  roles = setRoleForMany(roles, columns, names, 'confounder', true, 'continuous');
  assert.deepEqual(selection(roles, columns).confounders, ['age', 'id', 'race', 'wt', 'notes']);
  roles = setRoleForMany(roles, columns, names, 'confounder', false, 'continuous');
  assert.deepEqual(selection(roles, columns).confounders, []);
  assert.throws(() => setRoleForMany(roles, columns, names, 'outcome', true, 'continuous'), /Cannot set outcome/);
});

test('text confounders are dummy-coded against their first level in the rows kept', () => {
  const roles = withRoles([['case', 'outcome'], ['pb', 'exposure'], ['cd', 'exposure'], ['race', 'confounder'],
                           ['age', 'confounder']]);
  const p = plan(roles, {family: 'binary', log: true, standardize: true});
  // Dropped: row 3 (pb NA), row 6 (race missing), row 7 (age missing), row 4 (cd = 0, log).
  assert.deepEqual(p.dropped, {missing: 3, nonfinite: 0, nonpositive: 1});
  assert.equal(p.n, 4);
  // Asian was only in a dropped row, so White and Black and Hispanic remain: 2 indicators vs Black.
  assert.deepEqual(p.terms, [{name: 'race', count: 2, levels: 3, reference: 'Black'}, {name: 'age', count: 1}]);
  assert.deepEqual([p.p, p.q, p.ready, p.reason], [2, 3, true, null]);
  assert.deepEqual([p.data.p, p.data.q, p.data.n], [2, 3, 4]);
  assert.deepEqual(p.data.names.z, ['race=Hispanic', 'race=White', 'age']);
  assert.deepEqual(p.data.names.x, ['log pb', 'log cd']);
  assert.equal(p.cases, 3);
  // The same matrix as prepareTable on its own.
  const direct = prepareTable(messy, {family: 'binary', outcome: 'case', exposures: ['pb', 'cd'], covariates: ['race', 'age'],
                                      log: true, standardize: true});
  assert.deepEqual([Array.from(direct.x), Array.from(direct.z), Array.from(direct.y)],
                   [Array.from(p.data.x), Array.from(p.data.z), Array.from(p.data.y)]);
});

test('a Cox event column is never the outcome', () => {
  let roles = withRoles([['age', 'outcome'], ['pb', 'exposure']], 'cox');
  roles = changeFamily(roles, columns, 'cox');
  assert.deepEqual([selection(roles, columns).outcome, selection(roles, columns).event], ['age', 'status']);
  assert.equal(plan(roles, {family: 'cox'}).ready, true);
  assert.equal(plan(roles, {family: 'cox'}).events, 2);
  // Giving the event role to the outcome column clears the outcome.
  roles = withRoles([['case', 'outcome'], ['pb', 'exposure']], 'cox');
  roles = assignRole(roles, columns, 'case', 'event', 'cox');
  const chosen = selection(roles, columns);
  assert.deepEqual([chosen.outcome, chosen.event], [null, 'case']);
  assert.equal(plan(roles, {family: 'cox'}).reason, 'Choose the survival time column (Time)');
  // Only 0/1 columns can be the event, and only for Cox.
  assert.match(blockedRole(col('age'), 'event', 'cox'), /coded 0\/1/);
  assert.match(blockedRole(col('case'), 'event', 'binary'), /Only a survival \(Cox\) outcome/);
  assert.throws(() => assignRole(roles, columns, 'case', 'event', 'continuous'), /Only a survival/);
});

test('invalid setups keep Fit disabled with the reason', () => {
  const reason = (pairs, family = 'continuous', options = {}) => {
    const p = plan(withRoles(pairs, family), {family, ...options});
    assert.equal(p.ready, p.reason === null);
    return p.reason;
  };
  assert.equal(reason([['pb', 'exposure']]), 'Choose the outcome column');
  assert.equal(reason([['age', 'outcome']]), 'Choose at least one chemical exposure (X)');
  assert.equal(reason([['age', 'outcome'], ['race', 'confounder']]), 'Choose at least one chemical exposure (X)');
  assert.equal(reason([['age', 'outcome'], ['pb', 'exposure']], 'cox'), 'Choose the event column (0/1)');
  assert.equal(reason([['age', 'outcome'], ['pb', 'exposure']], 'binary'),
               'The outcome age is not coded 0/1; choose another outcome or outcome type');
  assert.equal(reason([['age', 'outcome'], ['pb', 'exposure']]), null);
  assert.equal(reason([['age', 'outcome'], ['notes', 'exposure']]), 'Only 1 complete row remains');
  const constant = parseCsv('y,c,e\n1,2,1\n2,2,3\n3,2,2\n'), cc = describeColumns(constant);
  const roles = [['y', 'outcome'], ['c', 'exposure']].reduce((r, [n, role]) => assignRole(r, cc, n, role, 'continuous'),
                                                               initialRoles(cc));
  assert.match(planCsv(constant, cc, {family: 'continuous', roles}).reason, /Chemical exposure c is constant/);
});

test('suggested family and the summary counts', () => {
  assert.equal(suggestFamily(col('case')), 'binary');
  assert.equal(suggestFamily(col('age')), 'continuous');
  assert.equal(suggestFamily(null), 'continuous');
  const roles = withRoles([['case', 'outcome'], ['pb', 'exposure'], ['cd', 'exposure'], ['hg', 'exposure'],
                           ['race', 'confounder'], ['age', 'confounder']]);
  const rows = Object.fromEntries(summaryRows(planFacts(plan(roles, {family: 'binary', log: true, standardize: true}))));
  assert.deepEqual(rows, {
    'Outcome': 'case · binary (logistic), 2 cases',
    'Chemical exposures (X)': '3, logged, then standardized',
    'Confounders (Z)': '2 (race: 1 indicator vs Black)',
    'Rows': '3 of 8 used · dropped: 3 missing, 1 non-finite, 1 not positive (log)'});
  const none = Object.fromEntries(summaryRows(planFacts(plan(initialRoles(columns)))));
  assert.deepEqual(none, {'Outcome': 'not chosen', 'Chemical exposures (X)': '0', 'Confounders (Z)': '0', 'Rows': '8 of 8 used'});
});

test('X and Z columns are in the order they got their role', () => {
  let roles = withRoles([['hg', 'exposure'], ['pb', 'exposure'], ['age', 'confounder'], ['cd', 'exposure'],
                         ['race', 'confounder'], ['case', 'outcome']]);
  assert.deepEqual(selection(roles, columns), {outcome: 'case', event: null, exposures: ['hg', 'pb', 'cd'],
                                               confounders: ['age', 'race']});
  // Giving a column the role it has keeps its place; a new role puts it last.
  roles = assignRole(roles, columns, 'hg', 'exposure', 'continuous');
  assert.deepEqual(selection(roles, columns).exposures, ['hg', 'pb', 'cd']);
  roles = assignRole(assignRole(roles, columns, 'hg', 'ignore', 'continuous'), columns, 'hg', 'exposure', 'continuous');
  assert.deepEqual(selection(roles, columns).exposures, ['pb', 'cd', 'hg']);
  const p = plan(roles, {log: true});
  assert.deepEqual(p.data.names.x, ['log pb', 'log cd', 'log hg']);
  // Rows 0, 1 and 7 remain, so race has two levels there.
  assert.deepEqual(p.data.names.z, ['age', 'race=White']);
});

test('numeric confounders of codes can be dummy-coded against their lowest value', () => {
  const t = parseCsv(['y,x,grade,dose,sex,score,flag',
    ...Array.from({length: 24}, (_, i) => [i % 5+1, (i*7) % 11+1, [1, 2, 10][i % 3], [0.5, 1.5][i % 2], 1+(i % 2),
                                           i, i % 2].join(','))].join('\n'));
  const cc = describeColumns(t), c = name => cc.find(col => col.name === name);
  // Suggested: 2 to 10 distinct integer codes, not 0/1. Not suggested: fractions, many values, 0/1.
  assert.deepEqual(cc.map(col => [col.name, col.codes, col.distinct]), [['y', true, 5], ['x', false, 11],
    ['grade', true, 3], ['dose', false, 2], ['sex', true, 2], ['score', false, 21], ['flag', false, 2]]);
  assert.equal(blockedCategorical(c('dose')), null);
  assert.match(blockedCategorical(c('score')), /More than 20 distinct values/);
  const roles = [['y', 'outcome'], ['x', 'exposure'], ['grade', 'confounder'], ['dose', 'confounder'], ['sex', 'confounder']]
    .reduce((r, [name, role]) => assignRole(r, cc, name, role, 'continuous'), initialRoles(cc));
  const numeric = planCsv(t, cc, {family: 'continuous', roles});
  assert.deepEqual([numeric.q, numeric.data.names.z], [3, ['grade', 'dose', 'sex']]);
  // Levels sort as numbers (10 after 2), and the lowest is the reference.
  const coded = planCsv(t, cc, {family: 'continuous', roles, categorical: ['grade', 'sex', 'dose']});
  assert.deepEqual(coded.data.names.z, ['grade=2', 'grade=10', 'dose=1.5', 'sex=2']);
  assert.deepEqual(coded.terms, [{name: 'grade', count: 2, levels: 3, reference: 1}, {name: 'dose', count: 1, levels: 2, reference: 0.5},
                                 {name: 'sex', count: 1, levels: 2, reference: 1}]);
  assert.deepEqual(Array.from(coded.data.z.slice(0, 12)), [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0]);
  assert.deepEqual(coded.data.transforms.z[1], {name: 'grade=10', column: 'grade', equals: 10, reference: 1});
  assert.equal(planFacts(coded).zNote, 'grade: 2 indicators vs 1; dose: 1 indicator vs 0.5; sex: 1 indicator vs 1');
  // Exposures and the outcome are never dummy-coded; categorical names of other roles are ignored.
  assert.deepEqual(planCsv(t, cc, {family: 'continuous', roles, categorical: ['x', 'y']}).data.names.z, ['grade', 'dose', 'sex']);
});

test('each exposure is logged or not, and a continuous outcome can be logged and standardized', () => {
  const roles = withRoles([['wt', 'outcome'], ['pb', 'exposure'], ['cd', 'exposure'], ['hg', 'exposure']]);
  // cd has a zero: logging only pb and hg keeps that row.
  const some = plan(roles, {log: ['pb', 'hg'], standardize: true});
  assert.deepEqual(some.data.names.x, ['log pb', 'cd', 'log hg']);
  assert.deepEqual(some.dropped, {missing: 1, nonfinite: 1, nonpositive: 0});
  assert.equal(planFacts(some).xNote, 'logged except cd, then standardized');
  assert.equal(planFacts(plan(roles, {log: ['pb']})).xNote, 'logged except cd and hg');
  const four = withRoles([['wt', 'outcome'], ['pb', 'exposure'], ['cd', 'exposure'], ['hg', 'exposure'], ['age', 'exposure']]);
  assert.equal(planFacts(plan(four, {log: ['pb']})).xNote, '1 of 4 logged');
  assert.deepEqual(plan(roles, {log: ['cd']}).dropped, {missing: 1, nonfinite: 1, nonpositive: 1});
  const y = plan(roles, {logOutcome: true, standardizeOutcome: true});
  const kept = [0, 1, 3, 5, 6, 7].map(r => Math.log(col('wt').values[r]));
  const m = kept.reduce((s, v) => s+v)/6, sd = Math.sqrt(kept.reduce((s, v) => s+(v-m)**2, 0)/6);
  assert.deepEqual(Array.from(y.data.y), kept.map(v => (v-m)/sd));
  assert.deepEqual(y.data.names.y, ['log wt (z-score)']);
  assert.deepEqual(y.data.transforms.y, [{name: 'log wt (z-score)', column: 'wt', log: true, mean: m, sd}]);
  assert.equal(planFacts(y).outcomeLabel, 'log wt (z-score)');
  assert.equal(Object.fromEntries(summaryRows(planFacts(y)))['Outcome'], 'log wt (z-score) · continuous');
  // Only for a continuous outcome.
  const binary = withRoles([['case', 'outcome'], ['pb', 'exposure']], 'binary');
  const b = plan(binary, {family: 'binary', logOutcome: true, standardizeOutcome: true});
  assert.deepEqual([b.logOutcome, b.standardizeOutcome, b.data.names.y, b.dropped.nonpositive], [false, false, ['case'], 0]);
  assert.match(planCsv(parseCsv('y,x\n2,1\n2,3\n2,2\n'), describeColumns(parseCsv('y,x\n2,1\n2,3\n2,2\n')), {
    family: 'continuous', roles: new Map([['y', 'outcome'], ['x', 'exposure']]), standardizeOutcome: true}).reason,
    /The outcome y is constant/);
});

test('numbers beyond the float32 range count as non-finite in their column', () => {
  const t = parseCsv('y,x,z\n1,2,3\n2,1e39,4\n3,5,-4e38\n4,3.4e38,1\n5,4,2\n');
  const [, x, z] = describeColumns(t);
  assert.deepEqual([x.nonfinite, x.beyond, z.nonfinite, z.beyond, x.present], [1, 1, 1, 1, 4]);
  assert.ok(3.4e38 < FLOAT32_MAX && Number.isFinite(Math.fround(3.4e38)) && !Number.isFinite(Math.fround(1e39)));
  const p = planCsv(t, describeColumns(t), {family: 'continuous', roles: new Map([['y', 'outcome'], ['x', 'exposure'],
                                                                                  ['z', 'confounder']])});
  assert.deepEqual([p.ready, p.n, p.dropped], [true, 3, {missing: 0, nonfinite: 2, nonpositive: 0}]);
  assert.ok(Float32Array.from(p.data.x).every(Number.isFinite) && Float32Array.from(p.data.z).every(Number.isFinite));
});

// A preset setup as the page applies it (app.js loadTable).
const planPreset = (table, preset) => {
  const cc = describeColumns(table);
  return planCsv(table, cc, {family: preset.family, roles: presetRoles(cc, preset), log: preset.log,
    standardize: preset.standardize, categorical: preset.categorical, logOutcome: preset.logOutcome,
    standardizeOutcome: preset.standardizeOutcome});
};

test('the NHANES setup has the columns and transforms of reproduce/nhanes.py', () => {
  const t = parseCsv(readFileSync(join(root, 'NHANES', '12940_2020_644_MOESM2_ESM.csv'), 'utf8'));
  assert.deepEqual(describeColumns(t).filter(c => c.codes).map(c => c.name), ['sex', 'race']);
  const p = planPreset(t, NHANES_SETUP);
  assert.deepEqual([p.ready, p.n, p.p, p.q], [true, 800, 8, 6]);
  assert.deepEqual(p.exposures, NHANES_EXPOSURES);
  assert.deepEqual(p.data.names.x, NHANES_EXPOSURES.map(name => name === 'a5.Retinol' ? name : `log ${name}`));
  assert.deepEqual(p.data.names.z, ['age', 'sex=2', 'race=2', 'race=3', 'race=4', 'race=5']);
  assert.deepEqual(p.data.names.y, ['log triglyceride (z-score)']);
  assert.deepEqual(Object.fromEntries(summaryRows(planFacts(p))), {'Outcome': 'log triglyceride (z-score) · continuous',
    'Chemical exposures (X)': '8, logged except a5.Retinol, then standardized',
    'Confounders (Z)': '6 (sex: 1 indicator vs 1; race: 4 indicators vs 1)', 'Rows': '800 of 800 used'});
  // reproduce/nhanes.py z-scores over every value of a column; no value is missing, so that is every row.
  assert.ok(describeColumns(t).every(c => c.missing+c.nonfinite === 0));
});

test('the simulated table, set up as generated, gives the simulated data exactly', () => {
  for (const outcome of ['continuous', 'binary', 'cox']) {
    const options = {n: 300, g: 'threshold', outcome, seed: 7}, d = simulate(options), {table, setup} = simulatedTable(options);
    assert.deepEqual(table.header, outcome === 'cox' ? [...d.names.x, ...d.names.z, 'time', 'event'] : [...d.names.x, ...d.names.z, 'y']);
    const p = planPreset(table, setup);
    assert.equal(p.ready, true, p.reason);
    for (const k of ['x', 'z', 'y']) assert.deepEqual(Array.from(p.data[k]), Array.from(d[k]), `${outcome} ${k}`);
    assert.deepEqual([p.data.family, p.data.names.x, p.data.names.z], [outcome, d.names.x, d.names.z]);
    // z3 (+-1) is suggested as categorical, not set.
    assert.deepEqual(describeColumns(table).filter(c => c.codes).map(c => c.name), ['z3']);
  }
});

test('a binary outcome with no cases or no non-cases cannot be fitted', () => {
  const t = parseCsv(['y0,y1,yy,x,w', ...Array.from({length: 12}, (_, i) =>
    [0, 1, i < 2 ? 1 : 0, Math.sin(i).toFixed(3), i < 2 ? 'NA' : String(i)].join(','))].join('\n'));
  const cc = describeColumns(t);
  const reason = (outcome, exposures = ['x']) => {
    const roles = [[outcome, 'outcome'], ...exposures.map(name => [name, 'exposure'])]
      .reduce((r, [name, role]) => assignRole(r, cc, name, role, 'binary'), initialRoles(cc));
    const p = planCsv(t, cc, {family: 'binary', roles});
    assert.equal(p.ready, p.reason === null);
    return [p.reason, p.cases];
  };
  // A 0/1 column suggests the binary family even when every value is 0.
  assert.equal(suggestFamily(cc[0]), 'binary');
  assert.deepEqual(reason('y0'), ['The outcome y0 has no cases (all 0)', 0]);
  assert.deepEqual(reason('y1'), ['The outcome y1 has no non-cases (all 1)', 12]);
  assert.deepEqual(reason('yy'), [null, 2]);
  // Its two cases are in rows that w drops.
  assert.deepEqual(reason('yy', ['x', 'w']), ['The outcome yy has no cases (all 0) in the rows used', 0]);
  // With fewer than two rows, the row count is the reason.
  const tiny = parseCsv('y,x\n0,1\n'), tc = describeColumns(tiny);
  assert.equal(planCsv(tiny, tc, {family: 'binary', roles: new Map([['y', 'outcome'], ['x', 'exposure']])}).reason,
               'Only 1 complete row remains');
});

test('numeric confounders have their own log and z-score, and categorical ones have neither', () => {
  const t = parseCsv(['y,x,age,bmi,race', ...Array.from({length: 40}, (_, i) =>
    [(Math.sin(i)+i/20).toFixed(3), (1+i%7).toString(), 20+i, i === 3 ? '0' : (18+(i*7)%15).toString(), 1+i%3].join(','))].join('\n'));
  const cc = describeColumns(t), roles = [['y', 'outcome'], ['x', 'exposure'], ['age', 'confounder'], ['bmi', 'confounder'],
    ['race', 'confounder']].reduce((r, [name, role]) => assignRole(r, cc, name, role, 'continuous'), initialRoles(cc));
  const p = planCsv(t, cc, {family: 'continuous', roles, zLog: ['bmi', 'race'], zStandardize: ['age', 'bmi', 'race'],
                            categorical: ['race']});
  assert.equal(p.ready, true, p.reason);
  // race is categorical, so its log and z-score do not apply; bmi's log drops its zero.
  assert.deepEqual([p.zLogged, p.zStandardized, p.n, p.dropped.nonpositive], [['bmi'], ['age', 'bmi'], 39, 1]);
  assert.deepEqual(p.data.names.z, ['age', 'log bmi', 'race=2', 'race=3']);
  const keep = [...Array(40).keys()].filter(i => i !== 3), col = j => keep.map((_, i) => p.data.z[i*p.data.q+j]);
  const mean = v => v.reduce((a, b) => a+b, 0)/v.length, sd = v => Math.sqrt(mean(v.map(e => (e-mean(v))**2)));
  for (const j of [0, 1]) {
    assert.ok(Math.abs(mean(col(j))) < 1e-12 && Math.abs(sd(col(j))-1) < 1e-12, `z column ${j} is a z-score`);
  }
  const bmi = p.data.transforms.z[1], logs = keep.map(i => Math.log(cc[3].values[i]));
  assert.deepEqual([bmi.column, bmi.log], ['bmi', true]);
  assert.ok(Math.abs(bmi.mean-mean(logs)) < 1e-12 && Math.abs(bmi.sd-sd(logs)) < 1e-12);
  assert.deepEqual(p.data.transforms.z[0], {name: 'age', column: 'age', log: false, mean: p.data.transforms.z[0].mean,
                                            sd: p.data.transforms.z[0].sd});
  assert.equal(Object.fromEntries(summaryRows(planFacts(p)))['Confounders (Z)'],
               '4 (race: 2 indicators vs 1; logged: bmi; standardized: age, bmi)');
  // Without the options, numeric confounders enter as they are.
  const plain = planCsv(t, cc, {family: 'continuous', roles});
  assert.deepEqual([plain.n, plain.data.transforms.z[1]], [40, {name: 'bmi', column: 'bmi'}]);
});
