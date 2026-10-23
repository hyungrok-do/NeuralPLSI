import test from 'node:test';
import assert from 'node:assert/strict';
import {parseCsv, parseNumber, describeColumns, prepareTable} from '../src/data.js';

test('quotes, escapes, line endings, BOM and blank lines', () => {
  const text = '﻿id,"name, full",note\r\n1,"Do, H.","said ""hi""\nthen left"\r\n\r\n2, plain ,"  kept  "\n3,,\n';
  const t = parseCsv(text);
  assert.equal(t.delimiter, ',');
  assert.deepEqual(t.header, ['id', 'name, full', 'note']);
  assert.deepEqual(t.rows, [['1', 'Do, H.', 'said "hi"\nthen left'], ['2', 'plain', '  kept  '], ['3', '', '']]);
  assert.deepEqual(parseCsv('a,b\r1,2\r3,4').rows, [['1', '2'], ['3', '4']]);
  assert.deepEqual(parseCsv('a,b\n1,2').rows, [['1', '2']]);
});

test('delimiters, decimal commas, duplicate and empty headers, short rows', () => {
  const semicolon = parseCsv('x;y;x\n1,5;2;3\n4;;\n');
  assert.equal(semicolon.delimiter, ';');
  assert.deepEqual(semicolon.header, ['x', 'y', 'x.1']);
  const [x, y] = describeColumns(semicolon);
  assert.deepEqual(Array.from(x.values), [1.5, 4]);
  assert.ok(y.numeric && y.missing === 1);
  assert.equal(parseCsv('a\tb\n1\t2\n').delimiter, '\t');
  assert.deepEqual(parseCsv(',b,b,b\n1,2,3,4\n').header, ['column_1', 'b', 'b.1', 'b.2']);
  assert.deepEqual(parseCsv('a,b,c\n1\n').rows, [['1', '', '']]);
});

test('malformed files are rejected with the line number', () => {
  assert.throws(() => parseCsv('a,b\n1,2\n3,4,5\n'), /Line 3 has 3 fields but the header has 2/);
  assert.throws(() => parseCsv('a,b\n1,"open\n'), /ends inside a quoted field/);
  assert.throws(() => parseCsv('\n\n'), /empty/);
});

test('numbers, missing tokens and text', () => {
  for (const [s, v] of [['1e-3', 1e-3], ['.5', 0.5], ['-2.', -2], ['+3', 3], [' 7 ', 7], ['1E+2', 100]])
    assert.equal(parseNumber(s), v, s);
  for (const s of ['', 'NA', 'n/a', 'NaN', 'null', 'None', '.']) assert.ok(Number.isNaN(parseNumber(s)), s);
  assert.equal(parseNumber('-Inf'), -Infinity);
  for (const s of ['abc', '1,234', '1.2.3', '0x10', '--1']) assert.equal(parseNumber(s), undefined, s);
  assert.equal(parseNumber('1,25', true), 1.25);
  const cols = describeColumns(parseCsv('a,b\n1,x\n2,\n3,y\n'));
  assert.deepEqual(cols.map(c => c.numeric), [true, false]);
  assert.deepEqual(cols[1].levels, ['x', 'y']);
});

const table = parseCsv([
  'y,e1,e2,age,group,event',
  '1.0,1,2,30,a,1', '2.0,2,4,40,b,0', '1.5,4,1,35,c,1', 'NA,1,1,20,a,1',   // missing outcome
  '3.0,Inf,1,50,b,0',                                                       // non-finite exposure
  '2.5,0,3,45,a,1', '0.5,8,5,25,,1',                                        // e1 = 0; missing group
  '1.2,3,3,33,b,1'].join('\n'));

test('complete cases, logs, standardization and dummy coding', () => {
  const d = prepareTable(table, {family: 'continuous', outcome: 'y', exposures: ['e1', 'e2'], covariates: ['age', 'group'],
                                 log: true, standardize: true});
  assert.equal(d.rows, 8);
  assert.deepEqual(d.dropped, {missing: 2, nonfinite: 1, nonpositive: 1});
  assert.deepEqual([d.n, d.p, d.q], [4, 2, 3]);
  assert.deepEqual(d.names.z, ['age', 'group=b', 'group=c']);
  assert.deepEqual(Array.from(d.y), [1, 2, 1.5, 1.2]);
  assert.deepEqual(Array.from(d.z), [30, 0, 0, 40, 1, 0, 35, 0, 1, 33, 1, 0]);
  const e1 = [0, 1, 2, 3].map(i => d.x[2*i]), logs = [1, 2, 4, 3].map(Math.log);
  const m = logs.reduce((s, v) => s+v)/4, sd = Math.sqrt(logs.reduce((s, v) => s+(v-m)**2, 0)/4);
  e1.forEach((v, i) => assert.ok(Math.abs(v-(logs[i]-m)/sd) < 1e-15));
  const raw = prepareTable(table, {family: 'continuous', outcome: 'y', exposures: ['e1']});
  assert.deepEqual(Array.from(raw.x), [1, 2, 4, 0, 8, 3]);
  assert.deepEqual(raw.dropped, {missing: 1, nonfinite: 1, nonpositive: 0});
});

test('outcome checks follow the Python validation', () => {
  const pick = options => prepareTable(table, {exposures: ['e2'], ...options});
  assert.throws(() => pick({family: 'binary', outcome: 'y'}), /Binary outcomes must be coded 0 or 1/);
  assert.equal(pick({family: 'binary', outcome: 'event'}).n, 8);
  const cox = pick({family: 'cox', outcome: 'age', event: 'event'});
  assert.deepEqual(Array.from(cox.y.slice(0, 4)), [30, 1, 40, 0]);
  assert.throws(() => pick({family: 'cox', outcome: 'age', event: 'y'}), /binary event indicators/);
  assert.throws(() => pick({family: 'continuous', outcome: 'group'}), /contains text/);
  assert.throws(() => pick({family: 'continuous', outcome: 'y', covariates: ['e2']}), /only one role/);
  assert.throws(() => prepareTable(parseCsv('y,c\n1,2\n2,2\n'), {family: 'continuous', outcome: 'y', exposures: ['c']}),
                /constant/);
});
