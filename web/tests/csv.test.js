import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {web, noPython, runPython} from './helpers.js';
import {parseCsv, parseNumber, describeColumns, prepareTable, NA_TOKENS} from '../src/data.js';
import {displayText} from '../src/stats.js';

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
  // pandas' default missing strings, matched exactly as pandas does.
  for (const s of ['', 'NA', 'n/a', 'N/A', 'NaN', 'nan', '-nan', 'null', 'NULL', 'None', '<NA>', '#N/A', '-1.#IND'])
    assert.ok(Number.isNaN(parseNumber(s)), s);
  for (const s of ['.', 'missing', 'na', 'Na', 'NONE', 'Null']) assert.equal(parseNumber(s), undefined, s);
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

test('parsing a long run of digits takes linear time', () => {
  // The answers of the earlier pattern, /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i, which was quadratic.
  const earlier = /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i;
  for (const s of ['1', '12.', '1.5', '.5', '-0.25e-3', '+7E2', '1..2', '.', '-.', '1e', '1.2.3', 'e5', '1e+', '00012.3400'])
    assert.equal(parseNumber(s) === undefined, !earlier.test(s), s);
  for (const digits of [32000, 200000]) {
    const started = performance.now();
    assert.equal(parseNumber('1'.repeat(digits)+'x'), undefined);
    assert.equal(parseNumber('1'.repeat(digits)+'.5'), Number('1'.repeat(digits)+'.5'));
    const ms = performance.now()-started;
    assert.ok(ms < 100, `${digits} digits took ${ms.toFixed(0)} ms`);
  }
});

test('missing values: pandas strings in numeric columns; text columns keep None, missing and .', () => {
  const t = parseCsv(['smoke,edu,score,pb,dot',
    ...Array.from({length: 60}, (_, i) => [['None', 'Former', 'Current'][i % 3], ['missing', 'HS', 'College'][i % 3],
      i % 10 === 0 ? 'NA' : i % 10 === 1 ? '' : i % 10 === 2 ? 'None' : String(i), i % 15 === 4 ? '#N/A' : String(i/10),
      i % 20 === 3 ? '.' : String(i % 4)].join(','))].join('\n'));
  const [smoke, edu, score, pb, dot] = describeColumns(t);
  // Text: None, missing and . are levels, and nothing is missing.
  assert.deepEqual([smoke.kind, smoke.missing, smoke.levels], ['text', 0, ['Current', 'Former', 'None']]);
  assert.deepEqual([edu.kind, edu.missing, edu.levels], ['text', 0, ['College', 'HS', 'missing']]);
  assert.deepEqual(smoke.examples, ['None', 'Former', 'Current']);
  // Numeric: NA, blank and None are missing; the tokens counted are listed, most frequent first.
  assert.deepEqual([score.kind, score.missing, score.present, score.tokens], ['numeric', 18, 42, [['NA', 6], ['', 6], ['None', 6]]]);
  assert.deepEqual([pb.missing, pb.tokens], [4, [['#N/A', 4]]]);
  // '.' is not missing, so a column of numbers with '.' is text, and the notes can say which values are not numbers.
  assert.deepEqual([dot.kind, dot.numeric, dot.notNumbers, dot.commas], ['text', false, ['.'], undefined]);
  assert.equal(smoke.notNumbers, undefined);
  // The rows used keep every text level: no row is dropped for None, missing or '.'.
  const d = prepareTable(t, {family: 'continuous', outcome: 'pb', exposures: ['score'], confounders: ['smoke', 'edu', 'dot']});
  assert.deepEqual([d.n, d.dropped.missing], [60-18-4, 18+4]);
  assert.ok(['smoke=None', 'edu=missing', 'dot=1'].every(name => d.names.z.includes(name)), d.names.z.join());
  // A text column still drops pandas' strings other than None.
  const blank = describeColumns(parseCsv('g,k\na,1\nNA,2\nNone,3\n,4\nb,5\n'))[0];
  assert.deepEqual([blank.kind, blank.missing, blank.tokens, blank.levels], ['text', 2, [['NA', 1], ['', 1]], ['None', 'a', 'b']]);
});

test('the missing-value strings are pandas defaults', {skip: noPython}, () => {
  const py = JSON.parse(runPython(`
import io, json, sys
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
tokens = json.loads(sys.argv[1])
column = pd.read_csv(io.StringIO('x\\n1\\n' + '\\n'.join('"%s"' % t for t in tokens) + '\\n'), keep_default_na=True)['x']
print(json.dumps({'tokens': sorted(STR_NA_VALUES), 'missing': column.isna().tolist()[1:]}))`, [JSON.stringify(NA_TOKENS)]));
  assert.deepEqual([...NA_TOKENS].sort(), py.tokens);
  assert.ok(py.missing.every(Boolean), 'pandas reads every token as missing');
});

test('the page and the README list the missing-value strings', () => {
  const html = readFileSync(join(web, 'src', 'index.html'), 'utf8'), readme = readFileSync(join(web, 'README.md'), 'utf8');
  const help = html.slice(html.indexOf('id="csv-format"'), html.indexOf('</details>', html.indexOf('id="csv-format"')));
  const listed = [...help.matchAll(/<code>(.*?)<\/code>/g)].map(m => m[1].replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'));
  for (const token of NA_TOKENS.filter(Boolean)) {
    assert.ok(listed.includes(token), `index.html does not list ${token}`);
    assert.ok(readme.includes('`'+token+'`'), `README.md does not list ${token}`);
  }
});

test('decimal commas are decided per column', () => {
  // Copied from a spreadsheet: tab-separated, thousands separators in income.
  const tsv = describeColumns(parseCsv('id\tincome\trate\tmixed\n1\t1,250\t1,5\t1,5\n2\t2,500\t2,25\t2.5\n3\t12,000\t0,5\t3\n'));
  const [, income, rate, mixed] = tsv;
  assert.deepEqual([income.kind, income.notNumbers, income.commas], ['text', ['1,250', '2,500', '12,000'], 'ambiguous']);
  assert.deepEqual([rate.kind, rate.decimalComma, Array.from(rate.values)], ['numeric', true, [1.5, 2.25, 0.5]]);
  assert.deepEqual([rate.examples, rate.exampleValues], [['1,5', '2,25', '0,5'], [1.5, 2.25, 0.5]]);
  assert.deepEqual([mixed.kind, mixed.notNumbers, mixed.commas], ['text', ['1,5'], 'point']);
  // Semicolons: a value that cannot be a thousands number (0,5, 2,5, 0,125)
  // settles a column as decimal commas, so 1,000 there is 1.
  const semi = describeColumns(parseCsv('a;b;c;d;e;f\n0,5;1.5;1,000;0,125;1,250;1,250,000\n2;2;2,5;1,375;2,500;2\nNA;3;3;2;3;3\n'));
  assert.deepEqual(semi.map(c => c.kind), ['numeric', 'numeric', 'numeric', 'numeric', 'text', 'text']);
  assert.deepEqual([Array.from(semi[0].values), semi[0].decimalComma, semi[0].exampleValues], [[0.5, 2, NaN], true, [0.5, null]]);
  assert.deepEqual([semi[1].decimalComma, semi[1].exampleValues], [false, [null, null, null]]);
  assert.deepEqual(Array.from(semi[2].values), [1, 2.5, 3]);
  assert.deepEqual(Array.from(semi[3].values), [0.125, 1.375, 2]);
  assert.deepEqual([semi[4].commas, semi[5].commas], ['ambiguous', 'thousands']);
  // A comma-separated file never has decimal commas, even quoted.
  const comma = describeColumns(parseCsv('a,b\n"1,5",1\n"2,5",2\n'));
  assert.deepEqual([comma[0].kind, comma[0].commas, comma[1].kind], ['text', 'delimiter', 'numeric']);
  // Text columns that are mostly not numbers are not flagged.
  assert.equal(describeColumns(parseCsv('g;h\na;1\nb,5;2\nc;3\n'))[0].notNumbers, undefined);
});

test('bidi controls are dropped from text shown on the page', () => {
  const rlo = String.fromCharCode(0x202e), lri = String.fromCharCode(0x2066), pdi = String.fromCharCode(0x2069);
  const rlm = String.fromCharCode(0x200f);
  assert.equal(displayText(`evil${rlo}SCV.exe=2`), 'evilSCV.exe=2');
  assert.equal(displayText(`${lri}a${pdi}${rlm} · b`), 'a · b');
  // Names in the data are unchanged.
  assert.equal(parseCsv(`evil${rlo}SCV.exe,y\n1,2\n`).header[0], `evil${rlo}SCV.exe`);
  assert.equal(displayText(12), '12');
});
