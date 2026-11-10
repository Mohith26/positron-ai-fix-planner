import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import * as C from '../core/fixplan.js';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const models = JSON.parse(fs.readFileSync(path.join(root, 'data/hub_snapshot.json'))).models;
const gaps = JSON.parse(fs.readFileSync(path.join(root, 'data/gaps.json'))).gaps;
const gb = Object.fromEntries(gaps.map((g) => [g.id, g]));
const lit = JSON.parse(fs.readFileSync(path.join(root, 'data/litmus.json')));
const recs = C.prepare(models, gb, lit.snapshots[lit.current]);

test('beta and gamma samplers have the right means', () => {
  const r = C.mulberry32(3); let s = 0; const N = 20000;
  for (let i = 0; i < N; i++) s += C.betaSample(2, 6, r);
  assert.ok(Math.abs(s / N - 0.25) < 0.01);
  let t = 0; for (let i = 0; i < N; i++) t += C.betaSample(0.2, 0.8, r);
  assert.ok(Math.abs(t / N - 0.2) < 0.015);
});

test('noisy-OR EM recovers cause strengths on simulated rows', () => {
  const r = C.mulberry32(8), rows = [];
  const q = new Float64Array(C.NF); q[0] = 0.05; q[1] = 0.6; q[2] = 0.3;
  for (let i = 0; i < 6000; i++) {
    const x = new Float64Array(C.NF); x[0] = 1; x[1] = r() < 0.4 ? 1 : 0; x[2] = r() < 0.5 ? 1 : 0;
    rows.push([x, r() < C.hazard({ q }, x) ? 1 : 0]);
  }
  const f = C.fitNoisyOr(rows);
  assert.ok(Math.abs(f.q[0] - 0.05) < 0.02 && Math.abs(f.q[1] - 0.6) < 0.04 && Math.abs(f.q[2] - 0.3) < 0.04, Array.from(f.q).slice(0, 3).join(','));
});

const mk = (over) => ({ i: 0, id: 'a/b', downloads: 1, x: Float64Array.from([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), status: 'INGEST_FAIL', cls: 'missing_op', remote: false, sFail: 1, seen: ['A'], known: [[], ['A'], [], [], [], [], []], frontier: 1, complete: false, bugAt: -1, deep: false, unknownBlock: false, ...over });

test('survival rows: fixed blockers move a model forward, then it is censored', () => {
  const r = mk({});
  let rows = C.survivalRows([r], new Set());
  assert.deepEqual(rows.map((s) => s.map((x) => x[1])), [[0], [1], [], [], [], [], []]);
  rows = C.survivalRows([r], new Set(['A']));
  assert.deepEqual(rows.map((s) => s.map((x) => x[1])), [[0], [0], [], [], [], [], []]); // passes fxtypedfx, censored after
  const d = mk({ deep: true, known: [[], ['A'], [], [], ['K'], [], []], frontier: 4 });
  rows = C.survivalRows([d], new Set(['A']));
  assert.deepEqual(rows.map((s) => s.map((x) => x[1])), [[0], [0], [0], [0], [1], [], []]); // deep analysis saw K at loopy
  const done = mk({ deep: true, complete: true, frontier: 6 });
  rows = C.survivalRows([done], new Set(['A']));
  assert.equal(rows.flat().length, 7);
});

test('gate and pass probability semantics', () => {
  const fits = Array.from({ length: C.NST }, () => ({ q: Float64Array.from([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) }));
  assert.equal(C.passProb(mk({}), new Set(), null, fits), 0);
  assert.ok(Math.abs(C.passProb(mk({}), new Set(['A']), null, fits) - 0.9 ** 5) < 1e-12); // stages rewrite..cpp unknown
  assert.equal(C.passProb(mk({ deep: true, complete: true, frontier: 6 }), new Set(['A']), null, fits), 1);
  assert.equal(C.passProb(mk({ unknownBlock: true, seen: [], known: [[], [], [], [], [], [], []] }), new Set(), null, fits), 0);
  assert.equal(C.naivePass(mk({ deep: true, known: [[], ['A'], [], [], ['K'], [], []], frontier: 4 }), new Set(['A'])), 1);
  assert.equal(C.gate(mk({ deep: true, known: [[], ['A'], [], [], ['K'], [], []], frontier: 4 }), new Set(['A'])), false);
});

test('walk-forward backtest: every release inside the 80% interval, op counts far off', () => {
  const wf = C.walkForward(models, gb, lit, { S: 300 });
  assert.equal(wf.length, 4);
  let errPlan = 0, errNaive = 0;
  for (const r of wf) {
    assert.ok(r.actual >= r.lo && r.actual <= r.hi, `${r.to}: ${r.actual} not in [${r.lo}, ${r.hi}]`);
    errPlan += Math.abs(r.expected - r.actual); errNaive += Math.abs(r.naive - r.actual);
  }
  assert.ok(errPlan < 0.1 * errNaive, `${errPlan} vs ${errNaive}`);
  const moe = wf.find((r) => r.fixed.includes('X1'));
  assert.equal(moe.naive, 98); assert.equal(moe.actual, 7); assert.ok(Math.abs(moe.expected - 7) < 1.5);
});

test('planner: respects the budget and promises far less than the op counts', () => {
  const P = C.plan(recs, gb, { budgetDays: 40 });
  assert.ok(P.steps.length >= 2);
  const last = P.steps.at(-1);
  assert.ok(last.spent <= 40);
  for (const s of P.steps) assert.ok(s.cumNaive >= s.cumExpected - 1e-9);
  const again = C.plan(recs, gb, { budgetDays: 40 });
  assert.deepEqual(again.steps.map((s) => s.gaps), P.steps.map((s) => s.gaps));
});

test('ground truth: planner plan beats the count-driven plan, and its intervals cover the truth', () => {
  const oracle = (F) => JSON.parse(spawnSync('python3', [path.join(root, 'scripts/oracle.py'), [...F].join(',')], { encoding: 'utf8' }).stdout);
  const P = C.plan(recs, gb, { budgetDays: 40 });
  const N = C.planByCounts(recs, gb, { budgetDays: 40 });
  const tP = oracle(P.steps.at(-1).F).models, tN = oracle(N.steps.at(-1).F).models;
  assert.ok(tP > tN, `planner ${tP} vs counts ${tN}`);
  for (const s of P.steps) {
    const iv = C.interval(recs, s.F, s.surv);
    const t = oracle(s.F).models;
    assert.ok(t >= iv.lo && t <= iv.hi, `step ${s.gaps}: truth ${t} not in [${iv.lo}, ${iv.hi}]`);
  }
});

test('data integrity and vendored dashboard markup', () => {
  assert.equal(models.length, 481);
  assert.equal(Object.keys(lit.snapshots).length, 6);
  for (const v of Object.values(lit.snapshots)) assert.equal(v.length, models.length);
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  assert.ok(html.includes('<h1>HF Litmus Dashboard</h1>') && html.includes('id="fix-planner"') && html.includes('Top Missing Ops'));
});
