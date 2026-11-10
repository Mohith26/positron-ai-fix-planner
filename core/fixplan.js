// Fix Planner core: estimate how many Hub models a set of Tron fixes would actually
// unblock, accounting for blockers hidden behind each model's first failure.
// Pure ES module; runs in the browser and under `node --test`.

export const STAGES = ['export', 'fxtypedfx', 'rewrite', 'bulk', 'loopy', 'tron', 'cpp'];
export const NST = STAGES.length;
export const EFFORT_DAYS = { small: 3, medium: 8, large: 20 };
export const FEATURES = ['intercept', 'MoE', 'hybrid', 'sliding window', 'MLA', 'vision', 'YaRN', 'softcap', 'quantized', 'vocab > 200k'];
export const NF = FEATURES.length;

// ---------- rng ----------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function normal(r) { let u = r(); while (u < 1e-12) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

// ---------- linear algebra ----------
export function cholesky(A) {
  const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let j = 0; j < n; j++) {
    let s = A[j][j]; for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    L[j][j] = Math.sqrt(Math.max(s, 1e-12));
    for (let i = j + 1; i < n; i++) { let t = A[i][j]; for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]; L[i][j] = t / L[j][j]; }
  }
  return L;
}
export function cholSolve(L, b) {
  const n = b.length, y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) { let t = b[i]; for (let k = 0; k < i; k++) t -= L[i][k] * y[k]; y[i] = t / L[i][i]; }
  for (let i = n - 1; i >= 0; i--) { let t = y[i]; for (let k = i + 1; k < n; k++) t -= L[k][i] * x[k]; x[i] = t / L[i][i]; }
  return x;
}
export function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// ---------- data prep ----------
export function featureVector(m) {
  const f = m.feat;
  return Float64Array.from([1, f.moe ? 1 : 0, f.hybrid ? 1 : 0, f.sliding ? 1 : 0, f.mla ? 1 : 0, f.vision ? 1 : 0, f.yarn ? 1 : 0, f.softcap ? 1 : 0, f.quant ? 1 : 0, f.vocab > 200000 ? 1 : 0]);
}
// One record per model from an hf-litmus snapshot.
export function prepare(models, gapsById, obs) {
  return obs.map((o) => {
    const m = models[o.i];
    const sFail = o.status === 'SUCCESS' ? NST : STAGES.indexOf(o.stage);
    const known = Array.from({ length: NST }, () => []);
    for (const g of o.seen) known[sFail].push(g);
    let frontier = o.status === 'SUCCESS' ? NST - 1 : sFail, complete = o.status === 'SUCCESS', bugAt = -1;
    if (o.deep) {
      for (const g of o.deep.revealed) known[STAGES.indexOf(gapsById[g].stage)].push(g);
      frontier = STAGES.indexOf(o.deep.furthest);
      complete = o.deep.complete;
      if (o.deep.bug) bugAt = frontier;
      if (complete) frontier = NST - 1;
    }
    return {
      i: o.i, id: m.id, downloads: m.downloads, x: featureVector(m), status: o.status, cls: o.cls,
      remote: o.cls === 'trust_remote_code', sFail, seen: o.seen.slice(), known, frontier, complete, bugAt,
      deep: !!o.deep, unknownBlock: o.status !== 'SUCCESS' && o.seen.length === 0 && o.cls !== 'trust_remote_code',
    };
  });
}

// ---------- discrete-time survival over pipeline stages ----------
// Rows (stage k, x, y): y = 1 if the model is blocked at stage k. A model contributes a row
// for every stage it is known to have reached. Under a counterfactual fix set F, known
// blockers in F are treated as fixed, and the model is censored where knowledge ends.
export function survivalRows(recs, F) {
  const rows = Array.from({ length: NST }, () => []);
  for (const r of recs) {
    if (r.remote) continue;
    if (r.status === 'SUCCESS') { for (let k = 0; k < NST; k++) rows[k].push([r.x, 0]); continue; }
    for (let k = 0; k < NST; k++) {
      if (k < r.sFail) { rows[k].push([r.x, 0]); continue; }
      if (k > r.frontier) break; // censored
      const open = r.known[k].some((g) => !F.has(g));
      const blocked = open || (k === r.sFail && r.unknownBlock) || k === r.bugAt;
      if (blocked) { rows[k].push([r.x, 1]); break; }
      rows[k].push([r.x, 0]);
      if (k === r.frontier) break; // knowledge ends here
    }
  }
  return rows;
}
// Ridge (Gaussian prior) logistic regression by Newton's method; returns MAP and Cholesky of the Hessian.
export function fitLogistic(rows, { lambda = 1, lambda0 = 0.05, iters = 25, mu = 0 } = {}) {
  const b = new Float64Array(NF);
  for (let a = 1; a < NF; a++) b[a] = mu;
  let L = null;
  for (let it = 0; it < iters; it++) {
    const H = Array.from({ length: NF }, () => new Float64Array(NF));
    const g = new Float64Array(NF);
    for (const [x, y] of rows) {
      let eta = 0; for (let a = 0; a < NF; a++) eta += b[a] * x[a];
      const p = 1 / (1 + Math.exp(-eta)), w = p * (1 - p);
      for (let a = 0; a < NF; a++) { if (!x[a]) continue; g[a] += (y - p) * x[a]; for (let c = 0; c < NF; c++) if (x[c]) H[a][c] += w * x[a] * x[c]; }
    }
    for (let a = 0; a < NF; a++) { const lam = a === 0 ? lambda0 : lambda; H[a][a] += lam; g[a] -= lam * (b[a] - (a === 0 ? 0 : mu)); }
    L = cholesky(H);
    const step = cholSolve(L, g);
    let mx = 0; for (let a = 0; a < NF; a++) { b[a] += step[a]; mx = Math.max(mx, Math.abs(step[a])); }
    if (mx < 1e-8) break;
  }
  return { beta: b, L, n: rows.length, events: rows.reduce((s, r) => s + r[1], 0) };
}
// Noisy-OR hazard per stage: every feature a model has is an independent possible cause of
// a blocker at that stage (as compiler gaps tend to be), plus a baseline cause for the
// unexplained rest:  P(blocked at k | x) = 1 - (1 - q0k) * prod_{f in x} (1 - qfk).
// Fit by EM with Beta priors. Empirical Bayes: pass 1 uses a flat prior; cause strengths
// for features with enough at-risk models set the Beta prior for pass 2, so a feature that
// has never reached a stage inherits what features typically do at a stage.
export const MIN_AT_RISK = 20;
export function fitNoisyOr(rows, { a = 1, b = 1, a0 = 1, b0 = 1, iters = 200 } = {}) {
  const q = new Float64Array(NF).fill(0.05);
  const n = new Float64Array(NF);
  for (const [x] of rows) for (let j = 0; j < NF; j++) if (x[j]) n[j]++;
  let es = new Float64Array(NF);
  for (let it = 0; it < iters; it++) {
    es = new Float64Array(NF);
    for (const [x, y] of rows) {
      if (!y) continue;
      let surv = 1; for (let j = 0; j < NF; j++) if (x[j]) surv *= 1 - q[j];
      const pb = Math.max(1 - surv, 1e-12);
      for (let j = 0; j < NF; j++) if (x[j]) es[j] += q[j] / pb;
    }
    let mx = 0;
    for (let j = 0; j < NF; j++) {
      const A = j === 0 ? a0 : a, B = j === 0 ? b0 : b;
      const nq = Math.min(1 - 1e-6, Math.max(1e-6, (A - 1 + es[j]) / Math.max(A + B - 2 + n[j], 1e-9)));
      mx = Math.max(mx, Math.abs(nq - q[j])); q[j] = nq;
    }
    if (mx < 1e-7) break;
  }
  // Beta posterior for each cause from EM expected counts (used for intervals)
  const pa = new Float64Array(NF), pb = new Float64Array(NF);
  for (let j = 0; j < NF; j++) { const A = j === 0 ? a0 : a, B = j === 0 ? b0 : b; const s1 = Math.min(es[j], n[j]); pa[j] = A + s1; pb[j] = B + n[j] - s1; }
  return { q, n, pa, pb, rows: rows.length, events: rows.reduce((s, r) => s + r[1], 0) };
}
// Prior on cause strengths at stages a feature has (or has not) reached: Beta(0.2, 0.8),
// mean 0.2, U-shaped (most feature-stage pairs need nothing, some need real work). Chosen
// by walk-forward backtest over past Tron releases (see tuneprior in tests and README).
export const PRIOR = { priorMean: 0.2, priorConc: 1 };
export const PRIOR_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4];
export function fitHazards(recs, F, opts = PRIOR) {
  const rows = survivalRows(recs, F);
  const pass1 = rows.map((r) => fitNoisyOr(r));
  if (opts.flat) return pass1;
  if (opts.priorMean != null) {
    const c = opts.priorConc ?? 2, A = opts.priorMean * c, B = (1 - opts.priorMean) * c;
    const fits = rows.map((r) => fitNoisyOr(r, { a: Math.max(A, 1e-3), b: B }));
    fits.prior = { mean: opts.priorMean, a: A, b: B };
    return fits;
  }
  const ident = [];
  pass1.forEach((f) => { for (let j = 1; j < NF; j++) if (f.n[j] >= MIN_AT_RISK) ident.push(f.q[j]); });
  const m = ident.length ? ident.reduce((x, y) => x + y, 0) / ident.length : 0.1;
  const v = ident.length > 1 ? ident.reduce((x, y) => x + (y - m) ** 2, 0) / (ident.length - 1) : 0.02;
  const conc = Math.min(50, Math.max(2, (m * (1 - m)) / Math.max(v, 1e-6) - 1));
  const A = Math.max(1, m * conc), B = Math.max(1, (1 - m) * conc);
  const fits = rows.map((r) => fitNoisyOr(r, { a: A, b: B }));
  fits.prior = { mean: A / (A + B), a: A, b: B, nIdentified: ident.length };
  return fits;
}
export function hazard(fit, x, q = fit.q) {
  let surv = 1; for (let j = 0; j < NF; j++) if (x[j]) surv *= 1 - q[j];
  return 1 - surv;
}
function gammaSample(k, rng) {
  if (k < 1) return gammaSample(k + 1, rng) * Math.pow(rng(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { x = normal(rng); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = rng(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
}
export function betaSample(a, b, rng) { const x = gammaSample(a, rng), y = gammaSample(b, rng); return x / (x + y); }
// One posterior draw of every cause strength at a stage.
export function drawBeta(fit, rng) {
  const q = new Float64Array(NF);
  for (let j = 0; j < NF; j++) q[j] = betaSample(fit.pa[j], fit.pb[j], rng);
  return q;
}

// ---------- unlock probabilities ----------
// A model can pass under fixes F only if every blocker we know about is in F.
export function gate(r, F) {
  if (r.remote || r.status === 'SUCCESS' || r.unknownBlock || r.bugAt >= 0) return false;
  for (let k = r.sFail; k <= r.frontier; k++) for (const g of r.known[k]) if (!F.has(g)) return false;
  return true;
}
// Chance that no hidden blocker waits beyond what we know (stages after the frontier).
export function survivalAt(r, fits, betas) {
  let p = 1;
  for (let k = r.frontier + 1; k < NST; k++) p *= 1 - hazard(fits[k], r.x, betas ? betas[k] : undefined);
  return p;
}
export function passProb(r, F, betas, fits) { return gate(r, F) ? survivalAt(r, fits, betas) : 0; }
// Posterior survival per model from S common draws of every stage's coefficients.
export function posteriorSurvival(recs, fits, { S = 200, seed = 5 } = {}) {
  const rng = mulberry32(seed), n = recs.length;
  const draws = new Float32Array(S * n), mean = new Float64Array(n);
  for (let s = 0; s < S; s++) {
    const betas = fits.map((f) => drawBeta(f, rng));
    for (let i = 0; i < n; i++) { const v = survivalAt(recs[i], fits, betas); draws[s * n + i] = v; mean[i] += v / S; }
  }
  return { S, n, draws, mean };
}
// Naive reading of op counts: a model passes once the blockers it reported are fixed.
export function naivePass(r, F) {
  if (r.remote || r.status === 'SUCCESS' || r.unknownBlock) return 0;
  return r.seen.every((g) => F.has(g)) ? 1 : 0;
}
const wOf = (r, weight) => (weight === 'downloads' ? r.downloads : 1);
export function expected(recs, F, surv, weight = 'models') {
  let e = 0, n = 0;
  recs.forEach((r, i) => { const w = wOf(r, weight); if (gate(r, F)) e += w * surv.mean[i]; n += w * naivePass(r, F); });
  return { expected: e, naive: n };
}
// Predictive interval: each posterior draw, then a Bernoulli outcome per eligible model.
export function interval(recs, F, surv, { seed = 11, weight = 'models', q = [0.1, 0.9], extra = null } = {}) {
  const rng = mulberry32(seed), out = new Float64Array(surv.S);
  const idx = []; recs.forEach((r, i) => { if (gate(r, F)) idx.push(i); });
  for (let s = 0; s < surv.S; s++) {
    let tot = 0;
    for (const i of idx) if (rng() < surv.draws[s * surv.n + i]) tot += wOf(recs[i], weight);
    if (extra) tot += extra(rng);
    out[s] = tot;
  }
  const srt = Array.from(out).sort((a, b) => a - b);
  return { lo: quantile(srt, q[0]), hi: quantile(srt, q[1]), mean: srt.reduce((a, b) => a + b, 0) / surv.S };
}

// ---------- planning ----------
export function candidates(recs, gapsById) {
  const c = new Set();
  for (const r of recs) {
    if (r.remote || r.status === 'SUCCESS') continue;
    for (const k of r.known) for (const g of k) c.add(g);
  }
  return [...c].filter((g) => gapsById[g]).sort();
}
// Greedy by expected gain per engineer-day, also trying pairs so that two fixes a model
// needs together are not ignored. Hazards (and posterior survival) are re-fit under the
// counterfactual fix set after every accepted step.
export function plan(recs, gapsById, { budgetDays = 40, weight = 'models', S = 200, scoreS = 80, minGainPerDay = 0 } = {}) {
  const cand = candidates(recs, gapsById);
  const F = new Set();
  let fits = fitHazards(recs, F), surv = posteriorSurvival(recs, fits, { S }), spent = 0;
  const steps = [];
  let cur = expected(recs, F, posteriorSurvival(recs, fits, { S: scoreS, seed: 77 }), weight).expected;
  const cost = (g) => EFFORT_DAYS[gapsById[g].effort];
  for (;;) {
    const left = cand.filter((g) => !F.has(g));
    // Score a candidate with hazards re-fit as if it had shipped: models it unblocks move
    // forward in the data and reveal (or rule out) what waits at later stages.
    const memo = new Map();
    const gainOf = (gs) => {
      const key = gs.slice().sort().join('|');
      if (memo.has(key)) return memo.get(key);
      const F2 = new Set([...F, ...gs]);
      const f2 = fitHazards(recs, F2);
      const s2 = posteriorSurvival(recs, f2, { S: scoreS, seed: 77 });
      const g = expected(recs, F2, s2, weight).expected - cur;
      memo.set(key, g);
      return g;
    };
    let best = null;
    const single = new Map();
    for (const g of left) {
      if (spent + cost(g) > budgetDays) continue;
      const gain = gainOf([g]); single.set(g, gain);
      if (gain > 1e-9 && (!best || gain / cost(g) > best.gain / best.cost)) best = { gs: [g], gain, cost: cost(g) };
    }
    const pairGain = new Map();
    for (let a = 0; a < left.length; a++) for (let b = a + 1; b < left.length; b++) {
      const c2 = cost(left[a]) + cost(left[b]);
      const gain = gainOf([left[a], left[b]]);
      pairGain.set(left[a] + '|' + left[b], gain);
      if (spent + c2 > budgetDays) continue;
      // only a pair that beats doing its parts separately (complementary fixes)
      if (gain > (single.get(left[a]) || 0) + (single.get(left[b]) || 0) + 1e-9 && (!best || gain / c2 > best.gain / best.cost)) best = { gs: [left[a], left[b]], gain, cost: c2 };
    }
    // triples, for models that need three gaps closed at once (e.g. two ops plus a kernel)
    for (let a = 0; a < left.length; a++) for (let b = a + 1; b < left.length; b++) for (let c = b + 1; c < left.length; c++) {
      const c3 = cost(left[a]) + cost(left[b]) + cost(left[c]);
      if (spent + c3 > budgetDays) continue;
      const gain = gainOf([left[a], left[b], left[c]]);
      const sub = Math.max((pairGain.get(left[a] + '|' + left[b]) || 0) + (single.get(left[c]) || 0), (pairGain.get(left[a] + '|' + left[c]) || 0) + (single.get(left[b]) || 0), (pairGain.get(left[b] + '|' + left[c]) || 0) + (single.get(left[a]) || 0));
      if (gain > sub + 1e-9 && (!best || gain / c3 > best.gain / best.cost)) best = { gs: [left[a], left[b], left[c]], gain, cost: c3 };
    }
    if (!best || best.gain / best.cost < minGainPerDay) break;
    for (const g of best.gs) F.add(g);
    spent += best.cost;
    fits = fitHazards(recs, F);
    surv = posteriorSurvival(recs, fits, { S });
    const e = expected(recs, F, surv, weight);
    cur = expected(recs, F, posteriorSurvival(recs, fits, { S: scoreS, seed: 77 }), weight).expected;
    steps.push({ gaps: best.gs, cost: best.cost, spent, cumExpected: e.expected, cumNaive: e.naive, F: new Set(F), fits, surv });
  }
  return { steps, cand };
}

// ---------- backtest ----------
// Models stuck on an unnamed, model-specific error (type errors, or a deep analysis that
// stopped on a bug) sometimes start passing through ordinary bug-fix churn. Rate estimated
// from earlier releases only.
export function isChurnPool(r) { return !r.remote && r.status !== 'SUCCESS' && (r.unknownBlock || r.bugAt >= 0); }
export function churnCounts(recsA, passB) {
  let n = 0, x = 0;
  for (const r of recsA) if (isChurnPool(r)) { n++; if (passB.has(r.i)) x++; }
  return { n, x };
}
// Walk-forward backtest over consecutive snapshots: for release t, fit on snapshot t-1 only,
// with the churn rate learned from releases before t.
export function walkForward(models, gapsById, litmus, { S = 300, seed = 9, lambda = PRIOR, start = 2, end = Infinity } = {}) {
  const vs = litmus.versions, out = [];
  const recsBy = vs.map((v) => prepare(models, gapsById, litmus.snapshots[v.version]));
  const passBy = vs.map((v) => new Set(litmus.snapshots[v.version].filter((o) => o.status === 'SUCCESS').map((o) => o.i)));
  for (let t = start; t < Math.min(vs.length, end); t++) {
    const recs = recsBy[t - 1], F = new Set(vs[t].fixed);
    const fits = fitHazards(recs, F, typeof lambda === 'object' ? lambda : { flat: lambda === 'flat' });
    let cn = 0, cx = 0;
    for (let u = 1; u < t; u++) { const c = churnCounts(recsBy[u - 1], passBy[u]); cn += c.n; cx += c.x; }
    const pool = recs.filter(isChurnPool).length;
    const rate = (cx + 1) / (cn + 2);
    const surv = posteriorSurvival(recs, fits, { S, seed: seed + t });
    const e = expected(recs, F, surv);
    const iv = interval(recs, F, surv, { seed: seed + 100 + t, extra: (rng) => { const q = betaSample(cx + 1, cn - cx + 1, rng); let c = 0; for (let k = 0; k < pool; k++) if (rng() < q) c++; return c; } });
    let actual = 0, actualFix = 0, actualChurn = 0;
    for (const r of recs) {
      if (r.status === 'SUCCESS' || r.remote || !passBy[t].has(r.i)) continue;
      actual++;
      if (isChurnPool(r)) actualChurn++; else actualFix++;
    }
    out.push({ from: vs[t - 1].version, to: vs[t].version, date: vs[t].date, fixed: vs[t].fixed, naive: e.naive, expectedFix: e.expected, expectedChurn: rate * pool, expected: e.expected + rate * pool, lo: iv.lo, hi: iv.hi, actual, actualFix, actualChurn });
  }
  return out;
}

// Nested walk-forward: for each release t, pick the prior mean using only releases before t.
export function nestedBacktest(models, gapsById, litmus, { S = 200, first = 3 } = {}) {
  const out = [];
  for (let t = first; t < litmus.versions.length; t++) {
    let best = null;
    for (const m of PRIOR_GRID) {
      const past = walkForward(models, gapsById, litmus, { S: 60, lambda: { priorMean: m, priorConc: 1 }, start: 2, end: t });
      const mae = past.reduce((a, r) => a + Math.abs(r.expected - r.actual), 0) / past.length;
      if (!best || mae < best.mae - 1e-9) best = { m, mae };
    }
    const [r] = walkForward(models, gapsById, litmus, { S, lambda: { priorMean: best.m, priorConc: 1 }, start: t, end: t + 1 });
    out.push({ ...r, priorMean: best.m });
  }
  return out;
}

// Baseline for comparison: the same greedy, budget and pairing, but scored by the dashboard's
// own reading (a model passes once the blockers it reported are fixed).
export function planByCounts(recs, gapsById, { budgetDays = 40, weight = 'models' } = {}) {
  const cand = candidates(recs, gapsById), F = new Set(), steps = [];
  const cost = (g) => EFFORT_DAYS[gapsById[g].effort];
  const naiveOf = (FF) => recs.reduce((a, r) => a + wOf(r, weight) * naivePass(r, FF), 0);
  let spent = 0, cur = 0;
  for (;;) {
    const left = cand.filter((g) => !F.has(g));
    let best = null;
    const consider = (gs) => {
      const c = gs.reduce((a, g) => a + cost(g), 0);
      if (spent + c > budgetDays) return;
      const gain = naiveOf(new Set([...F, ...gs])) - cur;
      if (gain > 1e-9 && (!best || gain / c > best.gain / best.cost)) best = { gs, gain, cost: c };
    };
    for (let a = 0; a < left.length; a++) { consider([left[a]]); for (let b = a + 1; b < left.length; b++) consider([left[a], left[b]]); }
    if (!best) break;
    for (const g of best.gs) F.add(g);
    spent += best.cost; cur = naiveOf(F);
    steps.push({ gaps: best.gs, cost: best.cost, spent, cumNaive: cur, F: new Set(F) });
  }
  return { steps };
}
