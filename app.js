import * as C from './core/fixplan.js';

const EFFORT_COLORS = { small: '#84cc16', medium: '#eab308', large: '#ef4444' }; // hf-litmus _EFFORT_COLORS
const state = { budget: 40, weight: 'models' };
const qs = new URLSearchParams(location.search);
if ([20, 40, 60].includes(+qs.get('budget'))) state.budget = +qs.get('budget');
if (qs.get('weight') === 'downloads') state.weight = 'downloads';
let models, gaps, gb, lit, recs, wf;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtN = (x) => (state.weight === 'downloads' ? fmtD(x) : String(Math.round(x)));
const fmtD = (x) => (x >= 1e6 ? `${(x / 1e6).toFixed(1)}M` : x >= 1e3 ? `${Math.round(x / 1e3)}K` : String(Math.round(x)));
const unit = () => (state.weight === 'downloads' ? 'downloads/mo' : 'models');
const tick = () => new Promise((r) => setTimeout(r, 0));

async function main() {
  const [h, g, l] = await Promise.all(['data/hub_snapshot.json', 'data/gaps.json', 'data/litmus.json'].map((u) => fetch(u).then((r) => r.json())));
  models = h.models; gaps = g.gaps; gb = Object.fromEntries(gaps.map((x) => [x.id, x])); lit = l;
  recs = C.prepare(models, gb, lit.snapshots[lit.current]);
  await tick();
  wf = C.walkForward(models, gb, lit, { S: 300 });
  render();
}

function render() {
  const P = C.plan(recs, gb, { budgetDays: state.budget, weight: state.weight });
  const N = C.planByCounts(recs, gb, { budgetDays: state.budget, weight: state.weight });
  const last = P.steps.at(-1);
  const iv = last ? C.interval(recs, last.F, last.surv, { weight: state.weight }) : { lo: 0, hi: 0 };
  // what the planner expects from the plan the op counts would pick
  const NF = N.steps.length ? N.steps.at(-1).F : new Set();
  const nFits = C.fitHazards(recs, NF), nSurv = C.posteriorSurvival(recs, nFits, { S: 200 });
  const nExp = C.expected(recs, NF, nSurv, state.weight).expected;
  const nIv = C.interval(recs, NF, nSurv, { weight: state.weight });
  const failing = recs.filter((r) => r.status !== 'SUCCESS').length;

  const btn = (grp, v, label, on) => `<button class="tag-btn${on ? ' active' : ''}" data-grp="${grp}" data-v="${v}">${label}</button>`;
  let html = `<p class="subtitle fp-lede">Which Tron gaps to close next. The tables below count a model as one fix away once its reported ops are fixed. The planner also estimates the blockers hidden behind each model's first failure, learned from how ${recs.length} models moved through the ${C.NST} pipeline stages, and is checked against past Tron releases.</p>
  <div class="filters">
    <div class="filter-group"><span class="filter-label">Budget</span><div class="filter-opts">${[20, 40, 60].map((b) => btn('budget', b, `${b} engineer-days`, state.budget === b)).join(' ')}</div></div>
    <div class="filter-group"><span class="filter-label">Objective</span><div class="filter-opts">${btn('weight', 'models', 'Models', state.weight === 'models')} ${btn('weight', 'downloads', 'Downloads', state.weight === 'downloads')}</div></div>
    <div class="filter-group"><span class="filter-label">Effort scale</span><div class="filter-opts fp-muted">small ${C.EFFORT_DAYS.small} d · medium ${C.EFFORT_DAYS.medium} d · large ${C.EFFORT_DAYS.large} d</div></div>
  </div>
  <div class="cards">
    <div class="card"><div class="label">Plan</div><div class="value">${last ? last.spent : 0}<span class="fp-unit"> days</span></div><div class="fp-sub">${last ? last.F.size : 0} gaps of ${P.cand.length} open</div></div>
    <div class="card"><div class="label">Op counts promise</div><div class="value fail">${fmtN(last ? last.cumNaive : 0)}</div><div class="fp-sub">${unit()} newly passing</div></div>
    <div class="card"><div class="label">Expected to pass</div><div class="value pass">${fmtN(last ? last.cumExpected : 0)}</div><div class="fp-sub">80% interval ${fmtN(iv.lo)}–${fmtN(iv.hi)}</div></div>
    <div class="card"><div class="label">Count-driven plan</div><div class="value pct">${fmtN(nExp)}</div><div class="fp-sub">expected from the plan op counts would pick (${fmtN(nIv.lo)}–${fmtN(nIv.hi)})</div></div>
  </div>`;

  // plan table
  const maxV = Math.max(1, ...P.steps.map((s) => s.cumNaive));
  const rows = P.steps.map((s, i) => {
    const prev = i ? P.steps[i - 1].cumExpected : 0;
    const siv = C.interval(recs, s.F, s.surv, { weight: state.weight, seed: 31 + i });
    const w = (v) => `${Math.min(100, (v / maxV) * 100).toFixed(2)}%`;
    return s.gaps.map((g, k) => {
      const gp = gb[g];
      const first = k === 0;
      return `<tr>${first ? `<td class="num" rowspan="${s.gaps.length}">${i + 1}</td>` : ''}<td><span class="fp-mono">${esc(gp.name)}</span></td><td>${esc(gp.stage)}</td><td class="areas">${esc(gp.area)}</td>
      <td><span class="badge" style="background:${EFFORT_COLORS[gp.effort]}">${gp.effort}</span></td>
      ${first ? `<td class="num" rowspan="${s.gaps.length}">${s.cost}</td><td class="num" rowspan="${s.gaps.length}">${fmtN(s.cumNaive)}</td>
      <td rowspan="${s.gaps.length}" class="fp-barcell"><div class="fp-track"><div class="fp-naive" style="width:${w(s.cumNaive)}"></div><div class="fp-iv" style="left:${w(siv.lo)};width:calc(${w(siv.hi)} - ${w(siv.lo)})"></div><div class="fp-exp" style="width:${w(s.cumExpected)}"></div></div></td>
      <td class="num fp-nw" rowspan="${s.gaps.length}"><b>${fmtN(s.cumExpected)}</b> <span class="fp-muted">${fmtN(siv.lo)}–${fmtN(siv.hi)}</span></td>
      <td class="num fp-gain" rowspan="${s.gaps.length}">+${fmtN(s.cumExpected - prev)}</td>` : ''}</tr>`;
    }).join('');
  }).join('');
  html += `<h3>Recommended order</h3>
  <table class="fp-plan"><thead><tr><th style="width:2rem">#</th><th>Fix</th><th>Stage</th><th>Area</th><th>Effort</th><th style="text-align:right">Days</th><th style="text-align:right">Op counts</th><th style="width:17%">Cumulative ${state.weight === 'downloads' ? 'downloads' : 'models'}</th><th style="text-align:right">Expected (80%)</th><th style="text-align:right">Step</th></tr></thead><tbody>${rows}</tbody></table>
  <div class="legend fp-legend"><span class="legend-item"><span class="fp-sw" style="background:#475569"></span>op counts</span><span class="legend-item"><span class="fp-sw" style="background:rgba(34,197,94,.35)"></span>80% interval</span><span class="legend-item"><span class="fp-sw" style="background:#22c55e"></span>expected to pass</span><span class="legend-item fp-muted">Pairs and triples are planned together when models need them closed at once.</span></div>`;

  // backtest
  const tot = wf.reduce((a, r) => ({ n: a.n + r.naive, e: a.e + r.expected, x: a.x + r.actual }), { n: 0, e: 0, x: 0 });
  const covered = wf.filter((r) => r.actual >= r.lo && r.actual <= r.hi).length;
  html += `<h3>Backtest on past Tron releases <span class="fp-muted fp-h3note">each release predicted from the snapshot before it</span></h3>
  <table><thead><tr><th>Release</th><th>Shipped</th><th style="text-align:right">Op counts said</th><th style="text-align:right">Planner (80%)</th><th style="text-align:right">Actually passed</th><th>Of which</th></tr></thead><tbody>
  ${wf.map((r) => `<tr><td class="fp-nw">v${r.from} → v${r.to}<div class="fp-muted">${r.date}</div></td><td class="areas"><span class="fp-mono">${r.fixed.map((g) => esc(gb[g].name)).join(', ')}</span></td><td class="num">${r.naive}</td><td class="num fp-nw"><b>${r.expected.toFixed(0)}</b> <span class="fp-muted">${r.lo.toFixed(0)}–${r.hi.toFixed(0)}</span></td><td class="num">${r.actual} ${r.actual >= r.lo && r.actual <= r.hi ? '<span class="fp-ok">✓</span>' : '<span class="fp-miss">✗</span>'}</td><td class="areas">${r.actualFix} from the release, ${r.actualChurn} from bug-fix churn</td></tr>`).join('')}
  <tr class="fp-total"><td>Total</td><td></td><td class="num">${tot.n}</td><td class="num"><b>${tot.e.toFixed(0)}</b></td><td class="num">${tot.x}</td><td class="areas">${covered} of ${wf.length} releases inside the 80% interval</td></tr></tbody></table>`;

  // hidden-blocker map
  const fits = C.fitHazards(recs, new Set());
  const feats = C.FEATURES.map((f, j) => ({ f: j === 0 ? 'any model (baseline)' : f, j })).slice(1).concat([{ f: 'any model (baseline)', j: 0 }]);
  const cell = (fit, j) => {
    const prior = fit.n[j] < C.MIN_AT_RISK;
    const v = fit.pa[j] / (fit.pa[j] + fit.pb[j]);
    const a = Math.min(0.85, v * 0.9 + 0.04);
    return `<td class="fp-heat${prior ? ' fp-prior' : ''}" style="background:rgba(56,189,248,${prior ? 0 : a.toFixed(3)})" title="${fit.n[j]} at-risk models">${prior ? `<span>${(v * 100).toFixed(0)}%</span>` : `${(v * 100).toFixed(0)}%`}</td>`;
  };
  html += `<h3>Where hidden blockers wait <span class="fp-muted fp-h3note">chance a feature causes a blocker at a stage (noisy-OR, posterior mean)</span></h3>
  <table class="fp-heatmap"><thead><tr><th>Feature</th>${C.STAGES.map((s) => `<th style="text-align:center">${s}</th>`).join('')}</tr></thead><tbody>
  ${feats.map(({ f, j }) => `<tr><td>${esc(f)}</td>${fits.map((fit) => cell(fit, j)).join('')}</tr>`).join('')}</tbody></table>
  <p class="fp-note">Dashed cells: fewer than ${C.MIN_AT_RISK} models with that feature have reached the stage, so the estimate is mostly the prior (Beta(0.2, 0.8), chosen by the backtest above). ${failing} models are failing today; ${recs.filter((r) => r.deep).length} have deep-analysis results that reveal blockers past their first failure.</p>`;

  $('fp-root').innerHTML = html;
  $('fp-root').querySelectorAll('.tag-btn[data-grp]').forEach((b) => b.addEventListener('click', async () => {
    const v = b.dataset.grp === 'budget' ? +b.dataset.v : b.dataset.v;
    if (state[b.dataset.grp] === v) return;
    state[b.dataset.grp] = v;
    $('fp-root').querySelectorAll(`.tag-btn[data-grp="${b.dataset.grp}"]`).forEach((x) => x.classList.toggle('active', x === b));
    $('fp-root').style.opacity = 0.6; await tick(); await tick();
    render(); $('fp-root').style.opacity = 1;
  }));
}

main().catch((e) => { console.error(e); $('fp-root').innerHTML = `<p class="subtitle">Planner failed to load: ${esc(e.message)}</p>`; });
