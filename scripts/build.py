"""Render the page: hf-litmus's own dashboard renderer (vendored, BSD-3) on the synthetic
current snapshot, with the Fix Planner section injected after the summary cards.

Usage: python3 scripts/build.py   (writes index.html)
"""
import json, os, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'vendor'))
from hf_litmus import dashboard as D  # noqa: E402
from hf_litmus.feature_areas import classify_op_area, compute_model_assessment  # noqa: E402
from hf_litmus.models import FailureClass, FailureStage, ModelResult, ModelStatus  # noqa: E402

SEVERITY = {'large': 'critical', 'medium': 'high', 'small': 'medium'}

def reports_for(snapshot, models, gaps, version):
    out = []
    for o in snapshot:
        m = models[o['i']]
        seen = [gaps[g] for g in o['seen']]
        ops = [g['name'] for g in seen]
        status = ModelStatus(o['status'])
        res = ModelResult(
            model_id=m['id'], status=status,
            failure_stage=None if status == ModelStatus.SUCCESS else (FailureStage.EXPORT if o['stage'] == 'export' else FailureStage.INGEST),
            failure_class=FailureClass(o['cls']) if o['cls'] else None,
            missing_ops=ops, ingest_version=version, pipeline_tag=m['pipeline_tag'],
            downloads=m['downloads'], likes=m['likes'], retryable=o.get('retryable', False),
            model_tags=m['tags'], tested_at=datetime(2026, 10, 28, tzinfo=timezone.utc))
        gap_data = None
        if o.get('deep'):
            rev = [gaps[g] for g in o['deep']['revealed']]
            allg = seen + rev
            gap_data = {
                'furthest_stage': o['deep']['furthest'],
                'missing_ops': [g['name'] for g in allg if g['kind'] == 'op'],
                'missing_patterns': [g['name'] for g in allg if g['kind'] == 'pattern'],
                'missing_kernels': [g['name'] for g in allg if g['kind'] == 'kernel'],
                'blockers': [{'description': g['name'], 'severity': SEVERITY[g['effort']], 'effort': g['effort']} for g in allg],
            }
        assessment = compute_model_assessment(res, gap_data)
        areas = set()
        for g in seen:
            areas.add(classify_op_area(g['name']) if g['kind'] == 'op' else g['area'])
        if gap_data:
            areas.update(gap_data['missing_patterns'])
            areas.update(gap_data['missing_kernels'])
        areas.discard('other')
        out.append({
            'model_id': m['id'], 'pipeline_tag': m['pipeline_tag'], 'downloads': m['downloads'], 'likes': m['likes'],
            'status': o['status'], 'failure_stage': res.failure_stage.value if res.failure_stage else None,
            'failure_class': o['cls'], 'retryable': o.get('retryable', False), 'model_tags': m['tags'],
            'ingest_version': version, 'assessment': assessment,
            'platform_compatibility': {'missing_ops': ops, 'missing_feature_areas': sorted(areas)},
        })
    return out

def main():
    models = json.load(open(os.path.join(ROOT, 'data/hub_snapshot.json')))['models']
    gaps = {g['id']: g for g in json.load(open(os.path.join(ROOT, 'data/gaps.json')))['gaps']}
    lit = json.load(open(os.path.join(ROOT, 'data/litmus.json')))
    reports = reports_for(lit['snapshots'][lit['current']], models, gaps, lit['current'])
    data = D._compute_dashboard(reports, 'verdict', None)
    html = D._render_html(data)
    # Static hosting: the dashboard's /api endpoints do not exist, so answer them locally.
    shim = ("<script>(function(){var f=window.fetch;window.fetch=function(u,o){if(typeof u==='string'&&u.indexOf('/api/')===0){"
            "var body=u.indexOf('/api/analyze/status')===0?{running:false,lines:[],offset:0}:u.indexOf('/api/models/search')===0?[]:"
            "{error:'Static demo: litmus server actions are disabled'};return Promise.resolve(new Response(JSON.stringify(body),"
            "{status:200,headers:{'Content-Type':'application/json'}}));}return f.apply(this,arguments);};})();</script>")
    html = html.replace('<script>var LITMUS_TOKEN', shim + '\n<script>var LITMUS_TOKEN', 1)
    html = html.replace('<title>HF Litmus Dashboard</title>', '<title>HF Litmus Dashboard · Fix Planner (concept)</title>\n<link rel="stylesheet" href="planner.css">', 1)
    marker = '<section>\n<h2>Supported Models</h2>'
    assert marker in html
    html = html.replace(marker, '<section id="fix-planner"><h2>Fix Planner <span class="fp-new">new</span></h2><div id="fp-root"><p class="subtitle">Loading planner…</p></div></section>\n\n' + marker, 1)
    html = html.replace('</body>', '<footer class="fp-footer">Unofficial concept demo by Mohith Gajjela. Not affiliated with Positron AI. Model list, downloads and configs are public Hugging Face Hub data; litmus results are synthetic. Dashboard markup rendered by hf-litmus (BSD-3-Clause, &copy; 2026 John Wiegley).</footer>\n<script type="module" src="app.js"></script>\n</body>', 1)
    open(os.path.join(ROOT, 'index.html'), 'w').write(html)
    print('reports', len(reports), 'pass', data['pass_count'], 'fail', data['fail_count'], 'verdicts', dict(data['verdict_counts']))

if __name__ == '__main__':
    main()
