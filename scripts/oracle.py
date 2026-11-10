"""Ground-truth check for a fix set: how many models failing at the current release would
pass if Tron also shipped the given gap ids (model-specific bugs unchanged).
Uses the hidden rules in scripts/generate.py, which the planner never sees.
Usage: python3 scripts/oracle.py P1,G9,...   -> prints JSON {"models": n, "downloads": d}
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G

def main(ids):
    F = set(x for x in ids.split(',') if x)
    models = json.load(open(os.path.join(G.ROOT, 'data/hub_snapshot.json')))['models']
    lit = json.load(open(os.path.join(G.ROOT, 'data/litmus.json')))
    cur = lit['current']
    rng = G.random.Random(20261110)
    # replay the generator's model-specific bug draws exactly
    idio = {}
    for i, m in enumerate(models):
        if rng.random() < 0.16:
            stage = rng.choices(G.STAGES[1:], weights=[4, 3, 2, 2, 2, 1])[0]
            fixed = rng.choice(['0.32', '0.33', '0.34', '0.35', '0.36', None, None, None])
            idio[i] = (stage, fixed)
    remote = {i for i, m in enumerate(models) if G.h(m['id'] + 'remote', 20) == 0}
    n = d = 0
    for o in lit['snapshots'][cur]:
        i = o['i']
        if o['status'] == 'SUCCESS' or i in remote: continue
        m = models[i]
        open_gaps = [g for g in G.GAPS if g[7](m) and (g[6] is None or G.vnum(g[6]) > G.vnum(cur)) and g[0] not in F]
        bug = idio.get(i)
        bug_live = bug and (bug[1] is None or G.vnum(bug[1]) > G.vnum(cur))
        if not open_gaps and not bug_live:
            n += 1; d += m['downloads']
    print(json.dumps({'models': n, 'downloads': d}))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '')
