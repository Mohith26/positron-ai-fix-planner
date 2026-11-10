"""Synthetic hf-litmus history on real Hugging Face Hub models.

The model list, downloads and architecture features are real (data/hub_snapshot.json).
Which Tron stage each model fails at is SYNTHETIC: a hidden catalog of compiler gaps
with matching rules, fixed over six simulated Tron releases. The planner never reads
the rules below; it only sees what hf-litmus would report (first failing stage, the
missing ops or patterns at that stage, and deeper gaps for deep-analysed models).

Usage: python3 scripts/generate.py   (writes data/gaps.json, data/litmus.json)
"""
import hashlib, json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ['export', 'fxtypedfx', 'rewrite', 'bulk', 'loopy', 'tron', 'cpp']
VERSIONS = [('0.31', '2026-07-07'), ('0.32', '2026-07-28'), ('0.33', '2026-08-18'),
            ('0.34', '2026-09-08'), ('0.35', '2026-10-06'), ('0.36', '2026-10-27')]

def h(s, mod):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % mod

# (id, name, kind, stage, area, effort, fixed_in, rule)  kind: op | pattern | kernel | export
GAPS = [
    ('G1', 'aten.scatter_add.default', 'op', 'fxtypedfx', 'MoE', 'medium', '0.34', lambda m: m['feat']['moe']),
    ('G2', 'aten.topk.default', 'op', 'fxtypedfx', 'sorting', 'small', '0.33', lambda m: m['feat']['moe']),
    ('G3', 'aten.index_put.default', 'op', 'fxtypedfx', 'MoE', 'medium', '0.36', lambda m: m['feat']['moe'] and h(m['model_type'], 2) == 0),
    ('G4', 'aten.cumsum.default', 'op', 'fxtypedfx', 'scan_ops', 'medium', None, lambda m: m['feat']['hybrid']),
    ('G5', 'aten.conv1d.default', 'op', 'fxtypedfx', 'convolution', 'medium', None, lambda m: m['feat']['hybrid'] and h(m['model_type'] + 'c', 3) != 0),
    ('G6', 'aten.repeat_interleave.self_int', 'op', 'fxtypedfx', 'attention_variants', 'small', '0.35', lambda m: 'GQA' in m['tags'] and h(m['model_type'], 3) == 0),
    ('G7', 'aten.masked_fill.Scalar', 'op', 'fxtypedfx', 'conditional_ops', 'small', '0.33', lambda m: m['feat']['sliding']),
    ('G8', 'aten.tanh.default', 'op', 'fxtypedfx', 'activation', 'small', '0.35', lambda m: m['feat']['softcap']),
    ('G9', 'aten.roll.default', 'op', 'fxtypedfx', 'position_encoding', 'small', None, lambda m: h(m['id'], 13) == 0),
    ('G10', 'aten.new_ones.default', 'op', 'fxtypedfx', 'tensor_construction', 'small', '0.32', lambda m: h(m['model_type'] + 'n', 6) == 0),
    ('G11', 'aten.triu.default', 'op', 'fxtypedfx', 'causal_mask', 'small', '0.32', lambda m: not m['feat']['sliding'] and h(m['model_type'] + 't', 4) == 0),
    ('G12', 'aten.one_hot.default', 'op', 'fxtypedfx', 'embedding', 'small', None, lambda m: m['feat']['moe'] and m['feat']['n_experts'] >= 128),
    ('P1', 'sliding_window_attention', 'pattern', 'rewrite', 'sliding_window_attention', 'medium', None, lambda m: m['feat']['sliding']),
    ('P2', 'yarn_rope_scaling', 'pattern', 'rewrite', 'yarn_rope_scaling', 'small', '0.35', lambda m: m['feat']['yarn']),
    ('P3', 'mla_latent_attention', 'pattern', 'rewrite', 'mla_latent_attention', 'large', None, lambda m: m['feat']['mla']),
    ('P4', 'qk_norm_fusion', 'pattern', 'rewrite', 'qk_norm_fusion', 'small', '0.33', lambda m: m['model_type'].startswith(('qwen3', 'gemma'))),
    ('P5', 'gated_delta_rule', 'pattern', 'rewrite', 'gated_delta_rule', 'large', None, lambda m: m['feat']['hybrid'] and m['model_type'].startswith(('qwen3_5', 'qwen3_next'))),
    ('B1', 'vision_patch_embed', 'kernel', 'bulk', 'vision_patch_embed', 'medium', '0.36', lambda m: m['feat']['vision']),
    ('L1', 'linear_attention_recurrence', 'kernel', 'loopy', 'linear_attention_recurrence', 'large', None, lambda m: m['feat']['hybrid']),
    ('L2', 'sliding_kv_ring_buffer', 'kernel', 'loopy', 'sliding_kv_ring_buffer', 'medium', None, lambda m: m['feat']['sliding'] and h(m['model_type'] + 'r', 2) == 0),
    ('T1', 'moe_expert_dispatch', 'kernel', 'tron', 'moe_expert_dispatch', 'large', None, lambda m: m['feat']['moe'] and m['feat']['n_experts'] > 64),
    ('T2', 'mla_kv_decompress', 'kernel', 'tron', 'mla_kv_decompress', 'medium', None, lambda m: m['feat']['mla']),
    ('C1', 'fp8_block_dequant', 'kernel', 'cpp', 'fp8_block_dequant', 'medium', '0.36', lambda m: m['feat']['quant'] in ('fp8', 'compressed-tensors', 'modelopt', 'mxfp8')),
    ('C2', 'mxfp4_dequant', 'kernel', 'cpp', 'mxfp4_dequant', 'medium', None, lambda m: m['feat']['quant'] == 'mxfp4'),
    ('C3', 'large_vocab_logits_split', 'kernel', 'cpp', 'large_vocab_logits_split', 'small', '0.34', lambda m: m['feat']['vocab'] > 200000),
    ('C4', 'awq_int4_dequant', 'kernel', 'cpp', 'awq_int4_dequant', 'medium', None, lambda m: m['feat']['quant'] in ('awq', 'gptq')),
    ('X1', 'data-dependent expert routing', 'export', 'export', 'dynamic_shape', 'large', '0.34', lambda m: m['feat']['moe'] and h(m['model_type'] + 'x', 3) != 0),
]

def vnum(v):
    return tuple(int(x) for x in v.split('.'))

def main():
    rng = random.Random(20261110)
    models = json.load(open(os.path.join(ROOT, 'data/hub_snapshot.json')))['models']
    # Model-specific bugs (type errors) that no catalog fix resolves; each is fixed by
    # ordinary bug churn in some release (or never).
    idio = {}
    for i, m in enumerate(models):
        if rng.random() < 0.16:
            stage = rng.choices(STAGES[1:], weights=[4, 3, 2, 2, 2, 1])[0]
            fixed = rng.choice(['0.32', '0.33', '0.34', '0.35', '0.36', None, None, None])
            idio[i] = (stage, fixed)
    remote = {i for i, m in enumerate(models) if h(m['id'] + 'remote', 20) == 0}

    true_blockers = []
    for i, m in enumerate(models):
        bl = [g for g in GAPS if g[7](m)]
        true_blockers.append(bl)

    snapshots = {}
    for v, date in VERSIONS:
        obs = []
        failing_rank = []
        for i, m in enumerate(models):
            if i in remote:
                obs.append({'i': i, 'status': 'EXPORT_FAIL', 'stage': 'export', 'cls': 'trust_remote_code', 'seen': [], 'retryable': False})
                continue
            unfixed = [g for g in true_blockers[i] if g[6] is None or vnum(g[6]) > vnum(v)]
            stage_of = {}
            for g in unfixed:
                stage_of.setdefault(g[3], []).append(g)
            id_bug = idio.get(i)
            if id_bug and (id_bug[1] is None or vnum(id_bug[1]) > vnum(v)):
                stage_of.setdefault(id_bug[0], []).append(None)
            fail = next((s for s in STAGES if s in stage_of), None)
            if fail is None:
                obs.append({'i': i, 'status': 'SUCCESS', 'stage': 'success', 'cls': None, 'seen': []})
                continue
            here = stage_of[fail]
            seen = [g[0] for g in here if g is not None]
            if fail == 'export':
                cls = 'unsupported_dynamic'
            elif seen and any(g is not None and g[2] == 'op' for g in here):
                cls = 'missing_op'
            elif seen:
                cls = 'missing_op'
            else:
                cls = 'type_error'
            o = {'i': i, 'status': 'EXPORT_FAIL' if fail == 'export' else 'INGEST_FAIL', 'stage': fail, 'cls': cls, 'seen': seen,
                 'retryable': False}
            obs.append(o)
            failing_rank.append((m['downloads'], i))
        # Deep analysis on the most-downloaded quarter of failing models: Claude Code pushes
        # past the first failure and records gaps until it hits a stage it cannot patch around
        # (a kernel/pattern gap, or a model-specific bug).
        failing_rank.sort(reverse=True)
        deep = {i for _, i in failing_rank[: len(failing_rank) // 4]}
        for o in obs:
            if o['i'] not in deep or o['status'] == 'SUCCESS' or o['cls'] == 'trust_remote_code':
                continue
            i = o['i']
            unfixed = [g for g in true_blockers[i] if g[6] is None or vnum(g[6]) > vnum(v)]
            id_bug = idio.get(i)
            id_live = id_bug and (id_bug[1] is None or vnum(id_bug[1]) > vnum(v))
            revealed, furthest, complete, bug = [], o['stage'], True, False
            for s in STAGES[STAGES.index(o['stage']):]:
                here = [g for g in unfixed if g[3] == s]
                revealed += [g[0] for g in here]
                furthest = s
                bug_here = bool(id_live and id_bug[0] == s)
                if bug_here or any(g[2] in ('kernel', 'pattern') for g in here):
                    complete, bug = False, bug_here
                    break
            # complete: analysis patched through every stage, so no blockers remain beyond `furthest`.
            # bug: it stopped on a model-specific bug it could not name.
            o['deep'] = {'furthest': furthest, 'revealed': sorted(set(revealed) - set(o['seen'])), 'complete': complete, 'bug': bug}
        snapshots[v] = obs

    gaps = [{'id': g[0], 'name': g[1], 'kind': g[2], 'stage': g[3], 'area': g[4], 'effort': g[5], 'fixed_in': g[6]} for g in GAPS]
    json.dump({'stages': STAGES, 'gaps': gaps}, open(os.path.join(ROOT, 'data/gaps.json'), 'w'), indent=1)
    json.dump({'versions': [{'version': v, 'date': d, 'fixed': [g[0] for g in GAPS if g[6] == v]} for v, d in VERSIONS],
               'current': VERSIONS[-1][0], 'snapshots': snapshots}, open(os.path.join(ROOT, 'data/litmus.json'), 'w'), separators=(',', ':'))
    cur = snapshots[VERSIONS[-1][0]]
    for v, _ in VERSIONS:
        s = snapshots[v]
        print(v, 'pass', sum(o['status'] == 'SUCCESS' for o in s), 'of', len(s))
    from collections import Counter
    print('current stages', Counter(o['stage'] for o in cur))
    print('current classes', Counter(o['cls'] for o in cur))

if __name__ == '__main__':
    main()
