"""Build data/hub_snapshot.json from public Hugging Face Hub metadata.

Inputs (fetched once with curl, not committed): the Hub API model lists
  https://huggingface.co/api/models?pipeline_tag=text-generation&library=transformers&sort=trendingScore&direction=-1&limit=400
  https://huggingface.co/api/models?pipeline_tag=text-generation&sort=downloads&direction=-1&limit=500
and each model's public config.json. Tags use the same rules as hf_litmus.model_tags._tags_from_config_attrs.
Usage: python3 scripts/snapshot.py <trend.json> <downloads.json> <cfg_dir> data/hub_snapshot.json
"""
import json, os, sys

def tags_from(cfg):
    g = lambda k, d=None: cfg.get(k, d)
    tags = []
    nh = g('num_attention_heads', 0) or 0
    nkv = g('num_key_value_heads', 0) or 0
    if g('kv_lora_rank'): tags.append('MLA')
    elif nkv and nkv == 1: tags.append('MQA')
    elif nkv and nkv < nh: tags.append('GQA')
    elif nh: tags.append('MHA')
    if g('num_local_experts'): tags.append('MoE')
    rope = g('rope_scaling')
    if isinstance(rope, dict):
        rt = (rope.get('type') or '').lower()
        if 'yarn' in rt: tags.append('YaRN')
        elif 'linear' in rt: tags.append('LinearRoPE')
        else: tags.append('RoPE')
    elif g('rope_theta'): tags.append('RoPE')
    if g('sliding_window'): tags.append('SlidingWindow')
    return sorted(tags)

HYBRID_TYPES = {'nemotron_h', 'lfm2', 'lfm3', 'lfm2_moe', 'bailing_hybrid', 'qwen3_next', 'jamba', 'falcon_h1', 'mamba', 'mamba2', 'rwkv', 'zamba2', 'granitemoehybrid'}

def main(trend_path, dl_path, cfg_dir, out_path):
    lists = json.load(open(trend_path)) + json.load(open(dl_path))
    out, seen = [], set()
    for m in lists:
        mid = m['id']
        if mid in seen: continue
        try: c = json.load(open(os.path.join(cfg_dir, mid.replace('/', '__') + '.json')))
        except Exception: continue
        if not isinstance(c, dict): continue
        tc = c.get('text_config') if isinstance(c.get('text_config'), dict) else None
        base = {**c, **tc} if tc else dict(c)
        mt = c.get('model_type') or (tc or {}).get('model_type')
        if not mt: continue
        seen.add(mid)
        nexp = base.get('num_local_experts') or base.get('num_experts') or base.get('n_routed_experts') or 0
        lt = base.get('layer_types') or []
        if not isinstance(lt, list): lt = []
        q = c.get('quantization_config')
        feat = {
            'moe': bool(nexp and nexp > 1), 'n_experts': int(nexp or 0),
            'mla': bool(base.get('kv_lora_rank')),
            'sliding': bool(base.get('sliding_window')) or any('sliding' in str(x) for x in lt),
            'hybrid': mt in HYBRID_TYPES or any(any(k in str(x) for k in ('mamba', 'linear', 'conv', 'ssm', 'delta')) for x in lt),
            'vision': 'vision_config' in c,
            'yarn': isinstance(base.get('rope_scaling'), dict) and 'yarn' in json.dumps(base['rope_scaling']).lower(),
            'softcap': bool(base.get('attn_logit_softcapping') or base.get('final_logit_softcapping')),
            'quant': (q.get('quant_method') if isinstance(q, dict) else None),
            'vocab': int(base.get('vocab_size') or 0),
            'layers': int(base.get('num_hidden_layers') or 0),
        }
        out.append({'id': mid, 'downloads': int(m.get('downloads', 0) or 0), 'likes': int(m.get('likes', 0) or 0),
                    'pipeline_tag': m.get('pipeline_tag') or 'text-generation', 'model_type': mt,
                    'arch': (c.get('architectures') or [''])[0], 'tags': tags_from(base), 'feat': feat})
    json.dump({'source': 'Hugging Face Hub public API and config.json files', 'fetched': '2026-11-10', 'models': out}, open(out_path, 'w'), separators=(',', ':'))
    print(len(out), 'models')

if __name__ == '__main__':
    main(*sys.argv[1:5])
