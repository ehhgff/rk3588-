import json, os

WORK_DIR = "/data/rknn_analysis"
profiles = {
    "vocoder": "matcha_vocoder.onnx_1970-01-01_12-55-22.json",
}

for name, fname in profiles.items():
    path = os.path.join(WORK_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get('traceEvents', [])

    # Filter Node events (kernel execution)
    ops = []
    for e in events:
        if e.get('cat') == 'Node' and 'dur' in e:
            ops.append({
                'name': e.get('name','?'),
                'dur_us': e['dur'],
            })
    ops.sort(key=lambda x: -x['dur_us'])
    total = sum(o['dur_us'] for o in ops)

    print(f"\n=== {name} ({len(ops)} ops, {total}us = {total/1000:.2f}ms) ===")
    print(f"{'操作':<55} {'耗时(us)':<10} {'占比(%)':<8}")
    print("-"*75)
    for o in ops[:20]:
        pct = o['dur_us']/total*100
        print(f"{o['name'][:53]:<55} {o['dur_us']:<10} {pct:.1f}")

    # Group by op type
    from collections import Counter
    op_types = Counter()
    for o in ops:
        t = o['name'].rsplit('/',1)[-1].replace('_kernel_time','')
        op_types[t] += o['dur_us']
    print(f"\n按操作类型汇总:")
    for t, d in op_types.most_common(10):
        print(f"  {t:<30} {d:>8}us ({d/total*100:.1f}%)")