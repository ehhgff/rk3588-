#!/usr/bin/env python3
"""Check the model's phoneme_id_map for tone-related phonemes."""
import json, sys

with open('/data/paroli/model/zh_CN-huayan-medium.onnx.json') as f:
    d = json.load(f)

pm = d['phoneme_id_map']

# Show all phonemes that contain digits
print("Tone-related phonemes:")
for k in sorted(pm.keys()):
    if any(c.isdigit() for c in k):
        print('  {!r}: {}'.format(k, pm[k]))

print()
# Show stress marks
print("Stress marks:")
for k in sorted(pm.keys()):
    if k in "'ˈˌ":
        print('  {!r}: {}'.format(k, pm[k]))

# Show all s, S, sh related phonemes
print()
print("s/sh related phonemes:")
for k in sorted(pm.keys()):
    if k.startswith('s') or k.startswith('S') or k.startswith('z'):
        print('  {!r}: {}'.format(k, pm[k]))