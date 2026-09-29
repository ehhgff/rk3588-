#!/usr/bin/env python3
"""Parse cmn_dict to find entries for 是 (U+662F)."""
import struct

with open('cmn_dict', 'rb') as f:
    data = f.read()

target = b'\xe6\x98\xaf'  # 是 in UTF-8
pos = 0
entries = {}
while True:
    pos = data.find(target, pos)
    if pos < 0:
        break
    if pos >= 1:
        null_before = data.rfind(b'\x00', max(0, pos - 50), pos)
        if null_before >= 0 and null_before < pos:
            phoneme_str = data[null_before + 1:pos - 1]
            flag = data[pos - 1:pos]
            key = (pos, flag[0])
            if key not in entries:
                start = max(0, pos - 4)
                end = min(len(data), pos + 20)
                ctx = data[start:end]
                entries[key] = (phoneme_str, ctx)
    pos += 1

print('All entries for 是 (offset, flag, phoneme_str, context):')
for (offset, flag_byte), (phoneme_str, ctx) in sorted(entries.items()):
    if 32 <= flag_byte < 127:
        flag_char = chr(flag_byte)
    else:
        flag_char = '\\x{:02x}'.format(flag_byte)
    print('{}: flag={}, phoneme={}'.format(hex(offset), flag_char, phoneme_str))
    print('  ctx: {}'.format(ctx))
    print()