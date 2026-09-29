#!/usr/bin/env python3
"""Parse cmn_dict to find entries for 是 (U+662F) - focus on standalone entry."""
import struct, sys

with open('cmn_dict', 'rb') as f:
    data = f.read()

# The standalone entry for 是 appears around offset 0xdc7d4
# Format: \n\x03 + char_utf8 + phoneme_string + \x00
# Look at the region around this offset for more context
print("=== Standalone entry area for 是 ===")
chunk = data[0xdc7c0:0xdc800]
for i in range(0, len(chunk), 16):
    hex_str = ' '.join('{:02x}'.format(b) for b in chunk[i:i+16])
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk[i:i+16])
    addr = 0xdc7c0 + i
    print('{}: {}  {}'.format(hex(addr), hex_str, ascii_str))

print()
print("=== Standalone entry at 0xdc7d4 decoded ===")
# The entry: \n\x03 + 是(E6 98 AF) + shi4 + \x00
entry = data[0xdc7d2:0xdc7e8]
print('Raw bytes:', entry.hex())
print('Bytes:', entry)

# Let's also look at the "不是" entry in the 不 character's section
# Find 不 (E4 B8 8D) and look at its compound entries
target = b'\xe4\xb8\x8d'  # 不
pos = 0
bu_entries = []
while True:
    pos = data.find(target, pos)
    if pos < 0:
        break
    # Check context for 不是 (bu2shi4)
    end = min(len(data), pos + 30)
    ctx = data[pos:end]
    if b'shi' in ctx:
        bu_entries.append((pos, ctx))
    pos += 1

print()
print("=== Entries containing 不...shi... (可能是'不是') ===")
for pos, ctx in bu_entries:
    print('{}: {}'.format(hex(pos), repr(ctx)))
PYEOF