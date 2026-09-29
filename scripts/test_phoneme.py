#!/usr/bin/env python3
"""Test espeak-ng phonemization for Chinese characters."""
import ctypes
import json

lib_path = '/data/paroli/lib/libespeak-ng.so.1.52.0.1'
data_path = '/userdata/paroli/bin/espeak-ng-data'
model_config = '/data/paroli/model/zh_CN-huayan-medium.onnx.json'

# Load espeak-ng library
lib = ctypes.CDLL(lib_path)

# Initialize espeak
lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
lib.espeak_Initialize.restype = ctypes.c_int

# AUDIO_OUTPUT_SYNCHRONOUS = 0
result = lib.espeak_Initialize(0, 0, data_path.encode(), 0)
print(f'espeak_Initialize: {result}')

# Set voice to Chinese
lib.espeak_SetVoiceByName.argtypes = [ctypes.POINTER(ctypes.c_char)]
lib.espeak_SetVoiceByName.restype = ctypes.c_int

voice_result = lib.espeak_SetVoiceByName(b'cmn')
print(f'espeak_SetVoiceByName(cmn): {voice_result}')

# Now call espeak_TextToPhonemesWithTerminator
# espeak_TextToPhonemesWithTerminator(text, textmode, terminator, phonememode, phoneme_ipa_lengths)
# phonememode = 2 (phoneme as IPA)
lib.espeak_TextToPhonemesWithTerminator.restype = ctypes.POINTER(ctypes.c_short)

# First test: "是"
text = '是'
text_bytes = text.encode('utf-8')
print(f'\nTesting: "{text}" ({text.encode("utf-8").hex()})')

# Call with phonememode=2 for IPA-like output
phoneme_ptr = lib.espeak_TextToPhonemesWithTerminator(
    ctypes.c_char_p(text_bytes),
    ctypes.c_int(0),  # textmode
    ctypes.c_int(0),  # terminator (0 = no terminator)
    ctypes.c_int(2),  # phonememode (2 = IPA)
    None  # phoneme_ipa_lengths
)

# Read the phoneme array (null-terminated)
if phoneme_ptr:
    phonemes = []
    i = 0
    while True:
        val = phoneme_ptr[i]
        if val == 0:
            break
        phonemes.append(chr(val))
        i += 1
    phoneme_str = ''.join(phonemes)
    print(f'  Phonemes (mode=2): {phoneme_str}')
    print(f'  Codepoints: {[hex(ord(c)) for c in phoneme_str]}')
else:
    print('  No phonemes returned')

# Try with phonememode = 1 (unicode phonemes without IPA grouping)
phoneme_ptr2 = lib.espeak_TextToPhonemesWithTerminator(
    ctypes.c_char_p(text_bytes),
    ctypes.c_int(0),
    ctypes.c_int(0),
    ctypes.c_int(1),
    None
)

if phoneme_ptr2:
    phonemes2 = []
    i = 0
    while True:
        val = phoneme_ptr2[i]
        if val == 0:
            break
        phonemes2.append(chr(val))
        i += 1
    phoneme_str2 = ''.join(phonemes2)
    print(f'  Phonemes (mode=1): {phoneme_str2}')
    print(f'  Codepoints: {[hex(ord(c)) for c in phoneme_str2]}')

# Try with phonememode = 0 (as espeak phoneme mnemonics)
phoneme_ptr3 = lib.espeak_TextToPhonemesWithTerminator(
    ctypes.c_char_p(text_bytes),
    ctypes.c_int(0),
    ctypes.c_int(0),
    ctypes.c_int(0),
    None
)

if phoneme_ptr3:
    phonemes3 = []
    i = 0
    while True:
        val = phoneme_ptr3[i]
        if val == 0:
            break
        phonemes3.append(chr(val))
        i += 1
    phoneme_str3 = ''.join(phonemes3)
    print(f'  Phonemes (mode=0): {phoneme_str3}')
    print(f'  Codepoints: {[hex(ord(c)) for c in phoneme_str3]}')

lib.espeak_Terminate()
print('\nespeak_Terminate() done')