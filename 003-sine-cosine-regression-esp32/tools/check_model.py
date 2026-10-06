from pathlib import Path
import re
import hashlib

root = Path(__file__).resolve().parents[1]
header = (root / 'firmware/include/model_data.h').read_text()
array = header[header.index('{') + 1:header.index('}')]
embedded = bytes(int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{2})', array))
model = (root / 'models/sinus_cosinus_int8_arduino.tflite').read_bytes()
assert embedded == model, 'Header and TFLite bytes differ'
assert len(model) == 3816 and model[4:8] == b'TFL3'
h = 2166136261
for b in model:
    h = ((h ^ b) * 16777619) & 0xFFFFFFFF
assert h == 0xB64E1E67
print(f'OK: {len(model)} bytes; FNV-1a={h:08X}')
print('SHA256:', hashlib.sha256(model).hexdigest())
