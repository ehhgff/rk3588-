from rknnlite.api import RKNNLite
rknn = RKNNLite(verbose=False)
ret = rknn.load_rknn('/userdata/sensevoice/sensevoice_encoder_ctc_int8.rknn')
print('load_rknn:', ret)
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
print('init_runtime:', ret)
import numpy as np
# Try different input sizes based on error: 1120000 bytes = 280000 floats
# 280000 = 1*700*400 or 1*350*800 or similar
for h in [700, 350, 280, 200]:
    w = 280000 // h
    if w * h == 280000:
        print(f'Trying shape (1,{h},{w}) = {1*h*w} elements')
        dummy = np.random.randn(1, h, w).astype(np.float32)
        try:
            out = rknn.inference(inputs=[dummy])
            print(f'  SUCCESS! output shape: {[o.shape for o in out]}')
            break
        except Exception as e:
            err = str(e)[:100]
            print(f'  Failed: {err}')
rknn.release()