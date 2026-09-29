from rknnlite.api import RKNNLite
import numpy as np, traceback
rknn = RKNNLite(verbose=False)
rknn.load_rknn('/userdata/zipformer/model/encoder-epoch-99-avg-1.rknn')
rknn.init_runtime()
try:
    out = rknn.inference(inputs=[np.random.randn(1,200,256).astype(np.float32)])
except Exception as e:
    traceback.print_exc()
    print('---')
    print(repr(e))
rknn.release()