from rknnlite.api import RKNNLite
import numpy as np

print('=== ASR Encoder ===')
rknn = RKNNLite(verbose=False)
rknn.load_rknn('/userdata/zipformer/model/encoder-epoch-99-avg-1.rknn')
rknn.init_runtime()

# Try 3 inputs: 3D + 2D + ? or 3D + 2D + 2D
# input[0]=3D, input[1]=2D
for shape in [(1,200,256), (1,100,256), (1,50,256)]:
    for d2 in [256, 512, 1024]:
        for d3 in [64, 128, 256, 512]:
            try:
                out = rknn.inference(inputs=[
                    np.random.randn(*shape).astype(np.float32),
                    np.random.randn(1, d2).astype(np.float32),
                    np.random.randn(1, d3).astype(np.float32)
                ])
                print(f'  OK: 3D{shape} + 2D({d2}) + 2D({d3}) -> {[o.shape for o in out]}')
                break
            except Exception as e:
                err = str(e)[:100]
        else:
            continue
        break
    else:
        continue
    break
else:
    print('  3 inputs (3D+2D+2D) not working')

# Try 4 inputs: 3D + 2D + 2D + 2D
for shape in [(1,200,256), (1,100,256)]:
    for d2 in [256, 512]:
        for d3 in [256, 512]:
            for d4 in [256, 512]:
                try:
                    out = rknn.inference(inputs=[
                        np.random.randn(*shape).astype(np.float32),
                        np.random.randn(1, d2).astype(np.float32),
                        np.random.randn(1, d3).astype(np.float32),
                        np.random.randn(1, d4).astype(np.float32)
                    ])
                    print(f'  OK: 4 inputs -> {[o.shape for o in out]}')
                    break
                except Exception as e:
                    err = str(e)[:100]
            else:
                continue
            break
        else:
            continue
        break
    else:
        continue
    break
else:
    print('  4 inputs not working either')

rknn.release()