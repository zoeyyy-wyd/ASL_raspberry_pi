import numpy as np
import json
from ai_edge_litert.interpreter import Interpreter

# 先把 tensorflow 那两行删掉或注释掉
# 加载 PC 录的序列

seq = np.load("test_sequence_pc.npy").astype(np.float32)
print(f"Loaded shape: {seq.shape}")
print(f"NaN ratio: {np.isnan(seq).mean():.3f}")

# 加载模型
interp = Interpreter(model_path="model.tflite", num_threads=4)
interp.allocate_tensors()
ind = interp.get_input_details()
outd = interp.get_output_details()
print(f"Input dtype: {ind[0]['dtype']}")

# 推理
interp.resize_tensor_input(ind[0]['index'], list(seq.shape))
interp.allocate_tensors()

# seq = np.nan_to_num(seq, nan=0.0)  # ← 加这一行
# print(f"NaN ratio after nan_to_num: {np.isnan(seq).mean():.3f}")  # 应该是 0.000

interp.set_tensor(ind[0]['index'], seq)
interp.invoke()
logits = interp.get_tensor(outd[0]['index']).flatten()

probs = np.exp(logits - logits.max())
probs = probs / probs.sum()

with open("sign_to_prediction_index_map.json") as f:
    sign_to_idx = json.load(f)
idx_to_sign = {v: k for k, v in sign_to_idx.items()}

print("\nTop-5 predictions:")
top = np.argsort(probs)[-5:][::-1]
for i in top:
    print(f"  {idx_to_sign[int(i)]:25s}  {probs[i]*100:5.2f}%")