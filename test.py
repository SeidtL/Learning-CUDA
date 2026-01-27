import numpy as np 
from pathlib import Path 

with open("hack.log", "r") as _:
    data = _.readlines()

Q = np.load("tmp/55.q.0.npy")
K = np.load("tmp/55.k.0.npy")
S = np.load("tmp/55.s.0.npy")
K = np.repeat(K, 2, axis=2)
print(Q.shape)
print(K.shape)
print(S.shape)
Se = np.einsum("bihd,bjhd->bihj", Q, K)
Se /= np.sqrt(32.0)

def masked_softmax(x):
    B, S1, H, S2 = x.shape
    mask = np.tril(np.ones((S1, S2)))
    mask = mask[None, :, None, :]
    adder = (1.0 - mask) * -1.0e18
    x_masked = x + adder
    
    # 4. 执行数值稳定的 Softmax
    max_val = np.max(x_masked, axis=-1, keepdims=True) # 对 S2 找最大值
    e_x = np.exp(x_masked - max_val)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    
    return e_x / sum_e_x
softmax_se = masked_softmax(Se)
print(softmax_se[0, 0, 0], S[0, 0, -1, 0])

exit(0)

for i in range(64):
    for j in range(i + 1, 64):
        Se[:, i, :, j] = 0
print(S[0, 0, 0, 0], Se[0, 0, 0, 0])
diff = Se - S 
print(S.max(), S.min())
