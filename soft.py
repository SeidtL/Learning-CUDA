import numpy as np 
import sys 
idx = int(sys.argv[1])

name_fmt = "tmp/{}.{}.0.npy"

q = np.load(name_fmt.format(idx, "q"))
k = np.load(name_fmt.format(idx, "k"))
k = np.repeat(k, 2, axis=2)
v = np.load(name_fmt.format(idx, "v"))
v = np.repeat(v, 2, axis=2)
o = np.load(name_fmt.format(idx, "o"))
s = np.load(name_fmt.format(idx, "s"))
print(q.shape)
print(k.shape)
print(s.shape)

def check(q, k, s, v, o):
    p = np.matmul(q, np.transpose(k)) / np.sqrt(32.0)
    p = np.tril(p, k=0)
    mask = np.tril(np.ones_like(p), k=0)

    max_diff = 0.0
    row_max = np.max(np.where(mask, p, -np.inf), axis=1)
    sub = row_max.reshape(-1, 1) * mask
    exp_up = np.tril(np.exp(p - sub))
    row_sum = np.sum(exp_up, axis=1, keepdims=True)

    max_diff_l = 0.0
    max_diff_s = 0.0
    for i in range(64):
        max_diff_l = max(max_diff_l, abs(row_sum[i, 0] - s[i, -1]))
        max_diff = max(max_diff, abs(row_sum[i, 0] - s[i, -1]))
        for j in range(64):
            max_diff_s = max(max_diff_s, abs(exp_up[i, j] - s[i, j]))
            max_diff = max(max_diff, abs(exp_up[i, j] - s[i, j]))

    exp_up /= row_sum
    # print("-" * 30)
    # print(s)
    # exit(0)
    O = np.matmul(exp_up, v)

    for i in range(64):
        for j in range(32):
            max_diff = max(max_diff, abs(O[i, j] - o[i, j]))
            # print(O[i, j], o[i, j])
    # print(max_diff)
    return max_diff, max_diff_l, max_diff_s

max_diff = 0.0
max_diff_l = 0.0
max_diff_s = 0.0
for i in range(4):
    for j in range(64):
        it, it_l, it_s = check(q[i, :, j, :], k[i, :, j, :], s[i, :, j, :], v[i, :, j, :], o[i, :, j, :])
        max_diff = max(it, max_diff)
        max_diff_l = max(it_l, max_diff_l)
        max_diff_s = max(it_s, max_diff_s)

print(max_diff)
print(max_diff_l)
print(max_diff_s)
