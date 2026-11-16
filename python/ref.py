"""用 numpy 独立实现一遍同样的网络，跟 C++ 的输出对拍。

C++ 侧 `./micro-infer dump` 会导出权重/输入/输出：
  dump_input.bin  dump_w1.bin  dump_b1.bin  dump_w2.bin  dump_b2.bin
  dump_w3.bin     dump_b3.bin  dump_out.bin
这里重新算一遍，比较最大绝对误差与余弦相似度。
"""
import numpy as np
import pathlib
import sys


def load(path):
    raw = pathlib.Path(path).read_bytes()
    nd = int.from_bytes(raw[:4], "little", signed=True)
    off = 4
    shape = []
    for _ in range(nd):
        shape.append(int.from_bytes(raw[off:off + 4], "little", signed=True))
        off += 4
    cnt = int.from_bytes(raw[off:off + 4], "little", signed=True)
    off += 4
    data = np.frombuffer(raw[off:off + cnt * 4], dtype=np.float32).copy()
    return data.reshape(shape)


def conv2d_nhwc(x, w, b, pad=1, stride=1):
    N, H, W, C = x.shape
    OC, KH, KW, IC = w.shape
    assert KH == KW
    K = KH
    oh = (H + 2 * pad - K) // stride + 1
    ow = (W + 2 * pad - K) // stride + 1
    out = np.zeros((N, oh, ow, OC), dtype=np.float32)
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    for y in range(oh):
        for xx in range(ow):
            patch = xp[:, y * stride:y * stride + K, xx * stride:xx * stride + K, :]
            out[:, y, xx, :] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def relu(x):
    return np.maximum(x, 0)


def maxpool2x2(x):
    N, H, W, C = x.shape
    return x.reshape(N, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def gap(x):
    return x.mean(axis=(1, 2), keepdims=True)


def main():
    try:
        inp = load("dump_input.bin")
    except FileNotFoundError:
        print("先跑 ./micro-infer dump 生成 dump_*.bin")
        return 1
    w1, b1 = load("dump_w1.bin"), load("dump_b1.bin").reshape(-1)
    w2, b2 = load("dump_w2.bin"), load("dump_b2.bin").reshape(-1)
    w3, b3 = load("dump_w3.bin"), load("dump_b3.bin").reshape(-1)
    ref = load("dump_out.bin").reshape(-1)

    x = relu(conv2d_nhwc(inp, w1, b1))
    x = maxpool2x2(x)
    x = relu(conv2d_nhwc(x, w2, b2))
    x = maxpool2x2(x)
    x = gap(x).reshape(1, -1)
    logits = (x @ w3.reshape(w3.shape[0], -1).T).reshape(-1) + b3

    diff = np.abs(logits - ref)
    cos = float(np.dot(logits, ref) / (np.linalg.norm(logits) * np.linalg.norm(ref) + 1e-12))
    print("logits        :", np.round(logits, 4))
    print("C++ 输出      :", np.round(ref, 4))
    print("最大绝对误差  : %.3e" % diff.max())
    print("余弦相似度    : %.6f" % cos)
    ok = diff.max() < 1e-3 and cos > 0.9999
    print("对拍结果      : " + ("通过" if ok else "不一致，检查算子实现"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
