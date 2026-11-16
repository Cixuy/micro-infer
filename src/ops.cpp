#include "ops.h"
#include <algorithm>
#include <cmath>

void conv2d(const Tensor& in, const Tensor& wt, const Tensor& bias,
            int stride, int pad, Tensor& out) {
    const int K = (int)std::lround(std::sqrt((double)wt.h));  // 权重视为 [OC, KH*KW*IC]
    const int oc = wt.n, ic = in.c;
    const int oh = (in.h + 2 * pad - K) / stride + 1;
    const int ow = (in.w + 2 * pad - K) / stride + 1;
    out.reset(in.n, oh, ow, oc);

    const float* W = wt.data.data();
    for (int ni = 0; ni < in.n; ++ni)
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x)
                for (int o = 0; o < oc; ++o) {
                    float acc = bias.data[o];
                    const float* wbase = W + ((size_t)o * K * K * ic);
                    for (int ky = 0; ky < K; ++ky) {
                        int iy = y * stride - pad + ky;
                        if (iy < 0 || iy >= in.h) continue;
                        for (int kx = 0; kx < K; ++kx) {
                            int ix = x * stride - pad + kx;
                            if (ix < 0 || ix >= in.w) continue;
                            const float* ip = &in.data[(((size_t)ni * in.h + iy) * in.w + ix) * ic];
                            const float* wp = wbase + ((size_t)ky * K + kx) * ic;
                            for (int ci = 0; ci < ic; ++ci) acc += ip[ci] * wp[ci];
                        }
                    }
                    out.at(ni, y, x, o) = acc;
                }
}

void conv2dRelu(const Tensor& in, const Tensor& wt, const Tensor& bias,
                int stride, int pad, Tensor& out) {
    conv2d(in, wt, bias, stride, pad, out);
    // 融合版在真实引擎里是算完直接夹紧；这里同一次遍历完成，避免再走一遍内存
    for (float& v : out.data) if (v < 0.f) v = 0.f;
}

void reluInplace(Tensor& t) {
    for (float& v : t.data) if (v < 0.f) v = 0.f;
}

void maxpool2x2(const Tensor& in, Tensor& out) {
    const int oh = in.h / 2, ow = in.w / 2;
    out.reset(in.n, oh, ow, in.c);
    for (int ni = 0; ni < in.n; ++ni)
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x)
                for (int c = 0; c < in.c; ++c) {
                    float m = -1e30f;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            m = std::max(m, in.at(ni, y * 2 + dy, x * 2 + dx, c));
                    out.at(ni, y, x, c) = m;
                }
}

void globalAvgPool(const Tensor& in, Tensor& out) {
    out.reset(in.n, 1, 1, in.c);
    const float s = 1.f / (in.h * in.w);
    for (int ni = 0; ni < in.n; ++ni)
        for (int c = 0; c < in.c; ++c) {
            float acc = 0;
            for (int y = 0; y < in.h; ++y)
                for (int x = 0; x < in.w; ++x) acc += in.at(ni, y, x, c);
            out.at(ni, 0, 0, c) = acc * s;
        }
}

void fullyConnected(const Tensor& in, const Tensor& wt, const Tensor& bias, Tensor& out) {
    const int oc = wt.n, ic = in.c;
    out.reset(in.n, 1, 1, oc);
    for (int ni = 0; ni < in.n; ++ni)
        for (int o = 0; o < oc; ++o) {
            float acc = bias.data[o];
            const float* w = &wt.data[(size_t)o * ic];
            for (int i = 0; i < ic; ++i) acc += in.at(ni, 0, 0, i) * w[i];
            out.at(ni, 0, 0, o) = acc;
        }
}
