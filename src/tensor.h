#pragma once
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

// NHWC 布局：和大多数端侧推理引擎一致，最后一维是通道，访存局部性更好
struct Tensor {
    int n = 0, h = 0, w = 0, c = 0;
    std::vector<float> data;

    Tensor() = default;
    Tensor(int n_, int h_, int w_, int c_) { reset(n_, h_, w_, c_); }

    void reset(int n_, int h_, int w_, int c_) {
        n = n_; h = h_; w = w_; c = c_;
        data.assign((size_t)n * h * w * c, 0.f);
    }
    size_t size() const { return data.size(); }
    size_t bytes() const { return data.size() * sizeof(float); }

    float& at(int ni, int hi, int wi, int ci) {
        return data[(((size_t)ni * h + hi) * w + wi) * c + ci];
    }
    float at(int ni, int hi, int wi, int ci) const {
        return data[(((size_t)ni * h + hi) * w + wi) * c + ci];
    }
};
