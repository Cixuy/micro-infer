#pragma once
#include "tensor.h"

// 朴素实现，不做向量化，先保证正确性 —— 后面所有优化都拿它做对拍基线
void conv2d(const Tensor& in, const Tensor& wt, const Tensor& bias,
            int stride, int pad, Tensor& out);

void reluInplace(Tensor& t);

// conv + relu 融合：省掉一次完整的输出张量读写
void conv2dRelu(const Tensor& in, const Tensor& wt, const Tensor& bias,
                int stride, int pad, Tensor& out);

void maxpool2x2(const Tensor& in, Tensor& out);

void globalAvgPool(const Tensor& in, Tensor& out);

void fullyConnected(const Tensor& in, const Tensor& wt, const Tensor& bias, Tensor& out);
