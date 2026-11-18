#include "ops.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static Tensor randTensor(int n, int h, int w, int c, float scale) {
    Tensor t(n, h, w, c);
    for (auto& v : t.data) v = ((float)rand() / RAND_MAX - 0.5f) * 2.f * scale;
    return t;
}

static void dump(const char* path, const Tensor& t, const std::vector<int>& shape) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int nd = (int)shape.size();
    fwrite(&nd, sizeof(int), 1, f);
    for (int i = 0; i < nd; ++i) fwrite(&shape[i], sizeof(int), 1, f);
    int cnt = (int)t.data.size();
    fwrite(&cnt, sizeof(int), 1, f);
    fwrite(t.data.data(), sizeof(float), t.data.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    bool doDump = false;
    int benchN = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "dump") doDump = true;
        else if (a == "--bench" && i + 1 < argc) benchN = std::max(1, atoi(argv[++i]));
        else if (a == "--help" || a == "-h") {
            printf("用法: micro_infer [dump] [--bench N]\n");
            printf("  dump      导出 dump_*.bin 供 python 参考实现对拍\n");
            printf("  --bench N 基准重复次数（默认 10），报告 min/中位/均值/max\n");
            return 0;
        }
    }
    srand(42);  // 固定种子，保证 C++ 和 python 参考实现拿到同一组权重

    const int N = 8, IN = 64, IC = 3, OC1 = 16, OC2 = 32, CLS = 10;

    Tensor input = randTensor(N, IN, IN, IC, 1.0f);
    Tensor w1 = randTensor(OC1, 3, 3, IC, 0.3f);   // OC x 3x3 x IC
    Tensor b1 = randTensor(OC1, 1, 1, 1, 0.1f);
    Tensor w2 = randTensor(OC2, 3, 3, OC1, 0.2f);
    Tensor b2 = randTensor(OC2, 1, 1, 1, 0.1f);
    Tensor w3 = randTensor(CLS, 1, 1, OC2, 0.3f);
    Tensor b3 = randTensor(CLS, 1, 1, 1, 0.1f);
    b1.reset(OC1, 1, 1, 1); for (int i = 0; i < OC1; ++i) b1.data[i] = 0.05f * (i - 3);
    b2.reset(OC2, 1, 1, 1); for (int i = 0; i < OC2; ++i) b2.data[i] = 0.03f * (i - 8);
    b3.reset(CLS, 1, 1, 1); for (int i = 0; i < CLS; ++i) b3.data[i] = 0.01f * i;

    // ---------- 基线：每层都完整写一遍输出张量 ----------
    Tensor c1, p1, c2, p2, g, out;
    printf("\n=== 基线（逐层独立执行，conv/relu 分开） ===\n");
    printf("%-22s %10s %10s\n", "layer", "output", "ms");
    auto t0 = std::chrono::steady_clock::now();

    auto a = std::chrono::steady_clock::now();
    conv2d(input, w1, b1, 1, 1, c1);
    auto b = std::chrono::steady_clock::now();
    double tConv1 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "conv1 3x3 s1 p1", (std::to_string(c1.h) + "x" + std::to_string(c1.w) + "x" + std::to_string(c1.c)).c_str(), tConv1);

    a = std::chrono::steady_clock::now(); reluInplace(c1); b = std::chrono::steady_clock::now();
    double tRelu1 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "relu1", "-", tRelu1);

    a = std::chrono::steady_clock::now(); maxpool2x2(c1, p1); b = std::chrono::steady_clock::now();
    double tPool1 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "maxpool 2x2", (std::to_string(p1.h) + "x" + std::to_string(p1.w)).c_str(), tPool1);

    a = std::chrono::steady_clock::now(); conv2d(p1, w2, b2, 1, 1, c2); b = std::chrono::steady_clock::now();
    double tConv2 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "conv2 3x3 s1 p1", (std::to_string(c2.h) + "x" + std::to_string(c2.w) + "x" + std::to_string(c2.c)).c_str(), tConv2);

    a = std::chrono::steady_clock::now(); reluInplace(c2); b = std::chrono::steady_clock::now();
    double tRelu2 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "relu2", "-", tRelu2);

    a = std::chrono::steady_clock::now(); maxpool2x2(c2, p2); b = std::chrono::steady_clock::now();
    double tPool2 = ms(a, b);
    printf("%-22s %10s %10.3f\n", "maxpool 2x2", (std::to_string(p2.h) + "x" + std::to_string(p2.w)).c_str(), tPool2);

    a = std::chrono::steady_clock::now(); globalAvgPool(p2, g); b = std::chrono::steady_clock::now();
    double tGap = ms(a, b);
    printf("%-22s %10s %10.3f\n", "global avgpool", (std::to_string(g.c)).c_str(), tGap);

    a = std::chrono::steady_clock::now(); fullyConnected(g, w3, b3, out); b = std::chrono::steady_clock::now();
    double tFc = ms(a, b);
    printf("%-22s %10s %10.3f\n", "fc", (std::to_string(out.c)).c_str(), tFc);

    auto t1 = std::chrono::steady_clock::now();
    double totalBase = ms(t0, t1);
    printf("%-22s %10s %10.3f\n", "total", "", totalBase);

    std::vector<float> baseOut = out.data;   // 留一份基线结果，用来验证优化前后一致

    // ---------- 优化：conv+relu 融合 + 两块缓冲乒乓复用 ----------
    printf("\n=== 优化（conv+relu 融合，buffer 乒乓复用） ===\n");
    Tensor bufA, bufB;
    auto t2 = std::chrono::steady_clock::now();
    conv2dRelu(input, w1, b1, 1, 1, bufA);
    maxpool2x2(bufA, bufB);
    conv2dRelu(bufB, w2, b2, 1, 1, bufA);
    maxpool2x2(bufA, bufB);
    globalAvgPool(bufB, g);
    fullyConnected(g, w3, b3, out);
    auto t3 = std::chrono::steady_clock::now();
    double totalFused = ms(t2, t3);
    printf("total                  %10s %10.3f\n", "", totalFused);

    // ---------- 统计 ----------
    double flops = 0;
    flops += 2.0 * c1.size() * 3 * 3 * IC;      // conv1
    flops += 2.0 * c2.size() * 3 * 3 * OC1;     // conv2
    flops += 2.0 * out.size() * OC2;            // fc

    size_t naiveBytes = c1.bytes() + p1.bytes() + c2.bytes() + p2.bytes() + g.bytes() + out.bytes();
    size_t reuseBytes = std::max(bufA.bytes(), bufB.bytes()) + g.bytes() + out.bytes();

    // ---------- 多次运行取最优与均值，单次测量抖动太大 ----------
    // 基准次数可调：机器噪声大时多跑几次看中位数，比只看均值可靠
    const int REPS = benchN;
    std::vector<double> baseRuns, fusedRuns;
    baseRuns.reserve(REPS); fusedRuns.reserve(REPS);
    for (int rep = 0; rep < REPS; ++rep) {
        Tensor m1, m2, m3, m4, mg, mo;
        auto s0 = std::chrono::steady_clock::now();
        conv2d(input, w1, b1, 1, 1, m1); reluInplace(m1); maxpool2x2(m1, m2);
        conv2d(m2, w2, b2, 1, 1, m3); reluInplace(m3); maxpool2x2(m3, m4);
        globalAvgPool(m4, mg); fullyConnected(mg, w3, b3, mo);
        auto s1 = std::chrono::steady_clock::now();
        baseRuns.push_back(ms(s0, s1));

        auto s2 = std::chrono::steady_clock::now();
        conv2dRelu(input, w1, b1, 1, 1, m1); maxpool2x2(m1, m2);
        conv2dRelu(m2, w2, b2, 1, 1, m3); maxpool2x2(m3, m4);
        globalAvgPool(m4, mg); fullyConnected(mg, w3, b3, mo);
        auto s3 = std::chrono::steady_clock::now();
        fusedRuns.push_back(ms(s2, s3));
    }
    // 均值会被偶发的一次调度抖动拉偏，所以同时给中位数和最大值：
    // min 代表"这台机器能跑多快"，median 代表"通常多快"，max 暴露抖动有多大
    struct Stat { double mn, med, avg, mx; };
    auto stat = [](std::vector<double>& v) {
        Stat st{};
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        st.mn = s.front();
        st.mx = s.back();
        st.med = (s.size() % 2) ? s[s.size() / 2]
                                : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);
        double sum = 0;
        for (double x : s) sum += x;
        st.avg = sum / s.size();
        return st;
    };
    Stat bs = stat(baseRuns), fs = stat(fusedRuns);
    const double bmin = bs.mn, bavg = bs.avg, fmin = fs.mn, favg = fs.avg;

    float maxDiff = 0.f;
    for (size_t i = 0; i < baseOut.size(); ++i)
        maxDiff = std::max(maxDiff, std::fabs(baseOut[i] - out.data[i]));

    printf("\n=== 结果 ===\n");
    printf("优化前后输出一致性      : max abs diff = %.3e (%s)\n", maxDiff,
           maxDiff < 1e-4f ? "一致" : "不一致，检查融合实现");
    printf("MACs                    : %.2f M\n", flops / 2 / 1e6);
    printf("GFLOPS(基线)            : %.3f\n", flops / (totalBase * 1e6));
    printf("GFLOPS(优化)            : %.3f\n", flops / (totalFused * 1e6));
    printf("单次 基线/优化          : %.3f ms -> %.3f ms\n", totalBase, totalFused);
    printf("%d 次平均 基线/优化      : %.3f ms -> %.3f ms (%.1f%%)\n", REPS, bavg, favg,
           (1 - favg / bavg) * 100);
    printf("%d 次最优 基线/优化      : %.3f ms -> %.3f ms\n", REPS, bmin, fmin);
    // 只有"最优"这组数受系统噪声影响最小，拿它算 GFLOPS 才有可比性
    printf("GFLOPS(最优)            : %.3f -> %.3f\n", flops / (bmin * 1e6), flops / (fmin * 1e6));
    printf("%d 次中位 基线/优化      : %.3f ms -> %.3f ms (%.1f%%)\n", REPS, bs.med, fs.med,
           (1 - fs.med / bs.med) * 100);
    printf("%d 次最差 基线/优化      : %.3f ms -> %.3f ms  抖动 %.1f%% / %.1f%%\n", REPS, bs.mx, fs.mx,
           (bs.mx / bs.mn - 1) * 100, (fs.mx / fs.mn - 1) * 100);
    printf("中间张量显存 朴素/复用  : %.1f KB -> %.1f KB\n", naiveBytes / 1024.0, reuseBytes / 1024.0);

    if (doDump) {
        dump("dump_input.bin", input, {N, IN, IN, IC});
        dump("dump_w1.bin", w1, {OC1, 3, 3, IC});
        dump("dump_b1.bin", b1, {OC1});
        dump("dump_w2.bin", w2, {OC2, 3, 3, OC1});
        dump("dump_b2.bin", b2, {OC2});
        dump("dump_w3.bin", w3, {CLS, 1, 1, OC2});
        dump("dump_b3.bin", b3, {CLS});
        dump("dump_out.bin", out, {N, CLS});
        printf("\n已导出张量到 dump_*.bin，供 python 参考实现对拍\n");
    }
    return 0;
}
