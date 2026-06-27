//
// Unit tests for nu::VanillaRnn (nu_rnn.h / nu_rnn.cc).
//

#include "nu_rnn.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using nu::RnnOutput;
using nu::VanillaRnn;

// ── Construction & getters ────────────────────────────────────────────────────

TEST(VanillaRnnTest, DimensionsMatchConstruction)
{
    VanillaRnn rnn(3, 16, 2);
    EXPECT_EQ(rnn.getInputSize(), 3u);
    EXPECT_EQ(rnn.getHiddenSize(), 16u);
    EXPECT_EQ(rnn.getOutputSize(), 2u);
    EXPECT_EQ(rnn.getOutput().size(), 2u);
    EXPECT_EQ(rnn.getHidden().size(), 16u);
}

TEST(VanillaRnnTest, DefaultOutputModeIsLinear)
{
    VanillaRnn rnn(1, 4, 1);
    EXPECT_EQ(rnn.getOutputMode(), RnnOutput::Linear);
}

// ── resetState ────────────────────────────────────────────────────────────────

TEST(VanillaRnnTest, ResetStateZerosHidden)
{
    VanillaRnn rnn(2, 8, 2);
    rnn.step({ 1.0, 1.0 });

    // After a step the hidden state is non-zero with overwhelming probability
    bool any_nonzero = false;
    for (double v : rnn.getHidden())
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    EXPECT_TRUE(any_nonzero);

    rnn.resetState();
    for (double v : rnn.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}

// ── step ──────────────────────────────────────────────────────────────────────

TEST(VanillaRnnTest, StepChangesHiddenState)
{
    VanillaRnn rnn(2, 8, 1);
    const auto h0 = rnn.getHidden(); // all zeros initially
    rnn.step({ 1.0, 0.5 });
    EXPECT_NE(rnn.getHidden(), h0);
}

TEST(VanillaRnnTest, StepOutputSizeMatches)
{
    VanillaRnn rnn(3, 10, 5, 0.01, 5.0, RnnOutput::Softmax);
    rnn.step({ 1.0, 0.0, -1.0 });
    EXPECT_EQ(rnn.getOutput().size(), 5u);
}

TEST(VanillaRnnTest, SoftmaxOutputSumsToOne)
{
    VanillaRnn rnn(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    rnn.step({ 0.3, -0.7 });
    double sum = 0.0;
    for (double v : rnn.getOutput())
        sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(VanillaRnnTest, SoftmaxOutputAllPositive)
{
    VanillaRnn rnn(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    rnn.step({ 1.0, 2.0 });
    for (double v : rnn.getOutput())
        EXPECT_GT(v, 0.0);
}

// ── BPTT ──────────────────────────────────────────────────────────────────────

TEST(VanillaRnnTest, BpttReturnsNonNegativeLoss)
{
    VanillaRnn rnn(1, 8, 1);
    const std::vector<std::vector<double>> xs = { { 0.5 }, { -0.3 }, { 0.7 } };
    const std::vector<std::vector<double>> ys = { { 0.2 }, { -0.1 }, { 0.5 } };
    EXPECT_GE(rnn.bptt(xs, ys), 0.0);
}

TEST(VanillaRnnTest, BpttEmptySequenceReturnsZero)
{
    VanillaRnn rnn(1, 4, 1);
    EXPECT_DOUBLE_EQ(rnn.bptt({}, {}), 0.0);
}

TEST(VanillaRnnTest, BpttAdvancesHiddenState)
{
    VanillaRnn rnn(1, 8, 1);
    const auto h_before = rnn.getHidden();
    const std::vector<std::vector<double>> xs = { { 1.0 }, { -1.0 } };
    const std::vector<std::vector<double>> ys = { { 0.5 }, { -0.5 } };
    rnn.bptt(xs, ys);
    EXPECT_NE(rnn.getHidden(), h_before);
}

// ── Convergence ───────────────────────────────────────────────────────────────

// The RNN must learn to copy a constant value: y_t = x_t (memoryless mapping).
// This is the simplest possible regression task; loss should drop to near zero.
TEST(VanillaRnnTest, ConvergesOnIdentityMapping)
{
    VanillaRnn rnn(1, 16, 1, 0.02, 5.0, RnnOutput::Linear);

    // Training sequence: x_t is the target itself
    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        double v = (t % 2 == 0) ? 0.8 : -0.8;
        xs[t] = { v };
        ys[t] = { v };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 3000; ++ep) {
        rnn.resetState();
        loss = rnn.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.01);
}

// The RNN must predict the next element of a repeating binary sequence:
// 0 1 0 1 … — requires one step of memory.
TEST(VanillaRnnTest, ConvergesOnDelayedEcho)
{
    VanillaRnn rnn(1, 16, 1, 0.01, 5.0, RnnOutput::Linear);

    // Input: [0,1,0,1,…]  Target: [1,0,1,0,…]  (predict next)
    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        xs[t] = { static_cast<double>(t % 2) };
        ys[t] = { static_cast<double>((t + 1) % 2) };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 5000; ++ep) {
        rnn.resetState();
        loss = rnn.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.05);
}

// Softmax convergence: predict next character in "ababab…"
TEST(VanillaRnnTest, ConvergesOnSoftmaxBinarySequence)
{
    // Vocabulary: 0 → [1,0], 1 → [0,1]
    VanillaRnn rnn(2, 16, 2, 0.02, 5.0, RnnOutput::Softmax);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        int cur = t % 2;
        int next = (t + 1) % 2;
        xs[t] = { cur == 0 ? 1.0 : 0.0, cur == 1 ? 1.0 : 0.0 };
        ys[t] = { next == 0 ? 1.0 : 0.0, next == 1 ? 1.0 : 0.0 };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 5000; ++ep) {
        rnn.resetState();
        loss = rnn.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.1);
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

TEST(VanillaRnnTest, ReshuffleResetsHiddenToZero)
{
    VanillaRnn rnn(2, 8, 1);
    rnn.step({ 1.0, -1.0 });
    rnn.reshuffleWeights();
    for (double v : rnn.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}
