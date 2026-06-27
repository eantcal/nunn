//
// Unit tests for nu::Lstm (nu_lstm.h / nu_lstm.cc).
//

#include "nu_lstm.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using nu::Lstm;
using nu::RnnOutput;

// ── Construction & getters ────────────────────────────────────────────────────

TEST(LstmTest, DimensionsMatchConstruction)
{
    Lstm lstm(4, 16, 3);
    EXPECT_EQ(lstm.getInputSize(), 4u);
    EXPECT_EQ(lstm.getHiddenSize(), 16u);
    EXPECT_EQ(lstm.getOutputSize(), 3u);
    EXPECT_EQ(lstm.getOutput().size(), 3u);
    EXPECT_EQ(lstm.getHidden().size(), 16u);
}

TEST(LstmTest, DefaultOutputModeIsLinear)
{
    Lstm lstm(1, 4, 1);
    EXPECT_EQ(lstm.getOutputMode(), RnnOutput::Linear);
}

// ── resetState ────────────────────────────────────────────────────────────────

TEST(LstmTest, ResetStateZerosHidden)
{
    Lstm lstm(2, 8, 1);
    lstm.step({ 1.0, 1.0 });

    bool any_nonzero = false;
    for (double v : lstm.getHidden())
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    EXPECT_TRUE(any_nonzero);

    lstm.resetState();
    for (double v : lstm.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}

// ── step ──────────────────────────────────────────────────────────────────────

TEST(LstmTest, StepChangesHiddenState)
{
    Lstm lstm(2, 8, 1);
    const auto h0 = lstm.getHidden();
    lstm.step({ 1.0, 0.5 });
    EXPECT_NE(lstm.getHidden(), h0);
}

TEST(LstmTest, StepOutputSizeMatches)
{
    Lstm lstm(3, 10, 5, 0.01, 5.0, RnnOutput::Softmax);
    lstm.step({ 1.0, 0.0, -1.0 });
    EXPECT_EQ(lstm.getOutput().size(), 5u);
}

TEST(LstmTest, SoftmaxOutputSumsToOne)
{
    Lstm lstm(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    lstm.step({ 0.3, -0.7 });
    double sum = 0.0;
    for (double v : lstm.getOutput())
        sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(LstmTest, SoftmaxOutputAllPositive)
{
    Lstm lstm(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    lstm.step({ 1.0, 2.0 });
    for (double v : lstm.getOutput())
        EXPECT_GT(v, 0.0);
}

// ── BPTT ──────────────────────────────────────────────────────────────────────

TEST(LstmTest, BpttReturnsNonNegativeLoss)
{
    Lstm lstm(1, 8, 1);
    const std::vector<std::vector<double>> xs = { { 0.5 }, { -0.3 }, { 0.7 } };
    const std::vector<std::vector<double>> ys = { { 0.2 }, { -0.1 }, { 0.5 } };
    EXPECT_GE(lstm.bptt(xs, ys), 0.0);
}

TEST(LstmTest, BpttEmptySequenceReturnsZero)
{
    Lstm lstm(1, 4, 1);
    EXPECT_DOUBLE_EQ(lstm.bptt({}, {}), 0.0);
}

TEST(LstmTest, BpttAdvancesHiddenState)
{
    Lstm lstm(1, 8, 1);
    const auto h_before = lstm.getHidden();
    const std::vector<std::vector<double>> xs = { { 1.0 }, { -1.0 } };
    const std::vector<std::vector<double>> ys = { { 0.5 }, { -0.5 } };
    lstm.bptt(xs, ys);
    EXPECT_NE(lstm.getHidden(), h_before);
}

// ── Convergence ───────────────────────────────────────────────────────────────

// LSTM should learn the identity mapping (y_t = x_t) faster than a vanilla RNN.
TEST(LstmTest, ConvergesOnIdentityMapping)
{
    Lstm lstm(1, 16, 1, 0.01, 5.0, RnnOutput::Linear);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        double v = (t % 2 == 0) ? 0.8 : -0.8;
        xs[t] = { v };
        ys[t] = { v };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 2000; ++ep) {
        lstm.resetState();
        loss = lstm.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.01);
}

// Delayed echo: predict x_{t-1} (requires one step of memory).
// The LSTM's cell state makes this trivial; fewer epochs than VanillaRnn.
TEST(LstmTest, ConvergesOnDelayedEcho)
{
    Lstm lstm(1, 16, 1, 0.01, 5.0, RnnOutput::Linear);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        xs[t] = { static_cast<double>(t % 2) };
        ys[t] = { static_cast<double>((t + 1) % 2) };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 3000; ++ep) {
        lstm.resetState();
        loss = lstm.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.05);
}

// Softmax binary sequence: "ab ab ab…" next-char prediction.
TEST(LstmTest, ConvergesOnSoftmaxBinarySequence)
{
    Lstm lstm(2, 16, 2, 0.01, 5.0, RnnOutput::Softmax);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        int cur = t % 2;
        int next = (t + 1) % 2;
        xs[t] = { cur == 0 ? 1.0 : 0.0, cur == 1 ? 1.0 : 0.0 };
        ys[t] = { next == 0 ? 1.0 : 0.0, next == 1 ? 1.0 : 0.0 };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 3000; ++ep) {
        lstm.resetState();
        loss = lstm.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.1);
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

TEST(LstmTest, ReshuffleResetsHiddenToZero)
{
    Lstm lstm(2, 8, 1);
    lstm.step({ 1.0, -1.0 });
    lstm.reshuffleWeights();
    for (double v : lstm.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}
