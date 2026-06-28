//
// Unit tests for nu::Gru (nu_gru.h / nu_gru.cc).
//

#include "nu_gru.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using nu::Gru;
using nu::RnnOutput;

// ── Construction & getters ────────────────────────────────────────────────────

TEST(GruTest, DimensionsMatchConstruction)
{
    Gru gru(4, 16, 3);
    EXPECT_EQ(gru.getInputSize(), 4u);
    EXPECT_EQ(gru.getHiddenSize(), 16u);
    EXPECT_EQ(gru.getOutputSize(), 3u);
    EXPECT_EQ(gru.getOutput().size(), 3u);
    EXPECT_EQ(gru.getHidden().size(), 16u);
}

TEST(GruTest, DefaultOutputModeIsLinear)
{
    Gru gru(1, 4, 1);
    EXPECT_EQ(gru.getOutputMode(), RnnOutput::Linear);
}

// ── resetState ────────────────────────────────────────────────────────────────

TEST(GruTest, ResetStateZerosHidden)
{
    Gru gru(2, 8, 1);
    gru.step({ 1.0, 1.0 });

    bool any_nonzero = false;
    for (double v : gru.getHidden())
        if (v != 0.0) {
            any_nonzero = true;
            break;
        }
    EXPECT_TRUE(any_nonzero);

    gru.resetState();
    for (double v : gru.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}

// ── step ──────────────────────────────────────────────────────────────────────

TEST(GruTest, StepChangesHiddenState)
{
    Gru gru(2, 8, 1);
    const auto h0 = gru.getHidden();
    gru.step({ 1.0, 0.5 });
    EXPECT_NE(gru.getHidden(), h0);
}

TEST(GruTest, StepOutputSizeMatches)
{
    Gru gru(3, 10, 5, 0.01, 5.0, RnnOutput::Softmax);
    gru.step({ 1.0, 0.0, -1.0 });
    EXPECT_EQ(gru.getOutput().size(), 5u);
}

TEST(GruTest, SoftmaxOutputSumsToOne)
{
    Gru gru(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    gru.step({ 0.3, -0.7 });
    double sum = 0.0;
    for (double v : gru.getOutput())
        sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(GruTest, SoftmaxOutputAllPositive)
{
    Gru gru(2, 8, 4, 0.01, 5.0, RnnOutput::Softmax);
    gru.step({ 1.0, 2.0 });
    for (double v : gru.getOutput())
        EXPECT_GT(v, 0.0);
}

// ── BPTT ──────────────────────────────────────────────────────────────────────

TEST(GruTest, BpttReturnsNonNegativeLoss)
{
    Gru gru(1, 8, 1);
    const std::vector<std::vector<double>> xs = { { 0.5 }, { -0.3 }, { 0.7 } };
    const std::vector<std::vector<double>> ys = { { 0.2 }, { -0.1 }, { 0.5 } };
    EXPECT_GE(gru.bptt(xs, ys), 0.0);
}

TEST(GruTest, BpttEmptySequenceReturnsZero)
{
    Gru gru(1, 4, 1);
    EXPECT_DOUBLE_EQ(gru.bptt({}, {}), 0.0);
}

TEST(GruTest, BpttAdvancesHiddenState)
{
    Gru gru(1, 8, 1);
    const auto h_before = gru.getHidden();
    const std::vector<std::vector<double>> xs = { { 1.0 }, { -1.0 } };
    const std::vector<std::vector<double>> ys = { { 0.5 }, { -0.5 } };
    gru.bptt(xs, ys);
    EXPECT_NE(gru.getHidden(), h_before);
}

// ── Convergence ───────────────────────────────────────────────────────────────

TEST(GruTest, ConvergesOnIdentityMapping)
{
    Gru gru(1, 16, 1, 0.01, 5.0, RnnOutput::Linear);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        double v = (t % 2 == 0) ? 0.8 : -0.8;
        xs[t] = { v };
        ys[t] = { v };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 2000; ++ep) {
        gru.resetState();
        loss = gru.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.01);
}

TEST(GruTest, ConvergesOnDelayedEcho)
{
    Gru gru(1, 16, 1, 0.01, 5.0, RnnOutput::Linear);

    const size_t T = 20;
    std::vector<std::vector<double>> xs(T), ys(T);
    for (size_t t = 0; t < T; ++t) {
        xs[t] = { static_cast<double>(t % 2) };
        ys[t] = { static_cast<double>((t + 1) % 2) };
    }

    double loss = 1e9;
    for (int ep = 0; ep < 3000; ++ep) {
        gru.resetState();
        loss = gru.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.05);
}

TEST(GruTest, ConvergesOnSoftmaxBinarySequence)
{
    Gru gru(2, 16, 2, 0.01, 5.0, RnnOutput::Softmax);

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
        gru.resetState();
        loss = gru.bptt(xs, ys);
    }
    EXPECT_LT(loss, 0.1);
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

TEST(GruTest, ReshuffleResetsHiddenToZero)
{
    Gru gru(2, 8, 1);
    gru.step({ 1.0, -1.0 });
    gru.reshuffleWeights();
    for (double v : gru.getHidden())
        EXPECT_DOUBLE_EQ(v, 0.0);
}
