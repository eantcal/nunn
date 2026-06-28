//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#define _USE_MATH_DEFINES
#include "nu_rbf.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Construction ──────────────────────────────────────────────────────────────

TEST(RbfTest, ConstructionValidDimensions)
{
    nu::Rbf rbf(3, 10, 2);
    EXPECT_EQ(rbf.getInputSize(), 3u);
    EXPECT_EQ(rbf.getNumCenters(), 10u);
    EXPECT_EQ(rbf.getOutputSize(), 2u);
    EXPECT_FALSE(rbf.isFitted());
}

TEST(RbfTest, ZeroInputSizeThrows)
{
    EXPECT_THROW(nu::Rbf(0, 5, 2), std::invalid_argument);
}

TEST(RbfTest, ZeroCentersThrows)
{
    EXPECT_THROW(nu::Rbf(3, 0, 2), std::invalid_argument);
}

TEST(RbfTest, ZeroOutputSizeThrows)
{
    EXPECT_THROW(nu::Rbf(3, 5, 0), std::invalid_argument);
}

// ── fitCenters ────────────────────────────────────────────────────────────────

TEST(RbfTest, FitCentersMarksAsFitted)
{
    nu::Rbf rbf(2, 4, 1);
    const std::vector<std::vector<double>> data = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    rbf.fitCenters(data);
    EXPECT_TRUE(rbf.isFitted());
}

TEST(RbfTest, FitCentersEmptyThrows)
{
    nu::Rbf rbf(2, 4, 1);
    EXPECT_THROW(rbf.fitCenters({}), std::invalid_argument);
}

TEST(RbfTest, FitCentersWrongInputSizeThrows)
{
    nu::Rbf rbf(3, 4, 1);
    EXPECT_THROW(rbf.fitCenters({ { 0.0, 1.0 } }), std::invalid_argument);
}

// ── forward / train guards ────────────────────────────────────────────────────

TEST(RbfTest, ForwardBeforeFitThrows)
{
    nu::Rbf rbf(2, 4, 1);
    EXPECT_THROW(rbf.forward({ 0.5, 0.5 }), std::runtime_error);
}

TEST(RbfTest, TrainBeforeFitThrows)
{
    nu::Rbf rbf(2, 4, 1);
    EXPECT_THROW(rbf.train({ { 0.5, 0.5 } }, { { 1.0 } }, 10), std::runtime_error);
}

// ── Output sizes ──────────────────────────────────────────────────────────────

TEST(RbfTest, ForwardOutputSize)
{
    nu::Rbf rbf(2, 4, 3);
    const std::vector<std::vector<double>> data = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    rbf.fitCenters(data);
    const auto out = rbf.forward({ 0.5, 0.5 });
    EXPECT_EQ(out.size(), 3u);
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

TEST(RbfTest, ReshuffleDoesNotCrash)
{
    nu::Rbf rbf(2, 4, 1);
    const std::vector<std::vector<double>> data = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    rbf.fitCenters(data);
    EXPECT_NO_THROW(rbf.reshuffleWeights());
    const auto out = rbf.forward({ 0.5, 0.5 });
    EXPECT_EQ(out.size(), 1u);
}

// ── Convergence: sine regression ─────────────────────────────────────────────

static std::vector<std::vector<double>> makeSineInputs(int n)
{
    std::vector<std::vector<double>> in;
    for (int i = 0; i < n; ++i)
        in.push_back({ 2.0 * M_PI * i / (n - 1) });
    return in;
}

static std::vector<std::vector<double>> makeSineTargets(int n)
{
    std::vector<std::vector<double>> tgt;
    for (int i = 0; i < n; ++i)
        tgt.push_back({ std::sin(2.0 * M_PI * i / (n - 1)) });
    return tgt;
}

TEST(RbfTest, RegressionReducesMSEOnSine)
{
    // 1D → 1D sine regression.  Multiple trials to cope with random center placement.
    constexpr int N = 20;
    const auto inputs = makeSineInputs(N);
    const auto targets = makeSineTargets(N);

    double bestFinalMSE = 1e9;
    for (int trial = 0; trial < 5; ++trial) {
        nu::Rbf rbf(1, 8, 1, 0.05);
        rbf.fitCenters(inputs);

        // Baseline MSE before training.
        double mseBefore = 0.0;
        for (int i = 0; i < N; ++i) {
            const double diff = rbf.forward(inputs[i])[0] - targets[i][0];
            mseBefore += diff * diff;
        }
        mseBefore /= N;

        rbf.train(inputs, targets, 3000);

        double mseAfter = 0.0;
        for (int i = 0; i < N; ++i) {
            const double diff = rbf.forward(inputs[i])[0] - targets[i][0];
            mseAfter += diff * diff;
        }
        mseAfter /= N;

        if (mseBefore > 0.0)
            bestFinalMSE = std::min(bestFinalMSE, mseAfter / mseBefore);
    }
    // Expect at least 50% reduction in the best trial.
    EXPECT_LT(bestFinalMSE, 0.50);
}

// ── Convergence: 2-class softmax ─────────────────────────────────────────────

TEST(RbfTest, ClassificationAccuracyOnLinearProblem)
{
    // x[0] > 0.5 → class 0; else → class 1. 20 samples, 1D input.
    constexpr int N = 20;
    std::vector<std::vector<double>> inputs, targets;
    for (int i = 0; i < N; ++i) {
        const double x = static_cast<double>(i) / (N - 1);
        inputs.push_back({ x });
        targets.push_back(x > 0.5 ? std::vector<double>{ 1, 0 } : std::vector<double>{ 0, 1 });
    }

    int bestCorrect = 0;
    for (int trial = 0; trial < 5; ++trial) {
        nu::Rbf rbf(1, 6, 2, 0.1, nu::RnnOutput::Softmax);
        rbf.fitCenters(inputs);
        rbf.train(inputs, targets, 3000);

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            const auto y = rbf.forward(inputs[i]);
            const int pred = (y[0] > y[1]) ? 0 : 1;
            const int gt = (targets[i][0] > 0.5) ? 0 : 1;
            if (pred == gt)
                ++correct;
        }
        bestCorrect = std::max(bestCorrect, correct);
    }
    EXPECT_GE(bestCorrect, static_cast<int>(N * 0.85));
}
