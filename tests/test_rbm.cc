//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_rbm.h"

#include <gtest/gtest.h>
#include <random>
#include <vector>

// ── Construction ───────────────────────────────────────────────────────────────

TEST(RbmTest, ConstructionStoresDimensions)
{
    nu::Rbm rbm(8, 4);
    EXPECT_EQ(rbm.nVisible(), 8u);
    EXPECT_EQ(rbm.nHidden(), 4u);
}

TEST(RbmTest, InvalidDimensionThrows)
{
    EXPECT_THROW(nu::Rbm(0, 4), std::invalid_argument);
    EXPECT_THROW(nu::Rbm(8, 0), std::invalid_argument);
}

// ── hiddenProbs / visibleProbs ─────────────────────────────────────────────────

TEST(RbmTest, HiddenProbsHasCorrectSize)
{
    nu::Rbm rbm(6, 4);
    Eigen::VectorXd v = Eigen::VectorXd::Ones(6);
    EXPECT_EQ(rbm.hiddenProbs(v).size(), 4);
}

TEST(RbmTest, VisibleProbsHasCorrectSize)
{
    nu::Rbm rbm(6, 4);
    Eigen::VectorXd h = Eigen::VectorXd::Zero(4);
    EXPECT_EQ(rbm.visibleProbs(h).size(), 6);
}

TEST(RbmTest, HiddenProbsInUnitInterval)
{
    nu::Rbm rbm(8, 5, 0.01, 1);
    Eigen::VectorXd v = Eigen::VectorXd::Random(8);
    const auto p = rbm.hiddenProbs(v);
    for (Eigen::Index i = 0; i < p.size(); ++i) {
        EXPECT_GE(p(i), 0.0);
        EXPECT_LE(p(i), 1.0);
    }
}

TEST(RbmTest, VisibleProbsInUnitInterval)
{
    nu::Rbm rbm(8, 5, 0.01, 1);
    Eigen::VectorXd h = Eigen::VectorXd::Random(5);
    const auto p = rbm.visibleProbs(h);
    for (Eigen::Index i = 0; i < p.size(); ++i) {
        EXPECT_GE(p(i), 0.0);
        EXPECT_LE(p(i), 1.0);
    }
}

// ── sampleHidden / sampleVisible ──────────────────────────────────────────────

TEST(RbmTest, SampleHiddenReturnsBinary)
{
    nu::Rbm rbm(8, 5, 0.01, 7);
    Eigen::VectorXd v = Eigen::VectorXd::Constant(8, 0.5);
    const auto s = rbm.sampleHidden(v);
    for (Eigen::Index i = 0; i < s.size(); ++i)
        EXPECT_TRUE(s(i) == 0.0 || s(i) == 1.0);
}

TEST(RbmTest, SampleVisibleReturnsBinary)
{
    nu::Rbm rbm(8, 5, 0.01, 7);
    Eigen::VectorXd h = Eigen::VectorXd::Constant(5, 0.5);
    const auto s = rbm.sampleVisible(h);
    for (Eigen::Index i = 0; i < s.size(); ++i)
        EXPECT_TRUE(s(i) == 0.0 || s(i) == 1.0);
}

// ── reconstruct / encode ──────────────────────────────────────────────────────

TEST(RbmTest, ReconstructHasCorrectSize)
{
    nu::Rbm rbm(8, 4);
    const std::vector<double> x(8, 0.5);
    EXPECT_EQ(rbm.reconstruct(x).size(), 8u);
}

TEST(RbmTest, EncodeHasCorrectSize)
{
    nu::Rbm rbm(8, 4);
    const std::vector<double> x(8, 1.0);
    EXPECT_EQ(rbm.encode(x).size(), 4u);
}

TEST(RbmTest, ReconstructValueInUnitInterval)
{
    nu::Rbm rbm(8, 4, 0.01, 3);
    const std::vector<double> x = { 1, 0, 1, 0, 1, 0, 1, 0 };
    const auto r = rbm.reconstruct(x);
    for (double v : r) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST(RbmTest, SizeMismatchThrows)
{
    nu::Rbm rbm(8, 4);
    EXPECT_THROW(rbm.reconstruct({ 0.1, 0.2 }), std::invalid_argument);
    EXPECT_THROW(rbm.encode({ 0.1 }), std::invalid_argument);
    EXPECT_THROW(rbm.trainStep({ 1.0, 0.0 }), std::invalid_argument);
}

// ── reconstructionError ───────────────────────────────────────────────────────

TEST(RbmTest, ReconstructionErrorNonNegative)
{
    nu::Rbm rbm(8, 4);
    const std::vector<std::vector<double>> data = {
        { 1, 0, 1, 0, 1, 0, 1, 0 },
        { 0, 1, 0, 1, 0, 1, 0, 1 },
    };
    EXPECT_GE(rbm.reconstructionError(data), 0.0);
}

TEST(RbmTest, ReconstructionErrorEmptyDataReturnsZero)
{
    nu::Rbm rbm(8, 4);
    EXPECT_EQ(rbm.reconstructionError({}), 0.0);
}

// ── train ─────────────────────────────────────────────────────────────────────

TEST(RbmTest, TrainReducesReconstructionError)
{
    // 4 prototype 8-bit binary patterns, 50 noisy copies each
    const std::vector<std::vector<double>> protos = {
        { 1, 1, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 1, 1, 1 },
        { 1, 0, 1, 0, 1, 0, 1, 0 },
        { 0, 1, 0, 1, 0, 1, 0, 1 },
    };

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::vector<std::vector<double>> data;
    for (const auto& p : protos)
        for (int i = 0; i < 50; ++i) {
            std::vector<double> x = p;
            for (double& b : x)
                if (udist(rng) < 0.1)
                    b = 1.0 - b;
            data.push_back(x);
        }

    nu::Rbm rbm(8, 6, 0.05, 42);
    const double errBefore = rbm.reconstructionError(data);
    rbm.train(data, 200, 1);
    const double errAfter = rbm.reconstructionError(data);
    EXPECT_LT(errAfter, errBefore);
}
