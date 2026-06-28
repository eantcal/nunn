//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_vae.h"

#include <gtest/gtest.h>
#include <random>
#include <vector>

// ── Construction ───────────────────────────────────────────────────────────────

TEST(VaeTest, ConstructionStoresDimensions)
{
    nu::Vae vae(8, 16, 4);
    EXPECT_EQ(vae.inputDim(), 8u);
    EXPECT_EQ(vae.hiddenDim(), 16u);
    EXPECT_EQ(vae.latentDim(), 4u);
}

TEST(VaeTest, InvalidDimensionThrows)
{
    EXPECT_THROW(nu::Vae(0, 16, 4), std::invalid_argument);
    EXPECT_THROW(nu::Vae(8, 0, 4), std::invalid_argument);
    EXPECT_THROW(nu::Vae(8, 16, 0), std::invalid_argument);
}

// ── encode ─────────────────────────────────────────────────────────────────────

TEST(VaeTest, EncodeReturnsMuAndLogvarOfCorrectSize)
{
    nu::Vae vae(8, 16, 4);
    const std::vector<double> x(8, 0.5);
    const auto [mu, lv] = vae.encode(x);
    EXPECT_EQ(mu.size(), 4u);
    EXPECT_EQ(lv.size(), 4u);
}

TEST(VaeTest, EncodeSizeMismatchThrows)
{
    nu::Vae vae(8, 16, 4);
    EXPECT_THROW(vae.encode({ 0.1, 0.2 }), std::invalid_argument);
}

// ── decode ─────────────────────────────────────────────────────────────────────

TEST(VaeTest, DecodeReturnsCorrectSize)
{
    nu::Vae vae(8, 16, 4);
    const std::vector<double> z(4, 0.0);
    EXPECT_EQ(vae.decode(z).size(), 8u);
}

TEST(VaeTest, DecodeOutputInUnitInterval)
{
    nu::Vae vae(8, 16, 4, 0.001, 1);
    const std::vector<double> z(4, 1.0);
    const auto r = vae.decode(z);
    for (double v : r) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

TEST(VaeTest, DecodeSizeMismatchThrows)
{
    nu::Vae vae(8, 16, 4);
    EXPECT_THROW(vae.decode({ 0.0, 0.0 }), std::invalid_argument);
}

// ── reconstruct ───────────────────────────────────────────────────────────────

TEST(VaeTest, ReconstructReturnsCorrectSize)
{
    nu::Vae vae(8, 16, 4);
    const std::vector<double> x(8, 1.0);
    EXPECT_EQ(vae.reconstruct(x).size(), 8u);
}

TEST(VaeTest, ReconstructOutputInUnitInterval)
{
    nu::Vae vae(8, 16, 4, 0.001, 2);
    const std::vector<double> x = { 1, 0, 1, 0, 1, 0, 1, 0 };
    const auto r = vae.reconstruct(x);
    for (double v : r) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

// ── generate ──────────────────────────────────────────────────────────────────

TEST(VaeTest, GenerateReturnsCorrectSize)
{
    nu::Vae vae(8, 16, 4);
    EXPECT_EQ(vae.generate().size(), 8u);
}

TEST(VaeTest, GenerateOutputInUnitInterval)
{
    nu::Vae vae(8, 16, 4, 0.001, 3);
    const auto g = vae.generate();
    for (double v : g) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
}

// ── reconstructionError ───────────────────────────────────────────────────────

TEST(VaeTest, ReconstructionErrorNonNegative)
{
    nu::Vae vae(8, 16, 4);
    const std::vector<std::vector<double>> data = {
        { 1, 0, 1, 0, 1, 0, 1, 0 },
        { 0, 1, 0, 1, 0, 1, 0, 1 },
    };
    EXPECT_GE(vae.reconstructionError(data), 0.0);
}

TEST(VaeTest, ReconstructionErrorEmptyDataReturnsZero)
{
    nu::Vae vae(8, 16, 4);
    EXPECT_EQ(vae.reconstructionError({}), 0.0);
}

// ── trainStep ─────────────────────────────────────────────────────────────────

TEST(VaeTest, TrainStepReturnsNonNegativeLosses)
{
    nu::Vae vae(8, 16, 4, 0.001, 5);
    const std::vector<double> x = { 1, 1, 1, 1, 0, 0, 0, 0 };
    const auto [lr, lkl] = vae.trainStep(x);
    EXPECT_GE(lr, 0.0);
    EXPECT_GE(lkl, 0.0);
}

// ── train ─────────────────────────────────────────────────────────────────────

TEST(VaeTest, TrainReducesReconstructionError)
{
    // 4 prototype 8-bit patterns, 50 noisy copies each
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

    nu::Vae vae(8, 32, 4, 0.003, 42);
    const double errBefore = vae.reconstructionError(data);
    vae.train(data, 500, 0.4); // KL warm-up over first 40% of epochs
    const double errAfter = vae.reconstructionError(data);
    EXPECT_LT(errAfter, errBefore);
}
