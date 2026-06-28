//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_som.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// ── Construction ───────────────────────────────────────────────────────────────

TEST(SomTest, ConstructionStoresDimensions)
{
    nu::Som som(4, 5, 3);
    EXPECT_EQ(som.rows(), 4u);
    EXPECT_EQ(som.cols(), 5u);
    EXPECT_EQ(som.inputDim(), 3u);
}

TEST(SomTest, InvalidDimensionThrows)
{
    EXPECT_THROW(nu::Som(0, 5, 3), std::invalid_argument);
    EXPECT_THROW(nu::Som(4, 0, 3), std::invalid_argument);
    EXPECT_THROW(nu::Som(4, 5, 0), std::invalid_argument);
}

// ── getWeights ────────────────────────────────────────────────────────────────

TEST(SomTest, GetWeightsReturnsCorrectSize)
{
    nu::Som som(3, 4, 6);
    const auto w = som.getWeights(1, 2);
    EXPECT_EQ(w.size(), 6u);
}

TEST(SomTest, GetWeightsOutOfRangeThrows)
{
    nu::Som som(3, 4, 6);
    EXPECT_THROW(som.getWeights(3, 0), std::out_of_range);
    EXPECT_THROW(som.getWeights(0, 4), std::out_of_range);
}

TEST(SomTest, InitialWeightsInUnitInterval)
{
    nu::Som som(4, 4, 4, 0.5, 0.0, 0);
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c) {
            auto w = som.getWeights(r, c);
            for (double v : w) {
                EXPECT_GE(v, 0.0);
                EXPECT_LE(v, 1.0);
            }
        }
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

TEST(SomTest, ReshuffleChangesWeights)
{
    nu::Som som(4, 4, 4, 0.5, 0.0, 42);
    const auto w_before = som.getWeights(2, 2);
    som.reshuffleWeights();
    const auto w_after = som.getWeights(2, 2);
    // With high probability weights change after reshuffle with new RNG state
    bool changed = false;
    for (size_t i = 0; i < w_before.size(); ++i)
        if (w_before[i] != w_after[i]) {
            changed = true;
            break;
        }
    EXPECT_TRUE(changed);
}

// ── bmu ───────────────────────────────────────────────────────────────────────

TEST(SomTest, BmuReturnsValidPosition)
{
    nu::Som som(4, 5, 3);
    const auto [r, c] = som.bmu({ 0.5, 0.5, 0.5 });
    EXPECT_LT(r, 4u);
    EXPECT_LT(c, 5u);
}

TEST(SomTest, BmuSizeMismatchThrows)
{
    nu::Som som(4, 4, 3);
    EXPECT_THROW(som.bmu({ 0.1, 0.2 }), std::invalid_argument);
}

TEST(SomTest, BmuIsDeterministic)
{
    nu::Som som(4, 4, 2, 0.5, 0.0, 7);
    const std::vector<double> x = { 0.3, 0.7 };
    EXPECT_EQ(som.bmu(x), som.bmu(x));
}

// ── Single-neuron SOM ─────────────────────────────────────────────────────────

TEST(SomTest, SingleNeuronBmuAlwaysReturnsOrigin)
{
    nu::Som som(1, 1, 4);
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> x(4, static_cast<double>(k) / 10.0);
        EXPECT_EQ(som.bmu(x), (std::pair<size_t, size_t>{ 0, 0 }));
    }
}

// ── quantizationError ─────────────────────────────────────────────────────────

TEST(SomTest, QuantizationErrorNonNegative)
{
    nu::Som som(4, 4, 2);
    const std::vector<std::vector<double>> data = { { 0.1, 0.9 }, { 0.8, 0.2 }, { 0.5, 0.5 } };
    EXPECT_GE(som.quantizationError(data), 0.0);
}

TEST(SomTest, QuantizationErrorEmptyDataReturnsZero)
{
    nu::Som som(3, 3, 2);
    EXPECT_EQ(som.quantizationError({}), 0.0);
}

// ── train ─────────────────────────────────────────────────────────────────────

TEST(SomTest, TrainReducesQuantizationError)
{
    // 2D inputs from 4 tight Gaussian clusters at the corners of [0,1]^2
    std::mt19937 rng(0);
    std::normal_distribution<double> noise(0.0, 0.05);
    const std::vector<std::array<double, 2>> centres
        = { { 0.1, 0.1 }, { 0.1, 0.9 }, { 0.9, 0.1 }, { 0.9, 0.9 } };
    std::vector<std::vector<double>> data;
    for (const auto& c : centres)
        for (int i = 0; i < 50; ++i)
            data.push_back({ c[0] + noise(rng), c[1] + noise(rng) });

    nu::Som som(6, 6, 2, 0.5, 0.0, 42);
    const double qeBefore = som.quantizationError(data);
    som.train(data, 100);
    const double qeAfter = som.quantizationError(data);
    EXPECT_LT(qeAfter, qeBefore);
}
