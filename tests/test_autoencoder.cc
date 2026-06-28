//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#define _USE_MATH_DEFINES
#include "nu_autoencoder.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Construction ──────────────────────────────────────────────────────────────

TEST(AutoencoderTest, DimensionsMatchConstruction)
{
    nu::Autoencoder ae(8, { 4, 2 });
    EXPECT_EQ(ae.getInputSize(), 8u);
    EXPECT_EQ(ae.getLatentSize(), 2u);
}

TEST(AutoencoderTest, SingleLayerBottleneck)
{
    nu::Autoencoder ae(4, { 2 });
    EXPECT_EQ(ae.getInputSize(), 4u);
    EXPECT_EQ(ae.getLatentSize(), 2u);
}

TEST(AutoencoderTest, EmptyEncoderSizesThrows)
{
    EXPECT_THROW(nu::Autoencoder(4, {}), std::invalid_argument);
}

// ── Output sizes ──────────────────────────────────────────────────────────────

TEST(AutoencoderTest, EncodeOutputMatchesLatentSize)
{
    nu::Autoencoder ae(6, { 3, 2 });
    const std::vector<double> x = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const auto z = ae.encode(x);
    EXPECT_EQ(z.size(), 2u);
}

TEST(AutoencoderTest, DecodeOutputMatchesInputDim)
{
    nu::Autoencoder ae(6, { 3, 2 });
    const std::vector<double> z = { 0.5, -0.3 };
    const auto rec = ae.decode(z);
    EXPECT_EQ(rec.size(), 6u);
}

TEST(AutoencoderTest, ReconstructOutputMatchesInputDim)
{
    nu::Autoencoder ae(6, { 3, 2 });
    const std::vector<double> x = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const auto rec = ae.reconstruct(x);
    EXPECT_EQ(rec.size(), 6u);
}

// ── Reshuffleweights does not crash ───────────────────────────────────────────

TEST(AutoencoderTest, ReshuffleWeightsDoesNotCrash)
{
    nu::Autoencoder ae(4, { 2 });
    EXPECT_NO_THROW(ae.reshuffleWeights());
    const std::vector<double> x = { 0.1, 0.9, 0.4, 0.6 };
    EXPECT_EQ(ae.reconstruct(x).size(), 4u);
}

// ── encode + decode matches reconstruct after sync ───────────────────────────

TEST(AutoencoderTest, EncodeDecodeMimicsReconstruct)
{
    nu::Autoencoder ae(4, { 2 }, nu::Activation::Tanh, 0.01);

    // Build trivial dataset and train briefly to get non-trivial weights
    std::vector<std::vector<double>> ds
        = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
    ae.train(ds, 50);

    // After train(), decoder is synced: encode+decode should equal reconstruct
    const std::vector<double> x = ds[0];
    const auto z = ae.encode(x);
    const auto dec = ae.decode(z);
    const auto rec = ae.reconstruct(x);

    ASSERT_EQ(dec.size(), rec.size());
    for (size_t i = 0; i < rec.size(); ++i)
        EXPECT_NEAR(dec[i], rec[i], 1e-9);
}

// ── Convergence ───────────────────────────────────────────────────────────────

static double meanMSE(nu::Autoencoder& ae, const std::vector<std::vector<double>>& ds)
{
    double total = 0.0;
    for (const auto& x : ds)
        total += ae.reconstructionMSE(x);
    return total / static_cast<double>(ds.size());
}

TEST(AutoencoderTest, TrainingReducesMSEOnOneHotData)
{
    // 4D one-hot vectors, bottleneck = 2. Verify training reduces MSE by >= 30%.
    std::vector<std::vector<double>> ds
        = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };

    // Try multiple seeds; accept the run with greatest MSE reduction.
    double bestRatio = 1.0; // ratio = finalMSE / initialMSE; smaller is better
    for (int trial = 0; trial < 5; ++trial) {
        nu::Autoencoder ae(4, { 4, 2 }, nu::Activation::Tanh, 0.01);
        const double mseBefore = meanMSE(ae, ds);
        ae.train(ds, 3000);
        const double mseAfter = meanMSE(ae, ds);
        if (mseBefore > 0.0)
            bestRatio = std::min(bestRatio, mseAfter / mseBefore);
    }
    // Expect at least 30% reduction in at least one trial
    EXPECT_LT(bestRatio, 0.70);
}

TEST(AutoencoderTest, TrainingReducesMSEOnSineData)
{
    // 8-point sine wave snippets at 4 phases, bottleneck = 2.
    constexpr size_t N = 8;
    std::vector<std::vector<double>> ds;
    for (int p = 0; p < 4; ++p) {
        std::vector<double> x(N);
        for (size_t t = 0; t < N; ++t)
            x[t] = 0.5 + 0.5 * std::sin(2.0 * M_PI * t / N + p * M_PI / 2.0);
        ds.push_back(x);
    }

    double bestRatio = 1.0;
    for (int trial = 0; trial < 5; ++trial) {
        nu::Autoencoder ae(N, { 4, 2 }, nu::Activation::Tanh, 0.005);
        const double mseBefore = meanMSE(ae, ds);
        ae.train(ds, 2000);
        const double mseAfter = meanMSE(ae, ds);
        if (mseBefore > 0.0)
            bestRatio = std::min(bestRatio, mseAfter / mseBefore);
    }
    EXPECT_LT(bestRatio, 0.70);
}
