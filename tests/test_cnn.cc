//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#define _USE_MATH_DEFINES
#include "nu_conv.h"
#include "nu_convnet.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using LC = nu::MlpMatrixNN::LayerConfig;

// ── Conv1DLayer ───────────────────────────────────────────────────────────────

TEST(Conv1DLayerTest, OutputSize)
{
    // 2-channel, 10-step input; 4 filters of kernel=3
    // outLen = 10 - 3 + 1 = 8
    nu::Conv1DLayer layer(2, 10, 4, 3);
    EXPECT_EQ(layer.outChannels(), 4u);
    EXPECT_EQ(layer.outLength(), 8u);
    EXPECT_EQ(layer.outputSize(), 32u);
}

TEST(Conv1DLayerTest, BadDimensionThrows)
{
    EXPECT_THROW(nu::Conv1DLayer(0, 10, 4, 3), std::invalid_argument);
    EXPECT_THROW(nu::Conv1DLayer(1, 2, 4, 5), std::invalid_argument); // inLen < kSize
}

TEST(Conv1DLayerTest, ForwardOutputShape)
{
    nu::Conv1DLayer layer(1, 8, 4, 3);
    std::vector<double> in(8, 1.0);
    const auto& out = layer.forward(in);
    EXPECT_EQ(out.size(), 4u * 6u); // outLen = 8 - 3 + 1 = 6
}

TEST(Conv1DLayerTest, ForwardBackwardDoesNotCrash)
{
    nu::Conv1DLayer layer(1, 8, 2, 3);
    std::vector<double> in(8, 0.5);
    layer.forward(in);
    std::vector<double> grad(2 * 6, 0.1);
    const auto gradIn = layer.backward(grad, 0.01);
    EXPECT_EQ(gradIn.size(), 8u);
}

// ── MaxPool1DLayer ────────────────────────────────────────────────────────────

TEST(MaxPool1DLayerTest, OutputSize)
{
    // 3 channels, 12 steps, pool=3 → outLen=4
    nu::MaxPool1DLayer layer(3, 12, 3);
    EXPECT_EQ(layer.outChannels(), 3u);
    EXPECT_EQ(layer.outLength(), 4u);
    EXPECT_EQ(layer.outputSize(), 12u);
}

TEST(MaxPool1DLayerTest, SelectsMax)
{
    // 1 channel, 4 steps, pool=2 → outLen=2
    nu::MaxPool1DLayer layer(1, 4, 2);
    const std::vector<double> in{ 1.0, 3.0, 2.0, 0.5 };
    const auto& out = layer.forward(in);
    EXPECT_EQ(out.size(), 2u);
    EXPECT_DOUBLE_EQ(out[0], 3.0); // max(1.0, 3.0)
    EXPECT_DOUBLE_EQ(out[1], 2.0); // max(2.0, 0.5)
}

TEST(MaxPool1DLayerTest, BackwardRoutesGradient)
{
    // 1 channel, 4 steps, pool=2
    nu::MaxPool1DLayer layer(1, 4, 2);
    const std::vector<double> in{ 1.0, 3.0, 2.0, 0.5 };
    layer.forward(in);
    const std::vector<double> gradOut{ 0.5, 0.8 };
    const auto gradIn = layer.backward(gradOut, 0.0);
    ASSERT_EQ(gradIn.size(), 4u);
    EXPECT_DOUBLE_EQ(gradIn[0], 0.0); // pos 0 was not max
    EXPECT_DOUBLE_EQ(gradIn[1], 0.5); // pos 1 was max in window 0
    EXPECT_DOUBLE_EQ(gradIn[2], 0.8); // pos 2 was max in window 1
    EXPECT_DOUBLE_EQ(gradIn[3], 0.0); // pos 3 was not max
}

// ── ConvNet ───────────────────────────────────────────────────────────────────

TEST(ConvNetTest, FlatFeatureSize)
{
    // 1 channel, 16 steps → Conv1D(4,k=5) → [4,12] → MaxPool1D(4) → [4,3]
    nu::ConvNet cnn(1, 16);
    cnn.addConv1D(4, 5);
    cnn.addMaxPool1D(4);
    EXPECT_EQ(cnn.flatFeatureSize(), 12u); // 4 * 3
}

TEST(ConvNetTest, PredictBeforeSetFCThrows)
{
    nu::ConvNet cnn(1, 8);
    cnn.addConv1D(2, 3);
    EXPECT_THROW(cnn.predict(std::vector<double>(8, 0.0)), std::logic_error);
}

TEST(ConvNetTest, WrongFCInputSizeThrows)
{
    nu::ConvNet cnn(1, 8);
    cnn.addConv1D(2, 3); // outLen = 6, 2ch → flatFeatureSize=12
    EXPECT_THROW(cnn.setFCHead({ LC(5), LC(2, nu::Activation::Sigmoid) }), // 5 != 12
        std::invalid_argument);
}

TEST(ConvNetTest, PredictOutputSize)
{
    nu::ConvNet cnn(1, 16);
    cnn.addConv1D(4, 5);
    cnn.addMaxPool1D(4);
    cnn.setFCHead(
        { LC(cnn.flatFeatureSize()), LC(8, nu::Activation::Tanh), LC(2, nu::Activation::Sigmoid) });
    const auto out = cnn.predict(std::vector<double>(16, 0.0));
    EXPECT_EQ(out.size(), 2u);
}

TEST(ConvNetTest, TrainReducesLoss)
{
    // Frequency-detection problem: class 0 = 1 cycle, class 1 = 2 cycles.
    constexpr int T = 16;
    auto makeSample = [&](int freqCycles) {
        std::vector<double> x(T);
        for (int t = 0; t < T; ++t)
            x[t] = std::sin(2.0 * M_PI * freqCycles * t / T);
        return x;
    };
    const auto x0 = makeSample(1); // class 0
    const auto x1 = makeSample(2); // class 1
    const std::vector<double> t0{ 1.0, 0.0 };
    const std::vector<double> t1{ 0.0, 1.0 };

    nu::ConvNet cnn(1, T);
    cnn.addConv1D(4, 5, nu::Activation::Tanh, 0.01);
    cnn.addMaxPool1D(4);
    const auto fSize = cnn.flatFeatureSize(); // 4*(16-5+1)/4 = 4*3 = 12
    cnn.setFCHead({ LC(fSize), LC(8, nu::Activation::Tanh), LC(2, nu::Activation::Sigmoid) }, 0.01);

    // Record initial loss.
    const double loss0 = 0.5 * (cnn.train(x0, t0) + cnn.train(x1, t1));

    // Train 200 epochs.
    for (int ep = 0; ep < 200; ++ep) {
        cnn.train(x0, t0);
        cnn.train(x1, t1);
    }
    const double lossF = 0.5 * (cnn.train(x0, t0) + cnn.train(x1, t1));

    EXPECT_LT(lossF, loss0);
}
