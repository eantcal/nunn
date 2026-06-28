//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_dqn.h"
#include "nu_replay_buffer.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

// ── ExperienceReplayBuffer ────────────────────────────────────────────────────

TEST(ReplayBufferTest, ZeroCapacityThrows)
{
    EXPECT_THROW((nu::ExperienceReplayBuffer<std::vector<double>, int>(0)), std::invalid_argument);
}

TEST(ReplayBufferTest, PushAndSize)
{
    nu::ExperienceReplayBuffer<std::vector<double>, int> buf(100);
    EXPECT_EQ(buf.size(), 0u);
    buf.push({ 0.0 }, 0, 1.0, { 0.0 }, false);
    EXPECT_EQ(buf.size(), 1u);
    buf.push({ 0.0 }, 1, 0.0, { 0.0 }, true);
    EXPECT_EQ(buf.size(), 2u);
}

TEST(ReplayBufferTest, CapsAtCapacity)
{
    nu::ExperienceReplayBuffer<std::vector<double>, int> buf(3);
    for (int i = 0; i < 10; ++i)
        buf.push({ static_cast<double>(i) }, 0, 0.0, { 0.0 }, false);
    EXPECT_EQ(buf.size(), 3u);
}

TEST(ReplayBufferTest, SampleCorrectSize)
{
    nu::ExperienceReplayBuffer<std::vector<double>, int> buf(100);
    for (int i = 0; i < 20; ++i)
        buf.push({ static_cast<double>(i) }, 0, 0.0, { 0.0 }, false);
    std::mt19937 rng(42);
    const auto batch = buf.sample(8, rng);
    EXPECT_EQ(batch.size(), 8u);
}

TEST(ReplayBufferTest, SampleThrowsIfNotReady)
{
    nu::ExperienceReplayBuffer<std::vector<double>, int> buf(100);
    buf.push({ 0.0 }, 0, 1.0, { 0.0 }, false);
    std::mt19937 rng(0);
    EXPECT_THROW(buf.sample(10, rng), std::invalid_argument);
}

TEST(ReplayBufferTest, ReadyFlag)
{
    nu::ExperienceReplayBuffer<std::vector<double>, int> buf(100);
    EXPECT_FALSE(buf.ready(1));
    buf.push({ 0.0 }, 0, 0.0, { 0.0 }, false);
    EXPECT_TRUE(buf.ready(1));
    EXPECT_FALSE(buf.ready(2));
}

// ── Dqn construction ──────────────────────────────────────────────────────────

static std::vector<nu::MlpMatrixNN::LayerConfig> makeLayers()
{
    using LC = nu::MlpMatrixNN::LayerConfig;
    return { LC(2), LC(8, nu::Activation::Tanh), LC(4, nu::Activation::Linear) };
}

TEST(DqnTest, ConstructionValid)
{
    nu::Dqn dqn(makeLayers(), 0.01, 200, 16, 0.99, 50);
    EXPECT_EQ(dqn.getNumActions(), 4u);
    EXPECT_EQ(dqn.getLearnStepCount(), 0u);
}

TEST(DqnTest, TooFewLayersThrows)
{
    using LC = nu::MlpMatrixNN::LayerConfig;
    EXPECT_THROW(nu::Dqn({ LC(4, nu::Activation::Linear) }, 0.01), std::invalid_argument);
}

TEST(DqnTest, ZeroBatchSizeThrows)
{
    EXPECT_THROW(nu::Dqn(makeLayers(), 0.01, 200, 0), std::invalid_argument);
}

// ── Action selection ──────────────────────────────────────────────────────────

TEST(DqnTest, SelectActionInRange)
{
    nu::Dqn dqn(makeLayers(), 0.01, 200, 16, 0.99, 50);
    for (int trial = 0; trial < 50; ++trial) {
        const int a = dqn.selectAction({ 0.5, 0.5 }, 1.0); // fully random
        EXPECT_GE(a, 0);
        EXPECT_LT(a, 4);
    }
}

TEST(DqnTest, GreedySelectActionInRange)
{
    nu::Dqn dqn(makeLayers(), 0.01, 200, 16, 0.99, 50);
    const int a = dqn.selectAction({ 0.5, 0.5 }, 0.0); // greedy
    EXPECT_GE(a, 0);
    EXPECT_LT(a, 4);
}

// ── Learn ─────────────────────────────────────────────────────────────────────

TEST(DqnTest, LearnReturnsZeroBeforeBufferReady)
{
    nu::Dqn dqn(makeLayers(), 0.01, 200, 16, 0.99, 50);
    const double loss = dqn.learn({ 0.0, 0.0 }, 0, 1.0, { 0.0, 0.0 }, false);
    EXPECT_EQ(loss, 0.0);
    EXPECT_EQ(dqn.getLearnStepCount(), 0u);
}

TEST(DqnTest, LearnStepCountIncrements)
{
    nu::Dqn dqn(makeLayers(), 0.01, 200, 8, 0.99, 1000);
    // Fill buffer past batchSize.
    for (int i = 0; i < 20; ++i)
        dqn.learn({ 0.0, 0.0 }, 0, 0.0, { 0.0, 0.0 }, false);
    EXPECT_GT(dqn.getLearnStepCount(), 0u);
}

// ── Convergence: bandit problem ───────────────────────────────────────────────

// State is always [0.5]; action 0 → reward 1.0 (good), action 1 → reward 0.0.
// After training, the greedy action should be 0.
TEST(DqnTest, ConvergesOnBanditProblem)
{
    using LC = nu::MlpMatrixNN::LayerConfig;
    const std::vector<nu::MlpMatrixNN::LayerConfig> layers{ LC(1), LC(16, nu::Activation::Tanh),
        LC(2, nu::Activation::Linear) };
    nu::Dqn dqn(layers, 0.01, 500, 32, 0.0 /*no discounting*/, 1000);

    const std::vector<double> s{ 0.5 };
    const std::vector<double> s_next{ 0.5 };

    for (int step = 0; step < 2000; ++step) {
        // Epsilon-greedy with decay.
        const double eps = std::max(0.05, 1.0 - step / 1000.0);
        const int a = dqn.selectAction(s, eps);
        const double r = (a == 0) ? 1.0 : 0.0;
        dqn.learn(s, a, r, s_next, true);
    }

    EXPECT_EQ(dqn.selectAction(s, 0.0), 0);
}
