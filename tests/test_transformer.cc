//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_transformer.h"

#include <gtest/gtest.h>
#include <vector>

// ── LayerNorm ─────────────────────────────────────────────────────────────────

TEST(LayerNormTest, OutputShape)
{
    nu::LayerNorm ln(8);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 8);
    auto out = ln.forward(x);
    EXPECT_EQ(out.rows(), 4);
    EXPECT_EQ(out.cols(), 8);
}

TEST(LayerNormTest, ForwardNormalisesRows)
{
    nu::LayerNorm ln(16);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(6, 16) * 5.0;
    auto out = ln.forward(x);
    // After LayerNorm with gamma=1, beta=0 each row should have mean≈0, std≈1.
    for (Eigen::Index r = 0; r < out.rows(); ++r) {
        EXPECT_NEAR(out.row(r).mean(), 0.0, 1e-9);
        EXPECT_NEAR(out.row(r).norm() / std::sqrt(static_cast<double>(out.cols())), 1.0, 1e-6);
    }
}

TEST(LayerNormTest, BackwardShape)
{
    nu::LayerNorm ln(8);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 8);
    ln.forward(x);
    auto dx = ln.backward(Eigen::MatrixXd::Ones(4, 8), 0.001);
    EXPECT_EQ(dx.rows(), 4);
    EXPECT_EQ(dx.cols(), 8);
}

// ── SelfAttentionLayer ────────────────────────────────────────────────────────

TEST(SelfAttentionTest, OutputShape)
{
    nu::SelfAttentionLayer attn(8, 2);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(5, 8);
    auto out = attn.forward(x);
    EXPECT_EQ(out.rows(), 5);
    EXPECT_EQ(out.cols(), 8);
}

TEST(SelfAttentionTest, BackwardShape)
{
    nu::SelfAttentionLayer attn(8, 2, 0.001);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 8);
    attn.forward(x, /*causal=*/true);
    auto dx = attn.backward(Eigen::MatrixXd::Ones(4, 8), 0.001);
    EXPECT_EQ(dx.rows(), 4);
    EXPECT_EQ(dx.cols(), 8);
}

TEST(SelfAttentionTest, InvalidNumHeadsThrows)
{
    EXPECT_THROW(nu::SelfAttentionLayer(7, 3), std::invalid_argument);
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TEST(TransformerBlockTest, OutputShape)
{
    nu::TransformerBlock block(8, 2, 16);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 8);
    auto out = block.forward(x, /*causal=*/true);
    EXPECT_EQ(out.rows(), 4);
    EXPECT_EQ(out.cols(), 8);
}

TEST(TransformerBlockTest, BackwardShape)
{
    nu::TransformerBlock block(8, 2, 16, 0.001);
    Eigen::MatrixXd x = Eigen::MatrixXd::Random(4, 8);
    block.forward(x, /*causal=*/true);
    auto dx = block.backward(Eigen::MatrixXd::Ones(4, 8), 0.001);
    EXPECT_EQ(dx.rows(), 4);
    EXPECT_EQ(dx.cols(), 8);
}

// ── MiniTransformer ───────────────────────────────────────────────────────────

TEST(MiniTransformerTest, ForwardOutputShape)
{
    nu::MiniTransformer mt(10, 4, 8, 2, 16, 1, 0.001);
    auto logits = mt.forward({ 0, 1, 2, 3 });
    EXPECT_EQ(logits.rows(), 4);
    EXPECT_EQ(logits.cols(), 10);
}

TEST(MiniTransformerTest, GenerateLength)
{
    nu::MiniTransformer mt(10, 4, 8, 2, 16, 1, 0.001);
    std::mt19937 rng(42);
    auto gen = mt.generate({ 0, 1 }, 5, 1.0, &rng);
    EXPECT_EQ(gen.size(), 5u);
}

TEST(MiniTransformerTest, GeneratedTokensInVocab)
{
    nu::MiniTransformer mt(10, 4, 8, 2, 16, 1, 0.001);
    std::mt19937 rng(0);
    auto gen = mt.generate({ 3 }, 10, 1.0, &rng);
    for (int tok : gen)
        EXPECT_GE(tok, 0) << "token out of range";
    for (int tok : gen)
        EXPECT_LT(tok, 10) << "token out of range";
}

TEST(MiniTransformerTest, TrainDecreasesLoss)
{
    // Tiny model; trains on memorising a 4-token sequence.
    nu::MiniTransformer mt(5, 4, 16, 2, 32, 2, 0.01);

    const std::vector<int> inputs = { 0, 1, 2, 3 };
    const std::vector<int> targets = { 1, 2, 3, 4 };

    // Measure initial loss (first train call).
    const double loss0 = mt.train(inputs, targets);

    for (int ep = 0; ep < 500; ++ep)
        mt.train(inputs, targets);

    const double lossF = mt.train(inputs, targets);
    EXPECT_LT(lossF, loss0);
}
