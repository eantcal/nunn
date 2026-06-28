//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Decoder-only mini transformer for character-level language modelling.
//
// Architecture (Pre-LN style):
//   Input tokens → Embedding + Positional Encoding
//   → N × TransformerBlock( LN → MH-Attention → residual → LN → FFN → residual )
//   → Output projection → logits
//

#pragma once

#include "nu_activation.h"

#include <Eigen/Core>
#include <random>
#include <vector>

namespace nu {

// ── LayerNorm ─────────────────────────────────────────────────────────────────
// Normalises each row of the input matrix (one row = one token's embedding).

class LayerNorm {
public:
    // dModel: embedding dimension.
    explicit LayerNorm(size_t dModel, double eps = 1e-5);

    // x: [seqLen × dModel] → normalised [seqLen × dModel]
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x);

    // Backprop; updates gamma / beta and returns dL/dx.
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad, double lr);

private:
    size_t _d;
    double _eps;
    Eigen::VectorXd _gamma, _beta; // learnable scale and shift [dModel]
    Eigen::MatrixXd _xhat, _xin; // saved for backward
    Eigen::VectorXd _invStd; // 1/sqrt(var+eps) per row [seqLen]
};

// ── SelfAttentionLayer ────────────────────────────────────────────────────────
// Multi-head scaled dot-product self-attention.
// Each head operates on a dk=dModel/numHeads sub-space.

class SelfAttentionLayer {
public:
    // numHeads must evenly divide dModel.
    SelfAttentionLayer(size_t dModel, size_t numHeads, double lr = 0.001);

    // x: [seqLen × dModel] → [seqLen × dModel]
    // causal=true adds an upper-triangular mask (autoregressive).
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, bool causal = false);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOut, double lr = 0.0);

private:
    size_t _d, _h, _dk;
    double _lr;

    // Per-head projections [d × dk]; _WO is the output projection [d × d].
    std::vector<Eigen::MatrixXd> _WQh, _WKh, _WVh;
    Eigen::MatrixXd _WO;
    Eigen::VectorXd _bO;

    // Saved for backward.
    Eigen::MatrixXd _xin, _concat;
    std::vector<Eigen::MatrixXd> _Qh, _Kh, _Vh, _Attn_h, _headOut_h;
};

// ── TransformerBlock ──────────────────────────────────────────────────────────
// Pre-LN block:
//   x → LN → MH-Attn → + x (residual)
//     → LN → FFN(ReLU) → + (residual)

class TransformerBlock {
public:
    // dFF: hidden dimension of the two-layer feed-forward sublayer.
    TransformerBlock(size_t dModel, size_t numHeads, size_t dFF, double lr = 0.001);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, bool causal = false);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradOut, double lr = 0.0);

private:
    double _lr;
    LayerNorm _ln1, _ln2;
    SelfAttentionLayer _attn;
    Eigen::MatrixXd _W1, _W2; // FFN: [d × dFF], [dFF × d]
    Eigen::VectorXd _b1, _b2;

    // Saved for backward.
    Eigen::MatrixXd _xin, _ln1out, _attnOut, _ln2out, _h1act;
};

// ── MiniTransformer ───────────────────────────────────────────────────────────
// Decoder-only model for next-token prediction (character LM).

class MiniTransformer {
public:
    // vocabSize: number of distinct tokens.
    // seqLen:    fixed context window (number of tokens per forward pass).
    // dModel:    embedding / model dimension (must be divisible by numHeads).
    // numLayers: number of stacked TransformerBlocks.
    MiniTransformer(size_t vocabSize, size_t seqLen, size_t dModel, size_t numHeads, size_t dFF,
        size_t numLayers, double lr = 0.001);

    // Forward pass; returns logit matrix [seqLen × vocabSize].
    Eigen::MatrixXd forward(const std::vector<int>& tokens);

    // Train one (inputs, targets) pair with shift-by-1 LM loss.
    // Returns mean cross-entropy over the sequence.
    double train(const std::vector<int>& inputs, const std::vector<int>& targets);

    // Autoregressive generation starting from `prompt`.
    std::vector<int> generate(const std::vector<int>& prompt, size_t nTokens,
        double temperature = 1.0, std::mt19937* rng = nullptr);

    size_t vocabSize() const noexcept { return _V; }
    size_t seqLen() const noexcept { return _T; }
    size_t dModel() const noexcept { return _d; }

private:
    size_t _V, _T, _d;
    double _lr;
    Eigen::MatrixXd _embed; // [V × d]  token embeddings
    Eigen::MatrixXd _posEnc; // [T × d]  fixed sinusoidal positional encoding
    std::vector<TransformerBlock> _blocks;
    Eigen::MatrixXd _Wout; // [d × V]  output projection
    Eigen::VectorXd _bout; // [V]

    // Saved for backward.
    Eigen::MatrixXd _xemb, _xfinal;
    std::vector<int> _lastTokens;

    static Eigen::MatrixXd _makePosEnc(size_t T, size_t d);
};

} // namespace nu
