//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// GRU (Gated Recurrent Unit) with truncated Backpropagation Through Time.
//
// Architecture (Cho et al., 2014):
//   r_t = σ(Wr·x_t + Ur·h_{t-1} + b_r)           reset gate
//   z_t = σ(Wz·x_t + Uz·h_{t-1} + b_z)           update gate
//   g_t = tanh(Wh·x_t + Uh·(r_t ⊙ h_{t-1}) + b_h) candidate hidden state
//   h_t = (1 − z_t) ⊙ h_{t-1}  +  z_t ⊙ g_t     new hidden state
//   y_t = f_out(Wy·h_t + b_y)                      output
//
// Compared with LSTM:
//   - No separate cell state; z_t merges the forget/input gates into one.
//   - r_t gates how much of the previous hidden state influences the candidate.
//   - ~25% fewer parameters than an LSTM of equal hidden size.
//
// Weight layout:
//   _W   [3·nh × ni] — input weights stacked [r; z; h]
//   _Urz [2·nh × nh] — recurrent weights for r and z (single GEMV)
//   _Uh  [  nh × nh] — recurrent weight for candidate (applied to r⊙h_{t-1})
//   _b   [3·nh]      — biases stacked [br; bz; bh]
//
// RnnOutput (Linear or Softmax) is defined in nu_rnn.h.

#pragma once

#include "nu_rnn.h"

#include <Eigen/Core>
#include <vector>

namespace nu {

class Gru {
public:
    // inputSize   — dimensionality of x_t
    // hiddenSize  — number of GRU units
    // outputSize  — dimensionality of y_t
    // lr          — SGD learning rate
    // gradClip    — element-wise gradient clipping threshold
    // outMode     — Linear (regression, MSE) or Softmax (classification, CE)
    Gru(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr = 0.01,
        double gradClip = 5.0, RnnOutput outMode = RnnOutput::Linear);

    // Reset hidden state to zero.
    void resetState();

    // Feed one time step; updates h and output y.
    void step(const std::vector<double>& x);

    const std::vector<double>& getOutput() const noexcept { return _y; }
    const std::vector<double>& getHidden() const noexcept { return _h; }

    // Run truncated BPTT over a full sequence and update weights.
    // Returns the mean loss over T steps.
    double bptt(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets, size_t truncate = 25);

    // Reinitialise all weights (Xavier normal); biases zero.
    void reshuffleWeights();

    size_t getInputSize() const noexcept { return _ni; }
    size_t getHiddenSize() const noexcept { return _nh; }
    size_t getOutputSize() const noexcept { return _no; }
    double getLearningRate() const noexcept { return _lr; }
    double getGradClip() const noexcept { return _gradClip; }
    RnnOutput getOutputMode() const noexcept { return _outMode; }

    void setLearningRate(double lr) noexcept { _lr = lr; }

private:
    size_t _ni, _nh, _no;
    double _lr, _gradClip;
    RnnOutput _outMode;

    Eigen::MatrixXd _W; // [3·nh × ni]  input weights  [r; z; h]
    Eigen::MatrixXd _Urz; // [2·nh × nh]  recurrent weights for r and z
    Eigen::MatrixXd _Uh; // [  nh × nh]  recurrent weight for candidate
    Eigen::VectorXd _b; // [3·nh]        biases [br; bz; bh]

    Eigen::MatrixXd _Wy; // [no × nh]
    Eigen::VectorXd _by; // [no]

    Eigen::VectorXd _h_prev;

    std::vector<double> _y;
    std::vector<double> _h;

    // Per-step forward result.
    struct StepResult {
        Eigen::VectorXd r; // reset gate
        Eigen::VectorXd z; // update gate
        Eigen::VectorXd g; // candidate hidden state
        Eigen::VectorXd rh; // r ⊙ h_prev (needed for dUh in backward)
        Eigen::VectorXd h; // new hidden state
        Eigen::VectorXd y; // output
    };

    StepResult _stepEigen(const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev) const;

    static Eigen::VectorXd _sigmoid(const Eigen::VectorXd& z);
    static Eigen::VectorXd _softmax(const Eigen::VectorXd& z);
    static void _clip(Eigen::MatrixXd& m, double c);
    static void _clip(Eigen::VectorXd& v, double c);
};

} // namespace nu
