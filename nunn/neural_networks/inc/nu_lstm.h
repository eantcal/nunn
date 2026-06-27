//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// LSTM (Long Short-Term Memory) with truncated Backpropagation Through Time.
//
// Architecture:
//   i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)      input  gate
//   f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)      forget gate   (bias init = 1)
//   o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)      output gate
//   g_t = tanh(W_g·x_t + U_g·h_{t-1} + b_g)   cell candidate
//   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t            cell state
//   h_t = o_t ⊙ tanh(c_t)                       hidden state
//   y_t = f_out(W_y·h_t + b_y)                  output
//
// The four gate weight matrices are stacked vertically [i; f; o; g] to allow
// a single GEMV per step: pre = W·x + U·h + b, then split into 4 blocks.
//
// RnnOutput (Linear or Softmax) is defined in nu_rnn.h.

#pragma once

#include "nu_rnn.h"

#include <Eigen/Core>
#include <vector>

namespace nu {

class Lstm {
public:
    // inputSize   — dimensionality of x_t
    // hiddenSize  — number of LSTM units (= cell size)
    // outputSize  — dimensionality of y_t
    // lr          — SGD learning rate
    // gradClip    — element-wise gradient clipping threshold
    // outMode     — Linear (regression, MSE) or Softmax (classification, CE)
    Lstm(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr = 0.01,
        double gradClip = 5.0, RnnOutput outMode = RnnOutput::Linear);

    // Reset hidden state h and cell state c to zero.
    void resetState();

    // Feed one time step; updates h, c, and output y.
    void step(const std::vector<double>& x);

    const std::vector<double>& getOutput() const noexcept { return _y; }
    const std::vector<double>& getHidden() const noexcept { return _h; }

    // Run truncated BPTT over a full sequence and update weights.
    // Returns the mean loss over T steps.
    // The cell and hidden states are advanced to the end of the sequence.
    double bptt(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets, size_t truncate = 25);

    // Reinitialise all weights (Xavier normal); forget-gate bias set to 1.
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

    // Stacked gate weight matrices.
    // Row layout: [i gate rows (0..nh); f gate rows (nh..2nh);
    //              o gate rows (2nh..3nh); g rows (3nh..4nh)]
    Eigen::MatrixXd _W; // [4·nh × ni]  input-to-gate
    Eigen::MatrixXd _U; // [4·nh × nh]  recurrent-to-gate
    Eigen::VectorXd _b; // [4·nh]       gate biases

    Eigen::MatrixXd _Wy; // [no × nh]   hidden-to-output
    Eigen::VectorXd _by; // [no]         output bias

    Eigen::VectorXd _h_prev; // [nh]  last hidden state
    Eigen::VectorXd _c_prev; // [nh]  last cell state

    std::vector<double> _y; // last output (public accessor)
    std::vector<double> _h; // last hidden (public accessor)

    // Per-step forward result (gate activations + new states).
    struct StepResult {
        Eigen::VectorXd i, f, o, g; // gate activations (post-nonlinearity)
        Eigen::VectorXd c, h; // new cell and hidden states
        Eigen::VectorXd y; // network output
    };

    StepResult _stepEigen(const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev,
        const Eigen::VectorXd& c_prev) const;

    static Eigen::VectorXd _sigmoid(const Eigen::VectorXd& z);
    static Eigen::VectorXd _softmax(const Eigen::VectorXd& z);
    static void _clip(Eigen::MatrixXd& m, double c);
    static void _clip(Eigen::VectorXd& v, double c);
};

} // namespace nu
