//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Vanilla RNN (Elman network) with truncated Backpropagation Through Time.
//
// Architecture (one layer):
//   h_t = tanh(Wx·x_t + Wh·h_{t-1} + b_h)
//   y_t = f_out(Wy·h_t + b_y)
//
// f_out is Linear (regression/MSE) or Softmax (classification/CE).
//
// Weights are stored as Eigen matrices; the public API uses std::vector<double>
// to match the existing nunn interface conventions.

#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nu {

enum class RnnOutput {
    Linear, // identity output + MSE loss
    Softmax, // softmax output + cross-entropy loss
};

class VanillaRnn {
public:
    // inputSize   — dimensionality of x_t
    // hiddenSize  — number of hidden units
    // outputSize  — dimensionality of y_t
    // lr          — SGD learning rate
    // gradClip    — element-wise gradient clipping threshold
    // outMode     — Linear (regression) or Softmax (classification)
    VanillaRnn(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr = 0.01,
        double gradClip = 5.0, RnnOutput outMode = RnnOutput::Linear);

    // Reset hidden state to zero (call at the start of each new sequence).
    void resetState();

    // Feed one time step; updates hidden state and output.
    // x.size() must equal getInputSize().
    void step(const std::vector<double>& x);

    // Output after the last step().
    const std::vector<double>& getOutput() const noexcept { return _y; }

    // Hidden state after the last step().
    const std::vector<double>& getHidden() const noexcept { return _h; }

    // Run truncated BPTT over a full sequence and update weights.
    // inputs[t]  — input  at step t  (size == getInputSize())
    // targets[t] — target at step t  (size == getOutputSize())
    // truncate   — how many steps to unroll (TBPTT window)
    // Returns the mean loss over the sequence.
    // Hidden state is advanced to the end of the sequence.
    double bptt(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets, size_t truncate = 25);

    // Reinitialise all weights (Xavier normal) and zero the hidden state.
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
    double _lr;
    double _gradClip;
    RnnOutput _outMode;

    Eigen::MatrixXd _Wx; // [nh × ni]  input-to-hidden
    Eigen::MatrixXd _Wh; // [nh × nh]  hidden-to-hidden (recurrent)
    Eigen::VectorXd _bh; // [nh]       hidden bias
    Eigen::MatrixXd _Wy; // [no × nh]  hidden-to-output
    Eigen::VectorXd _by; // [no]       output bias

    Eigen::VectorXd _h_prev; // current (last) hidden state

    std::vector<double> _y; // last output (public accessor)
    std::vector<double> _h; // last hidden (public accessor)

    // One forward step returning (h_t, y_t) as Eigen vectors.
    std::pair<Eigen::VectorXd, Eigen::VectorXd> _stepEigen(
        const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev) const;

    static Eigen::VectorXd _softmax(const Eigen::VectorXd& z);
    static void _clip(Eigen::MatrixXd& m, double c);
    static void _clip(Eigen::VectorXd& v, double c);
};

} // namespace nu
