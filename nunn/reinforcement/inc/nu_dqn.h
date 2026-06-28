//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Deep Q-Network (DQN) with experience replay and a frozen target network.
//
// Algorithm outline:
//   1. Every env step: store (s, a, r, s', done) in replay buffer.
//   2. Once buffer has >= batchSize transitions: sample a mini-batch,
//      compute Bellman targets via the TARGET network (frozen), update
//      MAIN network weights via trainBatch().
//   3. Every targetUpdateFreq learn steps: copy main → target weights.
//
// State is std::vector<double>; Action is int (0-based action index).
//

#pragma once

#include "nu_mlpmatrixnn.h"
#include "nu_replay_buffer.h"

#include <memory>
#include <random>
#include <vector>

namespace nu {

class Dqn {
public:
    // netLayers: full topology including input descriptor at [0] and output at [N-1].
    //   The output layer must have Activation::Linear (Q-values are unbounded).
    //   Example: {{2}, {32, Activation::Tanh}, {32, Activation::Tanh}, {4, Activation::Linear}}
    // lr:               learning rate for the main Q-network
    // bufferCapacity:   maximum transitions stored in the replay buffer
    // batchSize:        transitions sampled per gradient step
    // gamma:            discount factor
    // targetUpdateFreq: copy main→target every N learn steps
    //
    // Throws std::invalid_argument if netLayers has fewer than 2 entries or batchSize == 0.
    Dqn(const std::vector<MlpMatrixNN::LayerConfig>& netLayers, double lr = 0.001,
        size_t bufferCapacity = 10000, size_t batchSize = 32, double gamma = 0.99,
        size_t targetUpdateFreq = 100);

    // epsilon-greedy action selection.
    // epsilon == 0 → always greedy; epsilon == 1 → always random.
    int selectAction(const std::vector<double>& state, double epsilon);

    // Store transition and, if buffer is ready, run one mini-batch gradient step.
    // Returns the pre-update batch MSE loss when training happened, 0.0 otherwise.
    double learn(const std::vector<double>& state, int action, double reward,
        const std::vector<double>& nextState, bool done);

    // Q-values for state from the main network.
    std::vector<double> qValues(const std::vector<double>& state);

    size_t getNumActions() const noexcept { return _numActions; }
    size_t getLearnStepCount() const noexcept { return _learnStep; }

private:
    using Buffer = ExperienceReplayBuffer<std::vector<double>, int>;

    std::unique_ptr<MlpMatrixNN> _qNet;
    std::unique_ptr<MlpMatrixNN> _targetNet;
    Buffer _buffer;
    size_t _batchSize;
    double _gamma;
    size_t _targetUpdateFreq;
    size_t _learnStep = 0;
    size_t _numActions;
    std::mt19937 _rng;

    void _syncTarget();
    double _trainBatch();
};

} // namespace nu
