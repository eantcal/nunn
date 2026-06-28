//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_dqn.h"

#include <algorithm>
#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Dqn::Dqn(const std::vector<MlpMatrixNN::LayerConfig>& netLayers, double lr, size_t bufferCapacity,
    size_t batchSize, double gamma, size_t targetUpdateFreq)
    : _qNet(std::make_unique<MlpMatrixNN>(netLayers, lr))
    , _targetNet(std::make_unique<MlpMatrixNN>(netLayers, lr))
    , _buffer(bufferCapacity)
    , _batchSize(batchSize)
    , _gamma(gamma)
    , _targetUpdateFreq(targetUpdateFreq)
    , _numActions(netLayers.back().size)
    , _rng(std::random_device{}())
{
    if (netLayers.size() < 2)
        throw std::invalid_argument("Dqn: netLayers must have at least input and output");
    if (batchSize == 0)
        throw std::invalid_argument("Dqn: batchSize must be > 0");
    _syncTarget();
}

// ── Action selection ──────────────────────────────────────────────────────────

int Dqn::selectAction(const std::vector<double>& state, double epsilon)
{
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (coin(_rng) < epsilon) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(_numActions) - 1);
        return pick(_rng);
    }
    const auto q = qValues(state);
    return static_cast<int>(std::max_element(q.begin(), q.end()) - q.begin());
}

// ── Learn step ────────────────────────────────────────────────────────────────

double Dqn::learn(const std::vector<double>& state, int action, double reward,
    const std::vector<double>& nextState, bool done)
{
    _buffer.push(state, action, reward, nextState, done);
    if (!_buffer.ready(_batchSize))
        return 0.0;

    ++_learnStep;
    const double loss = _trainBatch();

    if (_learnStep % _targetUpdateFreq == 0)
        _syncTarget();

    return loss;
}

// ── Q-values ──────────────────────────────────────────────────────────────────

std::vector<double> Dqn::qValues(const std::vector<double>& state)
{
    _qNet->setInputVector(state);
    _qNet->feedForward();
    std::vector<double> out;
    _qNet->copyOutputVector(out);
    return out;
}

// ── Private ───────────────────────────────────────────────────────────────────

void Dqn::_syncTarget()
{
    for (size_t i = 0; i < _qNet->numLayers(); ++i) {
        _targetNet->setLayerW(i, _qNet->getLayerW(i));
        _targetNet->setLayerB(i, _qNet->getLayerB(i));
    }
}

double Dqn::_trainBatch()
{
    const auto batch = _buffer.sample(_batchSize, _rng);

    std::vector<std::vector<double>> states, targets;
    states.reserve(_batchSize);
    targets.reserve(_batchSize);
    double loss = 0.0;

    for (const auto& tr : batch) {
        // Current Q from main network — used as baseline for non-taken actions.
        auto q = qValues(tr.s);

        // Max Q for s' from frozen target network.
        _targetNet->setInputVector(tr.s_next);
        _targetNet->feedForward();
        std::vector<double> q_next;
        _targetNet->copyOutputVector(q_next);
        const double max_q_next = tr.done ? 0.0 : *std::max_element(q_next.begin(), q_next.end());

        const double bellman = tr.r + _gamma * max_q_next;
        const double err = q[tr.a] - bellman;
        loss += err * err;

        // Only the taken action gets a non-zero gradient.
        q[tr.a] = bellman;

        states.push_back(tr.s);
        targets.push_back(std::move(q));
    }

    _qNet->trainBatch(states, targets);
    return loss / static_cast<double>(_batchSize);
}

} // namespace nu
