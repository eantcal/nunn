//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Fixed-capacity circular experience replay buffer used by DQN training.
// Header-only template: State and Action are user-supplied types.
//

#pragma once

#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace nu {

template <class State, class Action> class ExperienceReplayBuffer {
public:
    struct Transition {
        State s;
        Action a;
        double r;
        State s_next;
        bool done;
    };

    explicit ExperienceReplayBuffer(size_t capacity)
        : _capacity(capacity)
    {
        if (capacity == 0)
            throw std::invalid_argument("ExperienceReplayBuffer: capacity must be > 0");
    }

    void push(State s, Action a, double r, State s_next, bool done)
    {
        if (_buf.size() == _capacity)
            _buf.pop_front();
        _buf.push_back({ std::move(s), std::move(a), r, std::move(s_next), done });
    }

    std::vector<Transition> sample(size_t batchSize, std::mt19937& rng) const
    {
        if (batchSize > _buf.size())
            throw std::invalid_argument("ExperienceReplayBuffer::sample: not enough transitions");
        std::vector<size_t> idx(_buf.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<Transition> batch;
        batch.reserve(batchSize);
        for (size_t i = 0; i < batchSize; ++i)
            batch.push_back(_buf[idx[i]]);
        return batch;
    }

    size_t size() const noexcept { return _buf.size(); }
    bool ready(size_t minSize) const noexcept { return _buf.size() >= minSize; }

private:
    std::deque<Transition> _buf;
    size_t _capacity;
};

} // namespace nu
