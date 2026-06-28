//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// rnn_adding — the "adding problem" benchmark for recurrent networks.
//
// Each input sequence has length T.  Every element is a pair (value, marker):
//   value  ∈ [0, 1]  — a random scalar
//   marker ∈ {0, 1}  — 1 at exactly two positions (one in each half)
//
// The target is the sum of the two marked values, predicted as a running
// cumulative sum at each step (many-to-many formulation).
// At the final step the network's output should equal the total sum.
//
// This tests selective memory: the network must remember only the values
// at marked positions and accumulate them, ignoring all others.
//
// Three architectures are trained on the same task and compared:
//   VanillaRnn, GRU, LSTM
//
// Usage: rnn_adding [seq_len] [hidden] [epochs] [lr]
//

#include "nu_gru.h"
#include "nu_lstm.h"
#include "nu_rnn.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

// ── Dataset ───────────────────────────────────────────────────────────────────

struct Sample {
    std::vector<std::vector<double>> inputs; // T × 2  [value, marker]
    std::vector<std::vector<double>> targets; // T × 1  cumulative marked sum
};

// Generate one sequence of length T.
// One marker is placed in [0, T/2) and one in [T/2, T).
Sample makeSample(size_t T, std::mt19937& rng)
{
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);
    std::uniform_int_distribution<size_t> pos1(0, T / 2 - 1);
    std::uniform_int_distribution<size_t> pos2(T / 2, T - 1);

    const size_t m1 = pos1(rng);
    const size_t m2 = pos2(rng);

    Sample s;
    s.inputs.resize(T);
    s.targets.resize(T);

    double running = 0.0;
    for (size_t t = 0; t < T; ++t) {
        const double v = val_dist(rng);
        const double marker = (t == m1 || t == m2) ? 1.0 : 0.0;
        s.inputs[t] = { v, marker };
        running += marker * v;
        s.targets[t] = { running };
    }
    return s;
}

std::vector<Sample> makeDataset(size_t n, size_t T, std::mt19937& rng)
{
    std::vector<Sample> ds(n);
    for (auto& s : ds)
        s = makeSample(T, rng);
    return ds;
}

// ── Evaluation ────────────────────────────────────────────────────────────────

// Returns mean absolute error at the LAST step across the dataset.
template <typename Rnn> double evalMAE(Rnn& rnn, const std::vector<Sample>& dataset)
{
    double total = 0.0;
    for (const auto& s : dataset) {
        rnn.resetState();
        for (size_t t = 0; t + 1 < s.inputs.size(); ++t)
            rnn.step(s.inputs[t]);
        rnn.step(s.inputs.back());
        total += std::abs(rnn.getOutput()[0] - s.targets.back()[0]);
    }
    return total / static_cast<double>(dataset.size());
}

// ── Training loop ─────────────────────────────────────────────────────────────

template <typename Rnn>
void train(Rnn& rnn, const std::string& name, const std::vector<Sample>& train_set,
    const std::vector<Sample>& test_set, size_t epochs)
{
    std::cout << "\n── " << name << " ───────────────────────────────\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::setw(7) << "Epoch" << std::setw(12) << "Train loss" << std::setw(12)
              << "Test MAE\n";
    std::cout << std::string(32, '-') << "\n";

    const size_t REPORT = epochs / 10;

    for (size_t ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;
        for (const auto& s : train_set) {
            rnn.resetState();
            total_loss += rnn.bptt(s.inputs, s.targets);
        }

        if (ep % REPORT == 0 || ep == epochs - 1) {
            const double mae = evalMAE(rnn, test_set);
            std::cout << std::setw(7) << ep << std::setw(12) << total_loss / train_set.size()
                      << std::setw(12) << mae << "\n";
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    const size_t SEQ_LEN = argc > 1 ? std::stoul(argv[1]) : 20;
    const size_t HIDDEN = argc > 2 ? std::stoul(argv[2]) : 32;
    const size_t EPOCHS = argc > 3 ? std::stoul(argv[3]) : 500;
    const double LR = argc > 4 ? std::stod(argv[4]) : 0.005;

    constexpr size_t TRAIN_N = 256;
    constexpr size_t TEST_N = 64;
    constexpr double GRAD_CLIP = 5.0;

    std::mt19937 rng(42);
    const auto train_set = makeDataset(TRAIN_N, SEQ_LEN, rng);
    const auto test_set = makeDataset(TEST_N, SEQ_LEN, rng);

    // Baseline: always predict 0.5 (expected value of a uniform [0,1] sum × 2 markers)
    const double baseline_mae = [&] {
        double total = 0.0;
        for (const auto& s : test_set)
            total += std::abs(0.5 - s.targets.back()[0]);
        return total / TEST_N;
    }();

    std::cout << "Adding problem  |  seq_len=" << SEQ_LEN << "  hidden=" << HIDDEN
              << "  epochs=" << EPOCHS << "  lr=" << LR << "\n";
    std::cout << "Train=" << TRAIN_N << "  Test=" << TEST_N
              << "  Baseline MAE (predict 0.5)=" << std::fixed << std::setprecision(4)
              << baseline_mae << "\n";

    {
        nu::VanillaRnn rnn(2, HIDDEN, 1, LR, GRAD_CLIP, nu::RnnOutput::Linear);
        train(rnn, "VanillaRnn", train_set, test_set, EPOCHS);
    }
    {
        nu::Gru gru(2, HIDDEN, 1, LR, GRAD_CLIP, nu::RnnOutput::Linear);
        train(gru, "GRU", train_set, test_set, EPOCHS);
    }
    {
        nu::Lstm lstm(2, HIDDEN, 1, LR, GRAD_CLIP, nu::RnnOutput::Linear);
        train(lstm, "LSTM", train_set, test_set, EPOCHS);
    }

    return 0;
}
