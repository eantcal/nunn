//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// rnn_sine — predict the next value of a sine wave using a VanillaRnn or LSTM.
//
// The network sees one sample at a time (x_t = sin(t·dt)) and must predict
// the next sample (y_t = sin((t+1)·dt)).  After training it runs
// autoregressively: each predicted value is fed back as the next input,
// testing whether the learned dynamics are self-sustaining.
//
// Usage: rnn_sine [--lstm] [epochs] [hidden_size] [lr]
//   --lstm   select LSTM instead of VanillaRnn (default)
//

#include "nu_lstm.h"
#include "nu_rnn.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> makeSeq(
    double t0, size_t seqLen, double dt)
{
    std::vector<std::vector<double>> xs, ys;
    xs.reserve(seqLen);
    ys.reserve(seqLen);
    for (size_t t = 0; t < seqLen; ++t) {
        xs.push_back({ std::sin((t0 + t) * dt) });
        ys.push_back({ std::sin((t0 + t + 1) * dt) });
    }
    return { xs, ys };
}

template <typename Rnn> void run(Rnn& rnn, size_t epochs, size_t numSeq, size_t seqLen, double dt)
{
    constexpr size_t PRED_STEPS = 80;

    std::cout << std::fixed << std::setprecision(6);

    // ── Training ──────────────────────────────────────────────────────────────
    std::cout << "Epoch     Loss\n";
    std::cout << std::string(24, '-') << "\n";

    for (size_t ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;
        for (size_t s = 0; s < numSeq; ++s) {
            const double t0 = static_cast<double>(s) * (2.0 * PI / numSeq) / dt;
            auto [xs, ys] = makeSeq(t0, seqLen, dt);
            rnn.resetState();
            total_loss += rnn.bptt(xs, ys);
        }
        if (ep % (epochs / 10) == 0 || ep == epochs - 1)
            std::cout << std::setw(6) << ep << "    " << total_loss / numSeq << "\n";
    }

    // ── Autoregressive prediction ─────────────────────────────────────────────
    std::cout << "\nAutoregressive prediction (" << PRED_STEPS << " steps)\n";
    std::cout << std::string(44, '-') << "\n";
    std::cout << std::setw(5) << "step" << std::setw(12) << "predicted" << std::setw(12) << "actual"
              << std::setw(12) << "error\n";
    std::cout << std::string(44, '-') << "\n";

    rnn.resetState();
    double prev = std::sin(0.0);
    double max_abs_err = 0.0;

    for (size_t t = 0; t < PRED_STEPS; ++t) {
        rnn.step({ prev });
        const double predicted = rnn.getOutput()[0];
        const double actual = std::sin(static_cast<double>(t + 1) * dt);
        const double err = std::abs(predicted - actual);
        max_abs_err = std::max(max_abs_err, err);

        std::cout << std::setw(5) << t << std::setw(12) << predicted << std::setw(12) << actual
                  << std::setw(12) << err << "\n";

        prev = predicted;
    }

    std::cout << std::string(44, '-') << "\n";
    std::cout << "Max absolute error (autoregressive): " << max_abs_err << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_lstm = false;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--lstm")
            use_lstm = true;
        else
            pos.push_back(argv[i]);
    }

    const size_t EPOCHS = pos.size() > 0 ? std::stoul(pos[0]) : 1500;
    const size_t HIDDEN = pos.size() > 1 ? std::stoul(pos[1]) : 32;
    const double LR = pos.size() > 2 ? std::stod(pos[2]) : 0.005;

    constexpr double DT = 0.15;
    constexpr size_t SEQ_LEN = 40;
    constexpr size_t NUM_SEQ = 24;

    std::cout << "rnn_sine  |  model=" << (use_lstm ? "LSTM" : "VanillaRnn")
              << "  hidden=" << HIDDEN << "  lr=" << LR << "  epochs=" << EPOCHS << "\n\n";

    if (use_lstm) {
        nu::Lstm lstm(1, HIDDEN, 1, LR, 5.0, nu::RnnOutput::Linear);
        run(lstm, EPOCHS, NUM_SEQ, SEQ_LEN, DT);
    } else {
        nu::VanillaRnn rnn(1, HIDDEN, 1, LR, 5.0, nu::RnnOutput::Linear);
        run(rnn, EPOCHS, NUM_SEQ, SEQ_LEN, DT);
    }

    return 0;
}
