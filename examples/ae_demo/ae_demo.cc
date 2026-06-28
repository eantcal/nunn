//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// ae_demo — autoencoder demonstration.
//
// Trains an autoencoder on a dataset of 8-point sine-wave snippets sampled
// at 8 different phases.  The bottleneck is 2-dimensional, so the network
// must learn a compact 2D representation of periodic signals.
//
// After training, the demo:
//   1. Prints reconstruction MSE per sample.
//   2. Encodes each sample to its 2D latent code.
//   3. Decodes a hand-crafted latent code to show generation.
//
// Usage: ae_demo [epochs=2000] [hidden=8] [bottleneck=2] [lr=0.005]
//

#define _USE_MATH_DEFINES
#include "nu_autoencoder.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr size_t WAVE_LEN = 16; // samples per sine snippet

std::vector<double> makeSine(double phase)
{
    std::vector<double> v(WAVE_LEN);
    for (size_t t = 0; t < WAVE_LEN; ++t)
        v[t] = 0.5 + 0.5 * std::sin(2.0 * M_PI * t / WAVE_LEN + phase);
    return v;
}

double mse(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += (a[i] - b[i]) * (a[i] - b[i]);
    return s / static_cast<double>(a.size());
}

void printBar(double v, int width = 20)
{
    int filled = static_cast<int>(std::round(v * width));
    std::cout << '[';
    for (int i = 0; i < width; ++i)
        std::cout << (i < filled ? '#' : '.');
    std::cout << ']';
}

} // namespace

int main(int argc, char* argv[])
{
    const size_t EPOCHS = argc > 1 ? std::stoul(argv[1]) : 2000;
    const size_t HIDDEN = argc > 2 ? std::stoul(argv[2]) : 8;
    const size_t BOTTLENECK = argc > 3 ? std::stoul(argv[3]) : 2;
    const double LR = argc > 4 ? std::stod(argv[4]) : 0.005;

    // ── Dataset: 8 phases evenly spaced in [0, 2π) ───────────────────────────
    constexpr size_t N_PHASES = 8;
    std::vector<std::vector<double>> dataset;
    dataset.reserve(N_PHASES);
    for (size_t p = 0; p < N_PHASES; ++p)
        dataset.push_back(makeSine(2.0 * M_PI * p / N_PHASES));

    std::cout << "Autoencoder demo\n";
    std::cout << "  input=" << WAVE_LEN << "  hidden=" << HIDDEN << "  bottleneck=" << BOTTLENECK
              << "  epochs=" << EPOCHS << "  lr=" << LR << "\n\n";

    // Topology: [WAVE_LEN, HIDDEN, BOTTLENECK, HIDDEN, WAVE_LEN]
    nu::Autoencoder ae(WAVE_LEN, { HIDDEN, BOTTLENECK }, nu::Activation::Tanh, LR);

    // ── Baseline MSE ──────────────────────────────────────────────────────────
    double baseMSE = 0.0;
    for (const auto& x : dataset)
        baseMSE += ae.reconstructionMSE(x);
    baseMSE /= N_PHASES;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Baseline MSE (before training): " << baseMSE << "\n\n";

    // ── Training ──────────────────────────────────────────────────────────────
    constexpr size_t REPORT = 200;
    std::cout << std::setw(8) << "Epoch" << std::setw(14) << "Mean MSE\n";
    std::cout << std::string(22, '-') << "\n";

    for (size_t ep = 0; ep <= EPOCHS; ep += REPORT) {
        if (ep > 0)
            ae.train(dataset, REPORT);
        double mseSum = 0.0;
        for (const auto& x : dataset)
            mseSum += ae.reconstructionMSE(x);
        std::cout << std::setw(8) << ep << std::setw(14) << mseSum / N_PHASES << "\n";
    }

    // ── Per-sample reconstruction ─────────────────────────────────────────────
    std::cout << "\nReconstruction quality per phase:\n";
    std::cout << std::string(60, '-') << "\n";
    for (size_t p = 0; p < N_PHASES; ++p) {
        const auto& x = dataset[p];
        const auto rec = ae.reconstruct(x);
        const double err = mse(x, rec);
        std::cout << "  phase " << std::setw(2) << p << "  MSE=" << std::setw(10) << err << "  ";
        printBar(std::max(0.0, 1.0 - err * 50.0));
        std::cout << "\n";
    }

    // ── Latent codes ──────────────────────────────────────────────────────────
    std::cout << "\nLatent codes (z0, z1) per phase:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::setprecision(4);
    for (size_t p = 0; p < N_PHASES; ++p) {
        const auto z = ae.encode(dataset[p]);
        std::cout << "  phase " << std::setw(2) << p << "  z=[";
        for (size_t i = 0; i < z.size(); ++i)
            std::cout << (i ? ", " : "") << std::setw(8) << z[i];
        std::cout << "]\n";
    }

    // ── Decode a hand-crafted latent code ─────────────────────────────────────
    std::cout << "\nDecoding z=[0,0]:\n  ";
    const std::vector<double> z0(BOTTLENECK, 0.0);
    const auto gen = ae.decode(z0);
    for (double v : gen) {
        printBar(v, 1);
        std::cout << std::setw(6) << std::setprecision(3) << v << " ";
    }
    std::cout << "\n";

    return 0;
}
