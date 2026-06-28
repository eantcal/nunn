//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// RBM demo: learn 4 binary prototype patterns and reconstruct noisy versions.
//
// Visible layer: 8 units (one byte, visualised as two rows of 4 bits)
// Hidden layer:  6 units
// Training: CD-1, 300 epochs, lr=0.05
//

#include "nu_rbm.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static std::string bits(const std::vector<double>& v, double thr = 0.5)
{
    std::string s;
    for (double x : v)
        s += (x >= thr) ? '#' : '.';
    return s;
}

int main()
{
    // 4 prototype 8-bit patterns
    const std::vector<std::vector<double>> protos = {
        { 1, 1, 1, 1, 0, 0, 0, 0 }, // A  ####....
        { 0, 0, 0, 0, 1, 1, 1, 1 }, // B  ....####
        { 1, 0, 1, 0, 1, 0, 1, 0 }, // C  #.#.#.#.
        { 0, 1, 0, 1, 0, 1, 0, 1 }, // D  .#.#.#.#
    };
    const char labels[] = "ABCD";
    const double NOISE = 0.15;
    const int SAMPLES = 80;
    const size_t NV = 8, NH = 6, EPOCHS = 300;

    // Generate noisy training set
    std::mt19937 rng(2025);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::vector<std::vector<double>> dataset;
    for (const auto& p : protos)
        for (int i = 0; i < SAMPLES; ++i) {
            std::vector<double> x = p;
            for (double& b : x)
                if (udist(rng) < NOISE)
                    b = 1.0 - b;
            dataset.push_back(x);
        }

    nu::Rbm rbm(NV, NH, 0.05, 42);

    std::cout << "RBM demo  --  " << NV << " visible, " << NH << " hidden\n";
    std::cout << protos.size() << " prototypes, " << dataset.size()
              << " training samples, noise=" << NOISE << "\n\n";

    std::cout << "Reconstruction error  before training: " << std::fixed << std::setprecision(4)
              << rbm.reconstructionError(dataset) << "\n";

    rbm.train(dataset, EPOCHS, 1);

    std::cout << "Reconstruction error  after  training: " << rbm.reconstructionError(dataset)
              << "\n\n";

    // Reconstruct one noisy sample per prototype
    std::cout << "  proto | clean    | noisy    | reconstruction\n";
    std::cout << "--------+----------+----------+---------------\n";
    for (size_t pi = 0; pi < protos.size(); ++pi) {
        std::vector<double> noisy = protos[pi];
        for (double& b : noisy)
            if (udist(rng) < NOISE)
                b = 1.0 - b;
        const auto recon = rbm.reconstruct(noisy);
        std::cout << "    " << labels[pi] << "   | " << bits(protos[pi]) << " | " << bits(noisy)
                  << " | " << bits(recon) << "\n";
    }
    std::cout << "  ('#' = 1,  '.' = 0)\n";

    return 0;
}
