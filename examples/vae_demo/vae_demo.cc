//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// VAE demo: learn 4 binary prototype patterns; reconstruct and generate.
//
// Architecture: 8 visible -> 32 hidden -> z(4) -> 32 hidden -> 8 output
// Training: SGD with reparameterization trick + KL annealing, 2000 epochs
//

#include "nu_vae.h"

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
    const std::vector<std::vector<double>> protos = {
        { 1, 1, 1, 1, 0, 0, 0, 0 }, // A  ####....
        { 0, 0, 0, 0, 1, 1, 1, 1 }, // B  ....####
        { 1, 0, 1, 0, 1, 0, 1, 0 }, // C  #.#.#.#.
        { 0, 1, 0, 1, 0, 1, 0, 1 }, // D  .#.#.#.#
    };
    const char labels[] = "ABCD";
    const double NOISE = 0.15;
    const int SAMPLES = 80;
    const size_t EPOCHS = 2000;

    // Noisy training set
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

    // VAE: 8 visible, 32 hidden, 4 latent
    nu::Vae vae(8, 32, 4, 0.003, 42);

    std::cout << "VAE demo  --  8 visible, 32 hidden, 4 latent\n";
    std::cout << protos.size() << " prototypes, " << dataset.size()
              << " training samples, noise=" << NOISE << "\n\n";

    std::cout << "Reconstruction error  before training: " << std::fixed << std::setprecision(4)
              << vae.reconstructionError(dataset) << "\n";

    vae.train(dataset, EPOCHS, 0.2); // KL warm-up over first 20% of epochs

    std::cout << "Reconstruction error  after  training: " << vae.reconstructionError(dataset)
              << "\n\n";

    // Deterministic reconstructions (use mu)
    std::cout << "Reconstructions (deterministic via mu):\n";
    std::cout << "  proto | clean    | reconstruction\n";
    std::cout << "--------+----------+---------------\n";
    for (size_t pi = 0; pi < protos.size(); ++pi) {
        const auto recon = vae.reconstruct(protos[pi]);
        std::cout << "    " << labels[pi] << "   | " << bits(protos[pi]) << " | " << bits(recon)
                  << "\n";
    }

    // Latent codes (mu) for clean prototypes
    std::cout << "\nLatent codes (mu) for clean prototypes:\n";
    std::cout << "  proto |  mu\n";
    std::cout << "--------+------------------------------------------\n";
    for (size_t pi = 0; pi < protos.size(); ++pi) {
        const auto [mu, lv] = vae.encode(protos[pi]);
        std::cout << "    " << labels[pi] << "   | ";
        for (double v : mu)
            std::cout << std::setw(7) << std::setprecision(2) << v;
        std::cout << "\n";
    }

    // Generated samples from prior N(0,I)
    std::cout << "\nGenerated samples from N(0,I) prior:\n";
    for (int i = 0; i < 4; ++i)
        std::cout << "  gen " << (i + 1) << ":  " << bits(vae.generate()) << "\n";

    std::cout << "\n  ('#' = 1,  '.' = 0)\n";
    return 0;
}
