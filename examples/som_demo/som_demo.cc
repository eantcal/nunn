//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// som_demo — Self-Organizing Map demonstration.
//
// Generates 2D data from four Gaussian clusters placed near the corners of
// the unit square, trains a 6x6 SOM, and prints an ASCII grid showing which
// cluster owns the BMU of each neuron.  Quantization error is reported before
// and after training.
//

#include "nu_som.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// Cluster labels and their 2D centroids
static constexpr int NUM_CLUSTERS = 4;
static constexpr char LABELS[] = { 'A', 'B', 'C', 'D' };
static const double CENTRES[4][2]
    = { { 0.15, 0.15 }, { 0.15, 0.85 }, { 0.85, 0.15 }, { 0.85, 0.85 } };

// Generate N samples per cluster from a Gaussian with given sigma
static std::vector<std::vector<double>> makeDataset(
    int samplesPerCluster, double sigma, std::mt19937& rng)
{
    std::normal_distribution<double> noise(0.0, sigma);
    std::vector<std::vector<double>> data;
    data.reserve(NUM_CLUSTERS * samplesPerCluster);
    for (int k = 0; k < NUM_CLUSTERS; ++k)
        for (int i = 0; i < samplesPerCluster; ++i)
            data.push_back({ CENTRES[k][0] + noise(rng), CENTRES[k][1] + noise(rng) });
    return data;
}

// Nearest cluster label for a 2D point
static char nearestCluster(double x, double y)
{
    double best = std::numeric_limits<double>::max();
    int bestK = 0;
    for (int k = 0; k < NUM_CLUSTERS; ++k) {
        const double dx = x - CENTRES[k][0];
        const double dy = y - CENTRES[k][1];
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < best) {
            best = d;
            bestK = k;
        }
    }
    return LABELS[bestK];
}

// Print the SOM grid: each cell shows which cluster the neuron weight is
// closest to (representative label), together with its weight coordinates.
static void printGrid(const nu::Som& som)
{
    const size_t R = som.rows(), C = som.cols();
    std::cout << "\nSOM grid  (cluster label / weight [x, y]):\n";
    std::cout << std::string(C * 16 + 2, '-') << '\n';
    for (size_t r = 0; r < R; ++r) {
        std::cout << '|';
        for (size_t c = 0; c < C; ++c) {
            const auto w = som.getWeights(r, c);
            const char lbl = nearestCluster(w[0], w[1]);
            std::cout << ' ' << lbl << ' ' << std::fixed << std::setprecision(2) << w[0] << ','
                      << w[1] << " |";
        }
        std::cout << '\n';
    }
    std::cout << std::string(C * 16 + 2, '-') << '\n';
}

int main()
{
    constexpr size_t ROWS = 6;
    constexpr size_t COLS = 6;
    constexpr int SAMPLES = 80; // samples per cluster
    constexpr size_t EPOCHS = 200;
    constexpr double SIGMA = 0.06;

    std::mt19937 rng(42);
    const auto dataset = makeDataset(SAMPLES, SIGMA, rng);

    nu::Som som(ROWS, COLS, 2, /*lr=*/0.5, /*radius=*/0.0, /*seed=*/1);

    std::cout << "SOM demo — " << ROWS << 'x' << COLS << " grid, " << NUM_CLUSTERS << " clusters, "
              << dataset.size() << " samples\n";
    std::cout << "Epochs: " << EPOCHS << "\n";

    const double qeBefore = som.quantizationError(dataset);
    std::cout << "\nQuantization error  before training: " << std::fixed << std::setprecision(4)
              << qeBefore << '\n';

    // Training with lr 0.5 → 0.01, radius sigma_0 → 0.5
    som.train(dataset, EPOCHS, 0.01, 0.5);

    const double qeAfter = som.quantizationError(dataset);
    std::cout << "Quantization error  after  training: " << std::fixed << std::setprecision(4)
              << qeAfter << '\n';

    printGrid(som);

    std::cout << "\nCluster centroids (reference):\n";
    for (int k = 0; k < NUM_CLUSTERS; ++k)
        std::cout << "  " << LABELS[k] << ": (" << CENTRES[k][0] << ", " << CENTRES[k][1] << ")\n";

    return 0;
}
