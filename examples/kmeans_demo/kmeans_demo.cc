//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  Licensed under the MIT License.
//
// kmeans_demo — cluster four 2D Gaussian blobs and print the ASCII grid.
//
// Usage:  kmeans_demo [k=4] [n_per_cluster=80] [seed=42]
//
// Output: a 20x20 ASCII grid where each cell shows the cluster label (0-k-1)
//         of the point nearest to the cell centre; centroids are shown as '*'.

#include "nu_kmeans.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static std::vector<std::vector<double>> generateBlobs(
    const std::vector<std::pair<double, double>>& centres, size_t nPerCluster, double sigma,
    unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    std::vector<std::vector<double>> X;
    for (const auto& [cx, cy] : centres)
        for (size_t i = 0; i < nPerCluster; ++i)
            X.push_back({ cx + noise(rng), cy + noise(rng) });
    return X;
}

int main(int argc, char* argv[])
{
    const size_t k = (argc > 1) ? static_cast<size_t>(std::atoi(argv[1])) : 4;
    const size_t n = (argc > 2) ? static_cast<size_t>(std::atoi(argv[2])) : 80;
    const unsigned seed = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42;

    const std::vector<std::pair<double, double>> centres{ { 2.0, 2.0 }, { 8.0, 2.0 }, { 2.0, 8.0 },
        { 8.0, 8.0 } };

    auto X = generateBlobs(centres, n, 0.8, seed);

    nu::KMeans km(k, 300, 1e-6, seed);
    km.fit(X);

    const double inertia = km.inertia(X);
    std::cout << "k=" << k << "  samples=" << X.size() << "  iterations=" << km.numIterations()
              << "  inertia=" << std::fixed << std::setprecision(2) << inertia << "\n\n";

    // ASCII grid: rows/cols map linearly to [0, 10]
    const int GRID = 20;
    const double lo = 0.0, hi = 10.0;
    const double step = (hi - lo) / GRID;

    // Centroid grid positions (for marking with '*')
    const auto& cents = km.centroids();
    auto toCell = [&](double v) { return static_cast<int>((v - lo) / step); };

    std::cout << "ASCII cluster map (" << GRID << "x" << GRID << "):\n";
    for (int row = GRID - 1; row >= 0; --row) {
        for (int col = 0; col < GRID; ++col) {
            const double x = lo + (col + 0.5) * step;
            const double y = lo + (row + 0.5) * step;
            const size_t label = km.predict({ x, y });

            // Check if any centroid maps to this cell
            bool isCentroid = false;
            for (const auto& c : cents)
                if (toCell(c[0]) == col && toCell(c[1]) == row) {
                    isCentroid = true;
                    break;
                }
            std::cout << (isCentroid ? '*' : static_cast<char>('0' + label));
        }
        std::cout << '\n';
    }

    std::cout << "\nCentroids:\n";
    for (size_t i = 0; i < cents.size(); ++i)
        std::cout << "  cluster " << i << ": (" << std::fixed << std::setprecision(3) << cents[i][0]
                  << ", " << cents[i][1] << ")\n";

    return 0;
}
