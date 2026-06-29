//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  Licensed under the MIT License.
//
// pca_demo — reduce a synthetic 4D dataset to 2D with PCA and show
//            the explained variance per component.
//
// Usage:  pca_demo [n_components=2] [n_samples=200] [seed=42]

#include "nu_pca.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static std::vector<std::vector<double>> makeData(size_t N, size_t D, unsigned seed)
{
    // Each column has a different scale so PCA has something to find
    std::mt19937 rng(seed);
    std::vector<std::vector<double>> X(N, std::vector<double>(D));
    std::vector<double> scale{ 3.0, 1.5, 0.5, 0.1 };
    scale.resize(D, 0.1);
    for (size_t i = 0; i < N; ++i) {
        double base = std::normal_distribution<double>(0.0, 1.0)(rng);
        for (size_t j = 0; j < D; ++j) {
            std::normal_distribution<double> noise(0.0, 0.2);
            X[i][j] = scale[j % scale.size()] * base + noise(rng);
        }
    }
    return X;
}

int main(int argc, char* argv[])
{
    const size_t nComp = (argc > 1) ? static_cast<size_t>(std::atoi(argv[1])) : 2;
    const size_t nSamples = (argc > 2) ? static_cast<size_t>(std::atoi(argv[2])) : 200;
    const unsigned seed = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42;
    const size_t D = 4;

    if (nComp == 0 || nComp > D) {
        std::cerr << "n_components must be in [1, " << D << "]\n";
        return 1;
    }

    auto X = makeData(nSamples, D, seed);

    nu::Pca pca(nComp);
    pca.fit(X);

    std::cout << "PCA  D=" << D << "  n_components=" << nComp << "  n_samples=" << nSamples
              << "\n\n";

    const auto& evr = pca.explainedVarianceRatio();
    std::cout << "Explained variance ratio per component:\n";
    double cumul = 0.0;
    for (size_t i = 0; i < nComp; ++i) {
        cumul += evr[i];
        std::cout << "  PC" << (i + 1) << ": " << std::fixed << std::setprecision(4) << evr[i]
                  << "  (cumulative: " << cumul << ")\n";
    }
    std::cout << "\nTotal explained variance: " << std::fixed << std::setprecision(4)
              << pca.totalExplainedVariance() << "\n\n";

    // Show first 5 projections
    auto Z = pca.transform(X);
    std::cout << "First 5 projections (rows = samples, cols = components):\n";
    for (size_t i = 0; i < std::min<size_t>(5, Z.size()); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < Z[i].size(); ++j)
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << Z[i][j]
                      << (j + 1 < Z[i].size() ? ", " : "");
        std::cout << "]\n";
    }

    // Reconstruction error on first 5 samples
    std::cout << "\nReconstruction MSE (first 5 samples):\n";
    for (size_t i = 0; i < std::min<size_t>(5, Z.size()); ++i) {
        auto xhat = pca.inverseTransform(Z[i]);
        double mse = 0.0;
        for (size_t j = 0; j < D; ++j) {
            double diff = X[i][j] - xhat[j];
            mse += diff * diff;
        }
        mse /= static_cast<double>(D);
        std::cout << "  sample " << i << ": mse=" << std::fixed << std::setprecision(6) << mse
                  << "\n";
    }
    return 0;
}
