//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// rbf_demo — RBF Network demonstration.
//
// Trains an RBF network on a 1D sine wave regression task:
//   input:  x in [0, 2*pi] discretized into N_TRAIN evenly spaced points
//   target: sin(x)
//
// After training the demo evaluates on a denser test grid and prints:
//   1. Per-point prediction vs ground truth (ASCII bar chart).
//   2. Final train MSE and test MSE.
//
// Usage: rbf_demo [centers=12] [epochs=3000] [lr=0.05]
//

#define _USE_MATH_DEFINES
#include "nu_rbf.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int N_TRAIN = 20; // training points
constexpr int N_TEST = 40; // evaluation points

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> makeSineDataset(int n)
{
    std::vector<std::vector<double>> X, Y;
    for (int i = 0; i < n; ++i) {
        const double x = 2.0 * M_PI * i / (n - 1);
        X.push_back({ x });
        Y.push_back({ std::sin(x) });
    }
    return { X, Y };
}

double evalMSE(nu::Rbf& rbf, const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& Y)
{
    double mse = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        const double diff = rbf.forward(X[i])[0] - Y[i][0];
        mse += diff * diff;
    }
    return mse / static_cast<double>(X.size());
}

void printBar(double v, int width = 20)
{
    // v in [-1,1]; map to [0,1] for display
    const double u = (v + 1.0) * 0.5;
    const int filled = static_cast<int>(std::round(u * width));
    std::cout << '[';
    for (int i = 0; i < width; ++i)
        std::cout << (i < filled ? '#' : '.');
    std::cout << ']';
}

} // namespace

int main(int argc, char* argv[])
{
    const size_t CENTERS = argc > 1 ? std::stoul(argv[1]) : 12;
    const size_t EPOCHS = argc > 2 ? std::stoul(argv[2]) : 3000;
    const double LR = argc > 3 ? std::stod(argv[3]) : 0.05;

    std::cout << "RBF demo — sine regression\n";
    std::cout << "  centers=" << CENTERS << "  epochs=" << EPOCHS << "  lr=" << LR << "\n\n";

    auto [trainX, trainY] = makeSineDataset(N_TRAIN);
    auto [testX, testY] = makeSineDataset(N_TEST);

    nu::Rbf rbf(1, CENTERS, 1, LR);
    rbf.fitCenters(trainX);

    // Baseline.
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Baseline train MSE: " << evalMSE(rbf, trainX, trainY) << "\n\n";

    // Training.
    constexpr size_t REPORT = 500;
    std::cout << std::setw(8) << "Epoch" << std::setw(14) << "Train MSE\n";
    std::cout << std::string(22, '-') << "\n";
    for (size_t ep = 0; ep <= EPOCHS; ep += REPORT) {
        if (ep > 0)
            rbf.train(trainX, trainY, REPORT);
        std::cout << std::setw(8) << ep << std::setw(14) << evalMSE(rbf, trainX, trainY) << "\n";
    }

    // Predictions on test grid.
    std::cout << "\nPredictions on test grid (x in [0, 2pi]):\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setprecision(3);
    for (int i = 0; i < N_TEST; ++i) {
        const double x = testX[i][0];
        const double pred = rbf.forward(testX[i])[0];
        const double gt = testY[i][0];
        std::cout << "  x=" << std::setw(5) << x << "  pred=" << std::setw(7) << pred
                  << "  gt=" << std::setw(7) << gt << "  ";
        printBar(pred);
        std::cout << "\n";
    }

    std::cout << "\nFinal train MSE: " << evalMSE(rbf, trainX, trainY);
    std::cout << "   test MSE: " << evalMSE(rbf, testX, testY) << "\n";
    return 0;
}
