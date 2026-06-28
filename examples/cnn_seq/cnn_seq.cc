//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// cnn_seq — ConvNet demonstration: frequency classification of 1D signals.
//
// Two-class problem:
//   Class 0: sin wave with 1 cycle over T samples  + Gaussian noise
//   Class 1: sin wave with 2 cycles over T samples + Gaussian noise
//
// A ConvNet with one Conv1D and one MaxPool1D layer learns local frequency
// patterns and achieves high classification accuracy.
//
// Network topology:
//   Input [1, T=16] → Conv1D(8 filters, k=5, Tanh) → MaxPool1D(4) → FC [12→16→2]
//
// Usage: cnn_seq [epochs=500] [lr=0.005] [samples=100]
//

#define _USE_MATH_DEFINES
#include "nu_convnet.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

constexpr int T = 16; // time steps

std::vector<double> makeSample(int freqCycles, double noiseStd, std::mt19937& rng)
{
    std::normal_distribution<double> noise(0.0, noiseStd);
    std::vector<double> x(T);
    for (int t = 0; t < T; ++t)
        x[t] = std::sin(2.0 * M_PI * freqCycles * t / T) + noise(rng);
    return x;
}

double accuracy(
    nu::ConvNet& cnn, const std::vector<std::vector<double>>& Xs, const std::vector<int>& labels)
{
    int correct = 0;
    for (size_t i = 0; i < Xs.size(); ++i) {
        const auto out = cnn.predict(Xs[i]);
        const int pred = (out[0] > out[1]) ? 0 : 1;
        if (pred == labels[i])
            ++correct;
    }
    return 100.0 * correct / static_cast<double>(Xs.size());
}

} // namespace

int main(int argc, char* argv[])
{
    const int EPOCHS = argc > 1 ? std::stoi(argv[1]) : 500;
    const double LR = argc > 2 ? std::stod(argv[2]) : 0.005;
    const int N_TRAIN = argc > 3 ? std::stoi(argv[3]) : 100; // per class

    std::cout << "ConvNet frequency-classification demo\n";
    std::cout << "  T=" << T << "  classes: 1-cycle sine vs 2-cycle sine\n";
    std::cout << "  epochs=" << EPOCHS << "  lr=" << LR << "  n_train=" << N_TRAIN << "/class\n\n";

    std::mt19937 rng(42);
    constexpr double NOISE = 0.15;

    // Build datasets.
    std::vector<std::vector<double>> trainX, testX;
    std::vector<int> trainY, testY;

    for (int i = 0; i < N_TRAIN; ++i) {
        trainX.push_back(makeSample(1, NOISE, rng));
        trainY.push_back(0);
        trainX.push_back(makeSample(2, NOISE, rng));
        trainY.push_back(1);
    }
    for (int i = 0; i < 20; ++i) {
        testX.push_back(makeSample(1, NOISE, rng));
        testY.push_back(0);
        testX.push_back(makeSample(2, NOISE, rng));
        testY.push_back(1);
    }

    // Build ConvNet.
    using LC = nu::MlpMatrixNN::LayerConfig;
    nu::ConvNet cnn(1, T);
    cnn.addConv1D(8, 5, nu::Activation::Tanh, LR);
    cnn.addMaxPool1D(4);
    const size_t flatSz = cnn.flatFeatureSize(); // 8 * ((16-5+1)/4) = 8*3 = 24
    cnn.setFCHead({ LC(flatSz), LC(16, nu::Activation::Tanh), LC(2, nu::Activation::Sigmoid) }, LR);

    std::cout << "Flat feature size: " << flatSz << "\n\n";

    // One-hot targets.
    const std::vector<double> T0{ 1.0, 0.0 };
    const std::vector<double> T1{ 0.0, 1.0 };

    // Shuffle index.
    std::vector<size_t> idx(trainX.size());
    std::iota(idx.begin(), idx.end(), 0);

    constexpr int REPORT = 50;
    std::cout << std::setw(8) << "Epoch" << std::setw(12) << "Train loss" << std::setw(12)
              << "Train acc%" << std::setw(11) << "Test acc%\n";
    std::cout << std::string(43, '-') << "\n";

    for (int ep = 1; ep <= EPOCHS; ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double totalLoss = 0.0;
        for (size_t j : idx) {
            const auto& tgt = (trainY[j] == 0) ? T0 : T1;
            totalLoss += cnn.train(trainX[j], tgt);
        }
        if (ep % REPORT == 0) {
            const double trainAcc = accuracy(cnn, trainX, trainY);
            const double testAcc = accuracy(cnn, testX, testY);
            std::cout << std::setw(8) << ep << std::fixed << std::setprecision(4) << std::setw(12)
                      << totalLoss / trainX.size() << std::setw(12) << trainAcc << std::setw(11)
                      << testAcc << "\n";
        }
    }

    std::cout << "\nFinal test accuracy: " << accuracy(cnn, testX, testY) << "%\n";
    return 0;
}
