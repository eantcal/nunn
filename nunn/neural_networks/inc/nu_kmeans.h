//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// k-Means clustering (Lloyd's algorithm with k-means++ initialisation).
//
// Algorithm:
//   1. k-means++ seed selection: choose first centroid uniformly at random;
//      each subsequent centroid is chosen with probability proportional to
//      D(x)^2 = squared distance to the nearest already-chosen centroid.
//   2. E-step: assign each sample to its nearest centroid.
//   3. M-step: recompute centroids as cluster means.
//   4. Repeat until max centroid shift < tol or maxIter reached.
//      Empty clusters keep their previous centroid (stable behaviour).
//
// Complexity per iteration: O(N * k * d)
//
// Usage:
//   KMeans km(3);
//   km.fit(X);
//   auto labels = km.predict(X);
//   double wsse  = km.inertia(X);

#pragma once

#include <random>
#include <stdexcept>
#include <vector>

namespace nu {

class KMeans {
public:
    struct NotFittedException : std::runtime_error {
        NotFittedException()
            : std::runtime_error("KMeans: call fit() before predict()")
        {
        }
    };

    struct SizeMismatchException : std::runtime_error {
        explicit SizeMismatchException(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    // k        — number of clusters
    // maxIter  — maximum Lloyd iterations
    // tol      — stop when all centroid shifts fall below this threshold
    // seed     — RNG seed for k-means++ initialisation
    explicit KMeans(size_t k, size_t maxIter = 300, double tol = 1e-4, unsigned seed = 42) noexcept;

    // Fit the model to X using k-means++ seeding.
    // Throws SizeMismatchException if X is empty, any sample is empty,
    // sample sizes are inconsistent, or k > N.
    void fit(const std::vector<std::vector<double>>& X);

    // Index of the nearest centroid for a single sample.
    // Throws NotFittedException / SizeMismatchException.
    size_t predict(const std::vector<double>& x) const;

    // Batch predict.
    std::vector<size_t> predict(const std::vector<std::vector<double>>& X) const;

    // Within-cluster sum of squared distances to centroids (WCSS / inertia).
    // Throws NotFittedException.
    double inertia(const std::vector<std::vector<double>>& X) const;

    // Learned centroids — valid only after fit().
    // Throws NotFittedException.
    const std::vector<std::vector<double>>& centroids() const;

    bool isFitted() const noexcept { return _fitted; }
    size_t k() const noexcept { return _k; }
    size_t maxIterations() const noexcept { return _maxIter; }
    double tolerance() const noexcept { return _tol; }
    size_t numIterations() const noexcept { return _nIter; }

private:
    size_t _k;
    size_t _maxIter;
    double _tol;
    unsigned _seed;

    std::vector<std::vector<double>> _centroids;
    bool _fitted = false;
    size_t _nIter = 0;

    void _initPlusPlus(const std::vector<std::vector<double>>& X, std::mt19937& rng);
    static double _sqDist(const std::vector<double>& a, const std::vector<double>& b) noexcept;
    static size_t _nearest(
        const std::vector<double>& x, const std::vector<std::vector<double>>& cents) noexcept;
};

} // namespace nu
