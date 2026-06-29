//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_kmeans.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

KMeans::KMeans(size_t k, size_t maxIter, double tol, unsigned seed) noexcept
    : _k(k)
    , _maxIter(maxIter)
    , _tol(tol)
    , _seed(seed)
{
}

// ── Helpers ───────────────────────────────────────────────────────────────────

double KMeans::_sqDist(const std::vector<double>& a, const std::vector<double>& b) noexcept
{
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

size_t KMeans::_nearest(
    const std::vector<double>& x, const std::vector<std::vector<double>>& cents) noexcept
{
    size_t best = 0;
    double bestD = std::numeric_limits<double>::max();
    for (size_t i = 0; i < cents.size(); ++i) {
        const double d = _sqDist(x, cents[i]);
        if (d < bestD) {
            bestD = d;
            best = i;
        }
    }
    return best;
}

// ── k-means++ initialisation ──────────────────────────────────────────────────

void KMeans::_initPlusPlus(const std::vector<std::vector<double>>& X, std::mt19937& rng)
{
    const size_t N = X.size();
    _centroids.clear();
    _centroids.reserve(_k);

    std::uniform_int_distribution<size_t> unif(0, N - 1);
    _centroids.push_back(X[unif(rng)]);

    std::vector<double> D(N);
    for (size_t c = 1; c < _k; ++c) {
        double total = 0.0;
        for (size_t i = 0; i < N; ++i) {
            D[i] = _sqDist(X[i], _centroids[_nearest(X[i], _centroids)]);
            total += D[i];
        }
        // Sample index proportional to D[i]
        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(rng);
        size_t chosen = N - 1;
        for (size_t i = 0; i < N; ++i) {
            r -= D[i];
            if (r <= 0.0) {
                chosen = i;
                break;
            }
        }
        _centroids.push_back(X[chosen]);
    }
}

// ── fit ───────────────────────────────────────────────────────────────────────

void KMeans::fit(const std::vector<std::vector<double>>& X)
{
    if (X.empty())
        throw SizeMismatchException("KMeans::fit: dataset is empty");
    const size_t N = X.size();
    const size_t D = X[0].size();
    if (D == 0)
        throw SizeMismatchException("KMeans::fit: samples have zero features");
    for (const auto& x : X)
        if (x.size() != D)
            throw SizeMismatchException("KMeans::fit: inconsistent sample sizes");
    if (_k == 0)
        throw SizeMismatchException("KMeans::fit: k must be > 0");
    if (_k > N)
        throw SizeMismatchException("KMeans::fit: k > number of samples");

    std::mt19937 rng(_seed);
    _initPlusPlus(X, rng);

    std::vector<size_t> labels(N, 0);
    std::vector<std::vector<double>> newCentroids(_k, std::vector<double>(D, 0.0));
    std::vector<size_t> counts(_k, 0);

    _nIter = 0;
    for (size_t iter = 0; iter < _maxIter; ++iter) {
        ++_nIter;

        // E-step: assign each sample to its nearest centroid
        for (size_t i = 0; i < N; ++i)
            labels[i] = _nearest(X[i], _centroids);

        // M-step: recompute centroids as cluster means
        for (auto& c : newCentroids)
            std::fill(c.begin(), c.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < N; ++i) {
            const size_t c = labels[i];
            ++counts[c];
            for (size_t d = 0; d < D; ++d)
                newCentroids[c][d] += X[i][d];
        }
        for (size_t c = 0; c < _k; ++c) {
            if (counts[c] > 0)
                for (size_t d = 0; d < D; ++d)
                    newCentroids[c][d] /= static_cast<double>(counts[c]);
            // Empty cluster: keep old centroid unchanged
        }

        // Convergence check: largest centroid shift
        double maxShift = 0.0;
        for (size_t c = 0; c < _k; ++c)
            maxShift = std::max(maxShift, std::sqrt(_sqDist(newCentroids[c], _centroids[c])));

        _centroids = newCentroids;
        if (maxShift < _tol)
            break;
    }
    _fitted = true;
}

// ── predict ───────────────────────────────────────────────────────────────────

size_t KMeans::predict(const std::vector<double>& x) const
{
    if (!_fitted)
        throw NotFittedException();
    if (x.size() != _centroids[0].size())
        throw SizeMismatchException("KMeans::predict: input size mismatch");
    return _nearest(x, _centroids);
}

std::vector<size_t> KMeans::predict(const std::vector<std::vector<double>>& X) const
{
    std::vector<size_t> out;
    out.reserve(X.size());
    for (const auto& x : X)
        out.push_back(predict(x));
    return out;
}

// ── inertia ───────────────────────────────────────────────────────────────────

double KMeans::inertia(const std::vector<std::vector<double>>& X) const
{
    if (!_fitted)
        throw NotFittedException();
    double total = 0.0;
    for (const auto& x : X)
        total += _sqDist(x, _centroids[predict(x)]);
    return total;
}

// ── centroids accessor ────────────────────────────────────────────────────────

const std::vector<std::vector<double>>& KMeans::centroids() const
{
    if (!_fitted)
        throw NotFittedException();
    return _centroids;
}

} // namespace nu
