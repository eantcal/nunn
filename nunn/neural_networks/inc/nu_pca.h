//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Principal Component Analysis (PCA) via truncated SVD on centred data.
//
// Algorithm (fit):
//   1. Centre X: subtract per-feature mean.
//   2. Compute thin SVD: X_c = U * S * V^T.
//   3. Principal components = first nComponents columns of V (rows of V^T).
//   4. Explained variance per component = S_i^2 / (N-1) / total_variance.
//
// Projection: z = V_k^T * (x - mu)   [nComponents-dimensional]
// Reconstruction: x_hat = V_k * z + mu
//
// Usage:
//   Pca pca(2);
//   pca.fit(X);
//   auto Z    = pca.transform(X);        // N x 2 projections
//   auto Xhat = pca.inverseTransform(Z); // approximate reconstruction

#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <stdexcept>
#include <vector>

namespace nu {

class Pca {
public:
    struct NotFittedException : std::runtime_error {
        NotFittedException()
            : std::runtime_error("Pca: call fit() before transform()")
        {
        }
    };

    struct SizeMismatchException : std::runtime_error {
        explicit SizeMismatchException(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    // nComponents — number of principal components to retain (>= 1).
    explicit Pca(size_t nComponents) noexcept;

    // Fit: centre X, run thin SVD, store nComponents right singular vectors.
    // Requires nComponents <= min(nSamples, nFeatures).
    // Throws SizeMismatchException on empty / inconsistent data or out-of-range nComponents.
    void fit(const std::vector<std::vector<double>>& X);

    // Project a single sample into the component space.
    // Returns a vector of length nComponents.
    // Throws NotFittedException / SizeMismatchException.
    std::vector<double> transform(const std::vector<double>& x) const;

    // Batch transform — returns one row per input sample.
    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>>& X) const;

    // Reconstruct an approximate original sample from a projected vector.
    // Throws NotFittedException / SizeMismatchException.
    std::vector<double> inverseTransform(const std::vector<double>& z) const;

    // Fraction of total variance captured by each retained component.
    // Valid only after fit(); throws NotFittedException otherwise.
    const std::vector<double>& explainedVarianceRatio() const;

    // Sum of all entries in explainedVarianceRatio().
    double totalExplainedVariance() const noexcept { return _totalExplVar; }

    size_t nComponents() const noexcept { return _nComp; }
    size_t inputDim() const noexcept { return _dim; }
    bool isFitted() const noexcept { return _fitted; }

private:
    size_t _nComp;
    size_t _dim = 0;
    bool _fitted = false;

    Eigen::VectorXd _mean; // [D]           per-feature mean
    Eigen::MatrixXd _components; // [nComp x D]   principal components (rows)
    std::vector<double> _explVar; // [nComp]       explained variance ratios
    double _totalExplVar = 0.0;
};

} // namespace nu
