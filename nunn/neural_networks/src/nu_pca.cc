//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_pca.h"

#include <algorithm>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Pca::Pca(size_t nComponents) noexcept
    : _nComp(nComponents)
{
}

// ── fit ───────────────────────────────────────────────────────────────────────

void Pca::fit(const std::vector<std::vector<double>>& X)
{
    if (X.empty())
        throw SizeMismatchException("Pca::fit: dataset is empty");
    const size_t N = X.size();
    const size_t D = X[0].size();
    if (D == 0)
        throw SizeMismatchException("Pca::fit: samples have zero features");
    for (const auto& x : X)
        if (x.size() != D)
            throw SizeMismatchException("Pca::fit: inconsistent sample sizes");
    if (_nComp == 0 || _nComp > std::min(N, D))
        throw SizeMismatchException(
            "Pca::fit: nComponents must be in [1, min(nSamples, nFeatures)]");

    const Eigen::Index iN = static_cast<Eigen::Index>(N);
    const Eigen::Index iD = static_cast<Eigen::Index>(D);

    // Build data matrix [N x D]
    Eigen::MatrixXd Xm(iN, iD);
    for (Eigen::Index i = 0; i < iN; ++i)
        for (Eigen::Index j = 0; j < iD; ++j)
            Xm(i, j) = X[static_cast<size_t>(i)][static_cast<size_t>(j)];

    // Centre
    _mean = Xm.colwise().mean(); // [D]
    Xm.rowwise() -= _mean.transpose();

    // Thin SVD: Xm = U * S * V^T
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xm, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& S = svd.singularValues(); // descending

    // Components = first nComp columns of V (stored as rows)
    const Eigen::Index nc = static_cast<Eigen::Index>(_nComp);
    _components = svd.matrixV().leftCols(nc).transpose(); // [nComp x D]

    // Explained variance: var_i = S_i^2 / (N-1)
    const double denom = (N > 1) ? static_cast<double>(N - 1) : 1.0;
    double totalVar = 0.0;
    for (Eigen::Index i = 0; i < S.size(); ++i)
        totalVar += S(i) * S(i);
    totalVar /= denom;
    if (totalVar < 1e-15)
        totalVar = 1.0; // constant data guard

    _explVar.resize(_nComp);
    double cumul = 0.0;
    for (size_t i = 0; i < _nComp; ++i) {
        const double v = S(static_cast<Eigen::Index>(i));
        _explVar[i] = (v * v / denom) / totalVar;
        cumul += _explVar[i];
    }
    _totalExplVar = cumul;
    _dim = D;
    _fitted = true;
}

// ── transform ─────────────────────────────────────────────────────────────────

std::vector<double> Pca::transform(const std::vector<double>& x) const
{
    if (!_fitted)
        throw NotFittedException();
    if (x.size() != _dim)
        throw SizeMismatchException("Pca::transform: input size mismatch");

    const Eigen::Map<const Eigen::VectorXd> xe(x.data(), static_cast<Eigen::Index>(_dim));
    const Eigen::VectorXd z = _components * (xe - _mean); // [nComp]

    std::vector<double> out(_nComp);
    for (size_t i = 0; i < _nComp; ++i)
        out[i] = z(static_cast<Eigen::Index>(i));
    return out;
}

std::vector<std::vector<double>> Pca::transform(const std::vector<std::vector<double>>& X) const
{
    std::vector<std::vector<double>> out;
    out.reserve(X.size());
    for (const auto& x : X)
        out.push_back(transform(x));
    return out;
}

// ── inverseTransform ──────────────────────────────────────────────────────────

std::vector<double> Pca::inverseTransform(const std::vector<double>& z) const
{
    if (!_fitted)
        throw NotFittedException();
    if (z.size() != _nComp)
        throw SizeMismatchException("Pca::inverseTransform: z size != nComponents");

    const Eigen::Map<const Eigen::VectorXd> ze(z.data(), static_cast<Eigen::Index>(_nComp));
    const Eigen::VectorXd xhat = _components.transpose() * ze + _mean; // [D]

    std::vector<double> out(_dim);
    for (size_t i = 0; i < _dim; ++i)
        out[i] = xhat(static_cast<Eigen::Index>(i));
    return out;
}

// ── explainedVarianceRatio ────────────────────────────────────────────────────

const std::vector<double>& Pca::explainedVarianceRatio() const
{
    if (!_fitted)
        throw NotFittedException();
    return _explVar;
}

} // namespace nu
