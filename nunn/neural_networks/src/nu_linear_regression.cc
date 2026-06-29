//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_linear_regression.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nu {

LinearRegression::LinearRegression(
    Method method, double learningRate, size_t maxIter, double tolerance) noexcept
    : _method(method)
    , _lr(learningRate)
    , _maxIter(maxIter)
    , _tol(tolerance)
{
}

// ── static helpers ──────────────────────────────────────────────────────────

Eigen::MatrixXd LinearRegression::toEigenMatrix(const std::vector<std::vector<double>>& X)
{
    if (X.empty())
        return Eigen::MatrixXd(0, 0);
    const size_t N = X.size();
    const size_t F = X[0].size();
    Eigen::MatrixXd M(N, F);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < F; ++j)
            M(i, j) = X[i][j];
    return M;
}

Eigen::VectorXd LinearRegression::toEigenVector(const std::vector<double>& v)
{
    return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
}

// ── OLS fit ─────────────────────────────────────────────────────────────────

// Solves the augmented normal equations [X|1]^T [X|1] w_aug = [X|1]^T y.
// w_aug = [w0 ... wF-1  b] — bias is the last component.
void LinearRegression::_fitOLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    const Eigen::Index N = X.rows();
    const Eigen::Index F = X.cols();

    // Build augmented matrix X_aug = [X  ones_column]
    Eigen::MatrixXd Xaug(N, F + 1);
    Xaug.leftCols(F) = X;
    Xaug.col(F).setOnes();

    // Solve via column-pivoting QR (numerically stable, handles rank-deficient cases)
    Eigen::VectorXd w_aug = Xaug.colPivHouseholderQr().solve(y);

    _w = w_aug.head(F);
    _b = w_aug(F);
    _cacheValid = false;
}

// ── Gradient Descent fit ────────────────────────────────────────────────────

void LinearRegression::_fitGD(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    const Eigen::Index N = X.rows();
    const Eigen::Index F = X.cols();

    _w.setZero(F);
    _b = 0.0;

    for (size_t iter = 0; iter < _maxIter; ++iter) {
        // Residuals: r = X*w + b*ones - y
        Eigen::VectorXd r = X * _w + Eigen::VectorXd::Constant(N, _b) - y;

        Eigen::VectorXd grad_w = (X.transpose() * r) / N;
        double grad_b = r.mean();

        Eigen::VectorXd delta_w = _lr * grad_w;
        _w -= delta_w;
        _b -= _lr * grad_b;

        if (delta_w.norm() < _tol)
            break;
    }
    _cacheValid = false;
}

// ── Public interface ─────────────────────────────────────────────────────────

void LinearRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y)
{
    if (X.empty() || y.empty())
        throw SizeMismatchException("LinearRegression::fit: empty dataset");
    if (X.size() != y.size())
        throw SizeMismatchException(
            "LinearRegression::fit: X and y must have the same number of rows");
    const size_t F = X[0].size();
    if (F == 0)
        throw SizeMismatchException("LinearRegression::fit: feature dimension must be > 0");
    for (size_t i = 1; i < X.size(); ++i)
        if (X[i].size() != F)
            throw SizeMismatchException(
                "LinearRegression::fit: all rows of X must have the same length");

    const Eigen::MatrixXd Xm = toEigenMatrix(X);
    const Eigen::VectorXd yv = toEigenVector(y);

    if (_method == Method::OLS)
        _fitOLS(Xm, yv);
    else
        _fitGD(Xm, yv);

    _inputSize = F;
    _fitted = true;
}

double LinearRegression::predict(const std::vector<double>& x) const
{
    if (!_fitted)
        throw NotFittedException();
    if (x.size() != _inputSize)
        throw SizeMismatchException("LinearRegression::predict: input size mismatch");

    const Eigen::VectorXd xv = toEigenVector(x);
    return _w.dot(xv) + _b;
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& X) const
{
    std::vector<double> out;
    out.reserve(X.size());
    for (const auto& x : X)
        out.push_back(predict(x));
    return out;
}

double LinearRegression::mse(
    const std::vector<std::vector<double>>& X, const std::vector<double>& y) const
{
    const auto yhat = predict(X);
    double acc = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double d = yhat[i] - y[i];
        acc += d * d;
    }
    return acc / y.size();
}

double LinearRegression::rSquared(
    const std::vector<std::vector<double>>& X, const std::vector<double>& y) const
{
    const double ybar = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    const auto yhat = predict(X);

    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double d_res = yhat[i] - y[i];
        double d_tot = y[i] - ybar;
        ss_res += d_res * d_res;
        ss_tot += d_tot * d_tot;
    }
    if (ss_tot == 0.0)
        return 1.0;
    return 1.0 - ss_res / ss_tot;
}

const std::vector<double>& LinearRegression::coefficients() const
{
    if (!_fitted)
        throw NotFittedException();
    if (!_cacheValid) {
        _coefCache.assign(_w.data(), _w.data() + _w.size());
        _cacheValid = true;
    }
    return _coefCache;
}

} // namespace nu
