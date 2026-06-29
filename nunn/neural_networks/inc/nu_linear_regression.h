//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Linear Regression model.
//
// Supports two training methods:
//   OLS            — closed-form Ordinary Least Squares via QR decomposition.
//                    Exact solution in one pass; requires the system to be
//                    well-conditioned.
//   GradientDescent — iterative MSE minimisation; connects directly to the
//                    backpropagation framework used by MLP networks.
//
// Model: y_hat = w^T x + b
//
// Usage:
//   LinearRegression lr(LinearRegression::Method::OLS);
//   lr.fit(X_train, y_train);
//   double pred = lr.predict(x_new);
//   double r2   = lr.rSquared(X_test, y_test);

#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <stdexcept>
#include <vector>

namespace nu {

class LinearRegression {
public:
    enum class Method { OLS, GradientDescent };

    struct NotFittedException : std::runtime_error {
        NotFittedException()
            : std::runtime_error("LinearRegression: call fit() before predict()")
        {
        }
    };

    struct SizeMismatchException : std::runtime_error {
        explicit SizeMismatchException(const char* msg)
            : std::runtime_error(msg)
        {
        }
    };

    // method       — OLS (default) or GradientDescent
    // learningRate — step size for GradientDescent (ignored for OLS)
    // maxIter      — maximum epochs for GradientDescent (ignored for OLS)
    // tolerance    — stop GradientDescent when weight change < tol
    explicit LinearRegression(Method method = Method::OLS, double learningRate = 0.01,
        size_t maxIter = 10000, double tolerance = 1e-9) noexcept;

    // Fit the model to (X, y).
    // X: N samples x F features.  y: N target values.
    // Throws SizeMismatchException if dimensions are inconsistent.
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    // Predict for a single sample.
    // Throws NotFittedException if called before fit().
    // Throws SizeMismatchException if x.size() != inputSize().
    double predict(const std::vector<double>& x) const;

    // Predict for a batch of samples.
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    // Mean Squared Error on (X, y).
    double mse(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;

    // Coefficient of determination R² on (X, y).
    // Returns 1.0 for a perfect fit; 0.0 for a mean-only predictor.
    double rSquared(const std::vector<std::vector<double>>& X, const std::vector<double>& y) const;

    // Learned coefficients (one per feature, excluding intercept).
    const std::vector<double>& coefficients() const;

    // Learned intercept (bias term).
    double intercept() const noexcept { return _b; }

    bool isFitted() const noexcept { return _fitted; }
    size_t inputSize() const noexcept { return _inputSize; }
    Method method() const noexcept { return _method; }
    double learningRate() const noexcept { return _lr; }
    size_t maxIterations() const noexcept { return _maxIter; }
    double tolerance() const noexcept { return _tol; }

    void setLearningRate(double lr) noexcept { _lr = lr; }
    void setMaxIterations(size_t n) noexcept { _maxIter = n; }
    void setTolerance(double tol) noexcept { _tol = tol; }

private:
    Method _method;
    double _lr;
    size_t _maxIter;
    double _tol;

    Eigen::VectorXd _w; // [F] feature weights
    double _b = 0.0; // intercept
    bool _fitted = false;
    size_t _inputSize = 0;

    mutable std::vector<double> _coefCache;
    mutable bool _cacheValid = false;

    void _fitOLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void _fitGD(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    static Eigen::MatrixXd toEigenMatrix(const std::vector<std::vector<double>>& X);
    static Eigen::VectorXd toEigenVector(const std::vector<double>& v);
};

} // namespace nu
