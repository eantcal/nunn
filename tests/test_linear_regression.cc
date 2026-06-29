//
// Tests for nu::LinearRegression
//

#include "nu_linear_regression.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using nu::LinearRegression;
using Method = LinearRegression::Method;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::vector<std::vector<double>> makeX1D(const std::vector<double>& xs)
{
    std::vector<std::vector<double>> X;
    X.reserve(xs.size());
    for (double x : xs)
        X.push_back({ x });
    return X;
}

// Build y = 3x + 2
static std::pair<std::vector<std::vector<double>>, std::vector<double>> linear1D()
{
    const std::vector<double> xs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<double> ys;
    ys.reserve(xs.size());
    for (double x : xs)
        ys.push_back(3.0 * x + 2.0);
    return { makeX1D(xs), ys };
}

// Build y = 2x1 - x2 + 5  (multivariate, 4x5 orthogonal grid)
static std::pair<std::vector<std::vector<double>>, std::vector<double>> linear2D()
{
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            double x1 = i * 0.5;
            double x2 = j * 0.3;
            X.push_back({ x1, x2 });
            y.push_back(2.0 * x1 - x2 + 5.0);
        }
    }
    return { X, y };
}

// ── Construction ─────────────────────────────────────────────────────────────

TEST(LinearRegressionTest, DefaultsToOLS)
{
    LinearRegression lr;
    EXPECT_EQ(lr.method(), Method::OLS);
    EXPECT_FALSE(lr.isFitted());
}

TEST(LinearRegressionTest, GDConstructor)
{
    LinearRegression lr(Method::GradientDescent, 0.005, 20000, 1e-10);
    EXPECT_EQ(lr.method(), Method::GradientDescent);
    EXPECT_DOUBLE_EQ(lr.learningRate(), 0.005);
    EXPECT_EQ(lr.maxIterations(), 20000u);
    EXPECT_DOUBLE_EQ(lr.tolerance(), 1e-10);
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST(LinearRegressionTest, PredictBeforeFitThrows)
{
    LinearRegression lr;
    EXPECT_THROW(lr.predict({ 1.0 }), LinearRegression::NotFittedException);
}

TEST(LinearRegressionTest, FitEmptyXThrows)
{
    LinearRegression lr;
    EXPECT_THROW(lr.fit({}, {}), LinearRegression::SizeMismatchException);
}

TEST(LinearRegressionTest, FitXYSizeMismatchThrows)
{
    LinearRegression lr;
    EXPECT_THROW(
        lr.fit(makeX1D({ 1, 2, 3 }), { 1.0, 2.0 }), LinearRegression::SizeMismatchException);
}

TEST(LinearRegressionTest, PredictWrongSizeThrows)
{
    LinearRegression lr;
    auto [X, y] = linear1D();
    lr.fit(X, y);
    EXPECT_THROW(lr.predict({ 1.0, 2.0 }), LinearRegression::SizeMismatchException);
}

TEST(LinearRegressionTest, CoefficientsBeforeFitThrows)
{
    LinearRegression lr;
    EXPECT_THROW(lr.coefficients(), LinearRegression::NotFittedException);
}

// ── OLS: 1D exact fit ────────────────────────────────────────────────────────

TEST(LinearRegressionTest, OLS_FitsExact1D)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear1D();

    lr.fit(X, y);

    EXPECT_TRUE(lr.isFitted());
    EXPECT_EQ(lr.inputSize(), 1u);
    EXPECT_NEAR(lr.coefficients()[0], 3.0, 1e-9);
    EXPECT_NEAR(lr.intercept(), 2.0, 1e-9);
}

TEST(LinearRegressionTest, OLS_PredictNewPoint)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear1D();
    lr.fit(X, y);

    EXPECT_NEAR(lr.predict({ 10.0 }), 32.0, 1e-9); // 3*10 + 2
    EXPECT_NEAR(lr.predict({ -1.0 }), -1.0, 1e-9); // 3*(-1) + 2
}

TEST(LinearRegressionTest, OLS_RSquaredPerfectFit)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear1D();
    lr.fit(X, y);

    EXPECT_NEAR(lr.rSquared(X, y), 1.0, 1e-12);
}

TEST(LinearRegressionTest, OLS_MSEZeroOnExactFit)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear1D();
    lr.fit(X, y);

    EXPECT_NEAR(lr.mse(X, y), 0.0, 1e-18);
}

// ── OLS: 2D exact fit ────────────────────────────────────────────────────────

TEST(LinearRegressionTest, OLS_FitsExact2D)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear2D();
    lr.fit(X, y);

    EXPECT_EQ(lr.inputSize(), 2u);
    EXPECT_NEAR(lr.coefficients()[0], 2.0, 1e-9); // x1 coef
    EXPECT_NEAR(lr.coefficients()[1], -1.0, 1e-9); // x2 coef
    EXPECT_NEAR(lr.intercept(), 5.0, 1e-9);
}

TEST(LinearRegressionTest, OLS_BatchPredict)
{
    LinearRegression lr(Method::OLS);
    auto [X, y] = linear1D();
    lr.fit(X, y);

    auto yhat = lr.predict(X);
    ASSERT_EQ(yhat.size(), y.size());
    for (size_t i = 0; i < y.size(); ++i)
        EXPECT_NEAR(yhat[i], y[i], 1e-9);
}

// ── GD: 1D convergence ───────────────────────────────────────────────────────

TEST(LinearRegressionTest, GD_Converges1D)
{
    // Feature-scale x to [0,1] so GD converges with lr=0.1
    const std::vector<double> xs = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    std::vector<double> ys;
    for (double x : xs)
        ys.push_back(3.0 * x + 2.0); // y = 3x + 2

    LinearRegression lr(Method::GradientDescent, 0.1, 50000, 1e-10);
    lr.fit(makeX1D(xs), ys);

    EXPECT_NEAR(lr.coefficients()[0], 3.0, 1e-4);
    EXPECT_NEAR(lr.intercept(), 2.0, 1e-4);
}

TEST(LinearRegressionTest, GD_RSquaredHighOnExactData)
{
    const std::vector<double> xs = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    std::vector<double> ys;
    for (double x : xs)
        ys.push_back(3.0 * x + 2.0);

    LinearRegression lr(Method::GradientDescent, 0.1, 50000, 1e-10);
    auto X = makeX1D(xs);
    lr.fit(X, ys);

    EXPECT_GT(lr.rSquared(X, ys), 0.9999);
}

// ── OLS vs GD agreement ──────────────────────────────────────────────────────

TEST(LinearRegressionTest, OLS_and_GD_AgreeOnScaledData)
{
    const std::vector<double> xs = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    std::vector<double> ys;
    for (double x : xs)
        ys.push_back(5.0 * x - 1.0);

    auto X = makeX1D(xs);

    LinearRegression ols(Method::OLS);
    ols.fit(X, ys);

    LinearRegression gd(Method::GradientDescent, 0.1, 100000, 1e-11);
    gd.fit(X, ys);

    EXPECT_NEAR(ols.coefficients()[0], gd.coefficients()[0], 1e-3);
    EXPECT_NEAR(ols.intercept(), gd.intercept(), 1e-3);
}

// ── Setter tests ─────────────────────────────────────────────────────────────

TEST(LinearRegressionTest, SettersWork)
{
    LinearRegression lr(Method::GradientDescent);
    lr.setLearningRate(0.005);
    lr.setMaxIterations(5000);
    lr.setTolerance(1e-8);

    EXPECT_DOUBLE_EQ(lr.learningRate(), 0.005);
    EXPECT_EQ(lr.maxIterations(), 5000u);
    EXPECT_DOUBLE_EQ(lr.tolerance(), 1e-8);
}
