//
// linear_regression_demo — demonstrates nu::LinearRegression
//
// Two datasets:
//   1. Synthetic 1D: y = 3x + 2  (exact fit to verify OLS vs GD parity)
//   2. Synthetic multi-feature: house price = 50*sqm + 30*rooms - 10*age + 80
//      (regression with 3 features; OLS exact, GD converges)
//

#include "nu_linear_regression.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using nu::LinearRegression;
using Method = LinearRegression::Method;

// ── helpers ──────────────────────────────────────────────────────────────────

static void printCoefficients(const LinearRegression& lr, const std::vector<std::string>& names)
{
    const auto& w = lr.coefficients();
    for (size_t i = 0; i < w.size(); ++i)
        std::cout << "  " << names[i] << " = " << std::fixed << std::setprecision(4) << w[i]
                  << "\n";
    std::cout << "  intercept = " << lr.intercept() << "\n";
}

// ── Dataset 1: 1D synthetic ───────────────────────────────────────────────────

static void demo1D()
{
    std::cout << "=== Dataset 1: y = 3x + 2 (exact) ===\n";

    // Build training data
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i <= 10; ++i) {
        double x = i * 0.1; // scale to [0, 1] so GD converges easily
        X.push_back({ x });
        y.push_back(3.0 * x + 2.0);
    }

    // OLS
    {
        LinearRegression lr(Method::OLS);
        lr.fit(X, y);
        std::cout << "\n[OLS]\n";
        printCoefficients(lr, { "x" });
        std::cout << "  R² = " << lr.rSquared(X, y) << "\n";
        std::cout << "  MSE = " << lr.mse(X, y) << "\n";
        std::cout << "  predict(x=1.5) = " << lr.predict({ 1.5 }) << "  (expected "
                  << 3.0 * 1.5 + 2.0 << ")\n";
    }

    // GD
    {
        LinearRegression lr(Method::GradientDescent, 0.1, 100000, 1e-10);
        lr.fit(X, y);
        std::cout << "\n[Gradient Descent  lr=0.1  maxIter=100000]\n";
        printCoefficients(lr, { "x" });
        std::cout << "  R² = " << lr.rSquared(X, y) << "\n";
        std::cout << "  MSE = " << lr.mse(X, y) << "\n";
        std::cout << "  predict(x=1.5) = " << lr.predict({ 1.5 }) << "  (expected "
                  << 3.0 * 1.5 + 2.0 << ")\n";
    }
}

// ── Dataset 2: house price ────────────────────────────────────────────────────
//
// Synthetic: price = 50*sqm + 30*rooms - 10*age + 80  (all in thousands)
//
static void demoHousePrice()
{
    std::cout << "\n=== Dataset 2: house price (3 features) ===\n";
    std::cout << "  model: price = 50*sqm + 30*rooms - 10*age + 80\n\n";

    // Generate 30 training samples
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    // Normalized features so GD converges without scaling issues:
    //   sqm in [0.5 .. 1.5]  (~50-150 m² / 100)
    //   rooms in [1..5]       normalized as [0.2..1.0]
    //   age in [0..40]        normalized as [0..0.4]

    const int N = 30;
    for (int i = 0; i < N; ++i) {
        double sqm = 0.5 + (i % 10) * 0.1;
        double rooms = 0.2 + (i % 5) * 0.2;
        double age = 0.0 + (i % 8) * 0.05;
        X.push_back({ sqm, rooms, age });
        y.push_back(50.0 * sqm + 30.0 * rooms - 10.0 * age + 80.0);
    }

    // OLS
    {
        LinearRegression lr(Method::OLS);
        lr.fit(X, y);
        std::cout << "[OLS]\n";
        printCoefficients(lr, { "sqm", "rooms", "age" });
        std::cout << "  R² = " << lr.rSquared(X, y) << "\n";
        std::cout << "  MSE = " << lr.mse(X, y) << "\n";

        // Predict a new house: 80 m², 3 rooms, 5 years old
        double pred = lr.predict({ 0.80, 0.60, 0.05 });
        double expected = 50.0 * 0.80 + 30.0 * 0.60 - 10.0 * 0.05 + 80.0;
        std::cout << "  predict(sqm=0.80, rooms=0.60, age=0.05) = " << std::fixed
                  << std::setprecision(2) << pred << "  (expected " << expected << ")\n";
    }

    // GD
    {
        LinearRegression lr(Method::GradientDescent, 0.05, 200000, 1e-10);
        lr.fit(X, y);
        std::cout << "\n[Gradient Descent  lr=0.05  maxIter=200000]\n";
        printCoefficients(lr, { "sqm", "rooms", "age" });
        std::cout << "  R² = " << lr.rSquared(X, y) << "\n";
        std::cout << "  MSE = " << lr.mse(X, y) << "\n";

        double pred = lr.predict({ 0.80, 0.60, 0.05 });
        double expected = 50.0 * 0.80 + 30.0 * 0.60 - 10.0 * 0.05 + 80.0;
        std::cout << "  predict(sqm=0.80, rooms=0.60, age=0.05) = " << std::fixed
                  << std::setprecision(2) << pred << "  (expected " << expected << ")\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    try {
        demo1D();
        demoHousePrice();
        std::cout << "\nDone.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
