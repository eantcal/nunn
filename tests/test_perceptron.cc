//
// Unit tests for nu::Perceptron (nu_perceptron.h / nu_perceptron.cc).
//

#include "nu_perceptron.h"
#include "nu_stepf.h"
#include "nu_vector.h"

#include <gtest/gtest.h>

#include <array>
#include <sstream>
#include <utility>

using nu::Perceptron;
using nu::StepFunction;
using nu::Vector;

namespace {

// Trains a perceptron on the AND truth table (a linearly separable problem)
// and returns true if it classifies all four samples correctly.
bool trainAnd()
{
    // Threshold 0.5 on the sigmoid output -> sharp {0,1} classification.
    Perceptron p(2, 0.5, StepFunction(0.5, 0.0, 1.0));

    const std::array<std::pair<Vector, double>, 4> samples = { {
        { Vector{ 0.0, 0.0 }, 0.0 },
        { Vector{ 0.0, 1.0 }, 0.0 },
        { Vector{ 1.0, 0.0 }, 0.0 },
        { Vector{ 1.0, 1.0 }, 1.0 },
    } };

    for (int epoch = 0; epoch < 5000; ++epoch) {
        for (const auto& [input, target] : samples) {
            p.setInputVector(input);
            p.backPropagate(target);
        }
    }

    for (const auto& [input, target] : samples) {
        p.setInputVector(input);
        p.feedForward();
        if (p.getSharpOutput() != target) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST(PerceptronTest, ConstructorRejectsZeroInputs)
{
    EXPECT_THROW(Perceptron(0), Perceptron::SizeMismatchException);
}

TEST(PerceptronTest, InputSizeMatchesConstruction)
{
    Perceptron p(3);
    EXPECT_EQ(p.getInputSize(), 3u);
}

TEST(PerceptronTest, SetInputVectorSizeMismatchThrows)
{
    Perceptron p(2);
    EXPECT_THROW(p.setInputVector(Vector{ 1.0, 2.0, 3.0 }), Perceptron::SizeMismatchException);
}

TEST(PerceptronTest, LearnsAndGate)
{
    // Random weight init can occasionally start in a bad spot; allow a few retries.
    bool ok = false;
    for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
        ok = trainAnd();
    }
    EXPECT_TRUE(ok) << "Perceptron failed to learn the AND function";
}

// With MSE the gradient is δ = (t−y)·y·(1−y).  At a perfect prediction
// (output ≈ target) the gradient approaches 0, so the weights must be
// stable after convergence (i.e., the output must remain close to the target
// after training stops).
TEST(PerceptronTest, MseGradientIncludesSigmoidDerivative)
{
    // Saturated sigmoid: output ≈ 1.0 → σ'(y) = y*(1−y) ≈ 0.
    // A gradient WITHOUT σ' would still try to push weights harder even when
    // near saturation; a correct MSE gradient decelerates naturally.
    Perceptron p(1, 0.5, StepFunction(0.5), nu::CostFunction::MSE);

    // Force a predictable state: one input, target = 1.0
    for (int i = 0; i < 10000; ++i) {
        p.setInputVector(Vector{ 1.0 });
        p.backPropagate(1.0);
    }

    p.setInputVector(Vector{ 1.0 });
    p.feedForward();
    // After many epochs the output should be close to 1.0
    EXPECT_GT(p.getOutput(), 0.90);
}

// CrossEntropy gradient: δ = (t−y), no σ' factor.
// Logistic regression converges faster than MSE on the same problem.
TEST(PerceptronTest, CrossEntropyGradientConvergesOnAnd)
{
    bool ok = false;
    for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
        Perceptron p(2, 0.5, StepFunction(0.5, 0.0, 1.0), nu::CostFunction::CrossEntropy);

        const std::array<std::pair<Vector, double>, 4> samples = { {
            { Vector{ 0.0, 0.0 }, 0.0 },
            { Vector{ 0.0, 1.0 }, 0.0 },
            { Vector{ 1.0, 0.0 }, 0.0 },
            { Vector{ 1.0, 1.0 }, 1.0 },
        } };

        for (int epoch = 0; epoch < 5000; ++epoch) {
            for (const auto& [input, target] : samples) {
                p.setInputVector(input);
                p.backPropagate(target);
            }
        }

        ok = true;
        for (const auto& [input, target] : samples) {
            p.setInputVector(input);
            p.feedForward();
            if (p.getSharpOutput() != target) {
                ok = false;
                break;
            }
        }
    }
    EXPECT_TRUE(ok) << "Perceptron (CE) failed to learn AND";
}

// Momentum must be persisted through save()/load() and toJson()/loadJson().
TEST(PerceptronTest, SaveLoadPreservesMomentumAndCostFunction)
{
    Perceptron p(2, 0.3, StepFunction(0.5), nu::CostFunction::CrossEntropy, 0.9);

    // stringstream round-trip
    std::stringstream ss;
    p.save(ss);
    Perceptron loaded(ss);
    EXPECT_DOUBLE_EQ(loaded.getMomentum(), 0.9);
    EXPECT_EQ(loaded.getCostFunction(), nu::CostFunction::CrossEntropy);

    // JSON round-trip
    std::ostringstream os;
    p.toJson(os);
    std::istringstream is(os.str());
    Perceptron fromJson;
    fromJson.loadJson(is);
    EXPECT_DOUBLE_EQ(fromJson.getMomentum(), 0.9);
    EXPECT_EQ(fromJson.getCostFunction(), nu::CostFunction::CrossEntropy);
}

TEST(PerceptronTest, SaveLoadRoundTrip)
{
    Perceptron p(2, 0.3, StepFunction(0.5, 0.0, 1.0));
    p.setInputVector(Vector{ 1.0, 1.0 });
    p.feedForward();
    const double before = p.getOutput();

    std::stringstream ss;
    p.save(ss);

    Perceptron loaded(ss); // constructs via load()
    loaded.setInputVector(Vector{ 1.0, 1.0 });
    loaded.feedForward();

    EXPECT_NEAR(loaded.getOutput(), before, 1e-4);
    EXPECT_EQ(loaded.getInputSize(), 2u);
}

// Regression (bug #14): save() must write doubles at full round-trip precision.
// It previously used the stream's default precision (~6 significant digits)
// with no std::setprecision, so a save/load round-trip silently lost precision.
TEST(PerceptronTest, SaveLoadRoundTripIsFullPrecision)
{
    Perceptron p(2, 0.3, StepFunction(0.5, 0.0, 1.0));
    p.setInputVector(Vector{ 1.0, 1.0 });
    p.feedForward();
    const double before = p.getOutput();

    std::stringstream ss;
    p.save(ss);

    Perceptron loaded(ss);
    loaded.setInputVector(Vector{ 1.0, 1.0 });
    loaded.feedForward();

    EXPECT_NEAR(loaded.getOutput(), before, 1e-12);
}
