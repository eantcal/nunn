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
        { Vector { 0.0, 0.0 }, 0.0 },
        { Vector { 0.0, 1.0 }, 0.0 },
        { Vector { 1.0, 0.0 }, 0.0 },
        { Vector { 1.0, 1.0 }, 1.0 },
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
    EXPECT_THROW(p.setInputVector(Vector { 1.0, 2.0, 3.0 }),
        Perceptron::SizeMismatchException);
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

TEST(PerceptronTest, SaveLoadRoundTrip)
{
    Perceptron p(2, 0.3, StepFunction(0.5, 0.0, 1.0));
    p.setInputVector(Vector { 1.0, 1.0 });
    p.feedForward();
    const double before = p.getOutput();

    std::stringstream ss;
    p.save(ss);

    Perceptron loaded(ss); // constructs via load()
    loaded.setInputVector(Vector { 1.0, 1.0 });
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
    p.setInputVector(Vector { 1.0, 1.0 });
    p.feedForward();
    const double before = p.getOutput();

    std::stringstream ss;
    p.save(ss);

    Perceptron loaded(ss);
    loaded.setInputVector(Vector { 1.0, 1.0 });
    loaded.feedForward();

    EXPECT_NEAR(loaded.getOutput(), before, 1e-12);
}
