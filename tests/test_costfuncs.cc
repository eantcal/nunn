//
// Unit tests for the cost functions (nu_costfuncs.h / nu_costfuncs.cc).
//

#include "nu_costfuncs.h"
#include "nu_vector.h"

#include <gtest/gtest.h>

using nu::Vector;

TEST(CostFuncTest, MseIsZeroForEqualVectors)
{
    Vector output { 0.2, 0.8, 0.5 };
    Vector target { 0.2, 0.8, 0.5 };
    EXPECT_DOUBLE_EQ(nu::cf::calcMSE(output, target), 0.0);
}

TEST(CostFuncTest, MseKnownValue)
{
    // 0.5 * || output - target ||^2 = 0.5 * (1^2 + 0^2) = 0.5
    Vector output { 1.0, 0.0 };
    Vector target { 0.0, 0.0 };
    EXPECT_DOUBLE_EQ(nu::cf::calcMSE(output, target), 0.5);
}

TEST(CostFuncTest, MseDoesNotMutateCaller)
{
    // calcMSE takes the output vector by value, so the caller's copy must be intact.
    Vector output { 1.0, 0.0 };
    Vector target { 0.0, 0.0 };
    (void)nu::cf::calcMSE(output, target);
    EXPECT_DOUBLE_EQ(output[0], 1.0);
    EXPECT_DOUBLE_EQ(output[1], 0.0);
}

TEST(CostFuncTest, CrossEntropyIsFiniteAndNonNegative)
{
    Vector output { 0.7, 0.2, 0.9 };
    Vector target { 1.0, 0.0, 1.0 };
    const double ce = nu::cf::calcCrossEntropy(output, target);
    EXPECT_TRUE(std::isfinite(ce));
    EXPECT_GE(ce, 0.0);
}

TEST(CostFuncTest, CrossEntropyPenalizesConfidentWrongMore)
{
    // A confident-but-wrong prediction must cost more than a confident-correct one.
    Vector target { 1.0 };
    Vector good { 0.99 };
    Vector bad { 0.01 };
    EXPECT_GT(nu::cf::calcCrossEntropy(bad, target),
        nu::cf::calcCrossEntropy(good, target));
}

TEST(CostFuncTest, CrossEntropyHandlesSaturatedOutputs)
{
    // Outputs of exactly 0 / 1 must not produce NaN/Inf (log clamping).
    Vector output { 1.0, 0.0 };
    Vector target { 1.0, 0.0 };
    const double ce = nu::cf::calcCrossEntropy(output, target);
    EXPECT_TRUE(std::isfinite(ce));
}
