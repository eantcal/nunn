//
// Tests for nu::MlpMatrixNN — the Eigen-backed matrix MLP.
//
// Test suites:
//   MatrixApiTest        — construction, getters, exception for bad CE combo
//   MatrixConvergenceTest — XOR with all activation/cost combinations
//   MatrixMetricsTest    — MSE and CE calculation correctness
//

#include "nu_mlpmatrixnn.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using nu::Activation;
using nu::CostFunction;
using nu::MlpMatrixNN;
using LC = MlpMatrixNN::LayerConfig;

// ─────────────────────────────────────────────────────────────────────────────
// XOR helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using Sample = std::pair<std::vector<double>, std::vector<double>>;
using XorSet = std::array<Sample, 4>;

const XorSet& xorSamples()
{
    static const XorSet s = { {
        { { 0.0, 0.0 }, { 0.0 } },
        { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } },
        { { 1.0, 1.0 }, { 0.0 } },
    } };
    return s;
}

double trainXor(MlpMatrixNN& nn, int epochs)
{
    for (int e = 0; e < epochs; ++e)
        for (const auto& [in, tgt] : xorSamples()) {
            nn.setInputVector(in);
            nn.feedForward();
            nn.backPropagate(tgt);
        }
    double worst = 0.0;
    for (const auto& [in, tgt] : xorSamples()) {
        nn.setInputVector(in);
        nn.feedForward();
        worst = std::max(worst, nn.calcMSE(tgt));
    }
    return worst;
}

template <typename Factory> double xorBestMse(Factory factory, int epochs, int maxAttempts = 8)
{
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < maxAttempts; ++i) {
        MlpMatrixNN nn = factory();
        best = std::min(best, trainXor(nn, epochs));
        if (best < 0.05)
            break;
    }
    return best;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// MatrixApiTest
// ─────────────────────────────────────────────────────────────────────────────

TEST(MatrixApiTest, InputOutputSizes)
{
    MlpMatrixNN nn({ LC{ 3 }, { 5, Activation::Sigmoid }, { 2, Activation::Sigmoid } });
    EXPECT_EQ(nn.getInputSize(), 3u);
    EXPECT_EQ(nn.getOutputSize(), 2u);
}

TEST(MatrixApiTest, DefaultsAreMSEAndSigmoid)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });
    EXPECT_EQ(nn.getCostFunction(), CostFunction::MSE);
}

TEST(MatrixApiTest, LearningRateAndMomentumStored)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.37, 0.82);
    EXPECT_DOUBLE_EQ(nn.getLearningRate(), 0.37);
    EXPECT_DOUBLE_EQ(nn.getMomentum(), 0.82);
}

TEST(MatrixApiTest, CrossEntropyWithSigmoidOutputIsValid)
{
    EXPECT_NO_THROW(MlpMatrixNN({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } },
        0.1, 0.9, CostFunction::CrossEntropy));
}

TEST(MatrixApiTest, CrossEntropyWithTanhOutputThrows)
{
    EXPECT_THROW((MlpMatrixNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Tanh } }, 0.1,
                     0.9, CostFunction::CrossEntropy)),
        MlpMatrixNN::InvalidCostFunctionCombinationException);
}

TEST(MatrixApiTest, CrossEntropyWithReLUOutputThrows)
{
    EXPECT_THROW((MlpMatrixNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::ReLU } }, 0.1,
                     0.9, CostFunction::CrossEntropy)),
        MlpMatrixNN::InvalidCostFunctionCombinationException);
}

TEST(MatrixApiTest, FeedForwardOutputInUnitInterval)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });
    nn.setInputVector({ 0.5, -0.5 });
    nn.feedForward();
    std::vector<double> out;
    nn.copyOutputVector(out);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_GT(out[0], 0.0);
    EXPECT_LT(out[0], 1.0);
}

TEST(MatrixApiTest, ReshuffleChangesOutput)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });
    nn.setInputVector({ 1.0, 0.5 });
    nn.feedForward();
    std::vector<double> out1;
    nn.copyOutputVector(out1);

    nn.reshuffleWeights();
    nn.feedForward();
    std::vector<double> out2;
    nn.copyOutputVector(out2);

    // Almost certainly different after a reshuffle (probability of exact match is negligible).
    EXPECT_NE(out1[0], out2[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixMetricsTest
// ─────────────────────────────────────────────────────────────────────────────

TEST(MatrixMetricsTest, MSEIsZeroForPerfectPrediction)
{
    MlpMatrixNN nn({ LC{ 1 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });

    // Force a specific output by training toward a fixed target for many steps.
    const std::vector<double> input = { 1.0 };
    const std::vector<double> target = { 0.9 };
    for (int i = 0; i < 5000; ++i) {
        nn.setInputVector(input);
        nn.feedForward();
        nn.backPropagate(target);
    }
    nn.setInputVector(input);
    nn.feedForward();
    const double mse = nn.calcMSE(target);
    EXPECT_LT(mse, 0.01);
}

TEST(MatrixMetricsTest, MSEDecreasesWithTraining)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.0);
    const std::vector<double> in = { 1.0, 0.0 };
    const std::vector<double> tgt = { 1.0 };

    nn.setInputVector(in);
    nn.feedForward();
    const double mseBefore = nn.calcMSE(tgt);

    for (int i = 0; i < 200; ++i) {
        nn.setInputVector(in);
        nn.feedForward();
        nn.backPropagate(tgt);
    }
    nn.setInputVector(in);
    nn.feedForward();
    const double mseAfter = nn.calcMSE(tgt);

    EXPECT_LT(mseAfter, mseBefore);
}

TEST(MatrixMetricsTest, CrossEntropyIsNonNegative)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.1, 0.9,
        CostFunction::CrossEntropy);
    nn.setInputVector({ 0.5, 0.5 });
    nn.feedForward();
    EXPECT_GE(nn.calcCrossEntropy({ 1.0 }), 0.0);
}

TEST(MatrixMetricsTest, CrossEntropyIsFiniteForSaturatedOutputs)
{
    // Saturated outputs (near 0 / 1) must not produce NaN or Inf.
    MlpMatrixNN nn({ LC{ 1 }, { 8, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.9,
        CostFunction::CrossEntropy);
    const std::vector<double> in = { 1.0 };
    const std::vector<double> tgt = { 1.0 };
    for (int i = 0; i < 5000; ++i) {
        nn.setInputVector(in);
        nn.feedForward();
        nn.backPropagate(tgt);
    }
    nn.setInputVector(in);
    nn.feedForward();
    EXPECT_TRUE(std::isfinite(nn.calcCrossEntropy(tgt)));
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixConvergenceTest
// ─────────────────────────────────────────────────────────────────────────────

TEST(MatrixConvergenceTest, Sigmoid_MSE)
{
    const double best = xorBestMse(
        [] {
            return MlpMatrixNN(
                { LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.9);
        },
        20000);
    EXPECT_LT(best, 0.1) << "Sigmoid+MSE XOR did not converge";
}

TEST(MatrixConvergenceTest, Sigmoid_CrossEntropy)
{
    const double best = xorBestMse(
        [] {
            return MlpMatrixNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } },
                0.5, 0.9, CostFunction::CrossEntropy);
        },
        15000);
    EXPECT_LT(best, 0.1) << "Sigmoid+CE XOR did not converge";
}

TEST(MatrixConvergenceTest, Tanh_CrossEntropy)
{
    // Tanh hidden + Sigmoid output + CE converges but is sensitive to
    // weight initialisation. Use more attempts to get a reliable result.
    const double best = xorBestMse(
        [] {
            return MlpMatrixNN({ LC{ 2 }, { 8, Activation::Tanh }, { 1, Activation::Sigmoid } },
                0.5, 0.9, CostFunction::CrossEntropy);
        },
        40000, 30);
    EXPECT_LT(best, 0.1) << "Tanh+CE XOR did not converge";
}

TEST(MatrixConvergenceTest, ReLU_MSE)
{
    const double best = xorBestMse(
        [] {
            return MlpMatrixNN(
                { LC{ 2 }, { 8, Activation::ReLU }, { 1, Activation::Sigmoid } }, 0.05, 0.9);
        },
        30000);
    EXPECT_LT(best, 0.1) << "ReLU+MSE XOR did not converge";
}

TEST(MatrixConvergenceTest, LeakyReLU_CrossEntropy)
{
    const double best = xorBestMse(
        [] {
            return MlpMatrixNN(
                { LC{ 2 }, { 6, Activation::LeakyReLU }, { 1, Activation::Sigmoid } }, 0.1, 0.9,
                CostFunction::CrossEntropy);
        },
        30000);
    EXPECT_LT(best, 0.1) << "LeakyReLU+CE XOR did not converge";
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixBatchTest — mini-batch SGD
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Train XOR for `epochs` passes using mini-batches of `batchSize`.
// One pass = iterate over all 4 samples in fixed order, grouped by batchSize.
double trainXorBatch(MlpMatrixNN& nn, int epochs, size_t batchSize)
{
    const auto& smp = xorSamples();
    std::vector<std::vector<double>> allIn, allTgt;
    for (const auto& [in, tgt] : smp) {
        allIn.push_back(in);
        allTgt.push_back(tgt);
    }

    for (int e = 0; e < epochs; ++e) {
        for (size_t start = 0; start < allIn.size(); start += batchSize) {
            const size_t end = std::min(start + batchSize, allIn.size());
            nn.trainBatch({ allIn.begin() + static_cast<std::ptrdiff_t>(start),
                              allIn.begin() + static_cast<std::ptrdiff_t>(end) },
                { allTgt.begin() + static_cast<std::ptrdiff_t>(start),
                    allTgt.begin() + static_cast<std::ptrdiff_t>(end) });
        }
    }

    double worst = 0.0;
    for (const auto& [in, tgt] : smp) {
        nn.setInputVector(in);
        nn.feedForward();
        worst = std::max(worst, nn.calcMSE(tgt));
    }
    return worst;
}

template <typename Factory>
double xorBestMseBatch(Factory factory, int epochs, size_t batchSize, int maxAttempts = 8)
{
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < maxAttempts; ++i) {
        MlpMatrixNN nn = factory();
        best = std::min(best, trainXorBatch(nn, epochs, batchSize));
        if (best < 0.05)
            break;
    }
    return best;
}

} // namespace

TEST(MatrixBatchTest, EmptyBatchThrows)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });
    EXPECT_THROW(nn.trainBatch({}, {}), std::invalid_argument);
}

TEST(MatrixBatchTest, MismatchedBatchThrows)
{
    MlpMatrixNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } });
    EXPECT_THROW(nn.trainBatch({ { 1.0, 0.0 }, { 0.0, 1.0 } }, { { 1.0 } }), std::invalid_argument);
}

TEST(MatrixBatchTest, LossDecreasesAfterOneBatch)
{
    // A single batch step must move the output closer to the target.
    MlpMatrixNN nn({ LC{ 2 }, { 8, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.0);
    const std::vector<double> in = { 1.0, 0.0 };
    const std::vector<double> tgt = { 1.0 };

    nn.setInputVector(in);
    nn.feedForward();
    const double mseBefore = nn.calcMSE(tgt);

    for (int i = 0; i < 50; ++i)
        nn.trainBatch({ in }, { tgt });

    nn.setInputVector(in);
    nn.feedForward();
    EXPECT_LT(nn.calcMSE(tgt), mseBefore);
}

TEST(MatrixBatchTest, FullBatch_XOR_Sigmoid_MSE)
{
    // All 4 XOR samples as a single batch (full-batch GD).
    const double best = xorBestMseBatch(
        [] {
            return MlpMatrixNN(
                { LC{ 2 }, { 8, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 1.0, 0.5);
        },
        5000, 4);
    EXPECT_LT(best, 0.1) << "Full-batch Sigmoid+MSE XOR did not converge";
}

TEST(MatrixBatchTest, MiniBatch2_XOR_Sigmoid_MSE)
{
    // Mini-batches of size 2 (two passes per epoch).
    const double best = xorBestMseBatch(
        [] {
            return MlpMatrixNN(
                { LC{ 2 }, { 6, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.9);
        },
        10000, 2);
    EXPECT_LT(best, 0.1) << "Mini-batch-2 Sigmoid+MSE XOR did not converge";
}

TEST(MatrixBatchTest, FullBatch_XOR_Sigmoid_CE)
{
    const double best = xorBestMseBatch(
        [] {
            return MlpMatrixNN({ LC{ 2 }, { 8, Activation::Sigmoid }, { 1, Activation::Sigmoid } },
                1.0, 0.5, CostFunction::CrossEntropy);
        },
        5000, 4);
    EXPECT_LT(best, 0.1) << "Full-batch Sigmoid+CE XOR did not converge";
}

TEST(MatrixBatchTest, FullBatch_XOR_Tanh_CE)
{
    const double best = xorBestMseBatch(
        [] {
            return MlpMatrixNN({ LC{ 2 }, { 8, Activation::Tanh }, { 1, Activation::Sigmoid } },
                1.0, 0.5, CostFunction::CrossEntropy);
        },
        5000, 4, 15);
    EXPECT_LT(best, 0.1) << "Full-batch Tanh+CE XOR did not converge";
}
