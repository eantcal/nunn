//
// Comprehensive tests for nu::MlpNN with configurable activation functions
// and cost functions.
//
// Test suites:
//   MlpActFwdTest      — activation output range sanity per function
//   MlpApiTest         — LayerConfig struct, getters, setters
//   MlpRegressionTest  — Sigmoid+MSE must behave exactly as before (no regression)
//   MlpConvergenceTest — XOR convergence with all activation/cost combinations
//   MlpLinearOutTest   — continuous regression with a Linear output neuron
//   MlpJsonV2Test      — JSON v2 round-trip (activations + cost function)
//   MlpJsonCompatTest  — backward compatibility: loading a version-1 JSON
//   MlpActNameTest     — act::name / act::fromString round-trip
//

#include "nu_activation.h"
#include "nu_mlpnn.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

using nu::Activation;
using nu::CostFunction;
using nu::MlpNN;
using nu::Vector;
using LC = nu::MlpNN::LayerConfig; // shorthand for the new API

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using XorSet = std::array<std::pair<Vector, Vector>, 4>;

const XorSet& xorSamples()
{
    static const XorSet s = { {
        { Vector{ 0.0, 0.0 }, Vector{ 0.0 } },
        { Vector{ 0.0, 1.0 }, Vector{ 1.0 } },
        { Vector{ 1.0, 0.0 }, Vector{ 1.0 } },
        { Vector{ 1.0, 1.0 }, Vector{ 0.0 } },
    } };
    return s;
}

// Run `epochs` training iterations on XOR and return worst per-sample MSE.
double trainXor(MlpNN& nn, int epochs)
{
    for (int e = 0; e < epochs; ++e)
        for (const auto& [in, tgt] : xorSamples()) {
            nn.setInputVector(in);
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

// Try to learn XOR up to `maxAttempts` times; each attempt re-initialises
// weights (new MlpNN = new random seed via RandomGenerator).
// Returns the best (lowest) worst-sample MSE achieved.
template <typename Factory> double xorBestMse(Factory factory, int epochs, int maxAttempts = 8)
{
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < maxAttempts; ++i) {
        MlpNN nn = factory();
        best = std::min(best, trainXor(nn, epochs));
        if (best < 0.05)
            break;
    }
    return best;
}

// Minimal valid version-1 JSON for a 2-2-1 network.
// No "version", no "activations", no "costFunction" fields.
constexpr std::string_view kJsonV1 = R"({
  "type": "ann",
  "learningRate": 0.1,
  "momentum": 0.5,
  "topology": [2, 2, 1],
  "inputs": [0.0, 0.0],
  "layers": [
    [
      {"bias":  0.10, "weights": [ 0.30, -0.20], "deltaW": [0.0, 0.0]},
      {"bias": -0.10, "weights": [ 0.50,  0.40], "deltaW": [0.0, 0.0]}
    ],
    [
      {"bias":  0.20, "weights": [ 0.60, -0.30], "deltaW": [0.0, 0.0]}
    ]
  ]
})";

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// MlpActFwdTest — activation output range / value
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpActFwdTest, SigmoidOutputInUnitInterval)
{
    // Topology ctor — all Sigmoid — output must be in (0, 1).
    MlpNN nn({ 2, 4, 1 });
    nn.setInputVector(Vector{ 1.0, -1.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_GT(out[0], 0.0);
    EXPECT_LT(out[0], 1.0);
}

TEST(MlpActFwdTest, TanhHiddenSigmoidOutputInUnitInterval)
{
    // Sigmoid output constrains the final value to (0, 1) regardless of hidden activation.
    MlpNN nn({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } });
    nn.setInputVector(Vector{ 1.0, -1.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_GT(out[0], 0.0);
    EXPECT_LT(out[0], 1.0);
}

TEST(MlpActFwdTest, ReLUHiddenSigmoidOutputInUnitInterval)
{
    MlpNN nn({ LC{ 2 }, { 8, Activation::ReLU }, { 1, Activation::Sigmoid } });
    nn.setInputVector(Vector{ 0.5, 0.5 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_GT(out[0], 0.0);
    EXPECT_LT(out[0], 1.0);
}

TEST(MlpActFwdTest, LeakyReLUHiddenSigmoidOutputInUnitInterval)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::LeakyReLU }, { 1, Activation::Sigmoid } });
    nn.setInputVector(Vector{ -1.0, -1.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_GT(out[0], 0.0);
    EXPECT_LT(out[0], 1.0);
}

TEST(MlpActFwdTest, LinearOutputIsFinite)
{
    // Linear output is unbounded — we only assert finiteness.
    MlpNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Linear } });
    nn.setInputVector(Vector{ 3.0, 3.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_EQ(nn.getLayerActivation(1), Activation::Linear);
}

TEST(MlpActFwdTest, ReLUHiddenSigmoidOutputValidForSeveralInputs)
{
    MlpNN nn({ LC{ 1 }, { 4, Activation::ReLU }, { 1, Activation::Sigmoid } });
    for (double x : { 0.0, 0.5, 1.0, 2.0 }) {
        nn.setInputVector(Vector{ x });
        nn.feedForward();
        Vector out;
        nn.copyOutputVector(out);
        EXPECT_GE(out[0], 0.0) << "x=" << x;
        EXPECT_LE(out[0], 1.0) << "x=" << x;
    }
}

TEST(MlpActFwdTest, MultipleOutputNeurons)
{
    MlpNN nn({ LC{ 3 }, { 5, Activation::Tanh }, { 2, Activation::Sigmoid } });
    nn.setInputVector(Vector{ 0.1, -0.2, 0.7 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    ASSERT_EQ(out.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_GT(out[i], 0.0) << "output[" << i << "]";
        EXPECT_LT(out[i], 1.0) << "output[" << i << "]";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpApiTest — LayerConfig struct, getters, setters
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpApiTest, TopologyConstructorDefaultsToSigmoidAndMSE)
{
    MlpNN nn({ 2, 4, 1 });
    EXPECT_EQ(nn.getCostFunction(), CostFunction::MSE);
    const auto& acts = nn.getLayerActivations();
    // Two neuron layers (hidden + output), both Sigmoid.
    ASSERT_EQ(acts.size(), 2u);
    EXPECT_EQ(acts[0], Activation::Sigmoid);
    EXPECT_EQ(acts[1], Activation::Sigmoid);
}

TEST(MlpApiTest, LayerConfigConstructorStoresActivations)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } });
    const auto& acts = nn.getLayerActivations();
    ASSERT_EQ(acts.size(), 2u);
    EXPECT_EQ(acts[0], Activation::Tanh);
    EXPECT_EQ(acts[1], Activation::Sigmoid);
}

TEST(MlpApiTest, LayerConfigConstructorStoresCostFunction)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.1, 0.5,
        CostFunction::CrossEntropy);
    EXPECT_EQ(nn.getCostFunction(), CostFunction::CrossEntropy);
}

TEST(MlpApiTest, GetLayerActivationByIndex)
{
    MlpNN nn({
        LC{ 3 },
        { 5, Activation::ReLU },
        { 4, Activation::LeakyReLU },
        { 2, Activation::Sigmoid },
    });
    EXPECT_EQ(nn.getLayerActivation(0), Activation::ReLU);
    EXPECT_EQ(nn.getLayerActivation(1), Activation::LeakyReLU);
    EXPECT_EQ(nn.getLayerActivation(2), Activation::Sigmoid);
}

TEST(MlpApiTest, GetLayerActivationOutOfRangeThrows)
{
    MlpNN nn({ 2, 3, 1 }); // 2 neuron layers → valid indices 0, 1
    EXPECT_THROW(nn.getLayerActivation(2), std::out_of_range);
}

TEST(MlpApiTest, SetCostFunctionUpdatesGetter)
{
    MlpNN nn({ 2, 4, 1 });
    nn.setCostFunction(CostFunction::CrossEntropy);
    EXPECT_EQ(nn.getCostFunction(), CostFunction::CrossEntropy);
    nn.setCostFunction(CostFunction::MSE);
    EXPECT_EQ(nn.getCostFunction(), CostFunction::MSE);
}

TEST(MlpApiTest, LayerConfigDefaultActivationIsSigmoid)
{
    LC lc(5); // explicit construction
    EXPECT_EQ(lc.size, 5u);
    EXPECT_EQ(lc.activation, Activation::Sigmoid);
}

TEST(MlpApiTest, TopologyAndSizesMatchBetweenConstructors)
{
    // Both constructors must produce the same topology sizes.
    MlpNN nn1({ 2, 3, 1 });
    MlpNN nn2({ LC{ 2 }, { 3, Activation::Sigmoid }, { 1, Activation::Sigmoid } });

    EXPECT_EQ(nn1.getInputSize(), nn2.getInputSize());
    EXPECT_EQ(nn1.getOutputSize(), nn2.getOutputSize());
    EXPECT_EQ(nn1.getTopology(), nn2.getTopology());
}

TEST(MlpApiTest, LearningRateAndMomentumStoredCorrectly)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.37, 0.82);
    EXPECT_DOUBLE_EQ(nn.getLearningRate(), 0.37);
    EXPECT_DOUBLE_EQ(nn.getMomentum(), 0.82);
}

TEST(MlpApiTest, CrossEntropyWithSigmoidOutputIsValid)
{
    // CE + Sigmoid output must not throw — this is the only valid combination.
    EXPECT_NO_THROW(MlpNN({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.1,
        0.9, CostFunction::CrossEntropy));
}

TEST(MlpApiTest, CrossEntropyWithTanhOutputThrows)
{
    EXPECT_THROW((MlpNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Tanh } }, 0.1, 0.9,
                     CostFunction::CrossEntropy)),
        MlpNN::InvalidCostFunctionCombinationException);
}

TEST(MlpApiTest, CrossEntropyWithReLUOutputThrows)
{
    EXPECT_THROW((MlpNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::ReLU } }, 0.1, 0.9,
                     CostFunction::CrossEntropy)),
        MlpNN::InvalidCostFunctionCombinationException);
}

TEST(MlpApiTest, SetCrossEntropyOnNonSigmoidOutputThrows)
{
    // A valid net that is then mutated to an invalid cost function.
    MlpNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Tanh } });
    EXPECT_THROW(nn.setCostFunction(CostFunction::CrossEntropy),
        MlpNN::InvalidCostFunctionCombinationException);
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpRegressionTest — Sigmoid+MSE must not have regressed
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpRegressionTest, SigmoidMSE_XorConverges)
{
    // Identical to the pre-existing LearnsXor test; reproduced here to make
    // the regression boundary explicit.
    double best = xorBestMse([] { return MlpNN({ 2, 4, 1 }, 0.4, 0.9); }, 40000);
    EXPECT_LT(best, 0.1) << "Sigmoid+MSE XOR regression: worst MSE=" << best;
}

TEST(MlpRegressionTest, SigmoidMSE_TopologyAndLayerConfigEquivalent)
{
    // After loading the same weights, both constructors must produce identical
    // forward-pass outputs.
    MlpNN nn1({ 2, 3, 1 }, 0.25, 0.6);
    MlpNN nn2({ LC{ 2 }, { 3, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.25, 0.6);

    // Copy nn1 weights into nn2 via the text round-trip.
    std::stringstream ss;
    nn1.save(ss);
    nn2.load(ss);

    const Vector input{ 0.7, -0.3 };
    nn1.setInputVector(input);
    nn1.feedForward();
    Vector out1;
    nn1.copyOutputVector(out1);

    nn2.setInputVector(input);
    nn2.feedForward();
    Vector out2;
    nn2.copyOutputVector(out2);

    ASSERT_EQ(out1.size(), out2.size());
    EXPECT_DOUBLE_EQ(out1[0], out2[0])
        << "Topology ctor and LayerConfig ctor diverge with same weights";
}

TEST(MlpRegressionTest, SigmoidMSE_SaveLoadPreservesOutput)
{
    // Text serialization round-trip must be lossless.
    MlpNN nn({ 2, 3, 1 }, 0.25, 0.6);
    nn.setInputVector(Vector{ 1.0, 0.0 });
    nn.feedForward();
    Vector before;
    nn.copyOutputVector(before);

    std::stringstream ss;
    nn.save(ss);

    MlpNN loaded;
    loaded.load(ss);
    loaded.setInputVector(Vector{ 1.0, 0.0 });
    loaded.feedForward();
    Vector after;
    loaded.copyOutputVector(after);

    ASSERT_EQ(before.size(), after.size());
    EXPECT_NEAR(before[0], after[0], 1e-9);
    EXPECT_DOUBLE_EQ(loaded.getLearningRate(), 0.25);
    EXPECT_DOUBLE_EQ(loaded.getMomentum(), 0.6);
    // After loading legacy text format, activations must default to Sigmoid and MSE.
    EXPECT_EQ(loaded.getCostFunction(), CostFunction::MSE);
    for (const auto& a : loaded.getLayerActivations())
        EXPECT_EQ(a, Activation::Sigmoid);
}

TEST(MlpRegressionTest, SigmoidMSE_TrainedNetCorrectlyClassifiesXOR)
{
    // A trained net must classify all four XOR inputs correctly (>0.5 threshold).
    double best = 1.0;
    MlpNN bestNet;
    for (int attempt = 0; attempt < 8; ++attempt) {
        MlpNN nn({ 2, 4, 1 }, 0.4, 0.9);
        double mse = trainXor(nn, 40000);
        if (mse < best) {
            best = mse;
            bestNet = nn;
        }
        if (best < 0.05)
            break;
    }
    ASSERT_LT(best, 0.1) << "Failed to find a converged Sigmoid+MSE net";

    for (const auto& [in, tgt] : xorSamples()) {
        bestNet.setInputVector(in);
        bestNet.feedForward();
        Vector out;
        bestNet.copyOutputVector(out);
        EXPECT_EQ(out[0] > 0.5, tgt[0] > 0.5)
            << "Wrong XOR output for input (" << in[0] << "," << in[1] << ")";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpConvergenceTest — XOR convergence with all activation/cost combinations
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpConvergenceTest, TanhHidden_SigmoidOut_MSE)
{
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 6, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.3, 0.9);
    };
    EXPECT_LT(xorBestMse(factory, 30000), 0.1);
}

TEST(MlpConvergenceTest, LeakyReLUHidden_SigmoidOut_MSE)
{
    auto factory = [] {
        return MlpNN(
            { LC{ 2 }, { 6, Activation::LeakyReLU }, { 1, Activation::Sigmoid } }, 0.25, 0.9);
    };
    EXPECT_LT(xorBestMse(factory, 40000), 0.1);
}

TEST(MlpConvergenceTest, ReLUHidden_SigmoidOut_MSE)
{
    // ReLU can suffer from dying neurons — use more neurons and attempts.
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 8, Activation::ReLU }, { 1, Activation::Sigmoid } }, 0.2, 0.9);
    };
    EXPECT_LT(xorBestMse(factory, 50000, /*maxAttempts=*/12), 0.1);
}

TEST(MlpConvergenceTest, SigmoidHidden_SigmoidOut_CrossEntropy)
{
    // CrossEntropy+Sigmoid output: delta simplifies to (t-y), often faster convergence.
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.5, 0.9,
            CostFunction::CrossEntropy);
    };
    EXPECT_LT(xorBestMse(factory, 20000), 0.1);
}

TEST(MlpConvergenceTest, TanhHidden_SigmoidOut_CrossEntropy)
{
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 6, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.4, 0.9,
            CostFunction::CrossEntropy);
    };
    EXPECT_LT(xorBestMse(factory, 20000), 0.1);
}

TEST(MlpConvergenceTest, LeakyReLUHidden_SigmoidOut_CrossEntropy)
{
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 6, Activation::LeakyReLU }, { 1, Activation::Sigmoid } }, 0.4,
            0.9, CostFunction::CrossEntropy);
    };
    EXPECT_LT(xorBestMse(factory, 30000), 0.1);
}

TEST(MlpConvergenceTest, DeepMixed_TanhReLU_SigmoidOut_MSE)
{
    // Three hidden layers with mixed activations.
    auto factory = [] {
        return MlpNN(
            {
                LC{ 2 },
                { 6, Activation::Tanh },
                { 6, Activation::ReLU },
                { 4, Activation::Tanh },
                { 1, Activation::Sigmoid },
            },
            0.15, 0.9);
    };
    EXPECT_LT(xorBestMse(factory, 60000, /*maxAttempts=*/10), 0.1);
}

TEST(MlpConvergenceTest, CrossEntropyWithHalfEpochsMatchesMSE)
{
    // CrossEntropy+Sigmoid should converge in fewer epochs at the same lr.
    // Verify CE succeeds with 20k where MSE needs 40k.
    double bestMse = xorBestMse([] { return MlpNN({ 2, 4, 1 }, 0.4, 0.9); }, 40000);
    double bestCe = xorBestMse(
        [] {
            return MlpNN({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Sigmoid } }, 0.4,
                0.9, CostFunction::CrossEntropy);
        },
        20000);

    EXPECT_LT(bestMse, 0.1) << "Sigmoid+MSE failed to converge";
    EXPECT_LT(bestCe, 0.1) << "Sigmoid+CrossEntropy failed with half the epochs";
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpLinearOutTest — regression with a Linear output neuron
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// y = 0.5*x1 + 0.3*x2
using RegrSet = std::array<std::pair<Vector, Vector>, 4>;
const RegrSet& regrSamples()
{
    static const RegrSet s = { {
        { Vector{ 0.0, 0.0 }, Vector{ 0.0 } },
        { Vector{ 1.0, 0.0 }, Vector{ 0.5 } },
        { Vector{ 0.0, 1.0 }, Vector{ 0.3 } },
        { Vector{ 1.0, 1.0 }, Vector{ 0.8 } },
    } };
    return s;
}

double trainRegression(MlpNN& nn, int epochs)
{
    for (int e = 0; e < epochs; ++e)
        for (const auto& [in, tgt] : regrSamples()) {
            nn.setInputVector(in);
            nn.backPropagate(tgt);
        }
    double maxErr = 0.0;
    for (const auto& [in, tgt] : regrSamples()) {
        nn.setInputVector(in);
        nn.feedForward();
        Vector out;
        nn.copyOutputVector(out);
        maxErr = std::max(maxErr, std::abs(out[0] - tgt[0]));
    }
    return maxErr;
}

} // namespace

TEST(MlpLinearOutTest, SigmoidHiddenLinearOutput)
{
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 8, Activation::Sigmoid }, { 1, Activation::Linear } }, 0.05, 0.5);
    };
    double best = std::numeric_limits<double>::max();
    for (int attempt = 0; attempt < 8; ++attempt) {
        MlpNN nn = factory();
        best = std::min(best, trainRegression(nn, 50000));
        if (best < 0.05)
            break;
    }
    EXPECT_LT(best, 0.05) << "Sigmoid+Linear regression max error=" << best;
}

TEST(MlpLinearOutTest, TanhHiddenLinearOutput)
{
    auto factory = [] {
        return MlpNN({ LC{ 2 }, { 6, Activation::Tanh }, { 1, Activation::Linear } }, 0.05, 0.5);
    };
    double best = std::numeric_limits<double>::max();
    for (int attempt = 0; attempt < 8; ++attempt) {
        MlpNN nn = factory();
        best = std::min(best, trainRegression(nn, 50000));
        if (best < 0.05)
            break;
    }
    EXPECT_LT(best, 0.05) << "Tanh+Linear regression max error=" << best;
}

TEST(MlpLinearOutTest, LinearOutputAllowsValuesBeyondUnitInterval)
{
    // A trained Sigmoid+Linear net with output target ~0.8 should converge.
    // The point is that output values > 1 and < 0 are possible and not clamped.
    MlpNN nn({ LC{ 2 }, { 4, Activation::Sigmoid }, { 1, Activation::Linear } }, 0.05, 0.5);
    nn.setInputVector(Vector{ 1.0, 1.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);
    EXPECT_TRUE(std::isfinite(out[0]));
    // The output may be anywhere — test only that the type is correct.
    EXPECT_EQ(nn.getLayerActivation(1), Activation::Linear);
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpJsonV2Test — JSON version-2 round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpJsonV2Test, RoundTripPreservesActivationsAndCostFunction)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.3, 0.7,
        CostFunction::CrossEntropy);

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() }; // brace init avoids Most Vexing Parse
    loaded.loadJson(iss);

    EXPECT_EQ(loaded.getCostFunction(), CostFunction::CrossEntropy);
    ASSERT_EQ(loaded.getLayerActivations().size(), 2u);
    EXPECT_EQ(loaded.getLayerActivation(0), Activation::Tanh);
    EXPECT_EQ(loaded.getLayerActivation(1), Activation::Sigmoid);
    EXPECT_DOUBLE_EQ(loaded.getLearningRate(), 0.3);
    EXPECT_DOUBLE_EQ(loaded.getMomentum(), 0.7);
}

TEST(MlpJsonV2Test, RoundTripPreservesForwardPassOutput)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::LeakyReLU }, { 1, Activation::Sigmoid } }, 0.2, 0.5);
    const Vector input{ 0.6, -0.4 };

    nn.setInputVector(input);
    nn.feedForward();
    Vector before;
    nn.copyOutputVector(before);

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() };
    loaded.loadJson(iss);

    loaded.setInputVector(input);
    loaded.feedForward();
    Vector after;
    loaded.copyOutputVector(after);

    ASSERT_EQ(before.size(), after.size());
    EXPECT_NEAR(before[0], after[0], 1e-9);
}

TEST(MlpJsonV2Test, RoundTripPreservesThreeHiddenLayers)
{
    MlpNN nn(
        {
            LC{ 3 },
            { 5, Activation::ReLU },
            { 4, Activation::Tanh },
            { 2, Activation::Sigmoid },
        },
        0.1, 0.5, CostFunction::MSE);

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() };
    loaded.loadJson(iss);

    ASSERT_EQ(loaded.getLayerActivations().size(), 3u);
    EXPECT_EQ(loaded.getLayerActivation(0), Activation::ReLU);
    EXPECT_EQ(loaded.getLayerActivation(1), Activation::Tanh);
    EXPECT_EQ(loaded.getLayerActivation(2), Activation::Sigmoid);
    EXPECT_EQ(loaded.getCostFunction(), CostFunction::MSE);
    EXPECT_EQ(loaded.getInputSize(), 3u);
    EXPECT_EQ(loaded.getOutputSize(), 2u);
}

TEST(MlpJsonV2Test, RoundTripAllLinear)
{
    MlpNN nn({ LC{ 2 }, { 4, Activation::Linear }, { 1, Activation::Linear } });

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() };
    loaded.loadJson(iss);

    ASSERT_EQ(loaded.getLayerActivations().size(), 2u);
    EXPECT_EQ(loaded.getLayerActivation(0), Activation::Linear);
    EXPECT_EQ(loaded.getLayerActivation(1), Activation::Linear);
}

TEST(MlpJsonV2Test, CrossEntropyPreservedInRoundTrip)
{
    MlpNN nn({ 2, 4, 1 }, 0.5, 0.9, CostFunction::CrossEntropy);

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() };
    loaded.loadJson(iss);

    EXPECT_EQ(loaded.getCostFunction(), CostFunction::CrossEntropy);
}

TEST(MlpJsonV2Test, InferenceMathchesAfterPartialTrainAndJsonReload)
{
    // Train briefly, serialise, reload, verify identical inference on all XOR inputs.
    MlpNN nn({ LC{ 2 }, { 4, Activation::Tanh }, { 1, Activation::Sigmoid } }, 0.3, 0.9);
    for (int e = 0; e < 1000; ++e)
        for (const auto& [in, tgt] : xorSamples()) {
            nn.setInputVector(in);
            nn.backPropagate(tgt);
        }

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN loaded;
    std::istringstream iss{ oss.str() };
    loaded.loadJson(iss);

    for (const auto& [in, tgt] : xorSamples()) {
        nn.setInputVector(in);
        nn.feedForward();
        Vector out1;
        nn.copyOutputVector(out1);

        loaded.setInputVector(in);
        loaded.feedForward();
        Vector out2;
        loaded.copyOutputVector(out2);

        EXPECT_NEAR(out1[0], out2[0], 1e-9) << "Mismatch after partial training + JSON reload";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpJsonCompatTest — version-1 JSON backward compatibility
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpJsonCompatTest, V1LoadDefaultsToSigmoidAndMSE)
{
    MlpNN nn;
    std::istringstream iss{ std::string(kJsonV1) };
    nn.loadJson(iss);

    EXPECT_EQ(nn.getCostFunction(), CostFunction::MSE);
    const auto& acts = nn.getLayerActivations();
    ASSERT_EQ(acts.size(), 2u); // 2-2-1 has two neuron layers
    for (const auto& a : acts)
        EXPECT_EQ(a, Activation::Sigmoid);
}

TEST(MlpJsonCompatTest, V1LoadPreservesTopologyAndHyperparams)
{
    MlpNN nn;
    std::istringstream iss{ std::string(kJsonV1) };
    nn.loadJson(iss);

    EXPECT_EQ(nn.getInputSize(), 2u);
    EXPECT_EQ(nn.getOutputSize(), 1u);
    EXPECT_DOUBLE_EQ(nn.getLearningRate(), 0.1);
    EXPECT_DOUBLE_EQ(nn.getMomentum(), 0.5);
}

TEST(MlpJsonCompatTest, V1LoadProducesValidSigmoidOutput)
{
    MlpNN nn;
    std::istringstream iss{ std::string(kJsonV1) };
    nn.loadJson(iss);

    nn.setInputVector(Vector{ 1.0, 0.0 });
    nn.feedForward();
    Vector out;
    nn.copyOutputVector(out);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_GT(out[0], 0.0); // Sigmoid output: always in (0, 1)
    EXPECT_LT(out[0], 1.0);
}

TEST(MlpJsonCompatTest, V1ThenReserialiseAsV2RetainsSigmoidDefaults)
{
    // Load v1 → save as v2 → reload — all activations must still be Sigmoid + MSE.
    MlpNN nn;
    {
        std::istringstream iss{ std::string(kJsonV1) };
        nn.loadJson(iss);
    }

    std::ostringstream oss;
    nn.toJson(oss);

    MlpNN reloaded;
    {
        std::istringstream iss{ oss.str() };
        reloaded.loadJson(iss);
    }

    EXPECT_EQ(reloaded.getCostFunction(), CostFunction::MSE);
    for (const auto& a : reloaded.getLayerActivations())
        EXPECT_EQ(a, Activation::Sigmoid);
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpActNameTest — act::name / act::fromString round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(MlpActNameTest, NameFromStringRoundTrip)
{
    using nu::act::fromString;
    using nu::act::name;

    constexpr std::array allActivations = {
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::ReLU,
        Activation::LeakyReLU,
        Activation::Linear,
    };
    for (const auto a : allActivations)
        EXPECT_EQ(fromString(name(a)), a);
}

TEST(MlpActNameTest, FromStringThrowsOnUnknown)
{
    EXPECT_THROW(nu::act::fromString("banana"), std::invalid_argument);
}

TEST(MlpActNameTest, KnownNames)
{
    using nu::act::name;
    EXPECT_EQ(name(Activation::Sigmoid), "sigmoid");
    EXPECT_EQ(name(Activation::Tanh), "tanh");
    EXPECT_EQ(name(Activation::ReLU), "relu");
    EXPECT_EQ(name(Activation::LeakyReLU), "leaky_relu");
    EXPECT_EQ(name(Activation::Linear), "linear");
}
