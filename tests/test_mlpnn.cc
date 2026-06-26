//
// Unit tests for nu::MlpNN (nu_mlpnn.h / nu_mlpnn.cc).
//

#include "nu_mlpnn.h"
#include "nu_stepf.h"
#include "nu_vector.h"

#include <gtest/gtest.h>

#include <array>
#include <map>
#include <sstream>
#include <vector>

using nu::MlpNN;
using nu::Vector;

namespace {

using XorSet = std::array<std::pair<Vector, Vector>, 4>;

const XorSet& xorSamples()
{
    static const XorSet samples = { {
        { Vector { 0.0, 0.0 }, Vector { 0.0 } },
        { Vector { 0.0, 1.0 }, Vector { 1.0 } },
        { Vector { 1.0, 0.0 }, Vector { 1.0 } },
        { Vector { 1.0, 1.0 }, Vector { 0.0 } },
    } };
    return samples;
}

// Trains an MLP on XOR and returns the worst per-sample MSE after training.
double trainXorWorstMse()
{
    MlpNN nn({ 2, 4, 1 }, 0.4, 0.9);

    for (int epoch = 0; epoch < 40000; ++epoch) {
        for (const auto& [input, target] : xorSamples()) {
            nn.setInputVector(input);
            nn.backPropagate(target);
        }
    }

    double worst = 0.0;
    for (const auto& [input, target] : xorSamples()) {
        nn.setInputVector(input);
        nn.feedForward();
        worst = std::max(worst, nn.calcMSE(target));
    }
    return worst;
}

} // namespace

TEST(MlpNNTest, TopologyTooSmallThrows)
{
    // _build requires at least input + hidden + output (>= 3 layers).
    EXPECT_THROW(MlpNN({ 2, 1 }), MlpNN::SizeMismatchException);
}

TEST(MlpNNTest, InputAndOutputSizeFromTopology)
{
    MlpNN nn({ 3, 5, 2 });
    EXPECT_EQ(nn.getInputSize(), 3u);
    EXPECT_EQ(nn.getOutputSize(), 2u);
}

TEST(MlpNNTest, SetInputVectorSizeMismatchThrows)
{
    MlpNN nn({ 2, 2, 1 });
    EXPECT_THROW(nn.setInputVector(Vector { 1.0, 2.0, 3.0 }),
        MlpNN::SizeMismatchException);
}

TEST(MlpNNTest, FeedForwardOutputsInUnitInterval)
{
    MlpNN nn({ 2, 3, 2 });
    nn.setInputVector(Vector { 0.5, -0.5 });
    nn.feedForward();

    Vector out;
    nn.copyOutputVector(out);
    ASSERT_EQ(out.size(), 2u);
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_GE(out[i], 0.0); // sigmoid range
        EXPECT_LE(out[i], 1.0);
    }
}

TEST(MlpNNTest, LearnsXor)
{
    // Random init may land in a local minimum; retry a few times before failing.
    double worst = 1.0;
    for (int attempt = 0; attempt < 5 && worst > 0.1; ++attempt) {
        worst = trainXorWorstMse();
    }
    EXPECT_LT(worst, 0.1) << "MLP failed to converge on XOR (worst MSE=" << worst << ")";
}

TEST(MlpNNTest, SaveLoadRoundTripPreservesOutput)
{
    MlpNN nn({ 2, 3, 1 }, 0.25, 0.6);
    nn.setInputVector(Vector { 1.0, 0.0 });
    nn.feedForward();
    Vector before;
    nn.copyOutputVector(before);

    std::stringstream ss;
    nn.save(ss);

    MlpNN loaded;
    loaded.load(ss);
    loaded.setInputVector(Vector { 1.0, 0.0 });
    loaded.feedForward();
    Vector after;
    loaded.copyOutputVector(after);

    ASSERT_EQ(before.size(), after.size());
    // Serialization is lossless (full round-trip precision), so the reloaded
    // output must match closely. See PerceptronTest.SaveLoadRoundTripIsFullPrecision.
    EXPECT_NEAR(before[0], after[0], 1e-9);
    EXPECT_DOUBLE_EQ(loaded.getLearningRate(), 0.25);
    EXPECT_DOUBLE_EQ(loaded.getMomentum(), 0.6);
}

TEST(MlpNNTest, LoadRejectsGarbageStream)
{
    std::stringstream ss;
    ss << "not-a-valid-net";
    MlpNN nn;
    EXPECT_THROW(nn.load(ss), MlpNN::InvalidSStreamFormatException);
}
