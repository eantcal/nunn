//
// Unit tests for nu::NNTrainer (nu_trainer.h), exercised through a Perceptron.
#include "nu_perceptron.h"
#include "nu_vector.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

using nu::Perceptron;
using nu::PerceptronTrainer;
using nu::Vector;

namespace {

using TrainingSet = std::vector<std::pair<Vector, double>>;

const TrainingSet& andSet()
{
    static const TrainingSet s = {
        { Vector{ 0.0, 0.0 }, 0.0 },
        { Vector{ 0.0, 1.0 }, 0.0 },
        { Vector{ 1.0, 0.0 }, 0.0 },
        { Vector{ 1.0, 1.0 }, 1.0 },
    };
    return s;
}

// Runs a single training epoch and returns how many samples were actually fed
// to the network (counted via the cost function, called once per trained sample).
int countTrainedSamples(bool withProgressCallback, double p2use)
{
    Perceptron p(2, 0.1);
    PerceptronTrainer trainer(p, /*epochs=*/1, /*minErr=*/-1.0);

    int trained = 0;
    auto cost = [&trained](Perceptron& net, const double& target) {
        ++trained;
        return net.error(target);
    };

    if (withProgressCallback) {
        auto noop = [](Perceptron&, const Vector&, const double&, size_t, size_t, double) {
            return false;
        };
        trainer.runTraining(andSet(), cost, noop, p2use);
    } else {
        trainer.runTraining(andSet(), cost, nullptr, p2use);
    }
    return trained;
}

} // namespace

TEST(TrainerTest, AccessorsReturnConstructionValues)
{
    Perceptron p(2);
    PerceptronTrainer trainer(p, 123, 0.05);
    EXPECT_EQ(trainer.getEpochs(), 123u);
    EXPECT_DOUBLE_EQ(trainer.getMinErr(), 0.05);
}

TEST(TrainerTest, FullEpochTrainsEverySample)
{
    EXPECT_EQ(countTrainedSamples(/*withProgressCallback=*/false, 1.0), 4);
    EXPECT_EQ(countTrainedSamples(/*withProgressCallback=*/true, 1.0), 4);
}

// Regression (bug #6): `sampleIdx` used to be advanced only inside the
// `if (progressCbk)` branch, so without a progress callback the partial-set
// early stop (`p2use < 1.0 && sampleIdx >= end_idx`) never fired and the WHOLE
// set was trained regardless of p2use. Supplying a progress callback must not
// change how many samples are trained.
TEST(TrainerTest, PartialTrainingIndependentOfCallbackPresence)
{
    const int withCbk = countTrainedSamples(/*withProgressCallback=*/true, 0.5);
    const int withoutCbk = countTrainedSamples(/*withProgressCallback=*/false, 0.5);
    EXPECT_EQ(withCbk, withoutCbk);
}
