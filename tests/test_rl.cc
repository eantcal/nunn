//
// Unit tests for the reinforcement-learning components:
//   - nu::QLearn   (nu_qlearn.h)
//   - nu::Sarsa    (nu_sarsa.h)
//   - nu::EGreedyPolicy / nu::SoftmaxPolicy (policies)
//

#include "nu_e_greedy_policy.h"
#include "nu_learner_listener.h"
#include "nu_qlearn.h"
#include "nu_sarsa.h"
#include "nu_softmax_policy.h"

#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// A tiny deterministic agent walking a line 0 -> goalPos.
// Every action advances the position by one, so an episode always terminates
// in exactly `goalPos` steps regardless of the policy's choices.
// ---------------------------------------------------------------------------
struct LineAgent {
    int pos = 0;
    int goalPos = 3;

    bool goal() const { return pos >= goalPos; }
    int getCurrentState() const { return pos; }
    void doAction(int /*action*/) { ++pos; }
    double reward() const { return goal() ? 1.0 : 0.0; }
    std::vector<int> getValidActions() const { return { 0, 1 }; }
};

// A fixed-state agent, useful to probe policy selection in isolation.
struct FixedAgent {
    int state = 0;
    std::vector<int> actions{ 0, 1, 2 };

    bool goal() const { return false; }
    int getCurrentState() const { return state; }
    void doAction(int) { }
    double reward() const { return 0.0; }
    std::vector<int> getValidActions() const { return actions; }
};

// Records the sequence of move counters reported to the learner.
struct RecordingListener : nu::LearnerListener {
    std::vector<size_t> moves;
    bool notify(const double& /*reward*/, const size_t& move) override
    {
        moves.push_back(move);
        return true; // keep going
    }
};

using EGreedy = nu::EGreedyPolicy<int, LineAgent>;
using EGreedyFixed = nu::EGreedyPolicy<int, FixedAgent>;
using SoftmaxFixed = nu::SoftmaxPolicy<int, FixedAgent>;
using QMap = std::unordered_map<int, std::unordered_map<int, double>>;

} // namespace

// --------------------------------- Policies --------------------------------

TEST(PolicyTest, EGreedyExploitsBestKnownAction)
{
    FixedAgent agent;
    QMap q;
    q[0][0] = 0.0;
    q[0][1] = 5.0; // clearly the best
    q[0][2] = 1.0;

    EGreedyFixed policy;
    EXPECT_EQ(policy.getLearnedAction(agent, q), 1);
}

TEST(PolicyTest, EGreedyAlwaysReturnsValidAction)
{
    FixedAgent agent;
    QMap q; // all zeros -> falls back to random selection
    EGreedyFixed policy;

    for (int i = 0; i < 50; ++i) {
        const int a = policy.getLearnedAction(agent, q);
        EXPECT_GE(a, 0);
        EXPECT_LE(a, 2);
    }
}

TEST(PolicyTest, SoftmaxExploitsBestKnownAction)
{
    FixedAgent agent;
    QMap q;
    q[0][0] = 0.0;
    q[0][1] = 1.0;
    q[0][2] = 9.0; // clearly the best
    SoftmaxFixed policy;
    EXPECT_EQ(policy.getLearnedAction(agent, q), 2);
}

// ---------------------------------- QLearn ---------------------------------

TEST(QLearnTest, LearnTerminatesAndAccumulatesReward)
{
    nu::QLearn<int, int, LineAgent, EGreedy> q;
    LineAgent agent;
    const double total = q.learn(agent);

    EXPECT_TRUE(agent.goal());
    EXPECT_GT(total, 0.0); // reached the goal at least once -> reward > 0
}

TEST(QLearnTest, ReportsConsecutiveMoveCounts)
{
    auto listener = std::make_shared<RecordingListener>();
    nu::QLearn<int, int, LineAgent, EGreedy> q(listener);
    LineAgent agent;
    q.learn(agent);

    // 3 steps to the goal, counter reported once per step: 0, 1, 2.
    const std::vector<size_t> expected{ 0, 1, 2 };
    EXPECT_EQ(listener->moves, expected);
}

// ----------------------------------- Sarsa ---------------------------------

TEST(SarsaTest, LearnTerminates)
{
    nu::Sarsa<int, int, LineAgent, EGreedy> sarsa;
    LineAgent agent;
    sarsa.learn(agent);
    EXPECT_TRUE(agent.goal());
}

// Regression (bug #4): Sarsa::learn() used to increment moveCnt twice per
// iteration (once in `notify(reward, moveCnt++)` and again with `++moveCnt`),
// so the move counter reported to the listener jumped 0, 2, 4, ... instead of
// 0, 1, 2, ... -- matching QLearn (the sibling algorithm).
TEST(SarsaTest, ReportsConsecutiveMoveCounts)
{
    auto listener = std::make_shared<RecordingListener>();
    nu::Sarsa<int, int, LineAgent, EGreedy> sarsa(listener);
    LineAgent agent;
    sarsa.learn(agent);

    const std::vector<size_t> expected{ 0, 1, 2 };
    EXPECT_EQ(listener->moves, expected);
}
