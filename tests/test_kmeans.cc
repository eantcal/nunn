//
// Tests for nu::KMeans
//

#include "nu_kmeans.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

using nu::KMeans;

// ── helpers ───────────────────────────────────────────────────────────────────

static std::vector<std::vector<double>> makeGrid(
    const std::vector<double>& xs, const std::vector<double>& ys)
{
    std::vector<std::vector<double>> X;
    for (double x : xs)
        for (double y : ys)
            X.push_back({ x, y });
    return X;
}

// Two well-separated clusters along x-axis
static std::vector<std::vector<double>> twoClusters()
{
    std::vector<std::vector<double>> X;
    for (int i = 0; i < 20; ++i)
        X.push_back({ static_cast<double>(i) * 0.1, 0.0 }); // cluster A: x in [0, 1.9]
    for (int i = 0; i < 20; ++i)
        X.push_back({ 10.0 + static_cast<double>(i) * 0.1, 0.0 }); // cluster B: x in [10, 11.9]
    return X;
}

// ── Construction ──────────────────────────────────────────────────────────────

TEST(KMeansTest, ConstructionStoresParams)
{
    KMeans km(3, 100, 1e-5, 7);
    EXPECT_EQ(km.k(), 3u);
    EXPECT_EQ(km.maxIterations(), 100u);
    EXPECT_DOUBLE_EQ(km.tolerance(), 1e-5);
    EXPECT_FALSE(km.isFitted());
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST(KMeansTest, PredictBeforeFitThrows)
{
    KMeans km(2);
    EXPECT_THROW(km.predict({ 1.0, 2.0 }), KMeans::NotFittedException);
}

TEST(KMeansTest, CentroidsBeforeFitThrows)
{
    KMeans km(2);
    EXPECT_THROW(km.centroids(), KMeans::NotFittedException);
}

TEST(KMeansTest, InertiaBeforeFitThrows)
{
    KMeans km(2);
    EXPECT_THROW(km.inertia({ { 1.0 } }), KMeans::NotFittedException);
}

TEST(KMeansTest, FitEmptyDatasetThrows)
{
    KMeans km(2);
    EXPECT_THROW(km.fit({}), KMeans::SizeMismatchException);
}

TEST(KMeansTest, FitKGreaterThanNThrows)
{
    KMeans km(5);
    EXPECT_THROW(km.fit({ { 1.0 }, { 2.0 } }), KMeans::SizeMismatchException);
}

TEST(KMeansTest, FitInconsistentSizesThrows)
{
    KMeans km(2);
    EXPECT_THROW(km.fit({ { 1.0, 2.0 }, { 3.0 } }), KMeans::SizeMismatchException);
}

TEST(KMeansTest, PredictWrongSizeThrows)
{
    KMeans km(2);
    km.fit(twoClusters());
    EXPECT_THROW(km.predict({ 1.0 }), KMeans::SizeMismatchException);
}

// ── Two-cluster separation ────────────────────────────────────────────────────

TEST(KMeansTest, TwoClustersCorrectlySeparated)
{
    auto X = twoClusters();
    KMeans km(2, 300, 1e-6, 0);
    km.fit(X);
    EXPECT_TRUE(km.isFitted());

    // All first 20 samples should map to the same label
    const size_t labelA = km.predict(X[0]);
    for (size_t i = 1; i < 20; ++i)
        EXPECT_EQ(km.predict(X[i]), labelA);

    // All last 20 samples should map to a different label
    const size_t labelB = km.predict(X[20]);
    EXPECT_NE(labelA, labelB);
    for (size_t i = 21; i < 40; ++i)
        EXPECT_EQ(km.predict(X[i]), labelB);
}

TEST(KMeansTest, TwoClustersInertiaIsLow)
{
    auto X = twoClusters();
    KMeans km(2, 300, 1e-6, 0);
    km.fit(X);
    // Inertia should be << 1000 for two tight clusters
    EXPECT_LT(km.inertia(X), 50.0);
}

// ── Centroids ─────────────────────────────────────────────────────────────────

TEST(KMeansTest, CentroidsHaveCorrectDimensions)
{
    auto X = twoClusters();
    KMeans km(2);
    km.fit(X);
    EXPECT_EQ(km.centroids().size(), 2u);
    for (const auto& c : km.centroids())
        EXPECT_EQ(c.size(), 2u);
}

TEST(KMeansTest, CentroidsAreNearClusterMeans)
{
    auto X = twoClusters();
    KMeans km(2, 300, 1e-8, 0);
    km.fit(X);

    // Cluster means: A ~ (0.95, 0), B ~ (10.95, 0)
    const auto& cents = km.centroids();
    std::vector<double> xs{ cents[0][0], cents[1][0] };
    std::sort(xs.begin(), xs.end());
    EXPECT_NEAR(xs[0], 0.95, 0.1);
    EXPECT_NEAR(xs[1], 10.95, 0.1);
}

// ── k=1 edge case ─────────────────────────────────────────────────────────────

TEST(KMeansTest, K1AllSameLabel)
{
    KMeans km(1);
    km.fit({ { 1.0 }, { 2.0 }, { 3.0 } });
    EXPECT_EQ(km.predict({ 1.0 }), 0u);
    EXPECT_EQ(km.predict({ 99.0 }), 0u);
    EXPECT_EQ(km.centroids().size(), 1u);
}

// ── Grid dataset, k=4 ────────────────────────────────────────────────────────

TEST(KMeansTest, FourCornersConverge)
{
    // 4 tight groups near the corners of [0,10]^2
    std::vector<std::vector<double>> X;
    for (double dx = -0.2; dx <= 0.2; dx += 0.1)
        for (double dy = -0.2; dy <= 0.2; dy += 0.1) {
            X.push_back({ 0.0 + dx, 0.0 + dy });
            X.push_back({ 10.0 + dx, 0.0 + dy });
            X.push_back({ 0.0 + dx, 10.0 + dy });
            X.push_back({ 10.0 + dx, 10.0 + dy });
        }
    KMeans km(4, 300, 1e-8, 42);
    km.fit(X);

    // All 4 labels must be used
    auto labels = km.predict(X);
    std::set<size_t> used(labels.begin(), labels.end());
    EXPECT_EQ(used.size(), 4u);
}

// ── Batch predict consistency ─────────────────────────────────────────────────

TEST(KMeansTest, BatchPredictMatchesSingle)
{
    auto X = twoClusters();
    KMeans km(2, 300, 1e-8, 0);
    km.fit(X);

    const auto batch = km.predict(X);
    ASSERT_EQ(batch.size(), X.size());
    for (size_t i = 0; i < X.size(); ++i)
        EXPECT_EQ(batch[i], km.predict(X[i]));
}

// ── numIterations ─────────────────────────────────────────────────────────────

TEST(KMeansTest, NumIterationsIsPositiveAfterFit)
{
    KMeans km(2);
    km.fit(twoClusters());
    EXPECT_GT(km.numIterations(), 0u);
}

TEST(KMeansTest, NumIterationsBoundedByMaxIter)
{
    KMeans km(2, 5);
    km.fit(twoClusters());
    EXPECT_LE(km.numIterations(), 5u);
}
