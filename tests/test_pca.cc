//
// Tests for nu::Pca
//

#include "nu_pca.h"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using nu::Pca;

// ── helpers ───────────────────────────────────────────────────────────────────

// Build a dataset where variation is entirely along one axis:
// X[i] = {a * i, 0, 0, ...}  — first component explains 100 % of variance.
static std::vector<std::vector<double>> oneDirData(size_t N, size_t D)
{
    std::vector<std::vector<double>> X(N, std::vector<double>(D, 0.0));
    for (size_t i = 0; i < N; ++i)
        X[i][0] = static_cast<double>(i);
    return X;
}

// ── Construction ──────────────────────────────────────────────────────────────

TEST(PcaTest, ConstructionStoresNComponents)
{
    Pca pca(3);
    EXPECT_EQ(pca.nComponents(), 3u);
    EXPECT_FALSE(pca.isFitted());
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST(PcaTest, TransformBeforeFitThrows)
{
    Pca pca(2);
    EXPECT_THROW(pca.transform({ 1.0, 2.0 }), Pca::NotFittedException);
}

TEST(PcaTest, InverseTransformBeforeFitThrows)
{
    Pca pca(2);
    EXPECT_THROW(pca.inverseTransform({ 0.0, 0.0 }), Pca::NotFittedException);
}

TEST(PcaTest, ExplainedVarianceRatioBeforeFitThrows)
{
    Pca pca(2);
    EXPECT_THROW(pca.explainedVarianceRatio(), Pca::NotFittedException);
}

TEST(PcaTest, FitEmptyDatasetThrows)
{
    Pca pca(1);
    EXPECT_THROW(pca.fit({}), Pca::SizeMismatchException);
}

TEST(PcaTest, FitInconsistentSizesThrows)
{
    Pca pca(1);
    EXPECT_THROW(pca.fit({ { 1.0, 2.0 }, { 3.0 } }), Pca::SizeMismatchException);
}

TEST(PcaTest, FitNComponentsTooLargeThrows)
{
    Pca pca(5);
    // min(N=3, D=2) = 2 < 5
    EXPECT_THROW(pca.fit({ { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } }), Pca::SizeMismatchException);
}

TEST(PcaTest, TransformWrongSizeThrows)
{
    Pca pca(1);
    pca.fit(oneDirData(10, 3));
    EXPECT_THROW(pca.transform({ 0.0, 0.0 }), Pca::SizeMismatchException);
}

TEST(PcaTest, InverseTransformWrongSizeThrows)
{
    Pca pca(2);
    pca.fit(oneDirData(10, 4));
    EXPECT_THROW(pca.inverseTransform({ 0.0 }), Pca::SizeMismatchException);
}

// ── fit correctness ───────────────────────────────────────────────────────────

TEST(PcaTest, FitSetsDimensionAndFlag)
{
    Pca pca(1);
    pca.fit(oneDirData(10, 3));
    EXPECT_TRUE(pca.isFitted());
    EXPECT_EQ(pca.inputDim(), 3u);
    EXPECT_EQ(pca.nComponents(), 1u);
}

TEST(PcaTest, OneDirDataFirstComponentExplainsAlmostAll)
{
    // All variance is in the first dimension → PC1 should explain ~100 %
    const size_t N = 20, D = 4;
    Pca pca(1);
    pca.fit(oneDirData(N, D));

    const auto& evr = pca.explainedVarianceRatio();
    ASSERT_EQ(evr.size(), 1u);
    EXPECT_NEAR(evr[0], 1.0, 1e-6);
    EXPECT_NEAR(pca.totalExplainedVariance(), 1.0, 1e-6);
}

TEST(PcaTest, ExplainedVarianceRatiosSumToTotalExplained)
{
    Pca pca(2);
    pca.fit(oneDirData(15, 4));
    const auto& evr = pca.explainedVarianceRatio();
    double sum = 0.0;
    for (double v : evr)
        sum += v;
    EXPECT_NEAR(sum, pca.totalExplainedVariance(), 1e-12);
}

TEST(PcaTest, ExplainedVarianceRatiosAreNonNegative)
{
    Pca pca(3);
    pca.fit(oneDirData(20, 5));
    for (double v : pca.explainedVarianceRatio())
        EXPECT_GE(v, 0.0);
}

// ── transform / inverseTransform ─────────────────────────────────────────────

TEST(PcaTest, TransformOutputHasCorrectSize)
{
    Pca pca(2);
    pca.fit(oneDirData(10, 5));
    const auto z = pca.transform(std::vector<double>(5, 1.0));
    EXPECT_EQ(z.size(), 2u);
}

TEST(PcaTest, BatchTransformMatchesSingle)
{
    auto X = oneDirData(10, 4);
    Pca pca(2);
    pca.fit(X);

    const auto batch = pca.transform(X);
    ASSERT_EQ(batch.size(), X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        const auto single = pca.transform(X[i]);
        ASSERT_EQ(batch[i].size(), single.size());
        for (size_t j = 0; j < single.size(); ++j)
            EXPECT_NEAR(batch[i][j], single[j], 1e-12);
    }
}

TEST(PcaTest, InverseTransformOutputHasCorrectSize)
{
    Pca pca(2);
    pca.fit(oneDirData(10, 5));
    const auto xhat = pca.inverseTransform({ 1.0, 0.5 });
    EXPECT_EQ(xhat.size(), 5u);
}

TEST(PcaTest, OneDirPerfectReconstructionWithOneComponent)
{
    // With one direction of variation and nComp=1, reconstruction is exact
    const size_t N = 10, D = 3;
    auto X = oneDirData(N, D);

    Pca pca(1);
    pca.fit(X);

    for (size_t i = 0; i < N; ++i) {
        const auto z = pca.transform(X[i]);
        const auto xhat = pca.inverseTransform(z);
        ASSERT_EQ(xhat.size(), D);
        for (size_t d = 0; d < D; ++d)
            EXPECT_NEAR(xhat[d], X[i][d], 1e-9);
    }
}

TEST(PcaTest, ReconstructionErrorDecreasesWithMoreComponents)
{
    // 3D data with variation along all axes — more components = lower error
    std::vector<std::vector<double>> X;
    for (int i = 0; i < 30; ++i)
        X.push_back({ static_cast<double>(i), static_cast<double>(i) * 0.5 + 1.0,
            static_cast<double>(i) * 0.1 - 2.0 });

    double mse1 = 0.0, mse2 = 0.0;
    for (size_t k = 1; k <= 2; ++k) {
        Pca pca(k);
        pca.fit(X);
        double mse = 0.0;
        for (const auto& x : X) {
            auto xhat = pca.inverseTransform(pca.transform(x));
            for (size_t d = 0; d < 3; ++d) {
                double diff = x[d] - xhat[d];
                mse += diff * diff;
            }
        }
        if (k == 1)
            mse1 = mse;
        else
            mse2 = mse;
    }
    EXPECT_LE(mse2, mse1);
}

// ── mean centring ─────────────────────────────────────────────────────────────

TEST(PcaTest, ConstantDataDoesNotThrow)
{
    // All samples equal — totalVar ~ 0, guard should prevent division by zero
    std::vector<std::vector<double>> X(5, { 3.0, 1.0, 4.0 });
    Pca pca(1);
    EXPECT_NO_THROW(pca.fit(X));
    EXPECT_TRUE(pca.isFitted());
}
