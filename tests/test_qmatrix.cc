//
// Unit tests for nu::QMatrix (nu_qmatrix.h / nu_qmatrix.cc).
//

#include "nu_qmatrix.h"

#include <gtest/gtest.h>

using nu::QMatrix;

TEST(QMatrixTest, ConstructedSquareAndZeroFillable)
{
    QMatrix m(3);
    EXPECT_EQ(m.size(), 3u);

    m.fill(0.0);
    for (size_t r = 0; r < m.size(); ++r) {
        for (size_t c = 0; c < m.size(); ++c) {
            EXPECT_DOUBLE_EQ(m[r][c], 0.0);
        }
    }
}

TEST(QMatrixTest, MaxAndMaxargPerRow)
{
    QMatrix m(3);
    m.fill(0.0);
    m[1][0] = 2.0;
    m[1][2] = 7.0;
    m[1][1] = 5.0;

    EXPECT_DOUBLE_EQ(m.max(1), 7.0);
    EXPECT_EQ(m.maxarg(1), 2u);
}

TEST(QMatrixTest, NormalizeScalesGlobalMaxTo100)
{
    QMatrix m(2);
    m.fill(0.0);
    m[0][0] = 10.0;
    m[1][1] = 20.0; // global max

    m.normalize();

    EXPECT_DOUBLE_EQ(m[1][1], 100.0);
    EXPECT_DOUBLE_EQ(m[0][0], 50.0);
}

TEST(QMatrixTest, OutOfRangeIndexThrows)
{
    QMatrix m(2);
    EXPECT_THROW(m[5], QMatrix::InvalidIndexException);

    const QMatrix& cm = m;
    EXPECT_THROW(cm[5], QMatrix::InvalidIndexException);
}

TEST(QMatrixTest, MaxOnInvalidRowThrows)
{
    QMatrix m(2);
    EXPECT_THROW(m.max(99), QMatrix::InvalidIndexException);
}
