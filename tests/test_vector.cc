//
// Unit tests for nu::Vector (nu_vector.h / nu_vector.cc).
#include "nu_vector.h"

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

using nu::Vector;

TEST(VectorTest, DefaultIsEmpty)
{
    Vector v;
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0u);
}

TEST(VectorTest, FillConstructor)
{
    Vector v(3, 2.5);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_DOUBLE_EQ(v[0], 2.5);
    EXPECT_DOUBLE_EQ(v[2], 2.5);
}

TEST(VectorTest, InitializerListConstructor)
{
    Vector v{ 1.0, 2.0, 3.0 };
    ASSERT_EQ(v.size(), 3u);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

TEST(VectorTest, StdVectorConstructor)
{
    std::vector<double> data{ 4.0, 5.0 };
    Vector v(data);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_DOUBLE_EQ(v[1], 5.0);
}

// Regression (bug #1): the (const double*, size_t) constructor must copy every
// element. It previously used memcpy(dst, src, v_len) -- copying v_len BYTES
// instead of v_len doubles -- which dropped ~7/8 of the data on a 64-bit build.
TEST(VectorTest, PointerConstructorCopiesAllElements)
{
    const double src[4] = { 1.0, 2.0, 3.0, 4.0 };
    Vector v(src, 4);

    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
    EXPECT_DOUBLE_EQ(v[3], 4.0);
}

TEST(VectorTest, FillAssignmentOperator)
{
    Vector v(3, 0.0);
    v = 7.0;
    EXPECT_DOUBLE_EQ(v[0], 7.0);
    EXPECT_DOUBLE_EQ(v[1], 7.0);
    EXPECT_DOUBLE_EQ(v[2], 7.0);
}

TEST(VectorTest, PushBackAndResize)
{
    Vector v;
    v.push_back(1.0);
    v.push_back(2.0);
    EXPECT_EQ(v.size(), 2u);

    v.resize(4, 9.0);
    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[3], 9.0);
}

TEST(VectorTest, DotProduct)
{
    Vector a{ 1.0, 2.0, 3.0 };
    Vector b{ 4.0, 5.0, 6.0 };
    EXPECT_DOUBLE_EQ(a.dot(b), 32.0); // 4 + 10 + 18
}

TEST(VectorTest, DotProductSizeMismatchThrows)
{
    Vector a{ 1.0, 2.0 };
    Vector b{ 1.0 };
    EXPECT_THROW(a.dot(b), Vector::SizeMismatchException);
}

TEST(VectorTest, ElementwiseOperators)
{
    Vector a{ 1.0, 2.0, 3.0 };
    Vector b{ 2.0, 2.0, 2.0 };

    Vector add = a + b;
    EXPECT_DOUBLE_EQ(add[0], 3.0);
    EXPECT_DOUBLE_EQ(add[2], 5.0);

    Vector sub = a - b;
    EXPECT_DOUBLE_EQ(sub[0], -1.0);
    EXPECT_DOUBLE_EQ(sub[2], 1.0);

    Vector h = a;
    h *= b; // Hadamard product
    EXPECT_DOUBLE_EQ(h[0], 2.0);
    EXPECT_DOUBLE_EQ(h[2], 6.0);
}

TEST(VectorTest, ElementwiseSizeMismatchThrows)
{
    Vector a{ 1.0, 2.0, 3.0 };
    Vector b{ 1.0, 2.0 };
    EXPECT_THROW(a += b, Vector::SizeMismatchException);
}

TEST(VectorTest, ScalarOperators)
{
    Vector v{ 1.0, 2.0, 3.0 };
    v += 1.0;
    EXPECT_DOUBLE_EQ(v[0], 2.0);

    v -= 1.0;
    EXPECT_DOUBLE_EQ(v[0], 1.0);

    v *= 2.0;
    EXPECT_DOUBLE_EQ(v[2], 6.0);

    v /= 2.0;
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

TEST(VectorTest, SumAndMean)
{
    Vector v{ 1.0, 2.0, 3.0, 4.0 };
    EXPECT_DOUBLE_EQ(v.sum(), 10.0);
    EXPECT_DOUBLE_EQ(v.mean(), 2.5);
}

TEST(VectorTest, MeanOfEmptyIsZero)
{
    Vector v;
    EXPECT_DOUBLE_EQ(v.mean(), 0.0);
}

TEST(VectorTest, ApplyAbsLogNegate)
{
    Vector v{ -1.0, -2.0, -3.0 };
    v.abs();
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);

    Vector n{ 1.0, 2.0 };
    n.negate();
    EXPECT_DOUBLE_EQ(n[0], -1.0);
    EXPECT_DOUBLE_EQ(n[1], -2.0);
}

TEST(VectorTest, Maxarg)
{
    Vector v{ 1.0, 9.0, 3.0 };
    EXPECT_EQ(v.maxarg(), 1u);
}

TEST(VectorTest, MaxargOfEmptyReturnsNpos)
{
    Vector v;
    EXPECT_EQ(v.maxarg(), size_t(-1));
}

TEST(VectorTest, EuclideanNorm)
{
    Vector v{ 3.0, 4.0 };
    EXPECT_DOUBLE_EQ(v.euclideanNorm2(), 25.0);
    EXPECT_DOUBLE_EQ(v.euclidean_norm(), 5.0);
}

TEST(VectorTest, RelationalOperators)
{
    Vector a{ 1.0, 2.0 };
    Vector b{ 1.0, 2.0 };
    Vector c{ 1.0, 3.0 };

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(c > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(a >= b);
}

TEST(VectorTest, OnesFactory)
{
    Vector v = Vector::ones(3);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[2], 1.0);
}

TEST(VectorTest, ToJson)
{
    Vector v{ 1.0, 2.0, 3.0 };
    std::ostringstream os;
    v.toJson(os);
    EXPECT_EQ(os.str(), "[1,2,3]");
}

TEST(VectorTest, StringStreamRoundTrip)
{
    Vector v{ 1.5, 2.5, 3.5 };
    std::stringstream ss;
    ss << v;

    Vector loaded;
    ss >> loaded;

    EXPECT_TRUE(v == loaded);
}
