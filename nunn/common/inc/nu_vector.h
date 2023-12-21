//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include <cmath>
#include <functional>
#include <ostream>
#include <ranges>
#include <span>
#include <sstream>
#include <vector>

template <typename T>
std::stringstream& operator>>(std::stringstream& ss,
    std::vector<T>& v) noexcept
{
    size_t size { 0 };

    ss >> size;
    v.resize(size);

    for (size_t i = 0; i < size; ++i) {
        ss >> v[i];
    }

    return ss;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) noexcept
{
    os << "[ ";

    for (auto i = v.cbegin(); i != v.cend(); ++i) {
        os << *i << " ";
    }

    os << " ]";

    return os;
}

template <typename T>
std::stringstream& operator<<(std::stringstream& ss,
    const std::vector<T>& v) noexcept
{
    ss << v.size() << std::endl;

    for (auto i = v.cbegin(); i != v.cend(); ++i) {
        ss << *i << std::endl;
    }

    return ss;
}

namespace nu {

template <typename T>
void toJson(std::ostream& os, const std::vector<T>& v) noexcept
{
    os << "[";

    for (auto it = v.cbegin(); it != v.cend(); ++it) {
        os << *it;
        if ((it + 1) != v.cend()) {
            os << ",";
        }
    }

    os << "]";
}

//! This class wraps a std::vector of double to basically make it capable to perform
//! math operations used by learning algorithms.
class Vector {
public:
    using VectorData = std::vector<double>;
    using iterator = typename VectorData::iterator;
    using const_iterator = typename VectorData::const_iterator;

    enum class Exception {
        size_mismatch
    };

    //! Construct an empty vector, with no elements.
    Vector() = default;

    //! Copy constructor
    Vector(const Vector&) = default;

    //! Constructor to use std::span for safer array handling
    Vector(const std::span<double> v) noexcept
        : _vectorData(v.begin(), v.end())
    {
    }

    //! Construct a vector coping elements of a given c-style vector
    Vector(const double* v, size_t v_len) noexcept
        : _vectorData(v_len)
    {
        memcpy(_vectorData.data(), v, v_len);
    }

    //! Initializer list constructor
    Vector(const std::initializer_list<double> l) noexcept
        : _vectorData(l)
    {
    }

    //! Move constructor
    Vector(Vector&& other) = default;

    //! Fill constructor
    Vector(const size_t& size, double v = 0.0) noexcept
        : _vectorData(size, v)
    {
    }

    //! std::vector constructor
    Vector(const VectorData& v) noexcept
        : _vectorData(v)
    {
    }

    //! Copy assignment operator
    Vector& operator=(const Vector&) = default;

    //! Fill assignment operator
    Vector& operator=(const double& value) noexcept;

    //! Move assignment operator
    Vector& operator=(Vector&& other) = default;

    //! Return size
    size_t size() const noexcept { return _vectorData.size(); }

    //! Return whether the vector is empty
    bool empty() const noexcept { return _vectorData.empty(); }

    //! Change size
    void resize(const size_t& size, double v = 0.0) noexcept { _vectorData.resize(size, v); }

    //! Return iterator to beginning
    iterator begin() noexcept { return _vectorData.begin(); }

    //! Return const_iterator to beginning
    const_iterator cbegin() const noexcept { return _vectorData.cbegin(); }

    //! Return iterator to end
    iterator end() noexcept { return _vectorData.end(); }

    //! Return const_iterator to end
    const_iterator cend() const noexcept { return _vectorData.cend(); }

    //! Access element
    double operator[](size_t idx) const noexcept { return _vectorData[idx]; }

    //! Access element
    double& operator[](size_t idx) noexcept { return _vectorData[idx]; }

    //! Add element at the end
    void push_back(const double& item) { _vectorData.push_back(item); }

    // Using ranges for operations like maxarg
    size_t maxarg() noexcept;

    //! deprecated - see maxarg
    size_t max_item_index() noexcept { return maxarg(); }

    //! Return dot product
    double dot(const Vector& other);

    //! Apply the function f to each vector element
    //! For each element x in vector, x=f(x)
    const Vector& apply(const std::function<double(double)>& f);

    //! Apply the function abs to each vector item
    const Vector& abs() noexcept
    {
        return apply([](double value) { return ::fabs(value); });
    }

    //! Apply the function std::log to each vector item
    const Vector& log() noexcept
    {
        return apply([](double x) { return std::log(x); });
    }

    //! Negates each vector item
    const Vector& negate() noexcept
    {
        return apply([](double x) { return -x; });
    }

    //! Returns the sum of all vector items
    double sum() const noexcept;

    //! Returns the sum of all vector items divided by size()
    double mean() const noexcept { return empty() ? 0 : sum() / double(size()); }

    //! Relational operator ==
    bool operator==(const Vector& other) const noexcept
    {
        return (this == &other) || _vectorData == other._vectorData;
    }

    //! Relational operator !=
    bool operator!=(const Vector& other) const noexcept
    {
        return (this != &other) && _vectorData != other._vectorData;
    }

    //! Relational operator <
    bool operator<(const Vector& other) const noexcept { return _vectorData < other._vectorData; }

    //! Relational operator <=
    bool operator<=(const Vector& other) const noexcept
    {
        return _vectorData <= other._vectorData;
    }

    //! Relational operator >=
    bool operator>=(const Vector& other) const noexcept
    {
        return _vectorData >= other._vectorData;
    }

    //! Relational operator >
    bool operator>(const Vector& other) const noexcept { return _vectorData > other._vectorData; }


    //! Operator +=
    Vector& operator+=(const Vector& other)
    {
        return _op(other, [](double& d, const double& s) { d += s; });
    }

    //! Operator (hadamard product) *=
    Vector& operator*=(const Vector& other)
    {
        return _op(other, [](double& d, const double& s) { d *= s; });
    }

    //! Operator -=
    Vector& operator-=(const Vector& other)
    {
        return _op(other, [](double& d, const double& s) { d -= s; });
    }

    //! Operator /= (entrywise division)
    Vector& operator/=(const Vector& other)
    {
        return _op(other, [](double& d, const double& s) { d /= s; });
    }

    //! Multiply a scalar s to the vector
    Vector& operator*=(const double& s);

    //! Sum scalar s to each vector element
    Vector& operator+=(const double& s);

    //! Subtract scalar s to each vector element
    Vector& operator-=(const double& s);

    //! Divide each vector element by s
    Vector& operator/=(const double& s);

    //! Writes the vector v status into the give string stream ss
    friend std::stringstream& operator<<(std::stringstream& ss, const Vector& v) noexcept
    {
        ss << v._vectorData;
        return ss;
    }

    //! Copies the vector status from the stream ss into vector v
    friend std::stringstream& operator>>(std::stringstream& ss, Vector& v) noexcept
    {
        ss >> v._vectorData;
        return ss;
    }

    //! Prints out to the os stream vector v
    friend std::ostream& operator<<(std::ostream& os, const Vector& v) noexcept
    {
        os << v._vectorData;
        return os;
    }

    //! Writes the JSON formatted content into os stream
    std::ostream& toJson(std::ostream& os) noexcept;

    //! Binary sum vector operator
    friend Vector operator+(const Vector& v1, const Vector& v2);

    //! Binary sub vector operator
    friend Vector operator-(const Vector& v1, const Vector& v2);

    //! Return the square euclidean norm of vector
    double euclideanNorm2() const noexcept;

    //! Return the euclidean norm of vector
    double euclidean_norm() const noexcept { return std::sqrt(euclideanNorm2()); }

    //! Return a const reference to standard vector
    const std::vector<double>& to_stdvec() const noexcept { return _vectorData; }

    //! Return a reference to standard vector
    std::vector<double>& to_stdvec() noexcept { return _vectorData; }

    // Static method to create a Vector with all elements set to 1.0
    static Vector ones(size_t size);

private:
    std::vector<double> _vectorData;

    Vector& _op(const Vector& other, std::function<void(double&, const double&)> func);
};

}
