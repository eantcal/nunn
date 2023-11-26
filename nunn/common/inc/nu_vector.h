//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//

#pragma once

#include <cmath>
#include <functional>
#include <ostream>
#include <sstream>
#include <vector>
#include <ranges>
#include <span>

namespace nu {

//! This class wraps a std::vector to basically make it capable to perform
//! math operations used by learning algorithms

// Define a concept for number-like data types
template<typename T>
concept Number = std::is_arithmetic_v<T>;

template <Number T = double>
class Vector {
public:
    using DataType = T;
    using VectorData = std::vector<DataType>;

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
    Vector(const std::span<T> v) noexcept : _v(v.begin(), v.end()) {}

    //! Construct a vector coping elements of a given c-style vector
    Vector(const double* v, size_t v_len) noexcept : _v(v_len) {
        memcpy(_v.data(), v, v_len);
    }


    //! Initializer list constructor
    Vector(const std::initializer_list<T> l) noexcept : 
        _v(l) 
    {}


    //! Move constructor
    Vector(Vector&& other) noexcept : 
        _v(std::move(other._v)) 
    {}


    //! Fill constructor
    Vector(const size_t& size, DataType v = 0.0) noexcept : _v(size, v) {}


    //! std::vector constructor
    Vector(const VectorData& v) noexcept : _v(v) {}


    //! Copy assignment operator
    Vector& operator=(const Vector&) = default;


    //! Fill assignment operator
    Vector& operator=(const T& value) noexcept {
        if (!_v.empty())
            std::fill(_v.begin(), _v.end(), value);

        return *this;
    }

    //! Move assignment operator
    Vector& operator=(Vector&& other) noexcept {
        if (this != &other)
            _v = std::move(other._v);

        return *this;
    }


    //! Return size
    size_t size() const noexcept { 
        return _v.size(); 
    }


    //! Return whether the vector is empty
    bool empty() const noexcept { 
        return _v.empty(); 
    }


    //! Change size
    void resize(const size_t& size, DataType v = 0.0) noexcept {
        _v.resize(size, v);
    }


    //! Return iterator to beginning
    iterator begin() noexcept { 
        return _v.begin(); 
    }


    //! Return const_iterator to beginning
    const_iterator cbegin() const noexcept { 
        return _v.cbegin(); 
    }


    //! Return iterator to end
    iterator end() noexcept {
        return _v.end(); 
    }


    //! Return const_iterator to end
    const_iterator cend() const noexcept { 
        return _v.cend(); 
    }


    //! Access element
    DataType operator[](size_t idx) const noexcept { 
        return _v[idx]; 
    }


    //! Access element
    DataType& operator[](size_t idx) noexcept { 
        return _v[idx]; 
    }


    //! Add element at the end
    void push_back(const DataType& item) { 
        _v.push_back(item); 
    }

    // Using ranges for operations like maxarg
    size_t maxarg() noexcept {
        if (empty())
            return size_t(-1);

        return std::distance(_v.begin(), std::ranges::max_element(_v));
    }

    //! deprecated - see maxarg
    size_t max_item_index() noexcept {
        return maxarg();
    }

    //! Return dot product
    DataType dot(const Vector& other) {
        if (other.size() != size())
            throw Exception::size_mismatch;

        DataType sum = 0;
        size_t idx = 0;

        for (auto& i : _v)
            sum += i * other[idx++];

        return sum;
    }

    //! Apply the function f to each vector element
    //! For each element x in vector, x=f(x)
    const Vector& apply(const std::function<T(T)>& f) {
        for (auto& i : _v)
            i = f(i);

        return *this;
    }

    //! Apply the function abs to each vector item
    const Vector& abs() noexcept { 
        return apply(::abs); 
    }

    //! Apply the function std::log to each vector item
    const Vector& log() noexcept {
        return apply(
            [](double x) { return std::log(x); }
        );
    }

    //! Negates each vector item
    const Vector& negate() noexcept {
        return apply(
            [](double x) { return -x; }
        );
    }

    //! Returns the sum of all vector items
    T sum() const noexcept {
        T res = T(0);
        for (auto& i : _v)
            res += i;

        return res;
    }

    //! Relational operator ==
    bool operator==(const Vector& other) const noexcept {
        return (this == &other) || _v == other._v;
    }


    //! Relational operator !=
    bool operator!=(const Vector& other) const noexcept {
        return (this != &other) && _v != other._v;
    }


    //! Relational operator <
    bool operator<(const Vector& other) const noexcept {
        return _v < other._v;
    }


    //! Relational operator <=
    bool operator<=(const Vector& other) const noexcept {
        return _v <= other._v;
    }

    //! Relational operator >=
    bool operator>=(const Vector& other) const noexcept {
        return _v >= other._v;
    }

    //! Relational operator >
    bool operator>(const Vector& other) const noexcept {
        return _v > other._v;
    }

    //! Operator +=
    Vector& operator+=(const Vector& other) {
        return _op(other, [](DataType& d, const DataType& s) { d += s; });
    }

    //! Operator (hadamard product) *=
    Vector& operator*=(const Vector& other) {
        return _op(other, [](DataType& d, const DataType& s) { d *= s; });
    }

    //! Operator -=
    Vector& operator-=(const Vector& other) {
        return _op(other, [](DataType& d, const DataType& s) { d -= s; });
    }

    //! Operator /= (entrywise division)
    Vector& operator/=(const Vector& other) {
        return _op(other, [](DataType& d, const DataType& s) { d /= s; });
    }

    //! Multiply a scalar s to the vector
    Vector& operator*=(const DataType& s) {
        Vector other(size(), s);
        return this->operator*=(other);
    }

    //! Sum scalar s to each vector element
    Vector& operator+=(const DataType& s) {
        Vector other(size(), s);
        return this->operator+=(other);
    }

    //! Subtract scalar s to each vector element
    Vector& operator-=(const DataType& s) {
        Vector other(size(), s);
        return this->operator-=(other);
    }

    //! Divide each vector element by s
    Vector& operator/=(const DataType& s) {
        Vector other(size(), s);
        return this->operator/=(other);
    }

    //! Writes the vector v status into the give string stream ss
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         const Vector& v) noexcept
    {
        ss << v.size() << std::endl;

        for (auto i = v.cbegin(); i != v.cend(); ++i)
            ss << *i << std::endl;

        return ss;
    }

    //! Copies the vector status from the stream ss into vector v
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         Vector& v) noexcept
    {
        size_t size = 0;
        ss >> size;

        v.resize(size);

        for (size_t i = 0; i < size; ++i)
            ss >> v[i];

        return ss;
    }

    //! Prints out to the os stream vector v
    friend std::ostream& operator<<(std::ostream& os,
                                    const Vector& v) noexcept
    {
        os << "[ ";

        for (auto i = v.cbegin(); i != v.cend(); ++i)
            os << *i << " ";

        os << " ]";

        return os;
    }

    //! Writes the JSON formatted content into os stream
    std::ostream& formatJson (std::ostream& os) noexcept {
        os << "[";

        for (auto i = cbegin(); i != cend(); ++i) {
            os << *i;
            if ((i + 1) != cend()) {
                os << ",";
            }
        }

        os << "]";
        return os;
    }

    //! Binary sum vector operator
    friend Vector operator+(const Vector& v1, const Vector& v2) {
        auto vr = v1;
        vr += v2;
        return vr;
    }

    //! Binary sub vector operator
    friend Vector operator-(const Vector& v1, const Vector& v2) {
        auto vr = v1;
        vr -= v2;
        return vr;
    }

    //! Return the square euclidean norm of vector
    DataType euclideanNorm2() const noexcept {
        DataType res = 0.0;

        for (size_t i = 0; i < _v.size(); ++i)
            res += _v[i] * _v[i];

        return res;
    }

    //! Return the euclidean norm of vector
    DataType euclidean_norm() const noexcept {
        return std::sqrt(euclideanNorm2());
    }

    //! Return a const reference to standard vector
    const std::vector<T>& to_stdvec() const noexcept { return _v; }


    //! Return a reference to standard vector
    std::vector<T>& to_stdvec() noexcept { return _v; }

private:
    VectorData _v;

    Vector& _op(const Vector& other,
                  std::function<void(DataType&, const DataType&)> f)
    {
        if (other.size() != size())
            throw Exception::size_mismatch;

        size_t idx = 0;
        for (auto& i : _v)
            f(i, other[idx++]);

        return *this;
    }
};

} // namespace nu