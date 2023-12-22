//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_vector.h"

namespace nu {

Vector& Vector::operator=(const double& value) noexcept
{
    if (!_vectorData.empty()) {
        std::fill(_vectorData.begin(), _vectorData.end(), value);
    }

    return *this;
}

size_t Vector::maxarg() noexcept
{
    if (empty()) {
        return size_t(-1);
    }

    return std::ranges::distance(_vectorData.begin(), std::ranges::max_element(_vectorData));
}

double Vector::dot(const Vector& other)
{
    if (other.size() != size()) {
        throw SizeMismatchException();
    }

    double sum = 0;
    size_t idx = 0;

    for (auto& data : _vectorData) {
        sum += data * other[idx++];
    }

    return sum;
}

const Vector& Vector::apply(const std::function<double(double)>& f)
{
    for (auto& data : _vectorData) {
        data = f(data);
    }

    return *this;
}

double Vector::sum() const noexcept
{
    double res { .0 };
    for (auto& data : _vectorData) {
        res += data;
    }

    return res;
}

Vector& Vector::operator*=(const double& s)
{
    Vector other(size(), s);
    return this->operator*=(other);
}

Vector& Vector::operator+=(const double& s)
{
    Vector other(size(), s);
    return this->operator+=(other);
}

Vector& Vector::operator-=(const double& s)
{
    Vector other(size(), s);
    return this->operator-=(other);
}

Vector& Vector::operator/=(const double& s)
{
    Vector other(size(), s);
    return this->operator/=(other);
}

std::ostream& Vector::toJson(std::ostream& os) noexcept
{
    nu::toJson(os, _vectorData);
    return os;
}

Vector operator+(const Vector& v1, const Vector& v2)
{
    auto vr = v1;
    vr += v2;
    return vr;
}

Vector operator-(const Vector& v1, const Vector& v2)
{
    auto vr = v1;
    vr -= v2;
    return vr;
}

double Vector::euclideanNorm2() const noexcept
{
    double res { .0 };

    for (size_t i = 0; i < _vectorData.size(); ++i) {
        res += _vectorData[i] * _vectorData[i];
    }

    return res;
}

Vector Vector::ones(size_t size)
{
    Vector vec(size);
    std::fill(vec._vectorData.begin(), vec._vectorData.end(), 1.0);
    return vec;
}

Vector& Vector::_op(const Vector& other,
    std::function<void(double&, const double&)> func)
{
    if (other.size() != size()) {
        throw SizeMismatchException();
    }

    for (size_t idx = 0; auto& data : _vectorData) {
        func(data, other[idx++]);
    }

    return *this;
}

} // namespace nu
