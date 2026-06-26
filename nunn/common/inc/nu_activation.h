//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nu {

//! Activation functions supported by the MLP.
enum class Activation {
    Sigmoid,   //!< Logistic: 1/(1+e^-x),  derivative: y*(1-y)
    Tanh,      //!< Hyperbolic tangent,     derivative: 1 - y^2
    ReLU,      //!< Rectified linear unit,  derivative: y>0 ? 1 : 0
    LeakyReLU, //!< Leaky ReLU (alpha=0.01), derivative: y>0 ? 1 : alpha
    Linear,    //!< Identity,               derivative: 1
};

namespace act {

constexpr double LEAKY_RELU_ALPHA = 0.01;

//! Apply activation function forward pass.
inline double forward(Activation a, double x) noexcept
{
    switch (a) {
    case Activation::Sigmoid:    return 1.0 / (1.0 + std::exp(-x));
    case Activation::Tanh:       return std::tanh(x);
    case Activation::ReLU:       return x > 0.0 ? x : 0.0;
    case Activation::LeakyReLU:  return x > 0.0 ? x : LEAKY_RELU_ALPHA * x;
    case Activation::Linear:     return x;
    }
    return x;
}

//! Derivative of the activation function with respect to its output y = f(x).
//! For ReLU/LeakyReLU the sign of y unambiguously determines the derivative.
inline double backward(Activation a, double y) noexcept
{
    switch (a) {
    case Activation::Sigmoid:    return y * (1.0 - y);
    case Activation::Tanh:       return 1.0 - y * y;
    case Activation::ReLU:       return y > 0.0 ? 1.0 : 0.0;
    case Activation::LeakyReLU:  return y > 0.0 ? 1.0 : LEAKY_RELU_ALPHA;
    case Activation::Linear:     return 1.0;
    }
    return 1.0;
}

//! String representation (used for JSON serialization).
inline std::string_view name(Activation a) noexcept
{
    switch (a) {
    case Activation::Sigmoid:    return "sigmoid";
    case Activation::Tanh:       return "tanh";
    case Activation::ReLU:       return "relu";
    case Activation::LeakyReLU:  return "leaky_relu";
    case Activation::Linear:     return "linear";
    }
    return "sigmoid";
}

//! Parse activation from string (throws std::invalid_argument on unknown name).
inline Activation fromString(std::string_view s)
{
    if (s == "sigmoid")    return Activation::Sigmoid;
    if (s == "tanh")       return Activation::Tanh;
    if (s == "relu")       return Activation::ReLU;
    if (s == "leaky_relu") return Activation::LeakyReLU;
    if (s == "linear")     return Activation::Linear;
    throw std::invalid_argument("Unknown activation: " + std::string(s));
}

} // namespace act
} // namespace nu
