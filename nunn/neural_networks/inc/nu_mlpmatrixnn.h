//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Matrix-based MLP using Eigen for layer-level GEMV operations.
// Mirrors the MlpNN public interface; can be used as a drop-in replacement.
// Weights are stored as dense matrices [out × in] per layer, enabling
// efficient SIMD on CPU and a clean path to GPU backends (ArrayFire/OpenCL).
//

#pragma once

#include "nu_activation.h"
#include "nu_costfuncs.h"

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#ifdef NUNN_HAS_ARRAYFIRE
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <arrayfire.h>
#include <optional>
#endif

namespace nu {

class MlpMatrixNN {
public:
    // ── Configuration ─────────────────────────────────────────────────────────

    struct LayerConfig {
        size_t size{ 0 };
        Activation activation{ Activation::Sigmoid };

        LayerConfig() = default;
        explicit LayerConfig(size_t s) noexcept
            : size(s)
        {
        }
        LayerConfig(size_t s, Activation a) noexcept
            : size(s)
            , activation(a)
        {
        }
    };

    // ── Compute backend ────────────────────────────────────────────────────────

    enum class ComputeBackend {
        Eigen, // CPU path via Eigen (default)
        OpenCL, // GPU path via ArrayFire/OpenCL (requires NUNN_HAS_ARRAYFIRE)
    };

    // ── Exceptions ────────────────────────────────────────────────────────────

    class InvalidCostFunctionCombinationException : public std::runtime_error {
    public:
        InvalidCostFunctionCombinationException()
            : std::runtime_error("CrossEntropy requires Sigmoid output activation: "
                                 "output must produce probabilities in (0, 1)")
        {
        }
    };

    // ── Construction ──────────────────────────────────────────────────────────

    // layers[0] is the input descriptor (size only; activation ignored).
    // layers[1..N] are the neuron layers.
    // Throws InvalidCostFunctionCombinationException if CrossEntropy is
    // paired with a non-Sigmoid output activation.
    // Throws std::runtime_error if backend == OpenCL but NUNN_HAS_ARRAYFIRE is
    // not defined at compile time.
    explicit MlpMatrixNN(const std::vector<LayerConfig>& layers, double learningRate = 0.1,
        double momentum = 0.0, CostFunction cf = CostFunction::MSE,
        ComputeBackend backend = ComputeBackend::Eigen);

    // ── Forward / backward — single sample ───────────────────────────────────

    void setInputVector(const std::vector<double>& input);
    void feedForward();
    void backPropagate(const std::vector<double>& target);
    void copyOutputVector(std::vector<double>& out) const;

    // ── Mini-batch SGD ────────────────────────────────────────────────────────

    // Run one mini-batch training step (forward + backward + weight update).
    // inputs[i]  — one sample of size getInputSize()
    // targets[i] — corresponding target of size getOutputSize()
    // Batch must be non-empty and inputs.size() == targets.size().
    // Gradients are averaged over the batch before the weight update.
    // Throws std::invalid_argument on empty or mismatched batch.
    void trainBatch(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets);

    // ── Metrics ───────────────────────────────────────────────────────────────

    [[nodiscard]] double calcMSE(const std::vector<double>& target) const;
    [[nodiscard]] double calcCrossEntropy(const std::vector<double>& target) const;

    // ── Getters ───────────────────────────────────────────────────────────────

    [[nodiscard]] size_t getInputSize() const noexcept { return _inputSize; }
    [[nodiscard]] size_t getOutputSize() const noexcept;
    [[nodiscard]] double getLearningRate() const noexcept { return _lr; }
    [[nodiscard]] double getMomentum() const noexcept { return _momentum; }
    [[nodiscard]] CostFunction getCostFunction() const noexcept { return _cf; }
    [[nodiscard]] ComputeBackend getBackend() const noexcept { return _backend; }

    void reshuffleWeights();

private:
    struct Layer {
        Eigen::MatrixXd W; // [out_size × in_size]  weight matrix
        Eigen::VectorXd b; // [out_size]             bias vector
        Eigen::VectorXd a; // [out_size]             activation output (host mirror)
        Eigen::VectorXd delta; // [out_size]             error signal (Eigen path)
        Eigen::MatrixXd dW; // [out_size × in_size]   momentum accumulator for W
        Eigen::VectorXd db; // [out_size]             momentum accumulator for b
        Activation act;

#ifdef NUNN_HAS_ARRAYFIRE
        // std::optional avoids default-constructing af::array (which would
        // call af_create_handle and require a backend even on the Eigen path).
        std::optional<af::array> W_af;
        std::optional<af::array> b_af;
        std::optional<af::array> a_af;
        std::optional<af::array> delta_af;
        std::optional<af::array> dW_af;
        std::optional<af::array> db_af;
#endif
    };

    std::vector<Layer> _layers;
    Eigen::VectorXd _input;
    size_t _inputSize = 0;
    double _lr = 0.1;
    double _momentum = 0.0;
    CostFunction _cf = CostFunction::MSE;
    ComputeBackend _backend = ComputeBackend::Eigen;

    static void _validateCostFunction(CostFunction cf, Activation outAct)
    {
        if (cf == CostFunction::CrossEntropy && outAct != Activation::Sigmoid)
            throw InvalidCostFunctionCombinationException();
    }
};

} // namespace nu
