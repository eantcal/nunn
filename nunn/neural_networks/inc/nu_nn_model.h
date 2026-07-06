//
// This file is part of the nunn Library
// Copyright (c) 2026 Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// NnModel — polymorphic wrapper for nunn classifier networks.
//
// Provides a common interface over MlpNN and MlpMatrixNN so that
// applications (e.g. ocr_test) can load any supported model type from
// a JSON file and use it without knowing the concrete class.
//
// JSON format for model_type dispatch:
//   "ann"        (or missing)  → MlpNN
//   "mlp_matrix"               → MlpMatrixNN

#pragma once

#include <memory>
#include <ostream>
#include <istream>
#include <string>
#include <vector>

namespace nu {

class NnModel {
public:
    virtual ~NnModel() = default;

    // ── Identity ──────────────────────────────────────────────────────────────

    virtual std::string typeName() const = 0;

    // ── Dimensions ────────────────────────────────────────────────────────────

    virtual size_t getInputSize() const noexcept = 0;
    virtual size_t getOutputSize() const noexcept = 0;

    // [input_size, hidden0, hidden1, ..., output_size]
    virtual std::vector<size_t> getTopology() const = 0;

    virtual double getLearningRate() const noexcept = 0;

    // ── Inference ─────────────────────────────────────────────────────────────

    virtual void setInputVector(const std::vector<double>& v) = 0;
    virtual void feedForward() = 0;
    virtual void copyOutputVector(std::vector<double>& out) const = 0;

    // ── Online training ───────────────────────────────────────────────────────

    // Runs one backprop step. Fills 'output' with the network's output after
    // the forward pass that precedes backpropagation.
    virtual void backPropagate(const std::vector<double>& target, std::vector<double>& output) = 0;

    virtual double calcMSE(const std::vector<double>& target) = 0;

    // ── Persistence ───────────────────────────────────────────────────────────

    virtual std::ostream& toJson(std::ostream& os) const = 0;
    virtual std::istream& loadJson(std::istream& is) = 0;

    // ── Factory ───────────────────────────────────────────────────────────────

    // Reads 'model_type' from the JSON at 'path' and returns the appropriate
    // concrete subtype. Throws std::runtime_error on I/O or parse errors.
    static std::unique_ptr<NnModel> load(const std::string& path);
};

} // namespace nu
