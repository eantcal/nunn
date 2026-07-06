//
// This file is part of the nunn Library
// Copyright (c) 2026 Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_nn_model.h"
#include "nu_mlpnn.h"
#include "nu_mlpmatrixnn.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace nu {

// ── MlpNN adapter ─────────────────────────────────────────────────────────────

class NnModelMlp final : public NnModel {
public:
    NnModelMlp()
        : _net(std::make_unique<MlpNN>())
    {
    }

    std::string typeName() const override { return "mlp"; }

    size_t getInputSize() const noexcept override { return _net->getInputSize(); }
    size_t getOutputSize() const noexcept override { return _net->getOutputSize(); }

    std::vector<size_t> getTopology() const override
    {
        const auto& t = _net->getTopology();
        return std::vector<size_t>(t.begin(), t.end());
    }

    double getLearningRate() const noexcept override { return _net->getLearningRate(); }

    void setInputVector(const std::vector<double>& v) override
    {
        _net->setInputVector(MlpNN::FpVector(v));
    }

    void feedForward() override { _net->feedForward(); }

    void copyOutputVector(std::vector<double>& out) const override
    {
        MlpNN::FpVector fp;
        _net->copyOutputVector(fp);
        out.assign(fp.begin(), fp.end());
    }

    void backPropagate(const std::vector<double>& target, std::vector<double>& output) override
    {
        MlpNN::FpVector t(target);
        MlpNN::FpVector o(output.size(), 0.0);
        _net->backPropagate(t, o);
        output.assign(o.begin(), o.end());
    }

    double calcMSE(const std::vector<double>& target) override
    {
        return _net->calcMSE(MlpNN::FpVector(target));
    }

    std::ostream& toJson(std::ostream& os) const override { return _net->toJson(os); }
    std::istream& loadJson(std::istream& is) override { return _net->loadJson(is); }

private:
    std::unique_ptr<MlpNN> _net;
};

// ── MlpMatrixNN adapter ───────────────────────────────────────────────────────

class NnModelMlpMatrix final : public NnModel {
public:
    // Default-constructed placeholder; loadJson must be called before use.
    NnModelMlpMatrix()
        : _net(std::make_unique<MlpMatrixNN>(std::vector<MlpMatrixNN::LayerConfig>{
              MlpMatrixNN::LayerConfig(static_cast<size_t>(1)),
              MlpMatrixNN::LayerConfig(static_cast<size_t>(1)) }))
    {
    }

    explicit NnModelMlpMatrix(std::unique_ptr<MlpMatrixNN> net)
        : _net(std::move(net))
    {
    }

    std::string typeName() const override { return "mlp_matrix"; }

    size_t getInputSize() const noexcept override { return _net->getInputSize(); }
    size_t getOutputSize() const noexcept override { return _net->getOutputSize(); }

    std::vector<size_t> getTopology() const override { return _net->getTopology(); }

    double getLearningRate() const noexcept override { return _net->getLearningRate(); }

    void setInputVector(const std::vector<double>& v) override { _net->setInputVector(v); }

    void feedForward() override { _net->feedForward(); }

    void copyOutputVector(std::vector<double>& out) const override { _net->copyOutputVector(out); }

    void backPropagate(const std::vector<double>& target, std::vector<double>& output) override
    {
        // MlpMatrixNN::backPropagate does not call feedForward internally.
        _net->feedForward();
        _net->backPropagate(target);
        _net->copyOutputVector(output);
    }

    double calcMSE(const std::vector<double>& target) override { return _net->calcMSE(target); }

    std::ostream& toJson(std::ostream& os) const override { return _net->toJson(os); }
    std::istream& loadJson(std::istream& is) override { return _net->loadJson(is); }

private:
    std::unique_ptr<MlpMatrixNN> _net;
};

// ── Factory ───────────────────────────────────────────────────────────────────

std::unique_ptr<NnModel> NnModel::load(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("NnModel::load: cannot open '" + path + "'");

    // Peek at the model_type field without consuming the stream.
    const nlohmann::json j = nlohmann::json::parse(f);
    std::string modelType;
    if (j.contains("model_type"))
        modelType = j["model_type"].get<std::string>();
    else
        modelType = "mlp"; // legacy MlpNN files have "type":"ann" but no "model_type"

    // Re-open because json::parse consumed the stream.
    std::ifstream f2(path);
    if (!f2.is_open())
        throw std::runtime_error("NnModel::load: cannot re-open '" + path + "'");

    std::unique_ptr<NnModel> model;
    if (modelType == "mlp_matrix") {
        model = std::make_unique<NnModelMlpMatrix>();
    } else {
        // Default: MlpNN (covers "mlp", "ann", legacy files without model_type).
        model = std::make_unique<NnModelMlp>();
    }

    model->loadJson(f2);
    return model;
}

} // namespace nu
