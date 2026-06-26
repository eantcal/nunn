//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_mlpnn.h"
#include "nu_random_gen.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ranges>

#include <nlohmann/json.hpp>

namespace nu {

// ── Constructors ──────────────────────────────────────────────────────────────

MlpNN::MlpNN(const Topology& topology, double learningRate, double momentum, CostFunction cf)
    : _costFunction(cf)
    , _topology(topology)
    , _learningRate(learningRate)
    , _momentum(momentum)
{
    _layerActivations.assign(topology.size() - 1, Activation::Sigmoid);
    _build(_topology, _neuronLayers, _inputVector);
    reshuffleWeights();
}

MlpNN::MlpNN(
    const std::vector<LayerConfig>& layers, double learningRate, double momentum, CostFunction cf)
    : _costFunction(cf)
    , _learningRate(learningRate)
    , _momentum(momentum)
{
    _topology.reserve(layers.size());
    _layerActivations.reserve(layers.size() - 1);

    for (size_t i = 0; i < layers.size(); ++i) {
        _topology.push_back(layers[i].size);
        if (i > 0)
            _layerActivations.push_back(layers[i].activation);
    }

    _build(_topology, _neuronLayers, _inputVector);
    reshuffleWeights();
}

// ── Getters ───────────────────────────────────────────────────────────────────

size_t MlpNN::getInputSize() const noexcept
{
    return _inputVector.size();
}

size_t MlpNN::getOutputSize() const noexcept
{
    return _topology.empty() ? 0 : _topology.back();
}

void MlpNN::setInputVector(const FpVector& inputs)
{
    if (inputs.size() != _inputVector.size())
        throw SizeMismatchException();
    _inputVector = inputs;
}

// ── Forward pass ──────────────────────────────────────────────────────────────

double MlpNN::_getInput(size_t layer, size_t idx) noexcept
{
    if (layer < 1)
        return _inputVector[idx];
    return _neuronLayers[layer - 1][idx].output;
}

void MlpNN::_fireNeuron(NeuronLayer& nlayer, size_t layerIdx, size_t outIdx) noexcept
{
    auto& neuron = nlayer[outIdx];
    double sum{ 0.0 };
    for (size_t idx = 0; const auto& wi : neuron.weights)
        sum += _getInput(layerIdx, idx++) * wi;
    sum += neuron.bias;
    neuron.output = act::forward(_layerActivations[layerIdx], sum);
}

void MlpNN::feedForward() noexcept
{
    for (size_t layerIdx = 0; layerIdx < _neuronLayers.size(); ++layerIdx) {
        auto& nlayer = _neuronLayers[layerIdx];
        for (size_t outIdx = 0; outIdx < nlayer.size(); ++outIdx)
            _fireNeuron(nlayer, layerIdx, outIdx);
    }
}

void MlpNN::copyOutputVector(FpVector& outputs) noexcept
{
    const auto& last = _neuronLayers.back();
    outputs.resize(last.size());
    for (size_t i = 0; const auto& n : last)
        outputs[i++] = n.output;
}

// ── Back-propagation ──────────────────────────────────────────────────────────

void MlpNN::_updateNeuronWeights(Neuron& neuron, size_t layerIdx)
{
    const double lr_err{ neuron.error * _learningRate };

    for (size_t inIdx = 0; inIdx < neuron.weights.size(); ++inIdx) {
        const double dw_prev = neuron.deltaW[inIdx];
        neuron.deltaW[inIdx] = _getInput(layerIdx - 1, inIdx) * lr_err + _momentum * dw_prev;
        neuron.weights[inIdx] += neuron.deltaW[inIdx];
    }

    neuron.deltaB = lr_err + _momentum * neuron.deltaB;
    neuron.bias += neuron.deltaB;
}

void MlpNN::_backPropagate(const FpVector& targetVector, const FpVector& outputVector)
{
    if (targetVector.size() != outputVector.size())
        throw SizeMismatchException();

    // ── Output layer errors ────────────────────────────────────────────────
    //
    // MSE + any activation:
    //   δ_i = act'(y_i) * (t_i - y_i)
    //
    // Cross-Entropy + Sigmoid  (simplification cancels sigmoid derivative):
    //   δ_i = t_i - y_i
    //
    const Activation outAct = _layerActivations.back();
    const bool ceSimplified
        = (_costFunction == CostFunction::CrossEntropy && outAct == Activation::Sigmoid);

    auto& outputLayer = _neuronLayers.back();
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        const double y = outputVector[i], t = targetVector[i];
        outputLayer[i].error = ceSimplified ? (t - y) : act::backward(outAct, y) * (t - y);
    }

    // ── Output layer weight update ─────────────────────────────────────────
    auto layerIdx = _topology.size() - 1; // 1-based index into neuron layers
    for (auto& neuron : _neuronLayers[layerIdx - 1])
        _updateNeuronWeights(neuron, layerIdx);

    // ── Hidden layer errors and weight updates ─────────────────────────────
    //
    // δ_h = act'(y_h) * Σ_k ( δ_k * w_{h→k} )
    //
    while (layerIdx > 1) {
        --layerIdx;

        auto& hiddenLayer = _neuronLayers[layerIdx - 1];
        const auto& nextLayer = _neuronLayers[layerIdx];
        const Activation hidAct = _layerActivations[layerIdx - 1];

        for (size_t nidx = 0; nidx < hiddenLayer.size(); ++nidx) {
            auto& neuron = hiddenLayer[nidx];

            double sum{ 0.0 };
            for (const auto& nextNeuron : nextLayer)
                sum += nextNeuron.error * nextNeuron.weights[nidx];

            neuron.error = act::backward(hidAct, neuron.output) * sum;
            _updateNeuronWeights(neuron, layerIdx);
        }
    }
}

void MlpNN::backPropagate(const FpVector& targetVector, FpVector& outputVector)
{
    feedForward();
    copyOutputVector(outputVector);
    _backPropagate(targetVector, outputVector);
}

void MlpNN::backPropagate(const FpVector& targetVector)
{
    FpVector outputVector;
    backPropagate(targetVector, outputVector);
}

// ── Weight initialisation ─────────────────────────────────────────────────────

void MlpNN::reshuffleWeights() noexcept
{
    // Count total weights across all layers with nested transform_reduce (C++17/20).
    const double weights_cnt = std::sqrt(std::transform_reduce(
        _neuronLayers.begin(), _neuronLayers.end(), 0.0, std::plus<>(), [](const auto& nl) {
            return std::transform_reduce(nl.begin(), nl.end(), 0.0, std::plus<>(),
                [](const auto& n) { return static_cast<double>(n.weights.size()); });
        }));

    RandomGenerator<> rndgen;

    for (auto& nl : _neuronLayers) {
        for (auto& neuron : nl) {
            std::ranges::generate(
                neuron.weights, [&] { return (-1.0 + 2.0 * rndgen()) / weights_cnt; });
            std::ranges::fill(neuron.deltaW, 0.0);
            neuron.deltaB = 0.0;
            neuron.bias = rndgen();
        }
    }
}

// ── Build ─────────────────────────────────────────────────────────────────────

void MlpNN::_build(
    const Topology& topology, std::vector<NeuronLayer>& neuronLayers, FpVector& inputs)
{
    if (topology.size() < 3)
        throw SizeMismatchException();

    neuronLayers.resize(topology.size() - 1);

    for (size_t idx = 0; const auto& count : topology) {
        if (idx < 1) {
            inputs.resize(count);
        } else {
            auto& nl = neuronLayers[idx - 1];
            nl.resize(count);
            for (auto& neuron : nl)
                neuron.resize(topology[idx - 1]);
        }
        ++idx;
    }
}

// ── Legacy text serialization ─────────────────────────────────────────────────

std::stringstream& MlpNN::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != getNetId())
        throw InvalidSStreamFormatException();

    ss >> _learningRate >> _momentum;

    ss >> s;
    if (s != getInputVectorId())
        throw InvalidSStreamFormatException();
    ss >> _inputVector;

    ss >> s;
    if (s != getTopologyId())
        throw InvalidSStreamFormatException();
    ss >> _topology;

    _build(_topology, _neuronLayers, _inputVector);
    _layerActivations.assign(_topology.size() - 1, Activation::Sigmoid);

    for (auto& nl : _neuronLayers) {
        ss >> s;
        if (s != getNeuronLayerId())
            throw InvalidSStreamFormatException();
        for (auto& neuron : nl) {
            ss >> s;
            if (s != getNeuronId())
                throw InvalidSStreamFormatException();
            ss >> neuron;
        }
    }

    return ss;
}

std::stringstream& MlpNN::save(std::stringstream& ss) noexcept
{
    ss.str({});
    ss.clear();
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Each statement starts from ss (stringstream&) so the stringstream-specific
    // operator<< overloads are selected, preserving the serialisation format.
    ss << std::string(getNetId()) << '\n';
    ss << _learningRate << '\n';
    ss << _momentum << '\n';
    ss << std::string(getInputVectorId()) << '\n';
    ss << _inputVector << '\n';
    ss << std::string(getTopologyId()) << '\n';
    ss << _topology << '\n';

    for (auto& nl : _neuronLayers) {
        ss << std::string(getNeuronLayerId()) << '\n';
        for (auto& neuron : nl) {
            ss << std::string(getNeuronId()) << '\n';
            ss << neuron << '\n';
        }
    }

    return ss;
}

// ── JSON serialization (version 2) ───────────────────────────────────────────

static std::string_view costFunctionName(CostFunction cf) noexcept
{
    return cf == CostFunction::CrossEntropy ? "cross_entropy" : "mse";
}

static CostFunction costFunctionFromString(std::string_view s) noexcept
{
    return s == "cross_entropy" ? CostFunction::CrossEntropy : CostFunction::MSE;
}

std::ostream& MlpNN::toJson(std::ostream& os) noexcept
{
    using json = nlohmann::json;

    json j;
    j["type"] = std::string(getNetId());
    j["version"] = 2;
    j["learningRate"] = _learningRate;
    j["momentum"] = _momentum;
    j["costFunction"] = std::string(costFunctionName(_costFunction));
    j["topology"] = static_cast<const std::vector<size_t>&>(_topology);

    json acts = json::array();
    for (const auto& a : _layerActivations)
        acts.push_back(std::string(act::name(a)));
    j["activations"] = std::move(acts);

    j["inputs"] = _inputVector.to_stdvec();

    json layers = json::array();
    for (const auto& nl : _neuronLayers) {
        json layer = json::array();
        for (const auto& n : nl) {
            layer.push_back({
                { "bias", n.bias },
                { "weights", n.weights.to_stdvec() },
                { "deltaW", n.deltaW.to_stdvec() },
            });
        }
        layers.push_back(std::move(layer));
    }
    j["layers"] = std::move(layers);

    os << j.dump(2);
    return os;
}

std::istream& MlpNN::loadJson(std::istream& is)
{
    using json = nlohmann::json;

    const json j = json::parse(is);

    if (j.value("type", "") != std::string(ID_ANN))
        throw InvalidSStreamFormatException();

    _learningRate = j.at("learningRate").get<double>();
    _momentum = j.at("momentum").get<double>();
    _topology = j.at("topology").get<std::vector<size_t>>();

    const int version = j.value("version", 1);

    // Cost function (v2+)
    if (version >= 2 && j.contains("costFunction"))
        _costFunction = costFunctionFromString(j["costFunction"].get<std::string>());
    else
        _costFunction = CostFunction::MSE;

    // Per-layer activations (v2+)
    if (version >= 2 && j.contains("activations")) {
        _layerActivations.clear();
        for (const auto& a : j["activations"])
            _layerActivations.push_back(act::fromString(a.get<std::string>()));
    } else {
        _layerActivations.assign(_topology.size() - 1, Activation::Sigmoid);
    }

    const auto inputs = j.at("inputs").get<std::vector<double>>();
    _inputVector = FpVector(inputs);

    _build(_topology, _neuronLayers, _inputVector);

    const auto& jlayers = j.at("layers");
    for (size_t li = 0; li < _neuronLayers.size(); ++li) {
        const auto& jlayer = jlayers.at(li);
        for (size_t ni = 0; ni < _neuronLayers[li].size(); ++ni) {
            const auto& jn = jlayer.at(ni);
            auto& neuron = _neuronLayers[li][ni];
            neuron.bias = jn.at("bias").get<double>();
            neuron.weights = FpVector(jn.at("weights").get<std::vector<double>>());
            neuron.deltaW = FpVector(jn.at("deltaW").get<std::vector<double>>());
        }
    }

    return is;
}

// ── Dump ──────────────────────────────────────────────────────────────────────

std::ostream& MlpNN::dump(std::ostream& os) noexcept
{
    os << "Net Inputs\n";
    for (size_t i = 0; const auto& v : _inputVector)
        os << "\t[" << i++ << "] = " << v << '\n';

    for (size_t li = 0; const auto& layer : _neuronLayers) {
        const bool isOutput = (li >= _topology.size() - 2);
        os << "\nNeuron layer " << li << " [" << act::name(_layerActivations[li]) << "] "
           << (isOutput ? "Output" : "Hidden") << '\n';

        for (size_t ni = 0; const auto& neuron : layer) {
            os << "\tNeuron " << ni++ << '\n';
            for (size_t inIdx = 0; inIdx < neuron.weights.size(); ++inIdx) {
                os << "\t\tInput  [" << inIdx << "] = " << _getInput(li, inIdx) << '\n';
                os << "\t\tWeight [" << inIdx << "] = " << neuron.weights[inIdx] << '\n';
            }
            os << "\t\tBias   = " << neuron.bias << '\n';
            os << "\t\tOutput = " << neuron.output << '\n';
            os << "\t\tError  = " << neuron.error << '\n';
        }
        ++li;
    }

    return os;
}

// ── Loss ──────────────────────────────────────────────────────────────────────

double MlpNN::calcMSE(const FpVector& targetVector)
{
    FpVector out;
    copyOutputVector(out);
    if (targetVector.size() != out.size())
        throw SizeMismatchException();
    return cf::calcMSE(out, targetVector);
}

double MlpNN::calcCrossEntropy(const FpVector& targetVector)
{
    FpVector out;
    copyOutputVector(out);
    if (targetVector.size() != out.size())
        throw SizeMismatchException();
    return cf::calcCrossEntropy(out, targetVector);
}

} // namespace nu
