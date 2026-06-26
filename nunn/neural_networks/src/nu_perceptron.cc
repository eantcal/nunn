#include "nu_perceptron.h"
#include "nu_random_gen.h"
#include "nu_sigmoid.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>

#include <nlohmann/json.hpp>

namespace nu {

Perceptron::Perceptron(size_t inputSize, double learningRate, StepFunction step_f)
    : _step_f(step_f)
    , _learningRate(learningRate)
{
    if (inputSize < 1) {
        throw SizeMismatchException();
    }

    _inputVector.resize(inputSize, 0.0);
    _neuron.deltaW.resize(inputSize, 0.0);
    _neuron.weights.resize(inputSize, 0.0);

    reshuffleWeights();
}

void Perceptron::setInputVector(const FpVector& inputs)
{
    if (inputs.size() != _inputVector.size()) {
        throw SizeMismatchException();
    }

    _inputVector = inputs;
}

void Perceptron::feedForward() noexcept
{
    double sum
        = std::inner_product(_inputVector.begin(), _inputVector.end(), _neuron.weights.begin(), 0.0)
        + _neuron.bias;
    _neuron.output = Sigmoid()(sum);
}

void Perceptron::backPropagate(const double& target, double& output) noexcept
{
    feedForward();
    output = getOutput();

    double error = target - output;
    double e = _learningRate * error;

    std::transform(_inputVector.begin(), _inputVector.end(), _neuron.weights.begin(),
        _neuron.weights.begin(), [&](double input, double weight) { return weight + e * input; });

    _neuron.bias += e;
}

void Perceptron::backPropagate(const double& target) noexcept
{
    double output;
    backPropagate(target, output);
}

double Perceptron::error(const double& target) const noexcept
{
    return std::abs(target - getOutput());
}

std::stringstream& Perceptron::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != Perceptron::ID_ANN) {
        throw InvalidSStreamFormatException();
    }

    ss >> _learningRate;

    ss >> s;
    if (s != Perceptron::ID_INPUTS) {
        throw InvalidSStreamFormatException();
    }

    ss >> _inputVector;

    ss >> s;
    if (s != Perceptron::ID_NEURON) {
        throw InvalidSStreamFormatException();
    }

    ss >> _neuron;

    return ss;
}

std::stringstream& Perceptron::save(std::stringstream& ss) noexcept
{
    ss.clear();

    // Write doubles at full round-trip precision so save()/load() is lossless.
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);

    ss << Perceptron::ID_ANN << std::endl;

    ss << _learningRate << std::endl;

    ss << Perceptron::ID_INPUTS << std::endl;
    ss << _inputVector << std::endl;

    ss << Perceptron::ID_NEURON << std::endl;
    ss << _neuron << std::endl;

    return ss;
}

void Perceptron::reshuffleWeights() noexcept
{
    RandomGenerator<> randgen;
    double weights_cnt = std::sqrt(static_cast<double>(_neuron.weights.size()));

    // Initialize all the network weights using random numbers within the range
    // [-1,1]
    for (auto& w : _neuron.weights) {
        w = (-1.0 + 2 * randgen()) / weights_cnt;
    }

    std::fill(_neuron.deltaW.begin(), _neuron.deltaW.end(), 0.0);
    _neuron.bias = randgen();
}

std::ostream& Perceptron::toJson(std::ostream& os) noexcept
{
    using json = nlohmann::json;

    json j;
    j["type"] = std::string(ID_ANN);
    j["version"] = 1;
    j["learningRate"] = _learningRate;
    j["inputs"] = _inputVector.to_stdvec();
    j["neuron"] = {
        { "bias", _neuron.bias },
        { "weights", _neuron.weights.to_stdvec() },
        { "deltaW", _neuron.deltaW.to_stdvec() },
    };

    os << j.dump(2);
    return os;
}

std::istream& Perceptron::loadJson(std::istream& is)
{
    using json = nlohmann::json;

    const json j = json::parse(is);

    if (j.value("type", "") != std::string(ID_ANN)) {
        throw InvalidSStreamFormatException();
    }

    _learningRate = j.at("learningRate").get<double>();
    _inputVector = FpVector(j.at("inputs").get<std::vector<double>>());

    const auto& jn = j.at("neuron");
    _neuron.bias = jn.at("bias").get<double>();
    _neuron.weights = FpVector(jn.at("weights").get<std::vector<double>>());
    _neuron.deltaW = FpVector(jn.at("deltaW").get<std::vector<double>>());

    return is;
}

std::ostream& Perceptron::dump(std::ostream& os) noexcept
{
    os << "Perceptron\n";

    for (size_t i = 0; i < _neuron.weights.size(); ++i) {
        os << "\tInput [" << i << "] = " << _inputVector[i] << ", Weight [" << i
           << "] = " << _neuron.weights[i] << "\n";
    }

    os << "\tBias = " << _neuron.bias << "\n\tOutput = " << _neuron.output
       << "\n\tError = " << _neuron.error << "\n";

    return os;
}

} // namespace nu
