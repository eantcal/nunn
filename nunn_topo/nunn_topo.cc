//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_mlpnn.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string inputFile;
    std::string outputFile;
    std::string topologyText;
    std::string format = "dot";
    std::string dotCommand = "dot";
    size_t maxNodesPerLayer = 16;
    bool full = false;
    bool summary = false;
};

using Topology = std::vector<size_t>;

void usage(const char* appname)
{
    std::cerr
        << "Usage:\n"
        << "  " << appname << " --load model.json --save topology.svg\n"
        << "  " << appname << " --load model.net --save topology.dot\n"
        << "  " << appname << " --topology 784,300,10 --save mnist.png\n"
        << "\n"
        << "Options:\n"
        << "  -l, --load <file>       Load a legacy .net model or a JSON model\n"
        << "  -s, --save <file>       Save DOT, SVG, PNG, or PDF output\n"
        << "  -t, --topology <list>   Use a topology list, e.g. 2,3,1 or 784x300x10\n"
        << "  -f, --format <fmt>      Output format: dot, svg, png, pdf (default: inferred from "
           "--save, or dot)\n"
        << "      --full              Draw every node and every connection\n"
        << "      --max-nodes <n>     Compact mode: visible nodes per layer before eliding "
           "(default: 16)\n"
        << "      --dot <command>     Graphviz dot command/path (default: dot)\n"
        << "      --summary           Print topology summary to stderr\n"
        << "  -v, --version           Show version\n"
        << "  -h, --help              Show this help\n"
        << "\n"
        << "Notes:\n"
        << "  DOT output does not require Graphviz. SVG/PNG/PDF output requires Graphviz dot in "
           "PATH.\n"
        << "  Large networks are compacted by default; use --full for exact complete graphs.\n";
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

std::string extensionOf(const std::string& filename)
{
    if (filename.empty())
        return {};
    auto ext = std::filesystem::path(filename).extension().string();
    if (!ext.empty() && ext.front() == '.')
        ext.erase(ext.begin());
    return lower(ext);
}

bool isRenderFormat(const std::string& format)
{
    return format == "svg" || format == "png" || format == "pdf";
}

void validateFormat(const std::string& format)
{
    if (format != "dot" && !isRenderFormat(format))
        throw std::runtime_error("Unsupported output format '" + format + "'");
}

Options parseCommandLine(int argc, char* argv[])
{
    Options opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto needValue = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + std::string(name));
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--version" || arg == "-v") {
            std::cout << "nunn_topo 2.0\n";
            std::exit(0);
        }
        if (arg == "--load" || arg == "-l") {
            opt.inputFile = needValue(arg);
            continue;
        }
        if (arg == "--save" || arg == "-s") {
            opt.outputFile = needValue(arg);
            continue;
        }
        if (arg == "--topology" || arg == "-t") {
            opt.topologyText = needValue(arg);
            continue;
        }
        if (arg == "--format" || arg == "-f") {
            opt.format = lower(needValue(arg));
            continue;
        }
        if (arg == "--dot") {
            opt.dotCommand = needValue(arg);
            continue;
        }
        if (arg == "--max-nodes") {
            opt.maxNodesPerLayer = static_cast<size_t>(std::stoull(needValue(arg)));
            continue;
        }
        if (arg == "--full") {
            opt.full = true;
            continue;
        }
        if (arg == "--summary") {
            opt.summary = true;
            continue;
        }
        if (arg.starts_with('-'))
            throw std::runtime_error("Unknown option '" + arg + "'");

        if (opt.inputFile.empty())
            opt.inputFile = arg;
        else if (opt.outputFile.empty())
            opt.outputFile = arg;
        else
            throw std::runtime_error("Unexpected positional argument '" + arg + "'");
    }

    if (opt.topologyText.empty() && opt.inputFile.empty())
        throw std::runtime_error("Missing input. Use --load <model> or --topology <list>.");

    if (!opt.outputFile.empty()) {
        const auto ext = extensionOf(opt.outputFile);
        if ((opt.format == "dot" || opt.format.empty())
            && (ext == "svg" || ext == "png" || ext == "pdf"))
            opt.format = ext;
    }

    opt.format = lower(opt.format);
    validateFormat(opt.format);

    if (opt.maxNodesPerLayer == 0)
        throw std::runtime_error("--max-nodes must be greater than zero");

    return opt;
}

std::string readFile(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open '" + filename + "'");

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Topology parseTopologyText(std::string text)
{
    for (auto& ch : text) {
        if (ch == 'x' || ch == 'X' || ch == ';')
            ch = ',';
    }

    Topology topology;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; }),
            item.end());
        if (item.empty())
            continue;
        const auto value = static_cast<size_t>(std::stoull(item));
        if (value == 0)
            throw std::runtime_error("Topology entries must be greater than zero");
        topology.push_back(value);
    }

    if (topology.size() < 2)
        throw std::runtime_error("Topology must contain at least input and output layers");

    return topology;
}

std::optional<Topology> topologyFromJson(const std::string& text)
{
    const auto first = std::find_if_not(
        text.begin(), text.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });

    if (first == text.end() || *first != '{')
        return std::nullopt;

    const auto j = nlohmann::json::parse(text);

    if (j.contains("topology"))
        return j.at("topology").get<Topology>();

    if (j.contains("layers")) {
        Topology topology;
        for (const auto& layer : j.at("layers"))
            topology.push_back(layer.at("size").get<size_t>());
        return topology;
    }

    throw std::runtime_error("JSON model does not contain 'topology' or 'layers[*].size'");
}

Topology topologyFromLegacyNet(const std::string& text)
{
    std::stringstream ss(text);
    nu::MlpNN net;
    net.load(ss);
    return net.getTopology();
}

Topology loadTopology(const Options& opt)
{
    Topology topology;

    if (!opt.topologyText.empty()) {
        topology = parseTopologyText(opt.topologyText);
    } else {
        const auto text = readFile(opt.inputFile);
        if (auto jsonTopology = topologyFromJson(text))
            topology = std::move(*jsonTopology);
        else
            topology = topologyFromLegacyNet(text);
    }

    if (topology.size() < 2)
        throw std::runtime_error("Topology must contain at least input and output layers");

    for (const auto n : topology) {
        if (n == 0)
            throw std::runtime_error("Topology entries must be greater than zero");
    }

    return topology;
}

std::vector<size_t> visibleNodeIndexes(size_t count, size_t maxNodes, bool full)
{
    std::vector<size_t> indexes;
    if (full || count <= maxNodes) {
        indexes.reserve(count);
        for (size_t i = 0; i < count; ++i)
            indexes.push_back(i);
        return indexes;
    }

    const size_t head = std::max<size_t>(1, maxNodes / 2);
    const size_t tail = std::max<size_t>(1, maxNodes - head);

    for (size_t i = 0; i < head && i < count; ++i)
        indexes.push_back(i);

    const size_t tailStart = count > tail ? count - tail : head;
    for (size_t i = tailStart; i < count; ++i) {
        if (std::find(indexes.begin(), indexes.end(), i) == indexes.end())
            indexes.push_back(i);
    }

    return indexes;
}

bool isCompacted(size_t count, size_t maxNodes, bool full)
{
    return !full && count > maxNodes;
}

std::string layerName(size_t layerIndex, size_t layerCount)
{
    if (layerIndex == 0)
        return "Input";
    if (layerIndex + 1 == layerCount)
        return "Output";
    return "Hidden " + std::to_string(layerIndex);
}

std::string nodeId(size_t layer, size_t node)
{
    return "l" + std::to_string(layer) + "_n" + std::to_string(node);
}

std::string ellipsisId(size_t layer)
{
    return "l" + std::to_string(layer) + "_ellipsis";
}

std::string nodeColor(size_t layer, size_t layerCount)
{
    if (layer == 0)
        return "#386cb0";
    if (layer + 1 == layerCount)
        return "#31a354";
    return "#de2d26";
}

std::string makeDot(const Topology& topology, const Options& opt)
{
    std::ostringstream dot;
    const size_t layerCount = topology.size();

    dot << "digraph nunn_topology {\n"
        << "  graph [rankdir=LR, splines=line, nodesep=0.45, ranksep=1.3, bgcolor=\"white\"];\n"
        << "  node [shape=circle, fixedsize=true, width=0.36, height=0.36, label=\"\", "
           "style=filled, fontname=\"Arial\"];\n"
        << "  edge [color=\"#8c8c8c\", arrowsize=0.45, penwidth=0.7];\n"
        << "  labelloc=\"t\";\n"
        << "  label=\"nuNN topology: ";

    for (size_t i = 0; i < topology.size(); ++i) {
        if (i)
            dot << " -> ";
        dot << topology[i];
    }
    dot << "\";\n\n";

    for (size_t layer = 0; layer < layerCount; ++layer) {
        const auto visible = visibleNodeIndexes(topology[layer], opt.maxNodesPerLayer, opt.full);
        const bool compact = isCompacted(topology[layer], opt.maxNodesPerLayer, opt.full);

        dot << "  subgraph cluster_" << layer << " {\n"
            << "    color=\"white\";\n"
            << "    label=\"" << layerName(layer, layerCount) << " (" << topology[layer] << ")\";\n"
            << "    rank=same;\n";

        for (const auto node : visible) {
            dot << "    " << nodeId(layer, node) << " [fillcolor=\"" << nodeColor(layer, layerCount)
                << "\"";
            if (topology[layer] <= opt.maxNodesPerLayer || opt.full)
                dot << ", xlabel=\"" << node << "\"";
            dot << "];\n";
        }

        if (compact) {
            dot << "    " << ellipsisId(layer)
                << " [shape=plaintext, fixedsize=false, label=\"...\", width=0.25, height=0.25, "
                   "fillcolor=\"white\"];\n";
        }

        dot << "  }\n";
    }

    dot << "\n";

    for (size_t layer = 0; layer + 1 < layerCount; ++layer) {
        const auto left = visibleNodeIndexes(topology[layer], opt.maxNodesPerLayer, opt.full);
        const auto right = visibleNodeIndexes(topology[layer + 1], opt.maxNodesPerLayer, opt.full);
        const bool leftCompact = isCompacted(topology[layer], opt.maxNodesPerLayer, opt.full);
        const bool rightCompact = isCompacted(topology[layer + 1], opt.maxNodesPerLayer, opt.full);

        for (const auto lnode : left) {
            for (const auto rnode : right)
                dot << "  " << nodeId(layer, lnode) << " -> " << nodeId(layer + 1, rnode) << ";\n";
        }

        if (rightCompact) {
            for (const auto lnode : left) {
                dot << "  " << nodeId(layer, lnode) << " -> " << ellipsisId(layer + 1)
                    << " [style=dashed, color=\"#bdbdbd\", arrowsize=0.35];\n";
            }
        }

        if (leftCompact) {
            for (const auto rnode : right) {
                dot << "  " << ellipsisId(layer) << " -> " << nodeId(layer + 1, rnode)
                    << " [style=dashed, color=\"#bdbdbd\", arrowsize=0.35];\n";
            }
        }

        if (leftCompact && rightCompact) {
            dot << "  " << ellipsisId(layer) << " -> " << ellipsisId(layer + 1)
                << " [style=dashed, color=\"#9e9e9e\", label=\"omitted all-to-all\", fontsize=9, "
                   "fontname=\"Arial\"];\n";
        }
    }

    dot << "}\n";
    return dot.str();
}

void writeTextFile(const std::string& filename, const std::string& text)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot create '" + filename + "'");
    out << text;
}

std::string shellQuote(const std::string& value)
{
    std::string out = "\"";
    for (const auto ch : value) {
        if (ch == '"')
            out += "\\\"";
        else
            out += ch;
    }
    out += "\"";
    return out;
}

std::string shellCommand(const std::string& value)
{
    const bool needsQuoting = value.find_first_of(" \t\\/.:") != std::string::npos;
    return needsQuoting ? shellQuote(value) : value;
}

void renderWithGraphviz(const std::string& dotText, const Options& opt)
{
    auto tmp = std::filesystem::temp_directory_path()
        / ("nunn_topo_" + std::to_string(static_cast<unsigned long long>(std::rand())) + ".dot");

    writeTextFile(tmp.string(), dotText);

    const auto command = shellCommand(opt.dotCommand) + " -T" + opt.format + " "
        + shellQuote(tmp.string()) + " -o " + shellQuote(opt.outputFile);

    const int rc = std::system(command.c_str());
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);

    if (rc != 0)
        throw std::runtime_error("Graphviz dot failed. Install Graphviz or use --format dot.");
}

void printSummary(const Topology& topology, bool compact)
{
    size_t weights = 0;
    for (size_t i = 0; i + 1 < topology.size(); ++i)
        weights += topology[i] * topology[i + 1];

    std::cerr << "Topology:";
    for (const auto n : topology)
        std::cerr << ' ' << n;
    std::cerr << "\nLayers: " << topology.size() << "\nWeighted connections: " << weights
              << "\nMode: " << (compact ? "compact" : "full") << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const auto opt = parseCommandLine(argc, argv);
        const auto topology = loadTopology(opt);
        const bool compact = !opt.full
            && std::any_of(topology.begin(), topology.end(),
                [&](size_t n) { return n > opt.maxNodesPerLayer; });

        if (opt.summary)
            printSummary(topology, compact);

        const auto dot = makeDot(topology, opt);

        if (opt.outputFile.empty()) {
            if (opt.format != "dot")
                throw std::runtime_error("Rendered output requires --save <file>");
            std::cout << dot;
            return 0;
        }

        if (opt.format == "dot")
            writeTextFile(opt.outputFile, dot);
        else
            renderWithGraphviz(dot, opt);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "nunn_topo: " << ex.what() << "\n\n";
        usage(argv[0]);
        return 1;
    }
}
