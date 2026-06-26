//
// net2json — converts nunn legacy .net files to JSON format.
//
// Usage:
//   net2json <input.net> [output.json]
//
// If output path is omitted the JSON file is written alongside the input
// with the same stem and the .json extension.
//

#include "nu_hopfieldnn.h"
#include "nu_mlpnn.h"
#include "nu_perceptron.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

static int convert(const fs::path& inPath, const fs::path& outPath)
{
    std::ifstream ifs(inPath, std::ios::binary);
    if (!ifs) {
        std::cerr << "error: cannot open '" << inPath.string() << "'\n";
        return 1;
    }

    // Read the whole file into a stringstream so we can peek at the type token
    // and then rewind before handing the stream to the load() method.
    std::ostringstream buf;
    buf << ifs.rdbuf();
    std::stringstream ss(buf.str());

    std::string typeToken;
    ss >> typeToken;
    ss.seekg(0);
    ss.clear();

    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "error: cannot create '" << outPath.string() << "'\n";
        return 1;
    }

    try {
        if (typeToken == "ann") {
            nu::MlpNN net;
            net.load(ss);
            net.toJson(ofs);
        } else if (typeToken == "perceptron") {
            nu::Perceptron net;
            net.load(ss);
            net.toJson(ofs);
        } else if (typeToken == "HopfieldNN") {
            nu::HopfieldNN net(0);
            net.load(ss);
            net.toJson(ofs);
        } else {
            std::cerr << "error: unknown network type '" << typeToken << "' in '" << inPath.string()
                      << "'\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << " (file: " << inPath.string() << ")\n";
        return 1;
    }

    std::cout << inPath.string() << "  ->  " << outPath.string() << "\n";
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: net2json <input.net> [output.json]\n";
        return 1;
    }

    fs::path inPath = argv[1];
    fs::path outPath = (argc >= 3) ? fs::path(argv[2]) : inPath.replace_extension(".json");

    return convert(inPath, outPath);
}
