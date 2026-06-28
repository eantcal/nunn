//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// transformer_char — MiniTransformer demonstration: character-level LM.
//
// Trains a small decoder-only transformer to predict the next character
// in a hardcoded text corpus (Shakespeare excerpt, ~200 chars).
// After training, generates a continuation from a user-chosen prompt.
//
// Network topology (defaults):
//   vocabSize = number of unique chars in corpus
//   seqLen    = 32   (context window)
//   dModel    = 64
//   numHeads  = 4    (dk = 16)
//   dFF       = 128
//   numLayers = 2
//
// Usage: transformer_char [epochs=1000] [lr=0.005] [genLen=80]
//

#include "nu_transformer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

const std::string CORPUS = "to be or not to be that is the question "
                           "whether tis nobler in the mind to suffer "
                           "the slings and arrows of outrageous fortune "
                           "or to take arms against a sea of troubles "
                           "and by opposing end them to die to sleep "
                           "no more and by a sleep to say we end "
                           "the heartache and the thousand natural shocks "
                           "that flesh is heir to";

// Build char <-> index mappings.
struct Vocab {
    std::map<char, int> c2i;
    std::vector<char> i2c;

    explicit Vocab(const std::string& text)
    {
        for (char c : text)
            if (c2i.find(c) == c2i.end()) {
                c2i[c] = static_cast<int>(i2c.size());
                i2c.push_back(c);
            }
    }

    int encode(char c) const { return c2i.at(c); }
    char decode(int i) const { return i2c[static_cast<size_t>(i)]; }
    size_t size() const { return i2c.size(); }
};

// Build (input, target) pairs with shift-by-1.
void buildDataset(const std::string& text, const Vocab& vocab, size_t seqLen,
    std::vector<std::vector<int>>& inputs, std::vector<std::vector<int>>& targets)
{
    if (text.size() <= seqLen)
        return;
    for (size_t i = 0; i + seqLen < text.size(); ++i) {
        std::vector<int> inp, tgt;
        for (size_t j = i; j < i + seqLen; ++j)
            inp.push_back(vocab.encode(text[j]));
        for (size_t j = i + 1; j <= i + seqLen; ++j)
            tgt.push_back(vocab.encode(text[j]));
        inputs.push_back(inp);
        targets.push_back(tgt);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    const int EPOCHS = argc > 1 ? std::stoi(argv[1]) : 1000;
    const double LR = argc > 2 ? std::stod(argv[2]) : 0.005;
    const int GEN_LEN = argc > 3 ? std::stoi(argv[3]) : 80;

    constexpr size_t SEQ_LEN = 32;
    constexpr size_t D_MODEL = 64;
    constexpr size_t N_HEADS = 4;
    constexpr size_t D_FF = 128;
    constexpr size_t N_LAYERS = 2;

    Vocab vocab(CORPUS);
    std::cout << "Corpus length : " << CORPUS.size() << " chars\n";
    std::cout << "Vocabulary    : " << vocab.size() << " unique chars\n\n";

    std::vector<std::vector<int>> inputs, targets;
    buildDataset(CORPUS, vocab, SEQ_LEN, inputs, targets);
    std::cout << "Training pairs: " << inputs.size() << "\n\n";

    nu::MiniTransformer model(vocab.size(), SEQ_LEN, D_MODEL, N_HEADS, D_FF, N_LAYERS, LR);

    std::mt19937 rng(42);
    std::vector<size_t> idx(inputs.size());
    std::iota(idx.begin(), idx.end(), 0);

    constexpr int REPORT = 100;
    std::cout << std::setw(8) << "Epoch" << std::setw(14) << "Mean loss\n";
    std::cout << std::string(22, '-') << "\n";

    for (int ep = 1; ep <= EPOCHS; ++ep) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double totalLoss = 0.0;
        for (size_t i : idx)
            totalLoss += model.train(inputs[i], targets[i]);

        if (ep % REPORT == 0) {
            std::cout << std::setw(8) << ep << std::fixed << std::setprecision(4) << std::setw(14)
                      << totalLoss / static_cast<double>(inputs.size()) << "\n";
        }
    }

    // Generate text starting from first SEQ_LEN chars of corpus.
    std::cout << "\nGeneration (seed: first " << SEQ_LEN << " chars):\n  ";
    std::vector<int> prompt;
    for (size_t i = 0; i < SEQ_LEN; ++i)
        prompt.push_back(vocab.encode(CORPUS[i]));

    // Print seed.
    for (int t : prompt)
        std::cout << vocab.decode(t);

    // Generate continuation.
    auto gen = model.generate(prompt, static_cast<size_t>(GEN_LEN), 0.8, &rng);
    for (int t : gen)
        std::cout << vocab.decode(t);
    std::cout << "\n";

    return 0;
}
