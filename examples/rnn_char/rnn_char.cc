//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// rnn_char — character-level language model using a VanillaRnn or LSTM.
//
// The network learns to predict the next character given the current one,
// accumulating context in its hidden/cell state.  After training it generates
// text autoregressively by sampling from the softmax output.
//
// Usage: rnn_char [--lstm] [epochs] [hidden_size] [gen_length] [temperature]
//   --lstm   select LSTM instead of VanillaRnn (default)
//

#include "nu_lstm.h"
#include "nu_rnn.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ── Vocabulary ────────────────────────────────────────────────────────────────

struct Vocab {
    std::map<char, int> ch2idx;
    std::vector<char> idx2ch;

    explicit Vocab(const std::string& corpus)
    {
        for (char c : corpus) {
            if (ch2idx.find(c) == ch2idx.end()) {
                ch2idx[c] = static_cast<int>(idx2ch.size());
                idx2ch.push_back(c);
            }
        }
    }

    int size() const { return static_cast<int>(idx2ch.size()); }

    std::vector<double> oneHot(char c) const
    {
        std::vector<double> v(size(), 0.0);
        v[ch2idx.at(c)] = 1.0;
        return v;
    }

    char sample(const std::vector<double>& probs, double temperature, std::mt19937& rng) const
    {
        std::vector<double> scaled(probs.size());
        double sum = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            scaled[i] = std::pow(probs[i], 1.0 / temperature);
            sum += scaled[i];
        }
        std::uniform_real_distribution<double> dist(0.0, sum);
        double r = dist(rng);
        for (size_t i = 0; i < scaled.size(); ++i) {
            r -= scaled[i];
            if (r <= 0.0)
                return idx2ch[i];
        }
        return idx2ch.back();
    }
};

// ── Training + generation (works for any Rnn type with the same API) ──────────

template <typename Rnn>
void run(Rnn& rnn, const std::string& corpus, const Vocab& vocab, size_t epochs, size_t genLen,
    double temperature)
{
    constexpr size_t SEQ_LEN = 40;
    constexpr size_t STEP = SEQ_LEN / 2;
    const size_t corpus_len = corpus.size();
    std::mt19937 rng_gen(42);

    std::cout << std::fixed << std::setprecision(4);

    // ── Training ──────────────────────────────────────────────────────────────
    std::cout << "Epoch     Loss\n";
    std::cout << std::string(24, '-') << "\n";

    for (size_t ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;
        size_t n_chunks = 0;

        rnn.resetState();
        for (size_t start = 0; start + SEQ_LEN + 1 < corpus_len; start += STEP) {
            std::vector<std::vector<double>> xs, ys;
            xs.reserve(SEQ_LEN);
            ys.reserve(SEQ_LEN);
            for (size_t i = 0; i < SEQ_LEN && start + i + 1 < corpus_len; ++i) {
                xs.push_back(vocab.oneHot(corpus[start + i]));
                ys.push_back(vocab.oneHot(corpus[start + i + 1]));
            }
            total_loss += rnn.bptt(xs, ys);
            ++n_chunks;
        }

        if (ep % (epochs / 10) == 0 || ep == epochs - 1)
            std::cout << std::setw(6) << ep << "    " << (n_chunks ? total_loss / n_chunks : 0.0)
                      << "\n";
    }

    // ── Text generation ───────────────────────────────────────────────────────
    std::cout << "\nGenerated text (seed=\"the \", temperature=" << temperature << "):\n";
    std::cout << std::string(60, '-') << "\n";

    const std::string SEED = "the ";
    rnn.resetState();
    for (char c : SEED) {
        rnn.step(vocab.oneHot(c));
        std::cout << c;
    }

    char prev = SEED.back();
    for (size_t i = 0; i < genLen; ++i) {
        rnn.step(vocab.oneHot(prev));
        prev = vocab.sample(rnn.getOutput(), temperature, rng_gen);
        std::cout << prev;
    }
    std::cout << "\n" << std::string(60, '-') << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_lstm = false;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--lstm")
            use_lstm = true;
        else
            pos.push_back(argv[i]);
    }

    const size_t EPOCHS = pos.size() > 0 ? std::stoul(pos[0]) : 800;
    const size_t HIDDEN = pos.size() > 1 ? std::stoul(pos[1]) : 64;
    const size_t GEN_LEN = pos.size() > 2 ? std::stoul(pos[2]) : 120;
    const double TEMPERATURE = pos.size() > 3 ? std::stod(pos[3]) : 0.8;

    const std::string BASE = "the quick brown fox jumps over the lazy dog. "
                             "pack my box with five dozen liquor jugs. ";
    std::string corpus;
    for (int r = 0; r < 6; ++r)
        corpus += BASE;

    const Vocab vocab(corpus);
    const int V = vocab.size();

    constexpr double LR = 0.005;
    constexpr double GRAD_CLIP = 5.0;

    std::cout << "rnn_char  |  model=" << (use_lstm ? "LSTM" : "VanillaRnn") << "  vocab=" << V
              << "  hidden=" << HIDDEN << "  lr=" << LR << "  epochs=" << EPOCHS << "\n\n";

    if (use_lstm) {
        nu::Lstm lstm(V, HIDDEN, V, LR, GRAD_CLIP, nu::RnnOutput::Softmax);
        run(lstm, corpus, vocab, EPOCHS, GEN_LEN, TEMPERATURE);
    } else {
        nu::VanillaRnn rnn(V, HIDDEN, V, LR, GRAD_CLIP, nu::RnnOutput::Softmax);
        run(rnn, corpus, vocab, EPOCHS, GEN_LEN, TEMPERATURE);
    }

    return 0;
}
