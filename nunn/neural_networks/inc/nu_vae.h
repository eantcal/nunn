//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Variational Autoencoder (VAE) with reparameterization trick.
//
// Architecture:
//   Encoder: x [nx] -> h_enc [nh] (ReLU) -> mu [nz] and logvar [nz] (linear)
//   Reparameterize: z = mu + exp(0.5*logvar) * eps,  eps ~ N(0,I)
//   Decoder: z [nz] -> h_dec [nh] (ReLU) -> recon [nx] (sigmoid)
//
// Loss (ELBO, sign-flipped for minimisation):
//   L = L_recon + L_KL
//   L_recon = -mean_i(x_i*log(r_i) + (1-x_i)*log(1-r_i))  (BCE for binary data)
//   L_KL    = -0.5 * mean_j(1 + lv_j - mu_j^2 - exp(lv_j))
//
// BCE + sigmoid gradient simplification: dL/d(pre_out_i) = (r_i - x_i) / nx
// (no sigmoid saturation, 4x larger gradient than MSE at the same error level)
//
// Parameters (10 Eigen matrices/vectors):
//   W_enc [nh x nx], b_enc [nh]
//   W_mu  [nz x nh], b_mu  [nz]
//   W_lv  [nz x nh], b_lv  [nz]   (logvar branch)
//   W_dec [nh x nz], b_dec [nh]
//   W_out [nx x nh], b_out [nx]
//
// Backprop:
//   All gradients are computed with original weights (no in-place aliasing),
//   then parameters are updated in a single step at the end of trainStep().
//

#pragma once

#include <Eigen/Core>
#include <random>
#include <utility>
#include <vector>

namespace nu {

class Vae {
public:
    // inputDim  -- visible / output dimension
    // hiddenDim -- encoder and decoder hidden layer width
    // latentDim -- size of latent vector z
    // lr        -- learning rate; default 0.001
    // seed      -- RNG seed for reparameterization and weight init
    // Throws std::invalid_argument if any dimension is zero.
    Vae(size_t inputDim, size_t hiddenDim, size_t latentDim, double lr = 0.001, unsigned seed = 42);

    // Encode: returns {mu [latentDim], logvar [latentDim]}.
    // Throws std::invalid_argument if x.size() != inputDim().
    std::pair<std::vector<double>, std::vector<double>> encode(const std::vector<double>& x) const;

    // Decode: z -> reconstruction in [0,1]^inputDim via sigmoid.
    // Throws std::invalid_argument if z.size() != latentDim().
    std::vector<double> decode(const std::vector<double>& z) const;

    // Reconstruct: x -> mu -> recon (deterministic; uses mu, no sampling).
    std::vector<double> reconstruct(const std::vector<double>& x) const;

    // Generate: sample z ~ N(0,I), decode.
    std::vector<double> generate();

    // Forward + backward for one sample; returns {L_recon, klWeight*L_KL}.
    //   klWeight = 0: pure reconstruction (autoencoder mode, no KL penalty)
    //   klWeight = 1: standard ELBO (KL fully active)
    // Throws std::invalid_argument if x.size() != inputDim().
    std::pair<double, double> trainStep(const std::vector<double>& x, double klWeight = 1.0);

    // Full online training with KL annealing: klWeight ramps linearly from 0
    // to 1 over the first warmupFrac fraction of epochs, then stays at 1.
    // warmupFrac = 0: KL always fully active (no annealing).
    // Throws std::invalid_argument if dataset is empty or size mismatches.
    void train(
        const std::vector<std::vector<double>>& dataset, size_t epochs, double warmupFrac = 0.5);

    // Mean binary cross-entropy between each input and its deterministic reconstruction.
    double reconstructionError(const std::vector<std::vector<double>>& dataset) const;

    size_t inputDim() const noexcept { return _nx; }
    size_t hiddenDim() const noexcept { return _nh; }
    size_t latentDim() const noexcept { return _nz; }

private:
    size_t _nx, _nh, _nz;
    double _lr;

    // Encoder
    Eigen::MatrixXd _W_enc; // [nh x nx]
    Eigen::VectorXd _b_enc; // [nh]
    // mu and logvar branches
    Eigen::MatrixXd _W_mu; // [nz x nh]
    Eigen::VectorXd _b_mu; // [nz]
    Eigen::MatrixXd _W_lv; // [nz x nh]
    Eigen::VectorXd _b_lv; // [nz]
    // Decoder
    Eigen::MatrixXd _W_dec; // [nh x nz]
    Eigen::VectorXd _b_dec; // [nh]
    Eigen::MatrixXd _W_out; // [nx x nh]
    Eigen::VectorXd _b_out; // [nx]

    std::mt19937 _rng;
    std::normal_distribution<double> _ndist;

    static Eigen::VectorXd _relu(const Eigen::VectorXd& x);
    static Eigen::VectorXd _sigmoid(const Eigen::VectorXd& x);
    Eigen::VectorXd _sampleNormal(size_t n);
    Eigen::VectorXd _toEigen(const std::vector<double>& v, size_t expectedSize) const;

    // Decoder forward (shared between decode(), reconstruct(), generate())
    Eigen::VectorXd _decodeEigen(const Eigen::VectorXd& z) const;
};

} // namespace nu
