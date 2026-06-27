//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/*
This test is performed by using the MNIST data set, which contains 60K+10K
scanned images of handwritten digits with their correct classifications.
The images are greyscale and 28x28 pixels in size.

60,000 images are used for training; 10,000 for evaluation.

The training input is a 28x28=784-dimensional vector of normalised grey values.
The output is a 10-dimensional vector (one neuron per digit class).

See also http://yann.lecun.com/exdb/mnist/

Extra flags vs. the classic build:
  --matrix / -M        Use MlpMatrixNN (Eigen-backed) instead of MlpNN
  --batch  / -b <N>    Mini-batch size (requires --matrix; default 1 = online SGD)
*/

#include "mnist.h"
#include "nu_mlpmatrixnn.h"
#include "nu_mlpnn.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

// ── Defaults ─────────────────────────────────────────────────────────────────

constexpr size_t HIDDEN_LAYER_SIZE = 300;
constexpr size_t OUTPUT_LAYER_SIZE = 10;
constexpr double NET_LEARNING_RATE = 0.025;
constexpr double NET_MOMENTUM = 0.50;
constexpr int TRAINING_EPOCH_NUMBER = 100;

std::string config_training_labels_fn = "train-labels.idx1-ubyte";
std::string config_training_images_fn = "train-images.idx3-ubyte";
std::string config_test_labels_fn = "t10k-labels.idx1-ubyte";
std::string config_test_image_fn = "t10k-images.idx3-ubyte";

// ── Argument parsing ──────────────────────────────────────────────────────────

static bool process_cl(int argc, char* argv[], std::string& files_path, std::string& load_file_name,
    std::string& save_file_name, bool& skip_training, double& learningRate, bool& change_lr,
    double& momentum, bool& change_m, int& epoch, std::vector<size_t>& hidden_layer,
    bool& use_cross_entropy, nu::Activation& activation, bool& use_matrix, size_t& batch_size)
{
    for (int pidx = 1; pidx < argc; ++pidx) {
        const std::string arg = argv[pidx];

        if (arg == "--help" || arg == "-h")
            return false;

        if (arg == "--version" || arg == "-v") {
            std::cout << "nunnlib MNIST Test 1.02 (c) antonino.calderone@gmail.com\n";
            continue;
        }

        if ((arg == "--training_files_path" || arg == "-p") && (pidx + 1) < argc) {
            files_path = argv[++pidx];
            if (!files_path.empty() && files_path.back() != '/')
                files_path += '/';
            continue;
        }
        if ((arg == "--training_imgsfn" || arg == "-tri") && (pidx + 1) < argc) {
            config_training_images_fn = argv[++pidx];
            continue;
        }
        if ((arg == "--training_lblsfn" || arg == "-trl") && (pidx + 1) < argc) {
            config_training_labels_fn = argv[++pidx];
            continue;
        }
        if ((arg == "--test_imgsfn" || arg == "-ti") && (pidx + 1) < argc) {
            config_test_image_fn = argv[++pidx];
            continue;
        }
        if ((arg == "--test_lblsfn" || arg == "-tl") && (pidx + 1) < argc) {
            config_test_labels_fn = argv[++pidx];
            continue;
        }

        if (arg == "--skip_training" || arg == "-n") {
            skip_training = true;
            continue;
        }
        if ((arg == "--load" || arg == "-l") && (pidx + 1) < argc) {
            load_file_name = argv[++pidx];
            continue;
        }
        if ((arg == "--save" || arg == "-s") && (pidx + 1) < argc) {
            save_file_name = argv[++pidx];
            continue;
        }
        if ((arg == "--learningRate" || arg == "-r") && (pidx + 1) < argc) {
            try {
                learningRate = std::stod(argv[++pidx]);
                change_lr = true;
            } catch (...) {
                return false;
            }
            continue;
        }
        if ((arg == "--momentum" || arg == "-m") && (pidx + 1) < argc) {
            try {
                momentum = std::stod(argv[++pidx]);
                change_m = true;
            } catch (...) {
                return false;
            }
            continue;
        }
        if (arg == "--use_cross_entropy" || arg == "-c") {
            use_cross_entropy = true;
            continue;
        }
        if ((arg == "--epoch_cnt" || arg == "--epoch_num" || arg == "-e") && (pidx + 1) < argc) {
            try {
                epoch = std::stoi(argv[++pidx]);
            } catch (...) {
                return false;
            }
            continue;
        }
        if ((arg == "--hidden_layer" || arg == "-hl") && (pidx + 1) < argc) {
            try {
                hidden_layer.push_back(static_cast<size_t>(std::stoi(argv[++pidx])));
            } catch (...) {
                return false;
            }
            continue;
        }
        if ((arg == "--activation" || arg == "-a") && (pidx + 1) < argc) {
            try {
                activation = nu::act::fromString(argv[++pidx]);
            } catch (...) {
                return false;
            }
            continue;
        }

        // ── New flags ─────────────────────────────────────────────────────────

        if (arg == "--matrix" || arg == "-M") {
            use_matrix = true;
            continue;
        }
        if ((arg == "--batch" || arg == "-b") && (pidx + 1) < argc) {
            try {
                const int v = std::stoi(argv[++pidx]);
                if (v < 1)
                    return false;
                batch_size = static_cast<size_t>(v);
            } catch (...) {
                return false;
            }
            continue;
        }

        return false;
    }
    return true;
}

// ── Console helpers ───────────────────────────────────────────────────────────

static int get_y_pos()
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info = {};
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
    return info.dwCursorPosition.Y;
#else
    return 0;
#endif
}

static void locate(int x, int y = 0)
{
    if (y == 0)
        y = get_y_pos();
#ifdef _WIN32
    COORD c = { short((x - 1) & 0xffff), short((y - 1) & 0xffff) };
    ::SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), c);
#else
    printf("%c[%d;%df", 0x1B, y, x);
#endif
}

// ── Usage ─────────────────────────────────────────────────────────────────────

static void usage(const char* appname)
{
    std::cerr
        << "Usage: " << appname << "\n"
        << "\t[--version|-v]\n"
        << "\t[--help|-h]\n"
        << "\t[--training_files_path|-p <path>]\n"
        << "\t[--training_imgsfn|-tri <filename>]  (default: " << config_training_images_fn << ")\n"
        << "\t[--training_lblsfn|-trl <filename>]  (default: " << config_training_labels_fn << ")\n"
        << "\t[--test_imgsfn|-ti <filename>]       (default: " << config_test_image_fn << ")\n"
        << "\t[--test_lblsfn|-tl <filename>]       (default: " << config_test_labels_fn << ")\n"
        << "\t[--save|-s <net_file>]\n"
        << "\t[--load|-l <net_file>]\n"
        << "\t[--skip_training|-n]\n"
        << "\t[--use_cross_entropy|-c]\n"
        << "\t[--learningRate|-r <rate>]           (default: " << NET_LEARNING_RATE << ")\n"
        << "\t[--momentum|-m <value>]              (default: " << NET_MOMENTUM << ")\n"
        << "\t[--epoch_cnt|-e <count>]             (default: " << TRAINING_EPOCH_NUMBER << ")\n"
        << "\t[--hidden_layer|-hl <size>] ...      (default: " << HIDDEN_LAYER_SIZE << ")\n"
        << "\t[--activation|-a <name>]             (sigmoid|tanh|relu|leaky_relu|linear,"
           " default: sigmoid)\n"
        << "\t[--matrix|-M]                        Use MlpMatrixNN (Eigen) instead of MlpNN\n"
        << "\t[--batch|-b <size>]                  Mini-batch size for --matrix (default: 1)\n"
        << "\n"
        << "Notes:\n"
        << "  --activation applies to all hidden layers; output layer is always Sigmoid.\n"
        << "  --use_cross_entropy is recommended together with Sigmoid hidden/output layers.\n"
        << "  --batch requires --matrix; batch=1 is online SGD (same as no --batch).\n"
        << "  --save/--load are not available in --matrix mode.\n";
}

// ── Test functions ────────────────────────────────────────────────────────────

// Evaluate MlpNN on the test set.
static double test_net(std::unique_ptr<nu::MlpNN>& net, const TrainingData::data_t& test_data,
    double& mean_square_error, double& entropy_cost)
{
    size_t cnt = 0, err_cnt = 0;
    mean_square_error = 0.0;
    entropy_cost = 0.0;

    for (const auto& item : test_data) {
        nu::Vector inputs, target;
        item->toVect(inputs);
        item->labelToTarget(target);

        net->setInputVector(inputs);
        net->feedForward();

        nu::Vector outputs;
        net->copyOutputVector(outputs);

        mean_square_error += nu::cf::calcMSE(outputs, target);
        entropy_cost += nu::cf::calcCrossEntropy(outputs, target);

        if (item->getLabel() != static_cast<int>(outputs.maxarg()))
            ++err_cnt;
        ++cnt;

#ifdef _WIN32
        if ((cnt % 100) == 0)
            item->paint(0, 0);
#endif
    }

    mean_square_error /= cnt;
    entropy_cost /= cnt;
    return static_cast<double>(err_cnt) / static_cast<double>(cnt);
}

// Evaluate MlpMatrixNN on the test set.
static double test_net_mat(std::unique_ptr<nu::MlpMatrixNN>& net,
    const TrainingData::data_t& test_data, double& mean_square_error, double& entropy_cost)
{
    size_t cnt = 0, err_cnt = 0;
    mean_square_error = 0.0;
    entropy_cost = 0.0;

    for (const auto& item : test_data) {
        nu::Vector nv_inputs, nv_target;
        item->toVect(nv_inputs);
        item->labelToTarget(nv_target);

        net->setInputVector(nv_inputs.to_stdvec());
        net->feedForward();

        std::vector<double> out_vec;
        net->copyOutputVector(out_vec);

        const nu::Vector outputs(out_vec);
        mean_square_error += nu::cf::calcMSE(outputs, nv_target);
        entropy_cost += nu::cf::calcCrossEntropy(outputs, nv_target);

        const int predicted
            = static_cast<int>(std::max_element(out_vec.begin(), out_vec.end()) - out_vec.begin());
        if (item->getLabel() != predicted)
            ++err_cnt;
        ++cnt;

#ifdef _WIN32
        if ((cnt % 100) == 0)
            item->paint(0, 0);
#endif
    }

    mean_square_error /= cnt;
    entropy_cost /= cnt;
    return static_cast<double>(err_cnt) / static_cast<double>(cnt);
}

// ── Save (MlpNN only) ─────────────────────────────────────────────────────────

static bool save_the_net(const std::string& filename, nu::MlpNN& net)
{
    if (filename.empty())
        return true;
    std::ofstream nf(filename);
    if (!nf.is_open()) {
        std::cerr << "Cannot open '" << filename << "'\n";
        return false;
    }
    net.toJson(nf);
    return true;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string files_path;
    std::string load_file_name;
    std::string save_file_name;
    bool skip_training = false;
    double learningRate = NET_LEARNING_RATE;
    double momentum = NET_MOMENTUM;
    int epoch_cnt = TRAINING_EPOCH_NUMBER;
    bool use_ce = false;
    bool use_matrix = false;
    size_t batch_size = 1;
    bool change_lr = false;
    bool change_m = false;

    std::vector<size_t> hidden_layer;
    nu::Activation hidden_activation = nu::Activation::Sigmoid;

    if (argc > 1) {
        if (!process_cl(argc, argv, files_path, load_file_name, save_file_name, skip_training,
                learningRate, change_lr, momentum, change_m, epoch_cnt, hidden_layer, use_ce,
                hidden_activation, use_matrix, batch_size)) {
            usage(argv[0]);
            return 1;
        }
    }

    if (hidden_layer.empty())
        hidden_layer.push_back(HIDDEN_LAYER_SIZE);

    if (batch_size > 1 && !use_matrix) {
        std::cerr << "Warning: --batch requires --matrix; ignoring --batch.\n";
        batch_size = 1;
    }
    if (use_matrix && (!load_file_name.empty() || !save_file_name.empty()))
        std::cerr << "Notice: --load/--save are not supported in --matrix mode; ignoring.\n";

#ifdef _WIN32
    ::system("cls");
#else
    int dummy = ::system("clear");
    (void)dummy;
#endif

    std::cout << "\n\n\n\n\n";

    for (int i = 0; const auto& hl : hidden_layer)
        std::cout << "NN hidden neurons L" << ++i << "       : " << hl << "\n";

    std::cout << "Hidden activation          : " << nu::act::name(hidden_activation) << "\n"
              << "Cost function              : " << (use_ce ? "cross-entropy" : "MSE") << "\n"
              << "Net Learning rate  ( LR )  : " << learningRate << "\n"
              << "Net Momentum       ( M )   : " << momentum << "\n"
              << "Backend                    : " << (use_matrix ? "MlpMatrixNN (Eigen)" : "MlpNN")
              << "\n";
    if (use_matrix)
        std::cout << "Mini-batch size            : " << batch_size
                  << (batch_size == 1 ? " (online SGD)" : "") << "\n";

    try {
        const std::string training_labels_fn = files_path + config_training_labels_fn;
        const std::string training_images_fn = files_path + config_training_images_fn;
        const std::string testing_labels_fn = files_path + config_test_labels_fn;
        const std::string testing_images_fn = files_path + config_test_image_fn;

        std::cout << "Training labels : " << training_labels_fn << "\n"
                  << "Training images : " << training_images_fn << "\n";

        TrainingData test_set(testing_labels_fn, testing_images_fn);
        [[maybe_unused]] auto n_of_test_items = test_set.load();
        const auto& test_data = test_set.data();

        // ── MlpNN path ────────────────────────────────────────────────────────

        if (!use_matrix) {
            std::unique_ptr<nu::MlpNN> net;

            if (!skip_training) {
                TrainingData trainingSet(training_labels_fn, training_images_fn);
                [[maybe_unused]] auto n = trainingSet.load();
                const auto& data = trainingSet.data();
                assert(!data.empty());

                std::cout << "Test labels file: " << testing_labels_fn << "\n"
                          << "Test images file: " << testing_images_fn << "\n";

                const auto input_size = data.front()->get_dx() * data.front()->get_dy();
                const auto cf = use_ce ? nu::CostFunction::CrossEntropy : nu::CostFunction::MSE;

                std::vector<nu::MlpNN::LayerConfig> layers;
                layers.push_back(nu::MlpNN::LayerConfig{ input_size });
                for (auto hl : hidden_layer)
                    layers.push_back({ hl, hidden_activation });
                layers.push_back({ OUTPUT_LAYER_SIZE, nu::Activation::Sigmoid });

                net = std::make_unique<nu::MlpNN>(layers, learningRate, momentum, cf);

                double prev_loss = -1.0;
                double best_ber = 100.0;
                int best_epoch = 0;
                std::cout << "\n";

                for (int epoch = 0; epoch < epoch_cnt; ++epoch) {
                    locate(1);
                    std::cout << "Learning epoch " << epoch + 1 << " of " << epoch_cnt
                              << " ( LR = " << net->getLearningRate()
                              << ", M = " << net->getMomentum() << " )\n\n";

                    size_t cnt = 0;
                    trainingSet.reshuffle();
                    const auto t0 = std::chrono::steady_clock::now();

                    for (const auto& item : trainingSet.data()) {
                        nu::Vector inputs, target;
                        item->toVect(inputs);
                        item->labelToTarget(target);
                        net->setInputVector(inputs);
                        net->backPropagate(target);
                        ++cnt;
                        if (cnt % 120 == 0) {
                            locate(1);
                            std::cout << "Completed "
                                      << (double(cnt) / trainingSet.data().size()) * 100.0
                                      << "%   \n";
#ifdef _WIN32
                            if (cnt % 600)
                                item->paint(0, 0);
#endif
                        }
                    }

                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                                                .count();

                    double epochMSE{}, epochCE{};
                    const auto err_rate = test_net(net, test_data, epochMSE, epochCE);
                    const double delta_loss = (prev_loss >= 0.0) ? (epochMSE - prev_loss) : 0.0;
                    prev_loss = epochMSE;

                    std::cout << "Error rate   : " << err_rate * 100.0 << "%     \n"
                              << "MSE          : " << epochMSE * 100.0 << "%     \n"
                              << "Cross-entropy: " << epochCE * 100.0 << "%     \n"
                              << "dLoss/depoch : " << std::showpos << delta_loss * 100.0
                              << std::noshowpos << "%     \n"
                              << "Success rate : " << (1.0 - err_rate) * 100.0 << "%    \n"
                              << "Epoch time   : " << elapsed_ms << " ms  \n"
                              << "Throughput   : "
                              << (elapsed_ms > 0 ? static_cast<long long>(cnt) * 1000LL / elapsed_ms
                                                 : 0LL)
                              << " samples/s  \n";

                    if (err_rate < best_ber) {
                        best_ber = err_rate;
                        best_epoch = epoch;
                        save_the_net(save_file_name, *net);
                    }

                    std::cout << "BER          : " << best_ber * 100.0 << "%    \n"
                              << "Epoch BER    : " << best_epoch + 1 << "    \n\n";
                }
            }

            if (!load_file_name.empty()) {
                std::ifstream nf(load_file_name);
                if (!nf.is_open()) {
                    std::cerr << "Cannot open '" << load_file_name << "'\n";
                    return 1;
                }
                net = std::make_unique<nu::MlpNN>();
                if (net)
                    net->loadJson(nf);
            }

            if (!net) {
                std::cerr << "Error: net not initialized — check parameters and retry.\n";
                return 1;
            }
            if (change_lr)
                net->setLearningRate(learningRate);
            if (change_m)
                net->setMomentum(momentum);
        }

        // ── MlpMatrixNN path ──────────────────────────────────────────────────

        else {
            std::unique_ptr<nu::MlpMatrixNN> net;

            if (!skip_training) {
                TrainingData trainingSet(training_labels_fn, training_images_fn);
                [[maybe_unused]] auto n = trainingSet.load();
                const auto& data = trainingSet.data();
                assert(!data.empty());

                std::cout << "Test labels file: " << testing_labels_fn << "\n"
                          << "Test images file: " << testing_images_fn << "\n";

                const auto input_size = data.front()->get_dx() * data.front()->get_dy();
                const auto cf = use_ce ? nu::CostFunction::CrossEntropy : nu::CostFunction::MSE;

                std::vector<nu::MlpMatrixNN::LayerConfig> layers;
                layers.push_back(nu::MlpMatrixNN::LayerConfig{ input_size });
                for (auto hl : hidden_layer)
                    layers.push_back({ hl, hidden_activation });
                layers.push_back({ OUTPUT_LAYER_SIZE, nu::Activation::Sigmoid });

                net = std::make_unique<nu::MlpMatrixNN>(layers, learningRate, momentum, cf);

                double prev_loss = -1.0;
                double best_ber = 100.0;
                int best_epoch = 0;
                std::cout << "\n";

                for (int epoch = 0; epoch < epoch_cnt; ++epoch) {
                    locate(1);
                    std::cout << "Learning epoch " << epoch + 1 << " of " << epoch_cnt
                              << " ( LR = " << net->getLearningRate()
                              << ", M = " << net->getMomentum() << " )\n\n";

                    size_t cnt = 0;
                    trainingSet.reshuffle();
                    const auto t0 = std::chrono::steady_clock::now();

                    if (batch_size <= 1) {
                        // Online SGD — one sample at a time
                        for (const auto& item : trainingSet.data()) {
                            nu::Vector nv_in, nv_tgt;
                            item->toVect(nv_in);
                            item->labelToTarget(nv_tgt);
                            net->setInputVector(nv_in.to_stdvec());
                            net->feedForward();
                            net->backPropagate(nv_tgt.to_stdvec());
                            ++cnt;
                            if (cnt % 120 == 0) {
                                locate(1);
                                std::cout << "Completed "
                                          << (double(cnt) / trainingSet.data().size()) * 100.0
                                          << "%   \n";
                            }
                        }
                    } else {
                        // Mini-batch SGD
                        std::vector<std::vector<double>> batch_in, batch_tgt;
                        batch_in.reserve(batch_size);
                        batch_tgt.reserve(batch_size);

                        for (const auto& item : trainingSet.data()) {
                            nu::Vector nv_in, nv_tgt;
                            item->toVect(nv_in);
                            item->labelToTarget(nv_tgt);
                            batch_in.push_back(nv_in.to_stdvec());
                            batch_tgt.push_back(nv_tgt.to_stdvec());

                            if (batch_in.size() == batch_size) {
                                net->trainBatch(batch_in, batch_tgt);
                                cnt += batch_size;
                                batch_in.clear();
                                batch_tgt.clear();

                                if (cnt % (120 * batch_size) < batch_size) {
                                    locate(1);
                                    std::cout << "Completed "
                                              << (double(cnt) / trainingSet.data().size()) * 100.0
                                              << "%   \n";
                                }
                            }
                        }
                        // Flush remainder
                        if (!batch_in.empty()) {
                            net->trainBatch(batch_in, batch_tgt);
                            cnt += batch_in.size();
                        }
                    }

                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                                                .count();

                    double epochMSE{}, epochCE{};
                    const auto err_rate = test_net_mat(net, test_data, epochMSE, epochCE);
                    const double delta_loss = (prev_loss >= 0.0) ? (epochMSE - prev_loss) : 0.0;
                    prev_loss = epochMSE;

                    std::cout << "Error rate   : " << err_rate * 100.0 << "%     \n"
                              << "MSE          : " << epochMSE * 100.0 << "%     \n"
                              << "Cross-entropy: " << epochCE * 100.0 << "%     \n"
                              << "dLoss/depoch : " << std::showpos << delta_loss * 100.0
                              << std::noshowpos << "%     \n"
                              << "Success rate : " << (1.0 - err_rate) * 100.0 << "%    \n"
                              << "Epoch time   : " << elapsed_ms << " ms  \n"
                              << "Throughput   : "
                              << (elapsed_ms > 0 ? static_cast<long long>(cnt) * 1000LL / elapsed_ms
                                                 : 0LL)
                              << " samples/s  \n";

                    if (err_rate < best_ber) {
                        best_ber = err_rate;
                        best_epoch = epoch;
                    }

                    std::cout << "BER          : " << best_ber * 100.0 << "%    \n"
                              << "Epoch BER    : " << best_epoch + 1 << "    \n\n";
                }
            }

            if (!net) {
                std::cerr << "Error: net not initialized — check parameters and retry.\n";
                return 1;
            }
        }

    } catch (const TrainingData::LabelsFileNotFoundException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::ImagesFileNotFoundException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::LabelsFileReadErrorException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::ImagesFileReadErrorException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::LabelsFileWrongMagicException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::ImagesFileWrongMagicException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const TrainingData::NumberOfItemsMismatchException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const nu::MlpMatrixNN::InvalidCostFunctionCombinationException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const nu::MlpNN::InvalidCostFunctionCombinationException& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}
