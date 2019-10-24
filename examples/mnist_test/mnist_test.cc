//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

/*

This test is performed by using the MNIST data set, which contains 60K+10K
scanned images of handwritten digits with their correct classifications
The images are greyscale and 28 by 28 pixels in size.

A first part of 60,000 images are used as training data.
The second part of the MNIST data set is 10,000 images to be used as test data.
To make this a good test of performance, the test data was taken from a
different
set of people than the original training data.

The training input is treated as a 28x28=784-dimensional vector.
Each entry in the vector represents the grey value for a single pixel in the
image.
The corresponding desired output is a 10-dimensional vector.

See also http://yann.lecun.com/exdb/mnist/

*/


/* -------------------------------------------------------------------------- */


#include "mnist.h"
#include "nu_mlpnn.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>
#endif


/* -------------------------------------------------------------------------- */

using neural_net_t = nu::MlpNN;


/* -------------------------------------------------------------------------- */

const size_t HIDDEN_LAYER_SIZE = 300; // neurons
const size_t OUTPUT_LAYER_SIZE = 10;  // neurons
const double NET_LEARNING_RATE = 0.025;
const double NET_MOMENTUM = 0.50;

std::string TRAINING_LABELS_FN = "train-labels.idx1-ubyte";
std::string TRAINING_IMAGES_FN = "train-images.idx3-ubyte";
std::string TEST_LABELS_FN = "t10k-labels.idx1-ubyte";
std::string TEST_IMAGES_FN = "t10k-images.idx3-ubyte";

const int TRAINING_EPOCH_NUMBER = 100;


/* -------------------------------------------------------------------------- */

static bool process_cl(int argc, char* argv[], std::string& files_path,
                       std::string& load_file_name, std::string& save_file_name,
                       bool& skip_training, double& learningRate,
                       bool& change_lr, double& momentum, bool& change_m,
                       int& epoch, std::vector<size_t>& hidden_layer,
                       bool& use_cross_entropy)
{
    int pidx = 1;

    for (; pidx < argc; ++pidx) {
        std::string arg = argv[pidx];

        if ((arg == "--help" || arg == "-h")) {
            return false;
        }

        if ((arg == "--version" || arg == "-v")) {
            std::cout << "nunnlib MNIST Test 1.01 (c) antonino.calderone@gmail.com"
                      << std::endl;
            continue;
        }

        if ((arg == "--training_files_path" || arg == "-p") &&
            (pidx + 1) < argc) {
            files_path = argv[++pidx];

            if (!files_path.empty()) {
                if (files_path.c_str()[files_path.size() - 1] != '/')
                    files_path += "/";
            }

            continue;
        }

        if ((arg == "--training_imgsfn" || arg == "-tri") &&
            (pidx + 1) < argc) {
            TRAINING_IMAGES_FN = argv[++pidx];
            continue;
        }

        if ((arg == "--training_lblsfn" || arg == "-trl") &&
            (pidx + 1) < argc) {
            TRAINING_LABELS_FN = argv[++pidx];
            continue;
        }

        if ((arg == "--test_imgsfn" || arg == "-ti") && (pidx + 1) < argc) {
            TEST_IMAGES_FN = argv[++pidx];
            continue;
        }

        if ((arg == "--test_lblsfn" || arg == "-tl") && (pidx + 1) < argc) {
            TEST_LABELS_FN = argv[++pidx];
            continue;
        }


        if ((arg == "--skip_training" || arg == "-n")) {
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

        if ((arg == "--momentun" || arg == "-m") && (pidx + 1) < argc) {
            try {
                momentum = std::stod(argv[++pidx]);
                change_m = true;
            } catch (...) {
                return false;
            }
            continue;
        }

        if ((arg == "--use_cross_entropy" || arg == "-c")) {
            use_cross_entropy = true;
            continue;
        }

        if ((arg == "--epoch_num" || arg == "-e") && (pidx + 1) < argc) {
            try {
                epoch = std::stoi(argv[++pidx]);
            } catch (...) {
                return false;
            }
            continue;
        }

        if ((arg == "--hidden_layer" || arg == "-hl") && (pidx + 1) < argc) {
            try {
                hidden_layer.push_back(std::stoi(argv[++pidx]));
            } catch (...) {
                return false;
            }
            continue;
        }

        return false;
    }

    return true;
}


/* -------------------------------------------------------------------------- */

static int get_y_pos()
{
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info = { 0 };
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);

    return info.dwCursorPosition.Y;
#else
    return 0;
#endif
}

/* -------------------------------------------------------------------------- */

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


/* -------------------------------------------------------------------------- */

static void usage(const char* appname)
{
    std::cerr
      << "Usage:" << std::endl
      << appname << std::endl
      << "\t[--version|-v] " << std::endl
      << "\t[--help|-h] " << std::endl
      << "\t[--training_files_path|-p <path>] " << std::endl
      << "\t[--training_imgsfn|-tri <filename>] (default " << TRAINING_IMAGES_FN
      << ")" << std::endl
      << "\t[--training_lblsfn|-trl <filename>] (default " << TRAINING_LABELS_FN
      << ")" << std::endl
      << "\t[--test_imgsfn|-ti <filename>] (default " << TEST_IMAGES_FN << ")"
      << std::endl
      << "\t[--test_lblsfn|-tl <filename>] (default " << TEST_IMAGES_FN << ")"
      << std::endl
      << "\t[--save|-s <net_description_file_name>] " << std::endl
      << "\t[--load|-l <net_description_file_name>] " << std::endl
      << "\t[--skip_training|-n] " << std::endl
      << "\t[--use_cross_entropy|-c] " << std::endl
      << "\t[--learningRate|-r <rate>] " << std::endl
      << "\t[--momentum|-m <value>] " << std::endl
      << "\t[--epoch_cnt|-e <count>] " << std::endl
      << "\t[[--hidden_layer|-hl <size> [--hidden_layer|--hl <size] ... ]  "
      << std::endl
      << std::endl
      << "Where:" << std::endl
      << "--version or -v " << std::endl
      << "\tshows the program version" << std::endl
      << "--help or -h " << std::endl
      << "\tgenerates just this 'Usage' text " << std::endl
      << "--training_files_path or -p " << std::endl
      << "\tset training/test files set path" << std::endl
      << "--training_imgsfn or -tri " << std::endl
      << "\tset training images file name" << std::endl
      << "--training_lblsfn or -trl " << std::endl
      << "\tset training labels file name" << std::endl
      << "--test_imgsfn or -ti " << std::endl
      << "\tset test images file name" << std::endl
      << "--test_lblsfn or -tl " << std::endl
      << "\tset test labels file name" << std::endl
      << "--save or -s" << std::endl
      << "\tsave net data to file" << std::endl
      << "--load or -l" << std::endl
      << "\tload net data from file" << std::endl
      << "--skip_training or -n" << std::endl
      << "\tskip net training" << std::endl
      << "--use_cross_entropy or -c" << std::endl
      << "\tuse the cross entropy cost function instead of MSE" << std::endl
      << "--learningRate or -r" << std::endl
      << "\tset learning rate (default " << NET_LEARNING_RATE << ")"
      << std::endl
      << "--momentum or -m" << std::endl
      << "\tset momentum (default " << NET_MOMENTUM << ")" << std::endl
      << "--epoch_cnt or -e" << std::endl
      << "\tset epoch count (default " << TRAINING_EPOCH_NUMBER << ")"
      << std::endl
      << "--hidden_layer or -hl" << std::endl
      << "\tset hidden layer size (n. of neurons, default " << HIDDEN_LAYER_SIZE
      << ")" << std::endl;
}


/* -------------------------------------------------------------------------- */

static double test_net(std::unique_ptr<neural_net_t>& net,
                       const TrainingData::data_t& test_data,
                       double& mean_square_error, double& entropy_cost)
{
    size_t cnt = 0;
    size_t err_cnt = 0;

    mean_square_error = 0.0;
    entropy_cost = 0.0;

    for (auto i = test_data.begin(); i != test_data.end(); ++i) {
        nu::Vector<double> inputs;
        (*i)->toVect(inputs);

        nu::Vector<double> target;
        (*i)->labelToTarget(target);

        net->setInputVector(inputs);
        net->feedForward();

        nu::Vector<double> outputs;
        net->copyOutputVector(outputs);

        mean_square_error += nu::cf::calcMSE(outputs, target);

        entropy_cost += nu::cf::calcCrossEntropy(outputs, target);

        if ((*i)->getLabel() != outputs.maxarg())
            ++err_cnt;

        ++cnt;

#ifdef _WIN32
        if ((cnt % 100) == 0)
            (*i)->paint(0, 0);
#endif
    }

    mean_square_error /= cnt;
    entropy_cost /= cnt;

    double err_rate = double(err_cnt) / double(cnt);

    return err_rate;
}


/* -------------------------------------------------------------------------- */

bool save_the_net(const std::string& filename, neural_net_t& net)
{
    // Save the net status if needed //

    if (!filename.empty()) {
        std::stringstream ss;
        ss << net;

        std::ofstream nf(filename);
        if (nf.is_open()) {
            nf << ss.str() << std::endl;
            nf.close();
        } else {
            std::cerr << "Cannot open '" << filename << "'" << std::endl;
            return false;
        }
    }

    return true;
}


/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    // Parse arguments

    std::string files_path;
    std::string load_file_name;
    std::string save_file_name;
    bool save_to_file = false;
    bool skip_training = false;
    double learningRate = NET_LEARNING_RATE;
    double momentum = NET_MOMENTUM;
    int epoch_cnt = TRAINING_EPOCH_NUMBER;
    bool use_ce = false;

    std::vector<size_t> hidden_layer;

    bool change_lr = false;
    bool change_m = false;

    if (argc > 1) {
        if (!process_cl(argc, argv, files_path, load_file_name, save_file_name,
                        skip_training, learningRate, change_lr, momentum,
                        change_m, epoch_cnt, hidden_layer, use_ce)) {
            usage(argv[0]);
            return 1;
        }
    }

    if (hidden_layer.empty())
        hidden_layer.push_back(HIDDEN_LAYER_SIZE);

#ifdef _WIN32
    ::system("cls");
#else
    int dummy = ::system("clear");
    (void)dummy;
#endif

    std::cout << std::endl << std::endl << std::endl << std::endl << std::endl;

    int hl_cnt = 0;
    for (const auto& hl : hidden_layer) {
        std::cout << "NN hidden neurons L" << hl_cnt + 1;

        std::cout << "       : " << hidden_layer[hl_cnt++] << std::endl;
    }

    std::cout << "Net Learning rate  ( LR )  : " << learningRate << std::endl;

    std::cout << "Net Momentum       ( M )   : " << momentum << std::endl;


    try {
        const std::string training_labels_fn = files_path + TRAINING_LABELS_FN;
        const std::string training_images_fn = files_path + TRAINING_IMAGES_FN;

        std::cout << "Training labels : " << training_labels_fn << std::endl;
        std::cout << "Training images : " << training_images_fn << std::endl;

        std::unique_ptr<neural_net_t> net;
        TrainingData trainingSet(training_labels_fn, training_images_fn);

        const std::string testing_labels_fn = files_path + TEST_LABELS_FN;
        const std::string testing_images_fn = files_path + TEST_IMAGES_FN;

        TrainingData test_set(testing_labels_fn, testing_images_fn);

        auto n_of_test_items = test_set.load();

        const auto& test_data = test_set.data();


        if (!skip_training) {
            // Start Training ... //

            auto n_of_items = trainingSet.load();
            const auto& data = trainingSet.data();

            assert(!data.empty());

            std::cout << "Test labels file: " << testing_labels_fn << std::endl;
            std::cout << "Test images file: " << testing_images_fn << std::endl;


            // Input size depens on number of pixels
            auto input_size =
              (*data.cbegin())->get_dx() * (*data.cbegin())->get_dy();

            // Set up the topology
            neural_net_t::Topology topology;

            topology.push_back(input_size);

            for (auto hl : hidden_layer)
                topology.push_back(hl);

            topology.push_back(OUTPUT_LAYER_SIZE);

            net = std::unique_ptr<neural_net_t>(
              new neural_net_t(topology, learningRate, momentum));
        }

        if (!load_file_name.empty()) {
            std::ifstream nf(load_file_name);
            std::stringstream ss;
            if (!nf.is_open()) {
                std::cerr << "Cannot open '" << load_file_name << "'"
                          << std::endl;
                return 1;
            }

            ss << nf.rdbuf();
            nf.close();

            net = std::unique_ptr<neural_net_t>(new neural_net_t);
            if (net)
                net->load(ss);
        }

        if (net == nullptr) {
            std::cerr
              << "Error: net not initialized... change parameters and retry"
              << std::endl;
            return 1;
        }

        if (change_lr)
            net->setLearningRate(learningRate);

        if (change_m)
            net->setMomentum(momentum);


        size_t cnt = 0;

        const int max_epoch_number = epoch_cnt;
        double best_performance = 100.0;
        int best_epoch = 0;

        if (!skip_training) {
            std::cout << std::endl;

            for (int epoch = 0; epoch < max_epoch_number; ++epoch) {
                locate(1);

                double calcMSE = 0.0;
                double calcCrossEntropy = 0.0;

                std::cout << "Learning epoch " << epoch + 1 << " of "
                          << max_epoch_number
                          << " ( LR = " << net->getLearningRate()
                          << ", M = " << net->getMomentum() << " )"
                          << std::endl
                          << std::endl;

                cnt = 0;
                trainingSet.reshuffle();
                const auto& data = trainingSet.data();

                for (auto i = data.begin(); i != data.end(); ++i) {
                    nu::Vector<double> inputs;

                    (*i)->toVect(inputs);

                    nu::Vector<double> target;
                    (*i)->labelToTarget(target);

                    net->setInputVector(inputs);
                    net->runBackPropagationAlgo(target);

                    ++cnt;

                    // Use cnt to show progress
                    if (cnt % 120 == 0) {
                        locate(1);
                        std::cout << "Completed "
                                  << (double(cnt) / data.size()) * 100.0
                                  << "%   " << std::endl;

#ifdef _WIN32
                        if (cnt % 600)
                            (*i)->paint(0, 0);
#endif
                    }
                }

                auto err_rate =
                  test_net(net, test_data, calcMSE, calcCrossEntropy);

                std::cout << "Error rate   : " << err_rate * 100.0 << "%     "
                          << std::endl;

                std::cout << "MS Error rate: " << calcMSE * 100.0
                          << "%     " << std::endl;

                std::cout << "Cross entropy: " << calcCrossEntropy * 100.0
                          << "%     " << std::endl;

                std::cout << "Success rate : " << (1.0 - err_rate) * 100.0
                          << "%    " << std::endl;


                if (err_rate < best_performance) {
                    best_performance = err_rate;
                    best_epoch = epoch;
                    save_the_net(save_file_name, *net);
                }

                std::cout << "BER          : " << best_performance * 100.0
                          << "%    " << std::endl;
                std::cout << "Epoch BER    : " << best_epoch + 1 << "    "
                          << std::endl
                          << std::endl;
            }
        }

    } catch (TrainingData::Exception e) {
        switch (e) {
            case TrainingData::Exception::imgs_file_not_found:
                std::cerr << "Images file not found";
                break;
            case TrainingData::Exception::imgs_file_read_error:
                std::cerr << "Error reading images file";
                break;
            case TrainingData::Exception::lbls_file_not_found:
                std::cerr << "Labels file not found";
                break;
            case TrainingData::Exception::lbls_file_read_error:
                std::cerr << "Error reading labels file";
                break;
            case TrainingData::Exception::imgs_file_wrong_magic:
                std::cerr << "Cannot recognize images file";
                break;
            case TrainingData::Exception::lbls_file_wrong_magic:
                std::cerr << "Cannot recognize labels file";
                break;
            case TrainingData::Exception::n_of_items_mismatch:
                std::cerr << "Images and labels count mismatch";
                break;
        }

        std::cerr << std::endl << "Error Code " << int(e) << std::endl;
        return 1;
    }

    return 0;
}
