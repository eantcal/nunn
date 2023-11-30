//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//


#include "titanic.h"


void Passenger ::processNew(NN& nn)
{
    std::cout << "Your age        : ";
    std::cin >> age;

    if (age < 0)
        age = 0;
    if (age > 80)
        age = 80;

    do {
        std::cout << "Class 1,2,3     : ";
        std::cin >> pclass;
    } while (!(pclass == 1 || pclass == 2 || pclass == 3));

    do {
        std::cout << "Gender 0-M 1-F  : ";
        std::cin >> gender;
    } while (!(gender == 0 || gender == 1));

    std::cout << "Sblings/Spouse  : ";
    std::cin >> sibsp;

    if (sibsp < 0)
        sibsp = 0;
    if (sibsp > 10)
        sibsp = 10;

    std::cout << "Parents/Children: ";
    std::cin >> parch;

    if (parch < 0)
        parch = 0;
    if (parch > 10)
        parch = 10;

    fare = pclass == 1 ? 150 : (pclass == 2 ? 30 : 10);

    NN::FpVector input { getInputVector() };
    NN::FpVector output { 0 };

    nn.setInputVector(input);
    nn.feedForward();
    nn.copyOutputVector(output);

    std::cout << "Surviving chance: " << output[0] * 100 << "%" << std::endl
              << std::endl;

    std::cin.clear();
    std::fflush(stdin);

    std::string tmp;
    std::getline(std::cin, tmp);
}


void Passenger ::find(const Passenger* db, const std::string& searchFor, NN& nn)
{
    size_t i = 0;

    while (db[i].valid()) {
        const std::string name = db[i].name;

        if (name.find(searchFor) != std::string::npos) {
            std::cout << "  " << name << std::endl;
            std::cout << "  Age                            " << db[i].age
                      << std::endl;
            std::cout << "  Class                          " << db[i].pclass
                      << std::endl;
            std::cout << "  # of siblings / spouses aboard " << db[i].sibsp
                      << std::endl;
            std::cout << "  # of parents / children aboard " << db[i].parch
                      << std::endl;
            std::cout << "  Ticket Fare                    " << db[i].fare
                      << std::endl;
            std::cout << "  Survived:                      "
                      << std::string(db[i].survived ? "Yes" : "No")
                      << std::endl;

            NN::FpVector input { db[i].getInputVector() };
            NN::FpVector output { 0 };

            nn.setInputVector(input);
            nn.feedForward();
            nn.copyOutputVector(output);

            std::cout << "  Survived prediction:           " << output[0] * 100
                      << "%" << std::endl
                      << std::endl;
        }
        ++i;
    }
}


// Function which can be used to genrate training and test sets
// from Passenger database
void Passenger ::populateDataSet(const Passenger* db,
    TrainingSet& trainingSet,
    TestSet& testSet,
    double trainingSetRate)
{
    const size_t dataSetSize = [db] {
        size_t i = 0;
        while (db[i++].valid()) {
        }
        return i;
    }();
    std::set<size_t> usedIndex;

    const size_t trainingSetSize = size_t(double(dataSetSize) * trainingSetRate);

    auto reshuffle = [&usedIndex, dataSetSize]() {
        size_t rndIndex = 0;

        do {
            rndIndex = rand() % dataSetSize;
        } while (usedIndex.find(rndIndex) != usedIndex.end());

        usedIndex.insert(rndIndex);
        return rndIndex;
    };

    // The data has been split into two groups: training set and test set
    // The training set is used to train the neural network.
    // The training set is built on “features” like passengers’ gender, age,
    // class, etc.
    //
    // The test set is used to see how well a trained NN performs on unseen
    // data. For each passenger in the test set, is used an NN trained to
    // predict whether or not they survived the sinking of the Titanic.

    for (size_t i = 0; i < trainingSetSize; ++i) {
        const size_t rndIndex = reshuffle();
        trainingSet[db[rndIndex].getInputVector()] = db[rndIndex].getOutputVector();
    }

    for (size_t i = trainingSetSize; i < dataSetSize; ++i) {
        const size_t rndIndex = reshuffle();
        testSet[db[rndIndex].getInputVector()] = db[rndIndex].getOutputVector();
    }
}


// Test a NN against a given test set

void test(const TestSet& testSet, NN& nn)
{
    NN::FpVector output { 0 };
    NN::FpVector input { 0 };
    size_t sampleCnt = 0;
    size_t errCnt = 0;

    for (const auto& sample : testSet) {
        input = sample.first;
        nn.setInputVector(input);
        nn.feedForward();
        nn.copyOutputVector(output);

        output[0] = output[0] >= 0.5 ? 1.0 : 0.0;

        ++sampleCnt;
        if (sample.second[0] != output[0])
            ++errCnt;
    }

    std::cout << "Test result: (err/samples)=" << errCnt << "/" << sampleCnt
              << " Success Rate (%)="
              << (1.0 - double(errCnt) / double(sampleCnt)) * 100.0
              << std::endl;
}


static void printDivider()
{
    for (size_t i = 0; i < 80; ++i)
        std::cout << "-";
    std::cout << std::endl;
}


// Working in progress...
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    printDivider();

    std::cout << " RMS Titanic was a British passenger liner that sank in the "
                 "North Atlantic "
              << std::endl
              << " Ocean On April 15, 1912, during her maiden voyage, "
              << std::endl
              << " after colliding with an iceberg, killing 1502 out of 2224 "
                 "passengers and crew."
              << std::endl
              << " We created a database of 1046 passengers classified using "
                 "“features” like"
              << std::endl
              << " gender, age, ticket class, fare, etc, plus a survival "
                 "status and divieded"
              << std::endl
              << " such database in two groups using a pseudo random algorithm:"
              << std::endl
              << " - 946 passengers in a training set, used for training a "
                 "Neural Network"
              << std::endl
              << " - 100 passengers in a test set used on the trained NN to "
                 "measure accurency of NN"
              << std::endl;

    printDivider();

    // Create training set and test set
    TrainingSet trainingSet;
    TestSet testSet;
    Passenger::populateDataSet(titanicDB, trainingSet, testSet, 0.905);

    NN::Topology topology = {
        6, // number of inputs
        6, // hidden layer
        1 // output
    };

    // Construct the network using topology, learning rate and momentum
    NN nn {
        topology,
        0.10, // learning rate
        0 // momentum
    };

    printDivider();
    std::cout << "UNTRAINED neural network ";
    test(testSet, nn);
    printDivider();

    Trainer trainer(nn, 5000);

    std::clog << "Training the NN ( Epochs =" << trainer.getEpochs() << " )";

    // Train the net
    trainer.runTraining<DataSet>(
        trainingSet,

        // Error cost function
        [](NN& net, const NN::FpVector& target) { return net.calcMSE(target); },

        // Progress callback
        [&trainingSet]([[maybe_unused]] NN& nn,
            [[maybe_unused]] const nu::Vector<double>& i,
            [[maybe_unused]] const nu::Vector<double>& t,
            size_t epoch,
            size_t sample,
            [[maybe_unused]] double err) {
            if (sample == trainingSet.size() - 1 && epoch % 1000 == 0)
                std::clog << ".";
            return false;
        });

    std::cout << " Done." << std::endl;

    printDivider();
    std::cout << "Trained neural network   ";
    test(testSet, nn);
    printDivider();

    do {
        std::string searchFor;
        std::cout << "Search for name (or new for a new unknown prediction, "
                     "quit -> for exiting): ";
        std::getline(std::cin, searchFor);
        printDivider();

        if (searchFor == "quit")
            return 0;

        if (searchFor == "new") {
            Passenger p;
            p.processNew(nn);
        } else
            Passenger::find(titanicDB, searchFor, nn);
    } while (1);

    return 0;
}
