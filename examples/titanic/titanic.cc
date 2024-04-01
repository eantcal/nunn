//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "titanic.h"

#include <algorithm>
#include <iostream>
#include <limits> // For std::numeric_limits
#include <random>
#include <set>
#include <string>
#include <vector>

void Passenger::processNew(NN& nn)
{
    auto readIntWithPrompt = [](const std::string& prompt, int min, int max) -> int {
        int value;
        do {
            std::cout << prompt;
            if (!(std::cin >> value) || value < min || value > max) {
                std::cout << "Invalid input. Please try again.\n";
                std::cin.clear(); // Clear error flags
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard the line
            } else {
                break; // Valid input
            }
        } while (true);
        return value;
    };

    age = std::clamp(readIntWithPrompt("Your age        : ", 0, 80), 0, 80);
    pclass = readIntWithPrompt("Class (1, 2, 3)  : ", 1, 3);
    gender = readIntWithPrompt("Gender (0-M, 1-F): ", 0, 1);
    sibsp = std::clamp(readIntWithPrompt("Siblings/Spouse : ", 0, 10), 0, 10);
    parch = std::clamp(readIntWithPrompt("Parents/Children: ", 0, 10), 0, 10);

    fare = pclass == 1 ? 150 : (pclass == 2 ? 30 : 10);

    NN::FpVector input = getInputVector();
    NN::FpVector output { 0 };

    nn.setInputVector(input);
    nn.feedForward();
    nn.copyOutputVector(output);

    std::cout << "Surviving chance: " << output[0] * 100 << "%\n\n";

    // Prepare for next input
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard the rest of the line
    std::cin.clear(); // Clear error flags
}

void Passenger::find(const Passenger* db, const std::string& searchFor, NN& nn)
{
    for (size_t i = 0; db[i].valid(); ++i) {
        const std::string& name = db[i].name;

        if (name.find(searchFor) != std::string::npos) {
            std::cout << "  " << name << std::endl
                      << "  Age                            : " << db[i].age << std::endl
                      << "  Class                          : " << db[i].pclass << std::endl
                      << "  # of siblings / spouses aboard : " << db[i].sibsp << std::endl
                      << "  # of parents / children aboard : " << db[i].parch << std::endl
                      << "  Ticket Fare                    : " << db[i].fare << std::endl
                      << "  Survived:                      : " << (db[i].survived ? "Yes" : "No") << std::endl;

            NN::FpVector input = db[i].getInputVector();
            NN::FpVector output { 0 };

            nn.setInputVector(input);
            nn.feedForward();
            nn.copyOutputVector(output);

            std::cout << "  Survived prediction:           : " << output[0] * 100 << "%" << std::endl
                      << std::endl;
        }
    }
}

// Assuming definitions for TrainingSet and TestSet types
// as well as a valid() method for Passenger
void Passenger::populateDataSet(const Passenger* db, TrainingSet& trainingSet, TestSet& testSet, double trainingSetRate)
{
    // Calculate total number of valid entries in the database
    size_t dataSetSize = 0;
    while (db[dataSetSize].valid()) {
        ++dataSetSize;
    }

    // Create a vector of indices to represent the database entries
    std::vector<size_t> indices(dataSetSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0 to dataSetSize - 1

    // Shuffle the indices to randomly order the database entries
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Determine the split index for training and test sets based on the provided rate
    size_t splitIndex = static_cast<size_t>(dataSetSize * trainingSetRate);

    // Populate the training set
    for (size_t i = 0; i < splitIndex; ++i) {
        const Passenger& passenger = db[indices[i]];
        trainingSet[passenger.getInputVector()] = passenger.getOutputVector();
    }

    // Populate the test set
    for (size_t i = splitIndex; i < indices.size(); ++i) {
        const Passenger& passenger = db[indices[i]];
        testSet[passenger.getInputVector()] = passenger.getOutputVector();
    }
}

void testNN(const TestSet& testSet, NN& nn)
{
    size_t sampleCount = 0;
    size_t errorCount = 0;

    for (const auto& [input, expectedOutput] : testSet) {
        NN::FpVector predictedOutput;

        nn.setInputVector(input);
        nn.feedForward();
        nn.copyOutputVector(predictedOutput);

        // Interpret NN output as binary classification with a threshold of 0.5
        double prediction = predictedOutput[0] >= 0.5 ? 1.0 : 0.0;

        // Increment counters based on the prediction accuracy
        ++sampleCount;
        if (expectedOutput[0] != prediction) {
            ++errorCount;
        }
    }

    // Calculate success rate
    double successRate = (1.0 - static_cast<double>(errorCount) / sampleCount) * 100.0;

    // Improved output formatting
    std::cout << "Test Results:\n"
              << "Error Count: " << errorCount << " out of " << sampleCount << " samples.\n"
              << "Success Rate: " << successRate << "%\n";
}

static void printDivider(size_t length = 80)
{
    std::cout << std::string(length, '-') << '\n';
}

void initializeNetwork(NN& nn)
{
    NN::Topology topology = { 6, 6, 1 }; // Example topology
    nn = NN(topology, 0.10, 0.9); // Assuming NN constructor takes topology, learning rate, and momentum
    std::cout << "Network initialized with learning rate = 0.10 and momentum = 0.9\n";
}

void trainNetwork(NN& nn, const TrainingSet& trainingSet)
{
    Trainer trainer(nn, 5000); // Assuming a Trainer class exists
    auto errorCostFunction = [](NN& net, const NN::FpVector& target) { return net.calcMSE(target); };

    trainer.runTraining(trainingSet, errorCostFunction,
        [](NN&, const NN::FpVector&, const NN::FpVector&, size_t epoch, size_t sample, double err) {
            if (epoch % 1000 == 0 && sample == 0) {
                std::cout << "Training progress: epoch " << epoch << ", error = " << err << '\n';
            }
            return false; // Continue training
        });

    std::cout << "Network training completed.\n";
}

void testNetwork(NN& nn, const TestSet& testSet)
{
    size_t totalSamples = 0, correctPredictions = 0;
    for (const auto& [input, expectedOutput] : testSet) {
        nn.setInputVector(input);
        nn.feedForward();
        NN::FpVector output;
        nn.copyOutputVector(output);
        bool prediction = output[0] >= 0.5;
        if (prediction == expectedOutput[0])
            ++correctPredictions;
        ++totalSamples;
    }
    double successRate = (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    std::cout << "Test Success Rate: " << successRate << "%\n";
}

void interactiveSession(NN& nn)
{
    std::string userInput;
    while (true) {
        std::cout << "Enter 'new' for a new prediction, 'quit' to exit: ";
        std::getline(std::cin, userInput);
        if (userInput == "quit")
            break;
        if (userInput == "new") {
            Passenger newPassenger;
            newPassenger.processNew(nn);
        } else {
            Passenger::find(titanicDB, userInput, nn);
        }
    }
}

int main()
{
    printDivider();

    std::cout << "RMS Titanic, the British passenger liner that sank in the North Atlantic Ocean\n"
                 "on April 15, 1912, during her maiden voyage, after colliding with an iceberg.\n"
                 "Of the 2224 passengers and crew aboard, 1502 died. A database of 1046 passengers\n"
                 "has been created, classified with features like gender, age, and survival status.\n"
                 "This dataset is divided into a training set of 946 passengers and a test set of\n"
                 "100 passengers to measure the accuracy of a trained Neural Network.\n";

    printDivider();

    NN nn;
    TrainingSet trainingSet;
    TestSet testSet;

    initializeNetwork(nn);
    Passenger::populateDataSet(titanicDB, trainingSet, testSet, 0.905);

    std::cout << "Before training: ";
    testNetwork(nn, testSet);

    trainNetwork(nn, trainingSet);

    std::cout << "After training: ";
    testNetwork(nn, testSet);

    interactiveSession(nn);

    return 0;
}