//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_mlpnn.h"
#include "nu_stepf.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>

// DataSet: A mapping of input vectors to output vectors used in training and testing neural networks.
using DataSet = std::map<std::vector<double>, std::vector<double>>;
using TrainingSet = DataSet; // Represents a set of training data.
using TestSet = DataSet; // Represents a set of test data.
using NN = nu::MlpNN; // Shorthand for a Multi-Layer Perceptron Neural Network.
using Trainer = nu::MlpNNTrainer; // Shorthand for the MLP Neural Network Trainer.

// Represents data about a passenger, for use in machine learning models.
struct Passenger {
    // pclass: Ticket class (1 = 1st/Upper, 2 = 2nd/Middle, 3 = 3rd/Lower)
    double pclass = 0;

    // Passenger's name
    const char* name = nullptr;

    // gender: Coded as Male = 0, Female = 1
    double gender = 0;

    // age: Age of the passenger. Fractional if less than 1. If estimated, appears as xx.5
    double age = 0;

    // sibsp: Number of siblings/spouses aboard (siblings, step-siblings, spouses)
    double sibsp = 0;

    // parch: Number of parents/children aboard (parents, step-parents, children, step-children)
    // Note: children traveling with nannies are marked as parch=0.
    double parch = 0;

    // fare: Ticket cost
    double fare = 0;

    // survived: Survival status (0 = No, 1 = Yes)
    double survived = 0;

    // Checks if the passenger data is valid.
    bool valid() const noexcept { return pclass > 0; }

    // Constructs a normalized input vector from the passenger data.
    // Normalizes each data point to be in the range [0, 1.0].
    std::vector<double> getInputVector() const
    {
        constexpr double maxAge = 80.0;
        constexpr double maxSibSp = 10.0;
        constexpr double maxParch = 10.0;
        constexpr double maxFare = 512.3292;

        return {
            (pclass - 1) / 2.0, // Normalized ticket class
            gender, // 0 or 1, already normalized
            std::min(age / maxAge, 1.0), // Normalized and capped age
            std::min(sibsp / maxSibSp, 1.0), // Normalized and capped siblings/spouses count
            std::min(parch / maxParch, 1.0), // Normalized and capped parents/children count
            std::min(fare / maxFare, 1.0) // Normalized and capped fare
        };
    }

    // Constructs a normalized output vector from the passenger data.
    // Output vector contains a single value representing survival status.
    std::vector<double> getOutputVector() const { return { survived }; }

    // Process new input data and display predicted survival chances using the provided neural network.
    void processNew(NN& nn);

    // Searches for passengers by name in the given database and displays matched data with predicted survival chances.
    static void find(const Passenger* db, const std::string& searchFor, NN& nn);

    // Generates training and test datasets from a given passenger database.
    // Splits the database into training and test datasets based on the specified training set rate.
    static void populateDataSet(const Passenger* db,
        TrainingSet& trainingSet,
        TestSet& testSet,
        double trainingSetRate);
};

// Reference to a prepopulated database of Titanic passengers.
extern const Passenger titanicDB[];
