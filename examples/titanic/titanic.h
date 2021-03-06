#ifndef __TITANIC_H__
#define __TITANIC_H__
//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"
#include "nu_stepf.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>


/* -------------------------------------------------------------------------- */

using DataSet = std::map<std::vector<double>, std::vector<double>>;
using TrainingSet = DataSet;
using TestSet = DataSet;
using NN = nu::MlpNN;
using Trainer = nu::MlpNNTrainer;



/* -------------------------------------------------------------------------- */

struct Passenger {
    // pclass: Ticket class
    // 1 == 1st - Upper
    // 2 == 2nd - Middle
    // 3 == 3rd - Lower
    double pclass = 0;

    // Passenger name
    const char * name = nullptr;

    // gender: Male = 0, Female = 1
    double gender = 0;
    
    // age: Age is fractional if less than 1. 
    // If the age is estimated, is it in the form of xx.5
    double age = 0;

    // sibsp: The dataset defines family relations in this way...
    // Sibling = brother, sister, stepbrother, stepsister
    // Spouse = husband, wife (mistresses and fiancés were ignored)
    double sibsp = 0;

    // parch: The dataset defines family relations in this way...
    // Parent = mother, father
    // Child = daughter, son, stepdaughter, stepson
    // Some children travelled only with a nanny, therefore parch=0 for them.
    double parch = 0;

    // Ticket cost
    double fare = 0;

    // 0 = No, 1 = Yes
    double survived = 0;

    // Return true if the Passenger data is valid
    bool valid() const noexcept {
        return pclass > 0;
    }

    // Get normalized input vector for current Passenger data
    // The input vector is of 6 double precision numbers 
    // Each of them in the range form 0 to 1.0
    std::vector<double> getInputVector() const {
        return {
            (pclass - 1) / 2.0, // 1st->0, 2nd->0.5, 3rd->1
            gender,             // 0 or 1
            age / 80.0,         // 80 age of older person in the DB
            sibsp / 10.0,       // no more than 10 in the DB
            parch / 10.0,       // no more than 10 in the DB
            fare / 512.3292 };  // 512.3292 is hiest fare
    }

    // Get normalized output vector for current Passenger data
    // Which is represented here as a single double precition number 0 or 1.0
    std::vector<double> getOutputVector() const {
        return { survived };
    }

    // Ask for new input data and display the survival chance given by
    // the nn neural network prediction
    void processNew(NN& nn);

    // Search by name in the database (db) using a given string (searchFor)
    // Display any matching Passenger data and related survival chance
    // computed by a given neural network (nn)
    static void find(const Passenger* db, const std::string& searchFor, NN & nn);

    // Genrate training and test data sets from a given Passenger database
    static void populateDataSet(
        const Passenger * db, 
        TrainingSet & trainingSet,
        TestSet & testSet,
        double trainingSetRate);
};



/* -------------------------------------------------------------------------- */

// Reference to the prepopulated Titanic Database
extern const Passenger titanicDB[];


/* -------------------------------------------------------------------------- */

// Test a NN against a given test set
static void test(const TestSet& testSet, NN& nn);


/* -------------------------------------------------------------------------- */

#endif // __TITANIC_H__

