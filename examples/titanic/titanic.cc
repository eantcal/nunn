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

/* -------------------------------------------------------------------------- */
 

using TrainingSet = DataSet;
using TestSet = DataSet;


/* -------------------------------------------------------------------------- */

struct Passenger {
    double pclass = 0;
    const char * name = nullptr;
    double gender = 0;
    double age = 0;
    double sibsp = 0;
    double parch = 0;
    double fare = 0;
    double survived = 0;

    bool valid() const noexcept {
        return pclass > 0;
    }

    std::vector<double> getInputVector() const {
        return {
            (pclass - 1) / 2.0, // 1->0, 2->0.5, 3->1
            gender,
            age / 80.0,
            sibsp / 10.0,
            parch / 10.0,
            fare / 512.3292 };
    }

    std::vector<double> getOutputVector() const {
        return { survived };
    }

    static void populateDataSet(
        const Passenger * db, 
        TrainingSet & trainingSet,
        TrainingSet & testSet,
        double trainingSetRate) 
    {
        const size_t dataSetSize = [db] { size_t i = 0;  while(db[i++].valid()){} return i; }();
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
        // The training set is built on “features” like passengers’ gender, age, class, 
        // etc.
        //
        // The test set is used to see how well a trained NN performs on unseen data. 
        // For each passenger in the test set, is used an NN trained to predict whether 
        // or not they survived the sinking of the Titanic.

        for (size_t i = 0; i < trainingSetSize; ++i) {
            size_t rndIndex = reshuffle();
            trainingSet[db[rndIndex].getInputVector()] = db[rndIndex].getOutputVector();
        }

        for (size_t i = trainingSetSize; i < dataSetSize; ++i) {
            size_t rndIndex = reshuffle();
            testSet[db[rndIndex].getInputVector()] = db[rndIndex].getOutputVector();
        }
    }
};



/* -------------------------------------------------------------------------- */

extern const Passenger titanicDB[];


/* -------------------------------------------------------------------------- */

using neural_net_t = nu::mlp_neural_net_t;
using trainer_t = nu::mlp_nn_trainer_t;


static void test(const TestSet& testSet, neural_net_t& nn) {
    neural_net_t::rvector_t output{ 0 };
    neural_net_t::rvector_t input{ 0 };
    size_t sampleCnt = 0;
    size_t errCnt = 0;
    for (const auto & sample : testSet) {
        input = sample.first;
        nn.set_inputs(input);
        nn.feed_forward();
        nn.get_outputs(output);

        output[0] = output[0] >= 0.5 ? 1.0 : 0.0;

        ++sampleCnt;
        if (sample.second[0] != output[0]) {
            ++errCnt;
        }
    }

    std::cout
        << "Test (err/samples):" << errCnt << "/" << sampleCnt << " Success Rate (%):" 
        << (1.0-double(errCnt) / double(sampleCnt)) * 100.0
        << std::endl;
}

/* -------------------------------------------------------------------------- */

// Working in progress...
int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------
    // Create the training set
    TrainingSet trainingSet;
    TestSet testSet;
    Passenger::populateDataSet(titanicDB, trainingSet, testSet, 0.905);

    neural_net_t::topology_t topology = {
        6,  // number of inputs

        6,  // hidden layer

        1   // output
    };

    // Construct the network using topology, learning rate and momentum
    neural_net_t nn{
        topology,
        0.10,    // learning rate
        0        // momentum
    };

    std::cout << "Non trained network test" << std::endl;
    test(testSet, nn);

    const size_t EPOCHS = 5000;
    trainer_t trainer(nn, EPOCHS);

    std::cout 
        << std::endl
        << "Training Start ( Max epochs =" << trainer.get_epochs() << " )"
        << std::endl;

    // Called to print out training progress
    auto progress_cbk = [&trainingSet](neural_net_t& nn,
        const nu::vector_t<double>& i,
        const nu::vector_t<double>& t,
        size_t epoch, size_t sample, double err) 
    {
        if (sample == trainingSet.size() - 1 && epoch % 100 == 0) {
            std::clog << ".";
            //std::clog << "Epoch: "
            //    << epoch
            //    << " Err= " << err << std::endl;
        }

        return false;
    };

    auto err_cost_f = [](neural_net_t& net, const neural_net_t::rvector_t& target) {
        return net.mean_squared_error(target);
    };

    // Train the net
    trainer.run_training<DataSet>(
        trainingSet, 
        err_cost_f, 
        progress_cbk);

    std::cout 
        << std::endl
        << "Trained network test" << std::endl;

    test(testSet, nn);

    do {
        std::string searchFor;
        std::cout << "Search for name (or new for a new unknown prediction, quit -> for exiting): ";
        std::getline( std::cin, searchFor );
        std::cout << "--------------------" << std::endl;

        if (searchFor == "quit") {
            return 0;
        }

        if (searchFor == "new") {
            Passenger p; 
            std::cout << "Your age        : "; std::cin >> p.age;
            std::cout << "Class           : "; std::cin >> p.pclass;
            std::cout << "Gender 0-M 1-F  : "; std::cin >> p.gender;
            std::cout << "Sblings/Spouse  : "; std::cin >> p.sibsp;
            std::cout << "Parents/Children: "; std::cin >> p.parch;

            p.fare = p.pclass == 1 ? 150 : (p.pclass==2 ? 30 : 10);
            
            neural_net_t::rvector_t input{ p.getInputVector() };
            neural_net_t::rvector_t output{ 0 };

            nn.set_inputs(input);
            nn.feed_forward();
            nn.get_outputs(output);

            std::cout 
                << "Surviving chance      : " << output[0] * 100 << "%" 
                << std::endl << std::endl;

            std::cin.clear();
            std::fflush(stdin);
            std::getline( std::cin, searchFor );
        }
        else {
            size_t i = 0;
            auto * db = & titanicDB[0];
            while(db[i].valid()) {
                const std::string name = db[i].name;
                if (name.find(searchFor) != std::string::npos) {
                    std::cout << "Found " << name << std::endl;
                    std::cout << "  Age                            " << db[i].age << std::endl;
                    std::cout << "  Class                          " << db[i].pclass << std::endl;
                    std::cout << "  # of siblings / spouses aboard " << db[i].sibsp << std::endl;
                    std::cout << "  # of parents / children aboard " << db[i].parch << std::endl;
                    std::cout << "  Ticket Fare                    " << db[i].fare << std::endl;
                    std::cout << "  Survived:                      " << 
                        std::string(db[i].survived ? "Yes":"No") << std::endl;

                    neural_net_t::rvector_t input{ db[i].getInputVector() };
                    neural_net_t::rvector_t output{ 0 };

                    nn.set_inputs(input);
                    nn.feed_forward();
                    nn.get_outputs(output);

                    std::cout 
                        << "  Survived prediction:           " << output[0] * 100 << "%" 
                        << std::endl << std::endl;
                }
                ++i;
            }
        }
    }
    while (1);

    return 0;
}


/* -------------------------------------------------------------------------- */

enum { Male = 0, Female = 1 };

/* -------------------------------------------------------------------------- */
// pclass,name,gender,age,sibsp,parch,fare,survived
const Passenger titanicDB[] = {
   { 1, "Crosby, Capt. Edward Gifford",Male,70,1,1,71.0000,0},
   { 2, "Mitchell, Mr. Henry Michael",Male,70,0,0,10.5000,0},
   { 1, "Straus, Mr. Isidor",Male,67,1,0,221.7792,0},
   { 2, "Wheadon, Mr. Edward H",Male,66,0,0,10.5000,0},
   { 1, "Ostby, Mr. Engelhart Cornelius",Male,65,0,1,61.9792,0},
   { 1, "Millet, Mr. Francis Davis",Male,65,0,0,26.5500,0},
   { 3, "Duane, Mr. Frank",Male,65,0,0,7.7500,0},
   { 2, "Myles, Mr. Thomas Francis",Male,62,0,0,9.6875,0},
   { 1, "Ryerson, Mr. Arthur Larned",Male,61,1,3,262.3750,0},
   { 1, "Van der hoef, Mr. Wyckoff",Male,61,0,0,33.5000,0},
   { 1, "Sutton, Mr. Frederick",Male,61,0,0,32.3208,0},
   { 2, "Lingane, Mr. John",Male,61,0,0,12.3500,0},
   { 3, "Nysveen, Mr. Johan Hansen",Male,61,0,0,6.2375,0},
   { 3, "Storey, Mr. Thomas",Male,60.5,0,0,0,0},
   { 1, "Barkworth, Mr. Algernon Henry Wilson",Male,80,0,0,30.0000,1 },
   { 1, "Cavendish, Mrs. Tyrell William (Julia Florence Siegel)",Female,76,1,0,78.8500,1 },
   { 3, "Svensson, Mr. Johan",Male,74,0,0,7.7750,0},
   { 1, "Artagaveytia, Mr. Ramon",Male,71,0,0,49.5042,0},
   { 1, "Goldschmidt, Mr. George B",Male,71,0,0,34.6542,0},
   { 3, "Connors, Mr. Patrick",Male,70.5,0,0,7.7500,0},
   { 1, "Fortune, Mrs. Mark (Mary McDougald)",Female,60,1,4,263.0000,1},
   { 1, "Frolicher-Stehli, Mr. Maxmillian",Male,60,1,1,79.2000,1},
   { 1, "Bucknell, Mrs. William Robert (Emma Eliza Ward)",Female,60,0,0,76.2917,1},
   { 1, "Warren, Mrs. Frank Manley (Anna Sophia Atkinson)",Female,60,1,0,75.2500,1},
   { 2, "Brown, Mr. Thomas William Solomon",Male,60,1,1,39.0000,0},
   { 1, "Weir, Col. John",Male,60,0,0,26.5500,0},
   { 2, "Howard, Mrs. Benjamin (Ellen Truelove Arman)",Female,60,1,0,26.0000,0},
   { 1, "Brown, Mrs. John Murray (Caroline Lane Lamson)",Female,59,2,0,51.4792,1},
   { 2, "Sjostedt, Mr. Ernst Adolf",Male,59,0,0,13.5000,0},
   { 3, "Coxon, Mr. Daniel",Male,59,0,0,7.2500,0},
   { 1, "Cardeza, Mrs. James Warburton Martinez (Charlotte Wardle Drake)",Female,58,0,1,512.3292,1},
   { 1, "Graham, Mrs. William Thompson (Edith Junkins)",Female,58,0,1,153.4625,1},
   { 1, "Lurette, Miss. Elise",Female,58,0,0,146.5208,1},
   { 1, "Newell, Mr. Arthur Webster",Male,58,0,2,113.2750,0},
   { 1, "Kent, Mr. Edward Austin",Male,58,0,0,29.7000,0},
   { 1, "Bonnell, Miss. Elizabeth",Female,58,0,0,26.5500,1},
   { 1, "Wick, Mr. George Dennick",Male,57,1,1,164.8667,0},
   { 1, "Spencer, Mr. William Augustus",Male,57,1,0,146.5208,0},
   { 2, "Ashby, Mr. John",Male,57,0,0,13.0000,0},
   { 2, "Kirkland, Rev. Charles Leonard",Male,57,0,0,12.3500,0},
   { 2, "Mack, Mrs. (Mary)",Female,57,0,0,10.5000,0},
   { 1, "Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)",Female,56,0,1,83.1583,1},
   { 1, "Simonius-Blumer, Col. Oberst Alfons",Male,56,0,0,35.5000,1},
   { 1, "Smith, Mr. James Clinch",Male,56,0,0,30.6958,0},
   { 1, "Smart, Mr. John Montgomery",Male,56,0,0,26.5500,0},
   { 3, "Meo, Mr. Alfonzo",Male,55.5,0,0,8.0500,0},
   { 1, "White, Mrs. John Stuart (Ella Holmes)",Female,55,0,0,135.6333,1},
   { 1, "Hays, Mr. Charles Melville",Male,55,1,1,93.5000,0},
   { 1, "Fortune, Mr. Mark",Male,64,1,4,263.0000,0},
   { 1, "Compton, Mrs. Alexander Taylor (Mary Eliza Ingersoll)",Female,64,0,2,83.1583,1},
   { 1, "Warren, Mr. Frank Manley",Male,64,1,0,75.2500,0},
   { 1, "Crosby, Mrs. Edward Gifford (Catherine Elizabeth Halstead)",Female,64,1,1,26.5500,1},
   { 1, "Nicholson, Mr. Arthur Ernest",Male,64,0,0,26.0000,0},
   { 1, "Straus, Mrs. Isidor (Rosalie Ida Blun)",Female,63,1,0,221.7792,0},
   { 1, "Andrews, Miss. Kornelia Theodosia",Female,63,1,0,77.9583,1},
   { 2, "Howard, Mr. Benjamin",Male,63,1,0,26.0000,0},
   { 3, "Turkula, Mrs. (Hedwig)",Female,63,0,0,9.5875,1},
   { 1, "Stone, Mrs. George Nelson (Martha Evelyn)",Female,62,0,0,80.0000,1},
   { 1, "Stead, Mr. William Thomas",Male,62,0,0,26.5500,0},
   { 1, "Wright, Mr. George",Male,62,0,0,26.5500,0},
   { 2, "Harris, Mr. George",Male,62,0,0,10.5000,1},
   { 1, "Rothschild, Mr. Martin",Male,55,1,0,59.4000,0},
   { 1, "Hipkins, Mr. William Edward",Male,55,0,0,50.0000,0},
   { 1, "Molson, Mr. Harry Markland",Male,55,0,0,30.5000,0},
   { 1, "Lindstrom, Mrs. Carl Johan (Sigrid Posse)",Female,55,0,0,27.7208,1},
   { 1, "Cornell, Mrs. Robert Clifford (Malvina Helen Lamson)",Female,55,2,0,25.7000,1},
   { 2, "Hewlett, Mrs. (Mary D Kingcome) ",Female,55,0,0,16.0000,1},
   { 1, "Dodge, Mrs. Washington (Ruth Vidaver)",Female,54,1,1,81.8583,1},
   { 1, "Eustis, Miss. Elizabeth Mussey",Female,54,1,0,78.2667,1},
   { 1, "White, Mr. Percival Wayland",Male,54,0,1,77.2875,0},
   { 1, "Rothschild, Mrs. Martin (Elizabeth L. Barrett)",Female,54,1,0,59.4000,1},
   { 1, "Stengel, Mr. Charles Emil Henry",Male,54,1,0,55.4417,1},
   { 1, "McCarthy, Mr. Timothy J",Male,54,0,0,51.8625,0},
   { 2, "Chapman, Mr. Charles Henry",Male,52,0,0,13.5000,0},
   { 2, "Greenberg, Mr. Samuel",Male,52,0,0,13.0000,0},
   { 1, "Hogeboom, Mrs. John C (Anna Andrews)",Female,51,1,0,77.9583,1},
   { 1, "Williams, Mr. Charles Duane",Male,51,0,1,61.3792,0},
   { 1, "Lines, Mrs. Ernest H (Elizabeth Lindsey James)",Female,51,0,1,39.4000,1},
   { 1, "Daly, Mr. Peter Denis ",Male,51,0,0,26.5500,1},
   { 2, "Bateman, Rev. Robert James",Male,51,0,0,12.5250,0},
   { 3, "Green, Mr. George Henry",Male,51,0,0,8.0500,0},
   { 3, "Widegren, Mr. Carl/Charles Peter",Male,51,0,0,7.7500,0},
   { 3, "Lundahl, Mr. Johan Svensson",Male,51,0,0,7.0542,0},
   { 1, "Baxter, Mrs. James (Helene DeLaudeniere Chaput)",Female,50,0,1,247.5208,1},
   { 1, "Widener, Mr. George Dunton",Male,50,1,1,211.5000,0},
   { 1, "Widener, Mrs. George Dunton (Eleanor Elkins)",Female,50,1,1,211.5000,1},
   { 1, "Frauenthal, Dr. Henry William",Male,50,2,0,133.6500,1},
   { 1, "Douglas, Mr. Walter Donald",Male,50,1,0,106.4250,0},
   { 1, "Silvey, Mr. William Baird",Male,50,1,0,55.9000,0},
   { 1, "Isham, Miss. Ann Elizabeth",Female,50,0,0,28.7125,0},
   { 1, "Julian, Mr. Henry Forbes",Male,50,0,0,26.0000,0},
   { 2, "Louch, Mr. Charles Alexander",Male,50,1,0,26.0000,0},
   { 2, "Parrish, Mrs. (Lutie Davis)",Female,50,0,1,26.0000,1},
   { 3, "Robins, Mr. Alexander A",Male,50,1,0,14.5000,0},
   { 2, "Hodges, Mr. Henry Price",Male,50,0,0,13.0000,0},
   { 2, "Ridsdale, Miss. Lucy",Female,50,0,0,10.5000,1},
   { 2, "Toomey, Miss. Ellen",Female,50,0,0,10.5000,1},
   { 3, "Rouse, Mr. Richard Henry",Male,50,0,0,8.0500,0},
   { 1, "Thayer, Mr. John Borland",Male,49,1,1,110.8833,0},
   { 1, "Goldenberg, Mr. Samuel L",Male,49,1,0,89.1042,1},
   { 1, "Harper, Mrs. Henry Sleeper (Myna Haxtun)",Female,49,1,0,76.7292,1},
   { 2, "Herman, Mr. Samuel",Male,49,1,2,65.0000,0},
   { 1, "Duff Gordon, Sir. Cosmo Edmund (Mr Morgan)",Male,49,1,0,56.9292,1},
   { 1, "Case, Mr. Howard Brown",Male,49,0,0,26.0000,0},
   { 1, "Leader, Dr. Alice (Farnham)",Female,49,0,0,25.9292,1},
   { 1, "Ismay, Mr. Joseph Bruce",Male,49,0,0,0.0000,1},
   { 3, "Johnson, Mr. Alfred",Male,49,0,0,0.0000,0},
   { 1, "Ryerson, Mrs. Arthur Larned (Emily Maria Borie)",Female,48,1,3,262.3750,1},
   { 1, "Douglas, Mrs. Walter Donald (Mahala Dutton)",Female,48,1,0,106.4250,1},
   { 1, "Frolicher-Stehli, Mrs. Maxmillian (Margaretha Emerentia Stehli)",Female,48,1,1,79.2000,1},
   { 1, "Harper, Mr. Henry Sleeper",Male,48,1,0,76.7292,1},
   { 2, "Herman, Mrs. Samuel (Jane Laver)",Female,48,1,2,65.0000,1},
   { 1, "Taylor, Mr. Elmer Zebley",Male,48,1,0,52.0000,1},
   { 1, "Brandeis, Mr. Emil",Male,48,0,0,50.4958,0},
   { 1, "Duff Gordon, Lady. (Lucille Christiana Sutherland) (Mrs Morgan)",Female,48,1,0,39.6000,1},
   { 2, "Davies, Mrs. John Morgan (Elizabeth Agnes Mary White) ",Female,48,0,2,36.7500,1},
   { 3, "Ford, Mrs. Edward (Margaret Ann Watson)",Female,48,1,3,34.3750,0},
   { 1, "Anderson, Mr. Harry",Male,48,0,0,26.5500,1},
   { 1, "Swift, Mrs. Frederick Joel (Margaret Welles Barron)",Female,48,0,0,25.9292,1},
   { 2, "Milling, Mr. Jacob Christian",Male,48,0,0,13.0000,0},
   { 3, "Jensen, Mr. Niels Peder",Male,48,0,0,7.8542,0},
   { 1, "Astor, Col. John Jacob",Male,47,1,0,227.5250,0},
   { 1, "Chaffee, Mrs. Herbert Fuller (Carrie Constance Toogood)",Female,47,1,0,61.1750,1},
   { 1, "Beckwith, Mrs. Richard Leonard (Sallie Monypeny)",Female,47,1,1,52.5542,1},
   { 1, "Porter, Mr. Walter Chamberlain",Male,47,0,0,52.0000,0},
   { 1, "Moore, Mr. Clarence Bloomfield",Male,47,0,0,42.4000,0},
   { 1, "Gee, Mr. Arthur H",Male,47,0,0,38.5000,0},
   { 1, "Walker, Mr. William Anderson",Male,47,0,0,34.0208,0},
   { 1, "Colley, Mr. Edward Pomeroy",Male,47,0,0,25.5875,0},
   { 2, "Jarvis, Mr. John Denzil",Male,47,0,0,15.0000,0},
   { 3, "Robins, Mrs. Alexander A (Grace Charity Laury)",Female,47,1,0,14.5000,0},
   { 2, "Gilbert, Mr. William",Male,47,0,0,10.5000,0},
   { 3, "Vander Cruyssen, Mr. Victor",Male,47,0,0,9.0000,0},
   { 3, "Elsbury, Mr. William James",Male,47,0,0,7.2500,0},
   { 3, "Wilkes, Mrs. James (Ellen Needs)",Female,47,1,0,7.0000,1},
   { 1, "Guggenheim, Mr. Benjamin",Male,46,0,0,79.2000,0},
   { 1, "Rosenshine, Mr. George (Mr George Thorne)",Male,46,0,0,79.2000,0},
   { 1, "McCaffry, Mr. Thomas Francis",Male,46,0,0,75.2417,0},
   { 1, "Chaffee, Mr. Herbert Fuller",Male,46,1,0,61.1750,0},
   { 1, "Jones, Mr. Charles Cresson",Male,46,0,0,26.0000,0},
   { 2, "McKane, Mr. Peter David",Male,46,0,0,26.0000,0},
   { 1, "Partner, Mr. Austen",Male,45.5,0,0,28.5000,0},
   { 3, "Youseff, Mr. Gerious",Male,45.5,0,0,7.2250,0},
   { 1, "Bowen, Miss. Grace Scott",Female,45,0,0,262.3750,1},
   { 2, "Carter, Rev. Ernest Courtenay",Male,54,1,0,26.0000,0 },
   { 2, "Downton, Mr. William James",Male,54,0,0,26.0000,0 },
   { 2, "Hocking, Mrs. Elizabeth (Eliza Needs)",Female,54,1,3,23.0000,1 },
   { 2, "Moraweck, Dr. Ernest",Male,54,0,0,14.0000,0 },
   { 1, "Dodge, Dr. Washington",Male,53,1,1,81.8583,1 },
   { 1, "Appleton, Mrs. Edward Dale (Charlotte Lamson)",Female,53,2,0,51.4792,1 },
   { 1, "Gracie, Col. Archibald IV",Male,53,0,0,28.5000,1 },
   { 1, "Candee, Mrs. Edward (Helen Churchill Hungerford)",Female,53,0,0,27.4458,1 },
   { 1, "Hays, Mrs. Charles Melville (Clara Jennings Gregg)",Female,52,1,1,93.5000,1 },
   { 1, "Taussig, Mr. Emil",Male,52,1,1,79.6500,0 },
   { 1, "Stephenson, Mrs. Walter Bertram (Martha Eustis)",Female,52,1,0,78.2667,1 },
   { 1, "Peuchen, Major. Arthur Godfrey",Male,52,0,0,30.5000,1 },
   { 1, "Wick, Mrs. George Dennick (Mary Hitchcock)",Female,45,1,1,164.8667,1},
   { 1, "Spedden, Mr. Frederic Oakley",Male,45,1,1,134.5000,1},
   { 1, "Harris, Mr. Henry Birkhardt",Male,45,1,0,83.4750,0},
   { 1, "Greenfield, Mrs. Leo David (Blanche Strouse)",Female,45,0,1,63.3583,1},
   { 1, "Gibson, Mrs. Leonard (Pauline C Boeson)",Female,45,0,1,59.4000,1},
   { 1, "Kimball, Mrs. Edwin Nelson Jr (Gertrude Parsons)",Female,45,1,0,52.5542,1},
   { 1, "Blackwell, Mr. Stephen Weart",Male,45,0,0,35.5000,0},
   { 2, "Christy, Mrs. (Alice Frances)",Female,45,0,2,30.0000,1},
   { 1, "Chevre, Mr. Paul Romaine",Male,45,0,0,29.7000,1},
   { 3, "Skoog, Mrs. William (Anna Bernhardina Karlsson)",Female,45,1,4,27.9000,0},
   { 1, "Butt, Major. Archibald Willingham",Male,45,0,0,26.5500,0},
   { 1, "Romaine, Mr. Charles Hallace (Mr C Rolmane)",Male,45,0,0,26.5500,1},
   { 2, "Hart, Mrs. Benjamin (Esther Ada Bloomfield)",Female,45,1,1,26.2500,1},
   { 3, "Barbara, Mrs. (Catherine David)",Female,45,0,1,14.4542,0},
   { 3, "Hansen, Mrs. Claus Peter (Jennie L Howard)",Female,45,1,0,14.1083,1},
   { 2, "Kelly, Mrs. Florence Fannie",Female,45,0,0,13.5000,1},
   { 3, "Dahl, Mr. Karl Edwart",Male,45,0,0,8.0500,1},
   { 3, "Lindblom, Miss. Augusta Charlotta",Female,45,0,0,7.7500,0},
   { 3, "Assaf Khalil, Mrs. Mariana (Miriam)",Female,45,0,0,7.2250,1},
   { 3, "Ekstrom, Mr. Johan",Male,45,0,0,6.9750,0},
   { 1, "Minahan, Dr. William Edward",Male,44,2,0,90.0000,0},
   { 1, "Hippach, Mrs. Louis Albert (Ida Sophia Fischer)",Female,44,0,1,57.9792,1},
   { 1, "Brown, Mrs. James Joseph (Margaret Tobin)",Female,44,0,0,27.7208,1},
   { 2, "Carter, Mrs. Ernest Courtenay (Lilian Hughes)",Female,44,1,0,26.0000,0},
   { 2, "Hold, Mr. Stephen",Male,44,1,0,26.0000,0},
   { 3, "Cribb, Mr. John Hatfield",Male,44,0,1,16.1000,0},
   { 2, "Harbeck, Mr. William H",Male,44,0,0,13.0000,0},
   { 3, "Kelly, Mr. James",Male,44,0,0,8.0500,0},
   { 3, "Torber, Mr. Ernst William",Male,44,0,0,8.0500,0},
   { 3, "Sundman, Mr. Johan Julian",Male,44,0,0,7.9250,1},
   { 1, "Robert, Mrs. Edward Scott (Elisabeth Walton McMillan)",Female,43,0,1,211.3375,1},
   { 1, "Stengel, Mrs. Charles Emil Henry (Annie May Morris)",Female,43,1,0,55.4417,1},
   { 3, "Goodwin, Mrs. Frederick (Augusta Tyler)",Female,43,1,6,46.9000,0},
   { 1, "Frauenthal, Mr. Isaac Gerald",Male,43,1,0,27.7208,1},
   { 2, "Hart, Mr. Benjamin",Male,43,1,1,26.2500,0},
   { 2, "Phillips, Mr. Escott Robert",Male,43,0,1,21.0000,0},
   { 3, "Cook, Mr. Jacob",Male,43,0,0,8.0500,0},
   { 3, "Dintcheff, Mr. Valtcho",Male,43,0,0,7.8958,0},
   { 3, "Holm, Mr. John Fredrik Alexander",Male,43,0,0,6.4500,0},
   { 1, "Bidois, Miss. Rosalie",Female,42,0,0,227.5250,1},
   { 1, "Kimball, Mr. Edwin Nelson Jr",Male,42,1,0,52.5542,1},
   { 1, "Holverson, Mr. Alexander Oskar",Male,42,1,0,52.0000,0},
   { 1, "Head, Mr. Christopher",Male,42,0,0,42.5000,0},
   { 2, "Drew, Mr. James Vivian",Male,42,1,1,32.5000,0},
   { 2, "Jacobsohn, Mr. Sidney Samuel",Male,42,1,0,27.0000,0},
   { 1, "Borebank, Mr. John James",Male,42,0,0,26.5500,0},
   { 1, "Lindeberg-Lind, Mr. Erik Gustaf (Mr Edward Lingrey)",Male,42,0,0,26.5500,0},
   { 1, "Calderhead, Mr. Edward Pennington",Male,42,0,0,26.2875,1},
   { 2, "Louch, Mrs. Charles Alexander (Alice Adelaide Slow)",Female,42,1,0,26.0000,1},
   { 2, "Bowenur, Mr. Solomon",Male,42,0,0,13.0000,0},
   { 2, "Byles, Rev. Thomas Roussel Davids",Male,42,0,0,13.0000,0},
   { 2, "Bystrom, Mrs. (Karolina)",Female,42,0,0,13.0000,1},
   { 2, "Hosono, Mr. Masabumi",Male,42,0,0,13.0000,1},
   { 3, "Dimic, Mr. Jovan",Male,42,0,0,8.6625,0},
   { 3, "Olsen, Mr. Karl Siegwart Andreas",Male,42,0,1,8.4042,0},
   { 3, "Humblen, Mr. Adolf Mathias Nicolai Olsen",Male,42,0,0,7.6500,0},
   { 3, "Abbing, Mr. Anthony",Male,42,0,0,7.5500,0},
   { 1, "Burns, Miss. Elizabeth Margaret",Female,41,0,0,134.5000,1},
   { 1, "Kenyon, Mr. Frederick R",Male,41,1,0,51.8625,0},
   { 3, "Panula, Mrs. Juha (Maria Emilia Ojala)",Female,41,0,5,39.6875,0},
   { 1, "Brady, Mr. John Bertram",Male,41,0,0,30.5000,0},
   { 3, "Rosblom, Mrs. Viktor (Helena Wilhelmina)",Female,41,0,2,20.2125,0},
   { 2, "Mellinger, Mrs. (Elizabeth Anne Maidment)",Female,41,0,1,19.5000,1},
   { 2, "Stanton, Mr. Samuel Ward",Male,41,0,0,15.0458,0},
   { 3, "Hansen, Mr. Claus Peter",Male,41,2,0,14.1083,0},
   { 2, "Peruschitz, Rev. Joseph Maria",Male,41,0,0,13.0000,0},
   { 3, "Goldsmith, Mr. Nathan",Male,41,0,0,7.8500,0},
   { 3, "Nirva, Mr. Iisakki Antino Aijo",Male,41,0,0,7.1250,0},
   { 3, "Everett, Mr. Thomas James",Male,40.5,0,0,15.1000,0},
   { 3, "van Billiard, Mr. Austin Blyler",Male,40.5,0,2,14.5000,0},
   { 3, "Farrell, Mr. James",Male,40.5,0,0,7.7500,0},
   { 1, "Shutes, Miss. Elizabeth W",Female,40,0,0,153.4625,1},
   { 1, "Spedden, Mrs. Frederic Oakley (Margaretta Corning Stone)",Female,40,1,1,134.5000,1},
   { 3, "Goodwin, Mr. Charles Frederick",Male,40,1,6,46.9000,0},
   { 2, "Brown, Mrs. Thomas William Solomon (Elizabeth Catherine Ford)",Female,40,1,1,39.0000,1},
   { 3, "Asplund, Mr. Carl Oscar Vilhelm Gustafsson",Male,40,1,5,31.3875,0},
   { 1, "Blank, Mr. Henry",Male,40,0,0,31.0000,1},
   { 3, "Skoog, Mr. Wilhelm",Male,40,1,4,27.9000,0},
   { 1, "Uruchurtu, Don. Manuel E",Male,40,0,0,27.7208,0},
   { 2, "Faunthorpe, Mr. Harry",Male,40,1,0,26.0000,0},
   { 2, "Maybery, Mr. Frank Hubert",Male,40,0,0,16.0000,0},
   { 2, "Watt, Mrs. James (Elizabeth Bessie Inglis Milne)",Female,40,0,0,15.7500,1},
   { 3, "Bourke, Mr. John",Male,40,1,1,15.5000,0},
   { 2, "Smith, Miss. Marion Elsie",Female,40,0,0,13.0000,1},
   { 2, "Veal, Mr. James",Male,40,0,0,13.0000,0},
   { 3, "Ahlin, Mrs. Johan (Johanna Persdotter Larsson)",Female,40,1,0,9.4750,0},
   { 3, "Sivic, Mr. Husein",Male,40,0,0,7.8958,0},
   { 3, "Badt, Mr. Mohamed",Male,40,0,0,7.2250,0},
   { 1, "Harrison, Mr. William",Male,40,0,0,0.0000,0},
   { 1, "Kreuchen, Miss. Emilie",Female,39,0,0,211.3375,1},
   { 1, "Thayer, Mrs. John Borland (Marian Longstreth Morris)",Female,39,1,1,110.8833,1},
   { 1, "Oliva y Ocana, Dona. Fermina",Female,39,0,0,108.9000,1},
   { 1, "Compton, Miss. Sara Rebecca",Female,39,1,1,83.1583,1},
   { 1, "Taussig, Mrs. Emil (Tillie Mandelbaum)",Female,39,1,1,79.6500,1},
   { 1, "Cumings, Mr. John Bradley",Male,39,1,0,71.2833,0},
   { 1, "Silvey, Mrs. William Baird (Alice Munger)",Female,39,1,0,55.9000,1},
   { 3, "Andersson, Mr. Anders Johan",Male,39,1,5,31.2750,0},
   { 3, "Andersson, Mrs. Anders Johan (Alfrida Konstantia Brogren)",Female,39,1,5,31.2750,0},
   { 1, "Dulles, Mr. William Crothers",Male,39,0,0,29.7000,0},
   { 3, "Rice, Mrs. William (Margaret Norton)",Female,39,0,5,29.1250,0},
   { 2, "Morley, Mr. Henry Samuel (Mr Henry Marshall)",Male,39,0,0,26.0000,0},
   { 3, "Lester, Mr. James",Male,39,0,0,24.1500,0},
   { 3, "Karun, Mr. Franz",Male,39,0,1,13.4167,1},
   { 2, "Meyer, Mr. August",Male,39,0,0,13.0000,0},
   { 2, "Otter, Mr. Richard",Male,39,0,0,13.0000,0},
   { 3, "Niskanen, Mr. Juha",Male,39,0,0,7.9250,1},
   { 3, "Salonen, Mr. Johan Werner",Male,39,0,0,7.9250,0},
   { 3, "Elias, Mr. Joseph",Male,39,0,2,7.2292,0},
   { 1, "Andrews, Mr. Thomas Jr",Male,39,0,0,0.0000,0},
   { 3, "Saether, Mr. Simon Sivertsen",Male,38.5,0,0,7.2500,0},
   { 1, "Endres, Miss. Caroline Louise",Female,38,0,0,227.5250,1},
   { 1, "Graham, Mr. George Edward",Male,38,0,1,153.4625,0},
   { 1, "Hoyt, Mr. Frederick Maxfield",Male,38,1,0,90.0000,1},
   { 1, "Icard, Miss. Amelie",Female,38,0,0,80.0000,1},
   { 1, "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",Female,38,1,0,71.2833,1},
   { 3, "Asplund, Mrs. Carl Oscar (Selma Augusta Emilia Johansson)",Female,38,1,5,31.3875,1},
   { 2, "Gale, Mr. Harry",Male,38,1,0,21.0000,0},
   { 2, "Funk, Miss. Annie Clemmer",Female,38,0,0,13.0000,0},
   { 3, "Cacic, Mr. Luka",Male,38,0,0,8.6625,0},
   { 3, "Rekic, Mr. Tido",Male,38,0,0,7.8958,0},
   { 3, "Andersson, Miss. Ida Augusta Margareta",Female,38,4,2,7.7750,0},
   { 3, "Whabee, Mrs. George Joseph (Shawneene Abi-Saab)",Female,38,0,0,7.2292,1},
   { 3, "Goncalves, Mr. Manuel Estanslas",Male,38,0,0,7.0500,0},
   { 1, "Reuchlin, Jonkheer. John George",Male,38,0,0,0.0000,0},
   { 1, "Minahan, Mrs. William Edward (Lillian E Thorpe)",Female,37,1,0,90.0000,1},
   { 1, "Compton, Mr. Alexander Taylor Jr",Male,37,1,1,83.1583,0},
   { 1, "Futrelle, Mr. Jacques Heath",Male,37,1,0,53.1000,0},
   { 1, "Beckwith, Mr. Richard Leonard",Male,37,1,1,52.5542,1},
   { 1, "Natsch, Mr. Charles H",Male,37,0,1,29.7000,0},
   { 2, "Chapman, Mr. John Henry",Male,37,1,0,26.0000,0},
   { 3, "Laitinen, Miss. Kristina Sofia",Female,37,0,0,9.5875,0},
   { 3, "Gustafsson, Mr. Anders Vilhelm",Male,37,2,0,7.9250,0},
   { 3, "Carr, Miss. Jeannie",Female,37,0,0,7.7500,0},
   { 2, "Navratil, Mr. Michel (Louis M Hoffman)",Male,36.5,0,2,26.0000,0},
   { 3, "de Messemaeker, Mr. Guillaume Joseph",Male,36.5,1,0,17.4000,1},
   { 1, "Cardeza, Mr. Thomas Drake Martinez",Male,36,0,1,512.3292,1},
   { 1, "Chaudanson, Miss. Victorine",Female,36,0,0,262.3750,1},
   { 1, "Young, Miss. Marie Grice",Female,36,0,0,135.6333,1},
   { 1, "Carter, Mr. William Ernest",Male,36,1,2,120.0000,1},
   { 1, "Carter, Mrs. William Ernest (Lucile Polk)",Female,36,1,2,120.0000,1},
   { 1, "Cavendish, Mr. Tyrell William",Male,36,1,0,78.8500,0},
   { 1, "Beattie, Mr. Thomson",Male,36,0,0,75.2417,0},
   { 1, "Crosby, Miss. Harriet R",Female,36,0,2,71.0000,1},
   { 1, "Ross, Mr. John Hugo",Male,36,0,0,40.1250,0},
   { 2, "Becker, Mrs. Allen Oliver (Nellie E Baumgardner)",Female,36,0,3,39.0000,1},
   { 1, "Evans, Miss. Edith Corse",Female,36,0,0,31.6792,0},
   { 2, "West, Mr. Edwy Arthur",Male,36,1,2,27.7500,0},
   { 1, "Flynn, Mr. John Irwin (Irving)",Male,36,0,0,26.3875,1},
   { 1, "McGough, Mr. James Robert",Male,36,0,0,26.2875,1},
   { 2, "Angle, Mrs. William A (Florence Mary Agnes Hughes)",Female,36,1,0,26.0000,1},
   { 3, "Van Impe, Mr. Jean Baptiste",Male,36,1,1,24.1500,0},
   { 3, "de Messemaeker, Mrs. Guillaume Joseph (Emma)",Female,36,1,0,17.4000,1},
   { 3, "Coutts, Mrs. William (Winnie Minnie Treanor)",Female,36,0,2,15.9000,1},
   { 3, "Lindell, Mr. Edvard Bengtsson",Male,36,1,0,15.5500,0},
   { 2, "Ball, Mrs. (Ada E Hall)",Female,36,0,0,13.0000,1},
   { 2, "Buss, Miss. Kate",Female,36,0,0,13.0000,1},
   { 2, "Fox, Mr. Stanley Hubert",Male,36,0,0,13.0000,0},
   { 2, "Hocking, Mr. Samuel James Metcalfe",Male,36,0,0,13.0000,0},
   { 2, "Levy, Mr. Rene Jacques",Male,36,0,0,12.8750,0},
   { 3, "Klasen, Mrs. (Hulda Kristina Eugenia Lofqvist)",Female,36,0,2,12.1833,0},
   { 2, "Reeves, Mr. David",Male,36,0,0,10.5000,0},
   { 3, "Wittevrongel, Mr. Camille",Male,36,0,0,9.5000,0},
   { 3, "Turcin, Mr. Stjepan",Male,36,0,0,7.8958,0},
   { 3, "Coleff, Mr. Peju",Male,36,0,0,7.4958,0},
   { 3, "Dennis, Mr. William",Male,36,0,0,7.2500,0},
   { 3, "Leonard, Mr. Lionel",Male,36,0,0,0.0000,0},
   { 1, "Lesurer, Mr. Gustave J",Male,35,0,0,512.3292,1},
   { 1, "Ward, Miss. Anna",Female,35,0,0,512.3292,1},
   { 1, "Geiger, Miss. Amalie",Female,35,0,0,211.5000,1},
   { 1, "Bissette, Miss. Amelia",Female,35,0,0,135.6333,1},
   { 1, "Hoyt, Mrs. Frederick Maxfield (Jane Anne Forby)",Female,35,1,0,90.0000,1},
   { 1, "Harris, Mrs. Henry Birkhardt (Irene Wallach)",Female,35,1,0,83.4750,1},
   { 1, "Schabert, Mrs. Paul (Emma Mock)",Female,35,1,0,57.7500,1},
   { 1, "Futrelle, Mrs. Jacques Heath (Lily May Peel)",Female,35,1,0,53.1000,1},
   { 1, "Holverson, Mrs. Alexander Oskar (Mary Aline Towner)",Female,35,1,0,52.0000,1},
   { 1, "Homer, Mr. Harry (Mr E Haven)",Male,35,0,0,26.5500,1},
   { 1, "Silverthorne, Mr. Spencer Victor",Male,35,0,0,26.2875,1},
   { 2, "Fynney, Mr. Joseph J",Male,35,0,0,26.0000,0},
   { 2, "Cameron, Miss. Clear Annie",Female,35,0,0,21.0000,1},
   { 3, "Abbott, Mrs. Stanton (Rosa Hunt)",Female,35,1,1,20.2500,1},
   { 2, "Keane, Mr. Daniel",Male,35,0,0,12.3500,0},
   { 2, "Slemen, Mr. Richard James",Male,35,0,0,10.5000,0},
   { 3, "Allen, Mr. William Henry",Male,35,0,0,8.0500,0},
   { 3, "Brocklebank, Mr. William Alfred",Male,35,0,0,8.0500,0},
   { 3, "Cor, Mr. Bartol",Male,35,0,0,7.8958,0},
   { 3, "Markoff, Mr. Marin",Male,35,0,0,7.8958,0},
   { 3, "McGowan, Miss. Katherine",Female,35,0,0,7.7500,0},
   { 3, "Rintamaki, Mr. Matti",Male,35,0,0,7.1250,0},
   { 3, "Asim, Mr. Adola",Male,35,0,0,7.0500,0},
   { 3, "Kelly, Mr. James",Male,34.5,0,0,7.8292,0},
   { 3, "Lemberopolous, Mr. Peter L",Male,34.5,0,0,6.4375,0},
   { 2, "Drew, Mrs. James Vivian (Lulu Thorne Christian)",Female,34,1,1,32.5000,1},
   { 1, "Seward, Mr. Frederic Kimber",Male,34,0,0,26.5500,1},
   { 2, "Angle, Mr. William A",Male,34,1,0,26.0000,0},
   { 2, "Kantor, Mr. Sinai",Male,34,1,0,26.0000,0},
   { 2, "Doling, Mrs. John T (Ada Julia Bone)",Female,34,0,1,23.0000,1},
   { 2, "Gale, Mr. Shadrach",Male,34,1,0,21.0000,0},
   { 2, "Renouf, Mr. Peter Henry",Male,34,1,0,21.0000,0},
   { 3, "Danbom, Mr. Ernst Gilbert",Male,34,1,1,14.4000,0},
   { 2, "Beesley, Mr. Lawrence",Male,34,0,0,13.0000,1},
   { 2, "Garside, Miss. Ethel",Female,34,0,0,13.0000,1},
   { 2, "Gillespie, Mr. William Henry",Male,34,0,0,13.0000,0},
   { 2, "Ponesell, Mr. Martin",Male,34,0,0,13.0000,0},
   { 2, "Lemore, Mrs. (Amelia Milley)",Female,34,0,0,10.5000,1},
   { 3, "Morley, Mr. William",Male,34,0,0,8.0500,0},
   { 3, "Theobald, Mr. Thomas Leonard",Male,34,0,0,8.0500,0},
   { 3, "Johanson, Mr. Jakob Alfred",Male,34,0,0,6.4958,0},
   { 1, "Daniels, Miss. Sarah",Female,33,0,0,151.5500,1},
   { 1, "Minahan, Miss. Daisy E",Female,33,1,0,90.0000,1},
   { 1, "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",Female,33,0,0,86.5000,1},
   { 1, "Chambers, Mrs. Norman Campbell (Bertha Griggs)",Female,33,1,0,53.1000,1},
   { 2, "West, Mrs. Edwy Arthur (Ada Mary Worth)",Female,33,1,2,27.7500,1},
   { 1, "Rosenbaum, Miss. Edith Louise",Female,33,0,0,27.7208,1},
   { 1, "Rowe, Mr. Alfred G",Male,33,0,0,26.5500,0},
   { 2, "Quick, Mrs. Frederick Charles (Jane Richards)",Female,33,0,2,26.0000,1},
   { 3, "Dean, Mrs. Bertram (Eva Georgetta Light)",Female,33,1,2,20.5750,1},
   { 3, "Goldsmith, Mr. Frank John",Male,33,1,1,20.5250,0},
   { 3, "Backstrom, Mrs. Karl Alfred (Maria Mathilda Gustafsson)",Female,33,3,0,15.8500,1},
   { 2, "Hunt, Mr. George Henry",Male,33,0,0,12.2750,0},
   { 3, "Vande Velde, Mr. Johannes Joseph",Male,33,0,0,9.5000,0},
   { 3, "Stankovic, Mr. Ivan",Male,33,0,0,8.6625,0},
   { 3, "Johansson, Mr. Gustaf Joel",Male,33,0,0,8.6542,0},
   { 3, "Nancarrow, Mr. William Henry",Male,33,0,0,8.0500,0},
   { 3, "Drazenoic, Mr. Jozef",Male,33,0,0,7.8958,0},
   { 3, "Markun, Mr. Johann",Male,33,0,0,7.8958,0},
   { 3, "Karlsson, Mr. Julius Konrad Eugen",Male,33,0,0,7.8542,0},
   { 3, "Johnson, Mr. Malkolm Joackim",Male,33,0,0,7.7750,0},
   { 1, "Carlsson, Mr. Frans Olof",Male,33,0,0,5.0000,0},
   { 1, "Keeping, Mr. Edwin",Male,32.5,0,0,211.5000,0},
   { 2, "Nasser, Mr. Nicholas",Male,32.5,1,0,30.0708,0},
   { 2, "Webber, Miss. Susan",Female,32.5,0,0,13.0000,1},
   { 3, "Wenzel, Mr. Linhart",Male,32.5,0,0,9.5000,0},
   { 1, "Bazzani, Miss. Albina",Female,32,0,0,76.2917,1},
   { 2, "Hickman, Mr. Lewis",Male,32,2,0,73.5000,0},
   { 3, "Bing, Mr. Lee",Male,32,0,0,56.4958,1},
   { 3, "Chip, Mr. Chang",Male,32,0,0,56.4958,1},
   { 1, "Stahelin-Maeglin, Dr. Max",Male,32,0,0,30.5000,1},
   { 2, "Beane, Mr. Edward",Male,32,1,0,26.0000,1},
   { 3, "Andersen, Mr. Albert Karvin",Male,32,0,0,22.5250,0},
   { 3, "Backstrom, Mr. Karl Alfred",Male,32,1,0,15.8500,0},
   { 3, "Bourke, Mrs. John (Catherine)",Female,32,1,1,15.5000,0},
   { 2, "McCrae, Mr. Arthur Gordon",Male,32,0,0,13.5000,0},
   { 2, "de Brito, Mr. Jose Joaquim",Male,32,0,0,13.0000,0},
   { 2, "Pinsky, Mrs. (Rosa)",Female,32,0,0,13.0000,1},
   { 2, "Jenkin, Mr. Stephen Curnow",Male,32,0,0,10.5000,0},
   { 3, "Gronnestad, Mr. Daniel Danielsen",Male,32,0,0,8.3625,0},
   { 3, "Pickard, Mr. Berk (Berk Trembisky)",Male,32,0,0,8.0500,1},
   { 3, "Spinner, Mr. Henry John",Male,32,0,0,8.0500,0},
   { 3, "Jussila, Mr. Eiriik",Male,32,0,0,7.9250,1},
   { 3, "Leinonen, Mr. Antti Gustaf",Male,32,0,0,7.9250,0},
   { 3, "Tikkanen, Mr. Juho",Male,32,0,0,7.9250,0},
   { 3, "Pavlovic, Mr. Stefo",Male,32,0,0,7.8958,0},
   { 3, "Jonsson, Mr. Carl",Male,32,0,0,7.8542,1},
   { 3, "Olsson, Mr. Oscar Wilhelm",Male,32,0,0,7.7750,1},
   { 3, "Dooley, Mr. Patrick",Male,32,0,0,7.7500,0},
   { 3, "Lundstrom, Mr. Thure Edvin",Male,32,0,0,7.5792,1},
   { 1, "Wick, Miss. Mary Natalie",Female,31,0,2,164.8667,1},
   { 1, "Wilson, Miss. Helen Alice",Female,31,0,0,134.5000,1},
   { 1, "Newell, Miss. Madeleine",Female,31,1,0,113.2750,1},
   { 1, "Dick, Mr. Albert Adrian",Male,31,1,0,57.0000,1},
   { 1, "Davidson, Mr. Thornton",Male,31,1,0,52.0000,0},
   { 1, "Roebling, Mr. Washington Augustus II",Male,31,0,0,50.4958,0},
   { 2, "Mallet, Mr. Albert",Male,31,1,1,37.0042,0},
   { 1, "Tucker, Mr. Gilbert Milligan Jr",Male,31,0,0,28.5375,1},
   { 2, "Collyer, Mr. Harvey",Male,31,1,1,26.2500,0},
   { 2, "Collyer, Mrs. Harvey (Charlotte Annie Tate)",Female,31,1,1,26.2500,1},
   { 2, "Walcroft, Miss. Nellie",Female,31,0,0,21.0000,1},
   { 2, "Ware, Mrs. John James (Florence Louise Long)",Female,31,0,0,21.0000,1},
   { 3, "Goldsmith, Mrs. Frank John (Emily Alice Brown)",Female,31,1,1,20.5250,1},
   { 3, "Vander Planke, Mr. Julius",Male,31,3,0,18.0000,0},
   { 3, "Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)",Female,31,1,0,18.0000,0},
   { 2, "Wilhelms, Mr. Charles",Male,31,0,0,13.0000,1},
   { 2, "Kvillner, Mr. Johan Henrik Johannesson",Male,31,0,0,10.5000,0},
   { 3, "Osman, Mrs. Mara",Female,31,0,0,8.6833,1},
   { 3, "Stranden, Mr. Juho",Male,31,0,0,7.9250,1},
   { 3, "Olsson, Miss. Elina",Female,31,0,0,7.8542,0},
   { 3, "Johansson, Mr. Karl Johan",Male,31,0,0,7.7750,0},
   { 3, "Connaghton, Mr. Michael",Male,31,0,0,7.7500,0},
   { 3, "Conlon, Mr. Thomas Henry",Male,31,0,0,7.7333,0},
   { 3, "Tomlin, Mr. Ernest Portage",Male,30.5,0,0,8.0500,0},
   { 3, "Mangan, Miss. Mary",Female,30.5,0,0,7.7500,0},
   { 1, "Bonnell, Miss. Caroline",Female,30,0,0,164.8667,1},
   { 1, "Allison, Mr. Hudson Joshua Creighton",Male,30,1,2,151.5500,0},
   { 1, "LeRoy, Miss. Bertha",Female,30,0,0,106.4250,1},
   { 1, "Perreault, Miss. Anne",Female,30,0,0,93.5000,1},
   { 1, "Cherry, Miss. Gladys",Female,30,0,0,86.5000,1},
   { 1, "Mock, Mr. Philipp Edmund",Male,30,1,0,57.7500,1},
   { 1, "Francatelli, Miss. Laura Mabel",Female,30,0,0,56.9292,1},
   { 1, "Loring, Mr. Joseph Holland",Male,30,0,0,45.5000,0},
   { 1, "Serepeca, Miss. Augusta",Female,30,0,0,31.0000,1},
   { 1, "Foreman, Mr. Benjamin Laventall",Male,30,0,0,27.7500,0},
   { 1, "Maguire, Mr. John Edward",Male,30,0,0,26.0000,0},
   { 2, "Lahtinen, Rev. William",Male,30,1,1,26.0000,0},
   { 3, "Van Impe, Mrs. Jean Baptiste (Rosalie Paula Govaert)",Female,30,1,1,24.1500,0},
   { 2, "Abelson, Mr. Samuel",Male,30,1,0,24.0000,0},
   { 2, "Renouf, Mrs. Peter Henry (Lillian Jefferys)",Female,30,3,0,21.0000,1},
   { 2, "Ware, Mr. John James",Male,30,1,0,21.0000,0},
   { 3, "Lobb, Mr. William Arthur",Male,30,1,0,16.1000,0},
   { 3, "Lindell, Mrs. Edvard Bengtsson (Elin Gerda Persson)",Female,30,1,0,15.5500,0},
   { 2, "Duran y More, Miss. Florentina",Female,30,1,0,13.8583,1},
   { 2, "Aldworth, Mr. Charles Augustus",Male,30,0,0,13.0000,0},
   { 2, "Corbett, Mrs. Walter H (Irene Colvin)",Female,30,0,0,13.0000,0},
   { 2, "Givard, Mr. Hans Kristensen",Male,30,0,0,13.0000,0},
   { 2, "Hale, Mr. Reginald",Male,30,0,0,13.0000,0},
   { 2, "Matthews, Mr. William John",Male,30,0,0,13.0000,0},
   { 2, "McCrie, Mr. James Matthew",Male,30,0,0,13.0000,0},
   { 2, "Sinkkonen, Miss. Anna",Female,30,0,0,13.0000,1},
   { 2, "Portaluppi, Mr. Emilio Ilario Giuseppe",Male,30,0,0,12.7375,1},
   { 3, "Dowdell, Miss. Elizabeth",Female,30,0,0,12.4750,1},
   { 2, "Slayter, Miss. Hilda Mary",Female,30,0,0,12.3500,1},
   { 2, "Harris, Mr. Walter",Male,30,0,0,10.5000,0},
   { 3, "de Mulder, Mr. Theodore",Male,30,0,0,9.5000,1},
   { 3, "Cacic, Miss. Marija",Female,30,0,0,8.6625,0},
   { 3, "Corn, Mr. Harry",Male,30,0,0,8.0500,0},
   { 3, "Somerton, Mr. Francis William",Male,30,0,0,8.0500,0},
   { 3, "Karaic, Mr. Milan",Male,30,0,0,7.8958,0},
   { 3, "Connolly, Miss. Kate",Female,30,0,0,7.6292,0},
   { 3, "Adahl, Mr. Mauritz Nils Martin",Male,30,0,0,7.2500,0},
   { 3, "Ibrahim Shawah, Mr. Yousseff",Male,30,0,0,7.2292,0},
   { 3, "Attalah, Mr. Sleiman",Male,30,0,0,7.2250,0},
   { 3, "Daly, Miss. Margaret Marcella Maggie",Female,30,0,0,6.9500,1},
   { 1, "Bird, Miss. Ellen",Female,29,0,0,221.7792,1},
   { 1, "Allen, Miss. Elisabeth Walton",Female,29,0,0,211.3375,1},
   { 1, "Pears, Mr. Thomas Clinton",Male,29,1,0,66.6000,0},
   { 1, "Long, Mr. Milton Clyde",Male,29,0,0,30.0000,0},
   { 2, "del Carlo, Mr. Sebastiano",Male,29,1,0,27.7208,0},
   { 2, "Chapman, Mrs. John Henry (Sara Elizabeth Lawry)",Female,29,1,0,26.0000,0},
   { 2, "Clarke, Mr. Charles Valentine",Male,29,1,0,26.0000,0},
   { 2, "Faunthorpe, Mrs. Lizzie (Elizabeth Anne Wilkinson)",Female,29,1,0,26.0000,1},
   { 2, "Hold, Mrs. Stephen (Annie Margaret Hill)",Female,29,1,0,26.0000,1},
   { 2, "Weisz, Mrs. Leopold (Mathilde Francoise Pede)",Female,29,1,0,26.0000,1},
   { 2, "Wells, Mrs. Arthur Henry (Addie Dart Trevaskis)",Female,29,0,2,23.0000,1},
   { 3, "Kink-Heilmann, Mr. Anton",Male,29,3,1,22.0250,1},
   { 3, "Palsson, Mrs. Nils (Alma Cornelia Berglund)",Female,29,0,4,21.0750,0},
   { 2, "Turpin, Mr. William John Robert",Male,29,1,0,21.0000,0},
   { 3, "Touma, Mrs. Darwis (Hanne Youssef Razi)",Female,29,0,2,15.2458,1},
   { 2, "Pallas y Castello, Mr. Emilio",Male,29,0,0,13.8583,1},
   { 2, "Coleridge, Mr. Reginald Charles",Male,29,0,0,10.5000,0},
   { 2, "Nye, Mrs. (Elizabeth Ramell)",Female,29,0,0,10.5000,1},
   { 3, "Strom, Mrs. Wilhelm (Elna Matilda Persson)",Female,29,1,1,10.4625,0},
   { 3, "Sheerlinck, Mr. Jan Baptist",Male,29,0,0,9.5000,1},
   { 3, "Larsson, Mr. August Viktor",Male,29,0,0,9.4833,0},
   { 3, "Christmann, Mr. Emil",Male,29,0,0,8.0500,0},
   { 3, "Makinen, Mr. Kalle Edvard",Male,29,0,0,7.9250,0},
   { 3, "Nieminen, Miss. Manta Josefina",Female,29,0,0,7.9250,0},
   { 3, "Jalsevac, Mr. Ivan",Male,29,0,0,7.8958,1},
   { 3, "Zimmerman, Mr. Leo",Male,29,0,0,7.8750,0},
   { 3, "Johansson, Mr. Nils",Male,29,0,0,7.8542,0},
   { 3, "Larsson, Mr. Bengt Edvin",Male,29,0,0,7.7750,0},
   { 3, "Daly, Mr. Eugene Patrick",Male,29,0,0,7.7500,1},
   { 3, "Braund, Mr. Lewis Richard",Male,29,1,0,7.0458,0},
   { 1, "Ovies y Rodriguez, Mr. Servando",Male,28.5,0,0,27.7208,0},
   { 3, "Williams, Mr. Leslie",Male,28.5,0,0,16.1000,0},
   { 3, "Novel, Mr. Mansouer",Male,28.5,0,0,7.2292,0},
   { 1, "Fortune, Miss. Ethel Flora",Female,28,3,2,263.0000,1},
   { 1, "Meyer, Mr. Edgar Joseph",Male,28,1,0,82.1708,0},
   { 3, "Ling, Mr. Lee",Male,28,0,0,56.4958,0},
   { 1, "Carrau, Mr. Francisco M",Male,28,0,0,47.1000,0},
   { 1, "Sloper, Mr. William Thompson",Male,28,0,0,35.5000,1},
   { 2, "Harper, Rev. John",Male,28,0,1,33.0000,0},
   { 1, "Bjornstrom-Steffansson, Mr. Mauritz Hakan",Male,28,0,0,26.5500,1},
   { 2, "Beauchamp, Mr. Henry James",Male,28,0,0,26.0000,0},
   { 2, "Clarke, Mrs. Charles V (Ada Maria Winfield)",Female,28,1,0,26.0000,1},
   { 2, "Abelson, Mrs. Samuel (Hannah Wizosky)",Female,28,1,0,24.0000,1},
   { 3, "Holthen, Mr. Johan Martin",Male,28,0,0,22.5250,0},
   { 3, "Olsen, Mr. Henry Margido",Male,28,0,0,22.5250,0},
   { 3, "Hakkarainen, Mr. Pekka Pietari",Male,28,1,0,15.8500,0},
   { 3, "Danbom, Mrs. Ernst Gilbert (Anna Sigrid Maria Brogren)",Female,28,1,1,14.4000,0},
   { 2, "Norman, Mr. Robert Douglas",Male,28,0,0,13.5000,0},
   { 2, "Collander, Mr. Erik Gustaf",Male,28,0,0,13.0000,0},
   { 2, "Davis, Miss. Mary",Female,28,0,0,13.0000,1},
   { 2, "Reynaldo, Ms. Encarnacion",Female,28,0,0,13.0000,1},
   { 2, "Trout, Mrs. William H (Jessie L)",Female,28,0,0,12.6500,1},
   { 2, "Banfield, Mr. Frederick James",Male,28,0,0,10.5000,0},
   { 2, "Parker, Mr. Clifford Richard",Male,28,0,0,10.5000,0},
   { 3, "Vande Walle, Mr. Nestor Cyriel",Male,28,0,0,9.5000,0},
   { 3, "Vanden Steen, Mr. Leo Peter",Male,28,0,0,9.5000,0},
   { 3, "Niklasson, Mr. Samuel",Male,28,0,0,8.0500,0},
   { 3, "Gustafsson, Mr. Johan Birger",Male,28,2,0,7.9250,0},
   { 3, "Hendekovic, Mr. Ignjac",Male,28,0,0,7.8958,0},
   { 3, "Mionoff, Mr. Stoytcho",Male,28,0,0,7.8958,0},
   { 3, "Petranec, Miss. Matilda",Female,28,0,0,7.8958,0},
   { 3, "Olsson, Mr. Nils Johan Goransson",Male,28,0,0,7.8542,0},
   { 3, "Carlsson, Mr. August Sigfrid",Male,28,0,0,7.7958,0},
   { 3, "Henriksson, Miss. Jenny Lovisa",Female,28,0,0,7.7750,0},
   { 3, "Carver, Mr. Alfred John",Male,28,0,0,7.2500,0},
   { 1, "Douglas, Mrs. Frederick Charles (Mary Helene Baxter)",Female,27,1,1,247.5208,1},
   { 1, "Widener, Mr. Harry Elkins",Male,27,0,2,211.5000,0},
   { 1, "Clark, Mr. Walter Miller",Male,27,1,0,136.7792,0},
   { 1, "Hassab, Mr. Hammad",Male,27,0,0,76.7292,1},
   { 1, "Chambers, Mr. Norman Campbell",Male,27,1,0,53.1000,1},
   { 1, "Davidson, Mrs. Thornton (Orian Hays)",Female,27,1,2,52.0000,1},
   { 1, "Daniel, Mr. Robert Williams",Male,27,0,0,30.5000,1},
   { 2, "Sharp, Mr. Percival James R",Male,27,0,0,26.0000,0},
   { 2, "Weisz, Mr. Leopold",Male,27,1,0,26.0000,0},
   { 2, "Turpin, Mrs. William John Robert (Dorothy Ann Wonnacott)",Female,27,1,0,21.0000,0},
   { 2, "Pulbaum, Mr. Franz",Male,27,0,0,15.0333,0},
   { 3, "Yasbeck, Mr. Antoni",Male,27,1,0,14.4542,0},
   { 2, "Duran y More, Miss. Asuncion",Female,27,1,0,13.8583,1},
   { 2, "Bracken, Mr. James H",Male,27,0,0,13.0000,0},
   { 2, "Montvila, Rev. Juozas",Male,27,0,0,13.0000,0},
   { 3, "Moor, Mrs. (Beila)",Female,27,0,1,12.4750,1},
   { 3, "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",Female,27,0,2,11.1333,1},
   { 2, "Troutt, Miss. Edwina Celia Winnie",Female,27,0,0,10.5000,1},
   { 3, "Lulic, Mr. Nikola",Male,27,0,0,8.6625,1},
   { 3, "Strilic, Mr. Ivan",Male,27,0,0,8.6625,0},
   { 3, "Wirz, Mr. Albert",Male,27,0,0,8.6625,0},
   { 3, "Honkanen, Miss. Eliina",Female,27,0,0,7.9250,1},
   { 3, "Ilmakangas, Miss. Ida Livija",Female,27,1,0,7.9250,0},
   { 3, "Cor, Mr. Ivan",Male,27,0,0,7.8958,0},
   { 3, "Danoff, Mr. Yoto",Male,27,0,0,7.8958,0},
   { 3, "Barry, Miss. Julia",Female,27,0,0,7.8792,0},
   { 3, "Jonsson, Mr. Nils Hilding",Male,27,0,0,7.8542,0},
   { 3, "Andersson, Mr. August Edvard (Wennerstrom)",Male,27,0,0,7.7958,1},
   { 3, "Zakarian, Mr. Ortin",Male,27,0,0,7.2250,0},
   { 3, "Hedman, Mr. Oskar Arvid",Male,27,0,0,6.9750,1},
   { 3, "Zakarian, Mr. Mapriededer",Male,26.5,0,0,7.2250,0},
   { 1, "Clark, Mrs. Walter Miller (Virginia McDowell)",Female,26,1,0,136.7792,1},
   { 1, "Barber, Miss. Ellen Nellie",Female,26,0,0,78.8500,1},
   { 3, "Lang, Mr. Fang",Male,26,0,0,56.4958,1},
   { 1, "Behr, Mr. Karl Howell",Male,26,0,0,30.0000,1},
   { 2, "Caldwell, Mr. Albert Francis",Male,26,1,1,29.0000,1},
   { 2, "Lahtinen, Mrs. William (Anna Sylfven)",Female,26,1,1,26.0000,0},
   { 3, "Kink-Heilmann, Mrs. Anton (Luise Heilmann)",Female,26,1,1,22.0250,1},
   { 3, "Dean, Mr. Bertram Frank",Male,26,1,2,20.5750,0},
   { 3, "Albimona, Mr. Nassef Cassem",Male,26,0,0,18.7875,1},
   { 3, "Lobb, Mrs. William Arthur (Cordelia K Stanlick)",Female,26,1,0,16.1000,0},
   { 3, "Chronopoulos, Mr. Apostolos",Male,26,1,0,14.4542,0},
   { 3, "Peacock, Mrs. Benjamin (Edith Nile)",Female,26,0,2,13.7750,0},
   { 2, "Wright, Miss. Marion",Female,26,0,0,13.5000,1},
   { 2, "Botsford, Mr. William Hull",Male,26,0,0,13.0000,0},
   { 2, "Nesson, Mr. Israel",Male,26,0,0,13.0000,0},
   { 2, "Schmidt, Mr. August",Male,26,0,0,13.0000,0},
   { 2, "Gavey, Mr. Lawrence",Male,26,0,0,10.5000,0},
   { 3, "Kink, Mr. Vincenz",Male,26,2,0,8.6625,0},
   { 3, "Adams, Mr. John",Male,26,0,0,8.0500,0},
   { 3, "Heikkinen, Miss. Laina",Female,26,0,0,7.9250,1},
   { 3, "Angheloff, Mr. Minko",Male,26,0,0,7.8958,0},
   { 3, "Balkic, Mr. Cerin",Male,26,0,0,7.8958,0},
   { 3, "Bostandyeff, Mr. Guentcho",Male,26,0,0,7.8958,0},
   { 3, "Alexander, Mr. William",Male,26,0,0,7.8875,0},
   { 3, "Foley, Mr. Joseph",Male,26,0,0,7.8792,0},
   { 3, "Hansen, Mr. Henrik Juul",Male,26,1,0,7.8542,0},
   { 3, "Nilsson, Miss. Helmina Josefina",Female,26,0,0,7.8542,1},
   { 3, "Andersson, Mr. Johan Samuel",Male,26,0,0,7.7750,0},
   { 3, "Bengtsson, Mr. John Viktor",Male,26,0,0,7.7750,0},
   { 3, "Johansson Palmquist, Mr. Oskar Leander",Male,26,0,0,7.7750,1},
   { 1, "Allison, Mrs. Hudson J C (Bessie Waldo Daniels)",Female,25,1,2,151.5500,0},
   { 1, "Bishop, Mr. Dickinson H",Male,25,1,0,91.0792,1},
   { 1, "Harder, Mr. George Achilles",Male,25,1,0,55.4417,1},
   { 1, "Harder, Mrs. George Achilles (Dorothy Annan)",Female,25,1,0,55.4417,1},
   { 2, "Laroche, Mr. Joseph Philippe Lemercier",Male,25,1,2,41.5792,0},
   { 2, "Denbury, Mr. Herbert",Male,25,0,0,31.5000,0},
   { 2, "Christy, Miss. Julie Rachel",Female,25,1,1,30.0000,1},
   { 1, "Birnbaum, Mr. Jakob",Male,25,0,0,26.0000,0},
   { 2, "Bryhl, Mr. Kurt Arnold Gottfrid",Male,25,1,0,26.0000,0},
   { 2, "Shelley, Mrs. William (Imanita Parrish Hall)",Female,25,0,1,26.0000,1},
   { 3, "Arnold-Franchi, Mr. Josef",Male,25,1,0,17.8000,0},
   { 2, "Butler, Mr. Reginald Fenton",Male,25,0,0,13.0000,0},
   { 2, "Sedgwick, Mr. Charles Frederick Waddington",Male,25,0,0,13.0000,0},
   { 2, "Sobey, Mr. Samuel James Hayden",Male,25,0,0,13.0000,0},
   { 2, "Andrew, Mr. Frank Thomas",Male,25,0,0,10.5000,0},
   { 2, "Stokes, Mr. Philip Joseph",Male,25,0,0,10.5000,0},
   { 3, "Sap, Mr. Julius",Male,25,0,0,9.5000,1},
   { 3, "Ilmakangas, Miss. Pieta Sofia",Female,25,1,0,7.9250,0},
   { 3, "Peltomaki, Mr. Nikolai Johannes",Male,25,0,0,7.9250,0},
   { 3, "Dantcheff, Mr. Ristiu",Male,25,0,0,7.8958,0},
   { 3, "Delalic, Mr. Redjo",Male,25,0,0,7.8958,0},
   { 3, "Tenglin, Mr. Gunnar Isidor",Male,25,0,0,7.7958,1},
   { 3, "Lindahl, Miss. Agda Thorilda Viktoria",Female,25,0,0,7.7750,0},
   { 3, "Persson, Mr. Ernst Ulrik",Male,25,1,0,7.7750,1},
   { 3, "Petterson, Mr. Johan Emil",Male,25,1,0,7.7750,0},
   { 3, "Gallagher, Mr. Martin",Male,25,0,0,7.7417,0},
   { 3, "Abelseth, Mr. Olaus Jorgensen",Male,25,0,0,7.6500,1},
   { 3, "Moen, Mr. Sigurd Hansen",Male,25,0,0,7.6500,0},
   { 3, "Harmer, Mr. Abraham (David Lishin)",Male,25,0,0,7.2500,0},
   { 3, "Krekorian, Mr. Neshan",Male,25,0,0,7.2292,1},
   { 3, "Saad, Mr. Khalil",Male,25,0,0,7.2250,0},
   { 3, "Ali, Mr. William",Male,25,0,0,7.0500,0},
   { 3, "Sutehall, Mr. Henry Jr",Male,25,0,0,7.0500,0},
   { 3, "Tornquist, Mr. William Henry",Male,25,0,0,0.0000,1},
   { 3, "Sawyer, Mr. Frederick Charles",Male,24.5,0,0,8.0500,0},
   { 1, "Fortune, Miss. Alice Elizabeth",Female,24,3,2,263.0000,1},
   { 1, "Baxter, Mr. Quigg Edmond",Male,24,0,1,247.5208,0},
   { 1, "Hays, Miss. Margaret Bechstein",Female,24,0,0,83.1583,1},
   { 1, "Snyder, Mr. John Pillsbury",Male,24,1,0,82.2667,1},
   { 1, "Giglio, Mr. Victor",Male,24,0,0,79.2000,0},
   { 2, "Hickman, Mr. Leonard Mark",Male,24,2,0,73.5000,0},
   { 1, "Aubart, Mme. Leontine Pauline",Female,24,0,0,69.3000,1},
   { 1, "Sagesser, Mlle. Emma",Female,24,0,0,69.3000,1},
   { 2, "Herman, Miss. Alice",Female,24,1,2,65.0000,1},
   { 2, "Herman, Miss. Kate",Female,24,1,2,65.0000,1},
   { 1, "Smith, Mr. Lucien Philip",Male,24,1,0,60.0000,0},
   { 1, "Mayne, Mlle. Berthe Antonine (Mrs de Villiers)",Female,24,0,0,49.5042,1},
   { 2, "Mallet, Mrs. Albert (Antoinette Magnin)",Female,24,1,1,37.0042,1},
   { 2, "Jefferys, Mr. Clifford Thomas",Male,24,2,0,31.5000,0},
   { 2, "del Carlo, Mrs. Sebastiano (Argenia Genovesi)",Female,24,1,0,27.7208,1},
   { 2, "Jacobsohn, Mrs. Sidney Samuel (Amy Frances Christy)",Female,24,2,1,27.0000,1},
   { 2, "Kantor, Mrs. Sinai (Miriam Sternin)",Female,24,1,0,26.0000,1},
   { 3, "Davies, Mr. Alfred J",Male,24,2,0,24.1500,0},
   { 3, "Baclini, Mrs. Solomon (Latifa Qurban)",Female,24,0,3,19.2583,1},
   { 2, "Richards, Mrs. Sidney (Emily Hocking)",Female,24,2,3,18.7500,1},
   { 3, "Sandstrom, Mrs. Hjalmar (Agnes Charlotta Bengtsson)",Female,24,0,2,16.7000,1},
   { 3, "McNamee, Mr. Neal",Male,24,1,0,16.1000,0},
   { 3, "Hakkarainen, Mrs. Pekka Pietari (Elin Matilda Dolck)",Female,24,1,0,15.8500,1},
   { 2, "Hamalainen, Mrs. William (Anna)",Female,24,0,2,14.5000,1},
   { 2, "Giles, Mr. Ralph",Male,24,0,0,13.5000,0},
   { 2, "Brown, Miss. Amelia Mildred",Female,24,0,0,13.0000,1},
   { 2, "Gill, Mr. John William",Male,24,0,0,13.0000,0},
   { 2, "Yrois, Miss. Henriette (Mrs Harbeck)",Female,24,0,0,13.0000,0},
   { 2, "Collett, Mr. Sidney C Stuart",Male,24,0,0,10.5000,1},
   { 2, "Leyson, Mr. Robert William Norman",Male,24,0,0,10.5000,0},
   { 3, "Lievens, Mr. Rene Aime",Male,24,0,0,9.5000,0},
   { 3, "Salander, Mr. Karl Johan",Male,24,0,0,9.3250,0},
   { 3, "Haas, Miss. Aloisia",Female,24,0,0,8.8500,0},
   { 3, "Pokrnic, Mr. Tome",Male,24,0,0,8.6625,0},
   { 3, "Celotti, Mr. Francesco",Male,24,0,0,8.0500,0},
   { 3, "Petersen, Mr. Marius",Male,24,0,0,8.0500,0},
   { 3, "Mineff, Mr. Ivan",Male,24,0,0,7.8958,0},
   { 3, "Carlsson, Mr. Carl Robert",Male,24,0,0,7.8542,0},
   { 3, "Svensson, Mr. Olof",Male,24,0,0,7.7958,0},
   { 3, "Aronsson, Mr. Ernst Axel Algot",Male,24,0,0,7.7750,0},
   { 3, "Doyle, Miss. Elizabeth",Female,24,0,0,7.7500,0},
   { 3, "Mulvihill, Miss. Bertha E",Female,24,0,0,7.7500,1},
   { 3, "Duquemin, Mr. Joseph",Male,24,0,0,7.5500,1},
   { 3, "Coleff, Mr. Satio",Male,24,0,0,7.4958,0},
   { 3, "Colbert, Mr. Patrick",Male,24,0,0,7.2500,0},
   { 3, "Madsen, Mr. Fridtjof Arne",Male,24,0,0,7.1417,1},
   { 3, "Ali, Mr. Ahmed",Male,24,0,0,7.0500,0},
   { 3, "Hanna, Mr. Mansour",Male,23.5,0,0,7.2292,0},
   { 1, "Fortune, Miss. Mabel Helen",Female,23,3,2,263.0000,1},
   { 1, "Newell, Miss. Marjorie",Female,23,1,0,113.2750,1},
   { 1, "Payne, Mr. Vivian Ponsonby",Male,23,0,0,93.5000,0},
   { 1, "Earnshaw, Mrs. Boulton (Olive Potter)",Female,23,0,1,83.1583,1},
   { 1, "Snyder, Mrs. John Pillsbury (Nelle Stevenson)",Female,23,1,0,82.2667,1},
   { 1, "Greenfield, Mr. William Bertram",Male,23,0,1,63.3583,1},
   { 2, "Richard, Mr. Emile",Male,23,0,0,15.0458,0},
   { 3, "Dyker, Mr. Adolf Fredrik",Male,23,1,0,13.9000,0},
   { 2, "Jerwan, Mrs. Amin S (Marie Marthe Thuillard)",Female,23,0,0,13.7917,1},
   { 2, "Berriman, Mr. William John",Male,23,0,0,13.0000,0},
   { 2, "Eitemiller, Mr. George Floyd",Male,23,0,0,13.0000,0},
   { 2, "Troupiansky, Mr. Moses Aaron",Male,23,0,0,13.0000,0},
   { 2, "Hocking, Mr. Richard George",Male,23,2,1,11.5000,0},
   { 2, "Baimbrigge, Mr. Charles Robert",Male,23,0,0,10.5000,0},
   { 2, "Pain, Dr. Alfred",Male,23,0,0,10.5000,0},
   { 2, "Ware, Mr. William Jeffery",Male,23,1,0,10.5000,0},
   { 3, "Odahl, Mr. Nils Martin",Male,23,0,0,9.2250,0},
   { 3, "Oreskovic, Miss. Jelka",Female,23,0,0,8.6625,0},
   { 3, "Drapkin, Miss. Jennie",Female,23,0,0,8.0500,1},
   { 3, "Heininen, Miss. Wendla Maria",Female,23,0,0,7.9250,0},
   { 3, "Jonkoff, Mr. Lalio",Male,23,0,0,7.8958,0},
   { 3, "Augustsson, Mr. Albert",Male,23,0,0,7.8542,0},
   { 3, "Lundin, Miss. Olga Elida",Female,23,0,0,7.8542,1},
   { 3, "Asplund, Mr. Johan Charles",Male,23,0,0,7.7958,1},
   { 3, "Stanley, Miss. Amy Zillah Elsie",Female,23,0,0,7.5500,1},
   { 3, "Assam, Mr. Ali",Male,23,0,0,7.0500,0},
   { 3, "Daher, Mr. Shedid",Male,22.5,0,0,7.2250,0},
   { 1, "Cleaver, Miss. Alice",Female,22,0,0,151.5500,1},
   { 1, "Ringhini, Mr. Sante",Male,22,0,0,135.6333,0},
   { 1, "Pears, Mrs. Thomas (Edith Wearne)",Female,22,1,0,66.6000,1},
   { 1, "Ostby, Miss. Helene Ragnhild",Female,22,0,1,61.9792,1},
   { 1, "Gibson, Miss. Dorothy Winifred",Female,22,0,1,59.4000,1},
   { 1, "Bowerman, Miss. Elsie Edith",Female,22,0,1,55.0000,1},
   { 1, "Frolicher, Miss. Hedwig Margaritha",Female,22,0,2,49.5000,1},
   { 2, "Laroche, Mrs. Joseph (Juliette Marie Louise Lafargue)",Female,22,1,2,41.5792,1},
   { 3, "Riihivouri, Miss. Susanna Juhantytar Sanni",Female,22,0,0,39.6875,0},
   { 2, "Jefferys, Mr. Ernest Wilfred",Male,22,2,0,31.5000,0},
   { 2, "Caldwell, Mrs. Albert Francis (Sylvia Mae Harbaugh)",Female,22,1,1,29.0000,1},
   { 2, "Karnes, Mrs. J Frank (Claire Bennett)",Female,22,0,0,21.0000,0},
   { 3, "Dyker, Mrs. Adolf Fredrik (Anna Elisabeth Judith Andersson)",Female,22,1,0,13.9000,1},
   { 3, "Hirvonen, Mrs. Alexander (Helga E Lindqvist)",Female,22,1,1,12.2875,1},
   { 3, "Dahlberg, Miss. Gerda Ulrika",Female,22,0,0,10.5167,0},
   { 2, "Cook, Mrs. (Selena Rogers)",Female,22,0,0,10.5000,1},
   { 2, "Oxenham, Mr. Percy Thomas",Male,22,0,0,10.5000,1},
   { 3, "Strandberg, Miss. Ida Sofia",Female,22,0,0,9.8375,0},
   { 3, "Berglund, Mr. Karl Ivar Sven",Male,22,0,0,9.3500,0},
   { 3, "Waelens, Mr. Achille",Male,22,0,0,9.0000,0},
   { 3, "Hellstrom, Miss. Hilda Maria",Female,22,0,0,8.9625,1},
   { 3, "Kink, Miss. Maria",Female,22,2,0,8.6625,0},
   { 3, "Barton, Mr. David John",Male,22,0,0,8.0500,0},
   { 3, "Davies, Mr. Evan",Male,22,0,0,8.0500,0},
   { 3, "Gilinski, Mr. Eliezer",Male,22,0,0,8.0500,0},
   { 3, "Naidenoff, Mr. Penko",Male,22,0,0,7.8958,0},
   { 3, "Vovk, Mr. Janko",Male,22,0,0,7.8958,0},
   { 3, "Brobeck, Mr. Karl Rudolf",Male,22,0,0,7.7958,0},
   { 3, "Johansson, Mr. Erik",Male,22,0,0,7.7958,0},
   { 3, "Larsson-Rondberg, Mr. Edvard A",Male,22,0,0,7.7750,0},
   { 3, "Ohman, Miss. Velin",Female,22,0,0,7.7750,1},
   { 3, "Connolly, Miss. Kate",Female,22,0,0,7.7500,1},
   { 3, "Nysten, Miss. Anna Sofia",Female,22,0,0,7.7500,1},
   { 3, "Bradley, Miss. Bridget Delia",Female,22,0,0,7.7250,1},
   { 3, "Karlsson, Mr. Nils August",Male,22,0,0,7.5208,0},
   { 3, "Braund, Mr. Owen Harris",Male,22,1,0,7.2500,0},
   { 3, "Dennis, Mr. Samuel",Male,22,0,0,7.2500,0},
   { 3, "Landergren, Miss. Aurora Adelia",Female,22,0,0,7.2500,1},
   { 3, "Perkin, Mr. John Henry",Male,22,0,0,7.2500,0},
   { 3, "Sirayanian, Mr. Orsen",Male,22,0,0,7.2292,0},
   { 3, "Leeni, Mr. Fahim (Philip Zenni)",Male,22,0,0,7.2250,1},
   { 3, "Vartanian, Mr. David",Male,22,0,0,7.2250,1},
   { 3, "Maenpaa, Mr. Matti Alexanteri",Male,22,0,0,7.1250,0},
   { 1, "Ryerson, Miss. Susan Parker Suzette",Female,21,2,2,262.3750,1},
   { 1, "Longley, Miss. Gretchen Fiske",Female,21,0,0,77.9583,1},
   { 1, "White, Mr. Richard Frasar",Male,21,0,1,77.2875,0},
   { 2, "Hickman, Mr. Stanley George",Male,21,2,0,73.5000,0},
   { 2, "Hood, Mr. Ambrose Jr",Male,21,0,0,73.5000,0},
   { 1, "Williams, Mr. Richard Norris II",Male,21,0,1,61.3792,1},
   { 3, "Ford, Miss. Doolina Margaret Daisy",Female,21,2,2,34.3750,0},
   { 1, "Willard, Miss. Constance",Female,21,0,0,26.5500,1},
   { 3, "Davies, Mr. John Samuel",Male,21,2,0,24.1500,0},
   { 2, "Phillips, Miss. Alice Frances Louisa",Female,21,0,1,21.0000,1},
   { 3, "Bowen, Mr. David John Dai",Male,21,0,0,16.1000,0},
   { 2, "Enander, Mr. Ingvar",Male,21,0,0,13.0000,0},
   { 2, "Cotterill, Mr. Henry Harry",Male,21,0,0,11.5000,0},
   { 2, "Giles, Mr. Edgar",Male,21,1,0,11.5000,0},
   { 2, "Giles, Mr. Frederick Edward",Male,21,1,0,11.5000,0},
   { 2, "Rugg, Miss. Emily",Female,21,0,0,10.5000,1},
   { 3, "Jussila, Miss. Mari Aina",Female,21,1,0,9.8250,0},
   { 3, "Cacic, Miss. Manda",Female,21,0,0,8.6625,0},
   { 3, "Pasic, Mr. Jakob",Male,21,0,0,8.6625,0},
   { 3, "Kalvik, Mr. Johannes Halvorsen",Male,21,0,0,8.4333,0},
   { 3, "Cann, Mr. Ernest Charles",Male,21,0,0,8.0500,0},
   { 3, "Reynolds, Mr. Harold J",Male,21,0,0,8.0500,0},
   { 3, "Stanley, Mr. Edward Roland",Male,21,0,0,8.0500,0},
   { 3, "Pekoniemi, Mr. Edvard",Male,21,0,0,7.9250,0},
   { 3, "Sivola, Mr. Antti Wilhelm",Male,21,0,0,7.9250,0},
   { 3, "Minkoff, Mr. Lazar",Male,21,0,0,7.8958,0},
   { 3, "Hansen, Mr. Henry Damsgaard",Male,21,0,0,7.8542,0},
   { 3, "Nilsson, Mr. August Ferdinand",Male,21,0,0,7.8542,0},
   { 3, "Buckley, Mr. Daniel",Male,21,0,0,7.8208,1},
   { 3, "Nosworthy, Mr. Richard Cater",Male,21,0,0,7.8000,0},
   { 3, "Jansson, Mr. Carl Olof",Male,21,0,0,7.7958,1},
   { 3, "Karlsson, Mr. Einar Gervasius",Male,21,0,0,7.7958,1},
   { 3, "Birkeland, Mr. Hans Martin Monsen",Male,21,0,0,7.7750,0},
   { 3, "Midtsjo, Mr. Karl Albert",Male,21,0,0,7.7750,1},
   { 3, "Canavan, Miss. Mary",Female,21,0,0,7.7500,0},
   { 3, "Canavan, Mr. Patrick",Male,21,0,0,7.7500,0},
   { 3, "Charters, Mr. David",Male,21,0,0,7.7333,0},
   { 3, "Salkjelsvik, Miss. Anna Kristine",Female,21,0,0,7.6500,1},
   { 3, "Windelov, Mr. Einar",Male,21,0,0,7.2500,0},
   { 3, "Assaf, Mr. Gerios",Male,21,0,0,7.2250,0},
   { 3, "Wiklund, Mr. Karl Johan",Male,21,1,0,6.4958,0},
   { 3, "Lovell, Mr. John Hall (Henry)",Male,20.5,0,0,7.2500,0},
   { 2, "Sincock, Miss. Maude",Female,20,0,0,36.7500,1},
   { 2, "Bryhl, Miss. Dagmar Jenny Ingeborg ",Female,20,1,0,26.0000,1},
   { 2, "Hocking, Miss. Ellen Nellie",Female,20,2,1,23.0000,1},
   { 3, "Nakid, Mr. Sahid",Male,20,1,1,15.7417,1},
   { 2, "Nourney, Mr. Alfred (Baron von Drachstedt)",Male,20,0,0,13.8625,1},
   { 3, "Gustafsson, Mr. Alfred Ossian",Male,20,0,0,9.8458,0},
   { 3, "Jussila, Miss. Katriina",Female,20,1,0,9.8250,0},
   { 3, "Hampe, Mr. Leon",Male,20,0,0,9.5000,0},
   { 3, "Olsvigen, Mr. Thor Anderson",Male,20,0,0,9.2250,0},
   { 3, "Oreskovic, Miss. Marija",Female,20,0,0,8.6625,0},
   { 3, "Oreskovic, Mr. Luka",Male,20,0,0,8.6625,0},
   { 3, "Saundercock, Mr. William Henry",Male,20,0,0,8.0500,0},
   { 3, "Abrahamsson, Mr. Abraham August Johannes",Male,20,0,0,7.9250,1},
   { 3, "Alhomaki, Mr. Ilmari Rudolf",Male,20,0,0,7.9250,0},
   { 3, "Lindqvist, Mr. Eino William",Male,20,1,0,7.9250,1},
   { 3, "Andreasson, Mr. Paul Edvin",Male,20,0,0,7.8542,0},
   { 3, "Braf, Miss. Elin Ester Maria",Female,20,0,0,7.8542,0},
   { 3, "Jensen, Mr. Hans Peder",Male,20,0,0,7.8542,0},
   { 3, "Vendel, Mr. Olof Edvin",Male,20,0,0,7.8542,0},
   { 3, "Barah, Mr. Hanna Assi",Male,20,0,0,7.2292,1},
   { 3, "Baccos, Mr. Raffull",Male,20,0,0,7.2250,0},
   { 3, "Coelho, Mr. Domingos Fernandeo",Male,20,0,0,7.0500,0},
   { 3, "Betros, Mr. Tannous",Male,20,0,0,4.0125,0},
   { 1, "Fortune, Mr. Charles Alexander",Male,19,3,2,263.0000,0},
   { 1, "Bishop, Mrs. Dickinson H (Helen Walton)",Female,19,1,0,91.0792,1},
   { 1, "Marvin, Mr. Daniel Warner",Male,19,1,0,53.1000,0},
   { 2, "Nicholls, Mr. Joseph Charles",Male,19,1,1,36.7500,0},
   { 1, "Graham, Miss. Margaret Edith",Female,19,0,0,30.0000,1},
   { 1, "Newsom, Miss. Helen Monypeny",Female,19,0,2,26.2833,1},
   { 2, "Beane, Mrs. Edward (Ethel Clarke)",Female,19,1,0,26.0000,1},
   { 2, "Phillips, Miss. Kate Florence (Mrs Kate Louise Phillips Marshall)",Female,19,0,0,26.0000,1},
   { 3, "McNamee, Mrs. Neal (Eileen O'Leary)",Female,19,1,0,16.1000,0},
   { 3, "Nakid, Mrs. Said (Waika Mary Mowad)",Female,19,1,1,15.7417,1},
   { 3, "Patchett, Mr. George",Male,19,0,0,14.5000,0},
   { 2, "Bentham, Miss. Lilian W",Female,19,0,0,13.0000,1},
   { 2, "Carbines, Mr. William",Male,19,0,0,13.0000,0},
   { 2, "Mellors, Mr. William John",Male,19,0,0,10.5000,1},
   { 2, "Pengelly, Mr. Frederick William",Male,19,0,0,10.5000,0},
   { 2, "Rogers, Mr. Reginald Harry",Male,19,0,0,10.5000,0},
   { 3, "Dakic, Mr. Branko",Male,19,0,0,10.1708,0},
   { 3, "Crease, Mr. Ernest James",Male,19,0,0,8.1583,0},
   { 3, "Beavan, Mr. William Thomas",Male,19,0,0,8.0500,0},
   { 3, "Dorking, Mr. Edward Arthur",Male,19,0,0,8.0500,1},
   { 3, "Cor, Mr. Liudevit",Male,19,0,0,7.8958,0},
   { 3, "Petroff, Mr. Nedelio",Male,19,0,0,7.8958,0},
   { 3, "Stoytcheff, Mr. Ilia",Male,19,0,0,7.8958,0},
   { 3, "Devaney, Miss. Margaret Delia",Female,19,0,0,7.8792,1},
   { 3, "Andersen-Jensen, Miss. Carla Christine Nielsine",Female,19,1,0,7.8542,1},
   { 3, "Gustafsson, Mr. Karl Gideon",Male,19,0,0,7.7750,0},
   { 3, "Soholt, Mr. Peter Andreas Lauritz Andersen",Male,19,0,0,7.6500,0},
   { 3, "Burke, Mr. Jeremiah",Male,19,0,0,6.7500,0},
   { 3, "Johnson, Mr. William Cahoone Jr",Male,19,0,0,0.0000,0},
   { 2, "Swane, Mr. George",Male,18.5,0,0,13.0000,0},
   { 3, "Buckley, Miss. Katherine",Female,18.5,0,0,7.2833,0},
   { 3, "Katavelas, Mr. Vassilios (Catavelas Vassilios)",Male,18.5,0,0,7.2292,0},
   { 1, "Ryerson, Miss. Emily Borie",Female,18,2,2,262.3750,1},
   { 1, "Astor, Mrs. John Jacob (Madeleine Talmadge Force)",Female,18,1,0,227.5250,1},
   { 1, "Penasco y Castellana, Mr. Victor de Satode",Male,18,1,0,108.9000,0},
   { 1, "Taussig, Miss. Ruth",Female,18,0,2,79.6500,1},
   { 2, "Davies, Mr. Charles Henry",Male,18,0,0,73.5000,0},
   { 2, "Dibden, Mr. William",Male,18,0,0,73.5000,0},
   { 1, "Smith, Mrs. Lucien Philip (Mary Eloise Hughes)",Female,18,1,0,60.0000,1},
   { 1, "Marvin, Mrs. Daniel Warner (Mary Graham Carmichael Farquarson)",Female,18,1,0,53.1000,1},
   { 3, "Ford, Mr. Edward Watson",Male,18,2,2,34.3750,0},
   { 2, "Doling, Miss. Elsie",Female,18,0,1,23.0000,1},
   { 3, "Rosblom, Mr. Viktor Richard",Male,18,1,1,20.2125,0},
   { 3, "Vander Planke, Miss. Augusta Maria",Female,18,2,0,18.0000,0},
   { 3, "Arnold-Franchi, Mrs. Josef (Josefine Franchi)",Female,18,1,0,17.8000,0},
   { 3, "Barbara, Miss. Saiide",Female,18,0,1,14.4542,0},
   { 3, "Chronopoulos, Mr. Demetrios",Male,18,1,0,14.4542,0},
   { 2, "Fahlstrom, Mr. Arne Jonas",Male,18,0,0,13.0000,0},
   { 2, "Hiltunen, Miss. Marta",Female,18,1,1,13.0000,0},
   { 2, "Silven, Miss. Lyyli Karoliina",Female,18,0,2,13.0000,1},
   { 2, "Andrew, Mr. Edgardo Samuel",Male,18,0,0,11.5000,0},
   { 2, "Bailey, Mr. Percy Andrew",Male,18,0,0,11.5000,0},
   { 2, "Fillbrook, Mr. Joseph Charles",Male,18,0,0,10.5000,0},
   { 3, "Turja, Miss. Anna Sofia",Female,18,0,0,9.8417,1},
   { 3, "Aks, Mrs. Sam (Leah Rosen)",Female,18,0,1,9.3500,1},
   { 3, "Cacic, Mr. Jego Grga",Male,18,0,0,8.6625,0},
   { 3, "Allum, Mr. Owen George",Male,18,0,0,8.3000,0},
   { 3, "Badman, Miss. Emily Louisa",Female,18,0,0,8.0500,1},
   { 3, "Cohen, Mr. Gurshon Gus",Male,18,0,0,8.0500,1},
   { 3, "Burns, Miss. Mary Delia",Female,18,0,0,7.8792,0},
   { 3, "Klasen, Mr. Klas Albin",Male,18,1,1,7.8542,0},
   { 3, "Fischer, Mr. Eberhard Thelander",Male,18,0,0,7.7958,0},
   { 3, "Edvardsson, Mr. Gustaf Hjalmar",Male,18,0,0,7.7750,0},
   { 3, "Nilsson, Miss. Berta Olivia",Female,18,0,0,7.7750,1},
   { 3, "Pettersson, Miss. Ellen Natalia",Female,18,0,0,7.7750,0},
   { 3, "Bjorklund, Mr. Ernst Herbert",Male,18,0,0,7.7500,0},
   { 3, "Myhrman, Mr. Pehr Fabian Oliver Malkolm",Male,18,0,0,7.7500,0},
   { 3, "Sjoblom, Miss. Anna Sofia",Female,18,0,0,7.4958,1},
   { 3, "Abrahim, Mrs. Joseph (Sophie Halaut Easu)",Female,18,0,0,7.2292,1},
   { 3, "Hegarty, Miss. Hanora Nora",Female,18,0,0,6.7500,0},
   { 3, "Wiklund, Mr. Jakob Alfred",Male,18,1,0,6.4958,0},
   { 1, "Thayer, Mr. John Borland Jr",Male,17,0,2,110.8833,1},
   { 1, "Penasco y Castellana, Mrs. Victor de Satode (Maria Josefa Perez de Soto y Vallejo)",Female,17,1,0,108.9000,1},
   { 2, "Deacon, Mr. Percy William",Male,17,0,0,73.5000,0},
   { 1, "Dick, Mrs. Albert Adrian (Vera Gillespie)",Female,17,1,0,57.0000,1},
   { 1, "Carrau, Mr. Jose Pedro",Male,17,0,0,47.1000,0},
   { 3, "Cribb, Miss. Laura Alice",Female,17,0,1,16.1000,1},
   { 3, "Attalah, Miss. Malake",Female,17,0,0,14.4583,0},
   { 2, "Lehmann, Miss. Bertha",Female,17,0,0,12.0000,1},
   { 2, "Ilett, Miss. Bertha",Female,17,0,0,10.5000,1},
   { 3, "Calic, Mr. Jovo",Male,17,0,0,8.6625,0},
   { 3, "Calic, Mr. Petar",Male,17,0,0,8.6625,0},
   { 3, "Culumovic, Mr. Jeso",Male,17,0,0,8.6625,0},
   { 3, "Pokrnic, Mr. Mate",Male,17,0,0,8.6625,0},
   { 3, "Davies, Mr. Joseph",Male,17,2,0,8.0500,0},
   { 3, "Andersson, Miss. Erna Alexandra",Female,17,4,2,7.9250,1},
   { 3, "Dika, Mr. Mirko",Male,17,0,0,7.8958,0},
   { 3, "Hagardon, Miss. Kate",Female,17,0,0,7.7333,0},
   { 3, "Elias, Mr. Joseph Jr",Male,17,1,1,7.2292,0},
   { 3, "Kallio, Mr. Nikolai Erland",Male,17,0,0,7.1250,0},
   { 3, "Jensen, Mr. Svend Lauritz",Male,17,1,0,7.0542,0},
   { 1, "Maioni, Miss. Roberta",Female,16,0,0,86.5000,1},
   { 1, "Hippach, Miss. Jean Gertrude",Female,16,0,1,57.9792,1},
   { 3, "Goodwin, Miss. Lillian Amy",Female,16,5,2,46.9000,0},
   { 3, "Panula, Mr. Ernesti Arvid",Male,16,4,1,39.6875,0},
   { 1, "Lines, Miss. Mary Conover",Female,16,0,1,39.4000,1},
   { 3, "Ford, Mr. William Neal",Male,16,1,3,34.3750,0},
   { 2, "Gaskell, Mr. Alfred",Male,16,0,0,26.0000,0},
   { 3, "Abbott, Mr. Rossmore Edward",Male,16,1,1,20.2500,0},
   { 3, "Vander Planke, Mr. Leo Edmondus",Male,16,2,0,18.0000,0},
   { 2, "Mudd, Mr. Thomas Charles",Male,16,0,0,10.5000,0},
   { 3, "de Pelsmaeker, Mr. Alfons",Male,16,0,0,9.5000,0},
   { 3, "Osen, Mr. Olaf Elon",Male,16,0,0,9.2167,0},
   { 3, "Thomas, Mrs. Alexander (Thamine Thelma)",Female,16,1,1,8.5167,1},
   { 3, "Rush, Mr. Alfred George John",Male,16,0,0,8.0500,0},
   { 3, "Sunderland, Mr. Victor Francis",Male,16,0,0,8.0500,1},
   { 3, "Eklund, Mr. Hans Linus",Male,16,0,0,7.7750,0},
   { 3, "Carr, Miss. Helen Ellen",Female,16,0,0,7.7500,1},
   { 3, "Gilnagh, Miss. Katherine Katie",Female,16,0,0,7.7333,1},
   { 3, "Abelseth, Miss. Karen Marie",Female,16,0,0,7.6500,1},
   { 1, "Madill, Miss. Georgette Alexandra",Female,15,0,1,211.3375,1},
   { 2, "Brown, Miss. Edith Eileen",Female,15,0,2,39.0000,1},
   { 3, "Yasbeck, Mrs. Antoni (Selini Alexander)",Female,15,1,0,14.4542,1},
   { 3, "McGowan, Miss. Anna Annie",Female,15,0,0,8.0292,1},
   { 3, "Elias, Mr. Tannous",Male,15,1,1,7.2292,0},
   { 3, "Najib, Miss. Adele Kiamie Jane",Female,15,0,0,7.2250,1},
   { 3, "Sage, Master. William Henry",Male,14.5,8,2,69.5500,0},
   { 3, "Zabour, Miss. Hileni",Female,14.5,1,0,14.4542,0},
   { 1, "Carter, Miss. Lucile Polk",Female,14,1,2,120.0000,1},
   { 2, "Sweet, Mr. George Frederick",Male,14,0,0,65.0000,0},
   { 3, "Goodwin, Mr. Charles Edward",Male,14,5,2,46.9000,0},
   { 3, "Panula, Mr. Jaako Arnold",Male,14,4,1,39.6875,0},
   { 2, "Nasser, Mrs. Nicholas (Adele Achem)",Female,14,1,0,30.0708,1},
   { 3, "Nicola-Yarred, Miss. Jamila",Female,14,1,0,11.2417,1},
   { 3, "Svensson, Mr. Johan Cervin",Male,14,0,0,9.2250,1},
   { 3, "Vestrom, Miss. Hulda Amanda Adolfina",Female,14,0,0,7.8542,0},
   { 1, "Ryerson, Master. John Borie",Male,13,2,2,262.3750,1},
   { 3, "Asplund, Master. Filip Oscar",Male,13,4,2,31.3875,0},
   { 3, "Abbott, Master. Eugene Joseph",Male,13,0,2,20.2500,0},
   { 2, "Mellinger, Miss. Madeleine Violet",Female,13,0,1,19.5000,1},
   { 3, "Ayoub, Miss. Banoura",Female,13,0,0,7.2292,1},
   { 2, "Becker, Miss. Ruth Elizabeth",Female,12,2,1,39.0000,1},
   { 2, "Watt, Miss. Bertha J",Female,12,0,0,15.7500,1},
   { 3, "Nicola-Yarred, Master. Elias",Male,12,1,0,11.2417,1},
   { 3, "van Billiard, Master. Walter John",Male,11.5,1,1,14.5000,0},
   { 1, "Carter, Master. William Thornton II",Male,11,1,2,120.0000,1},
   { 3, "Goodwin, Master. William Frederick",Male,11,5,2,46.9000,0},
   { 3, "Andersson, Miss. Sigrid Elisabeth",Female,11,4,2,31.2750,0},
   { 3, "Hassan, Mr. Houssein G N",Male,11,0,0,18.7875,0},
   { 3, "Goodwin, Miss. Jessie Allis",Female,10,5,2,46.9000,0},
   { 3, "Rice, Master. Albert",Male,10,4,1,29.1250,0},
   { 3, "Skoog, Master. Karl Thorsten",Male,10,3,2,27.9000,0},
   { 3, "Van Impe, Miss. Catharina",Female,10,0,2,24.1500,0},
   { 3, "Goodwin, Master. Harold Victor",Male,9,5,2,46.9000,0},
   { 3, "Ford, Miss. Robina Maggie Ruby",Female,9,2,2,34.3750,0},
   { 3, "Asplund, Master. Clarence Gustaf Hugo",Male,9,4,2,31.3875,0},
   { 3, "Andersson, Miss. Ingeborg Constanzia",Female,9,4,2,31.2750,0},
   { 3, "Skoog, Miss. Mabel",Female,9,3,2,27.9000,0},
   { 3, "Goldsmith, Master. Frank John William Frankie",Male,9,0,2,20.5250,1},
   { 3, "Coutts, Master. Eden Leslie Neville",Male,9,1,1,15.9000,1},
   { 3, "Boulos, Miss. Nourelain",Female,9,1,1,15.2458,0},
   { 3, "Touma, Miss. Maria Youssef",Female,9,1,1,15.2458,1},
   { 3, "Olsen, Master. Artur Karl",Male,9,0,1,3.1708,1},
   { 2, "Davies, Master. John Morgan Jr",Male,8,1,1,36.7500,1},
   { 2, "Drew, Master. Marshall Brines",Male,8,0,2,32.5000,1},
   { 3, "Rice, Master. George Hugh",Male,8,4,1,29.1250,0},
   { 2, "Collyer, Miss. Marjorie Lottie",Female,8,0,2,26.2500,1},
   { 2, "Quick, Miss. Winifred Vera",Female,8,1,1,26.0000,1},
   { 3, "Palsson, Miss. Torborg Danira",Female,8,3,1,21.0750,0},
   { 3, "Panula, Master. Juha Niilo",Male,7,4,1,39.6875,0},
   { 3, "Rice, Master. Eric",Male,7,4,1,29.1250,0},
   { 2, "Hart, Miss. Eva Miriam",Female,7,0,2,26.2500,1},
   { 3, "Touma, Master. Georges Youssef",Male,7,1,1,15.2458,1},
   { 1, "Spedden, Master. Robert Douglas",Male,6,0,2,134.5000,1},
   { 2, "Harper, Miss. Annie Jessie Nina",Female,6,0,1,33.0000,1},
   { 3, "Andersson, Miss. Ebba Iris Alfrida",Female,6,4,2,31.2750,0},
   { 3, "Palsson, Master. Paul Folke",Male,6,3,1,21.0750,0},
   { 3, "Boulos, Master. Akar",Male,6,1,1,15.2458,0},
   { 3, "Moor, Master. Meier",Male,6,0,1,12.4750,1},
   { 3, "Asplund, Master. Carl Edgar",Male,5,4,2,31.3875,0},
   { 3, "Asplund, Miss. Lillian Gertrud",Female,5,4,2,31.3875,1},
   { 2, "West, Miss. Constance Mirium",Female,5,1,2,27.7500,1},
   { 3, "Baclini, Miss. Marie Catherine",Female,5,2,1,19.2583,1},
   { 3, "Emanuel, Miss. Virginia Ethel",Female,5,0,0,12.4750,1},
   { 1, "Dodge, Master. Washington",Male,4,0,2,81.8583,1},
   { 2, "Becker, Miss. Marion Louise",Female,4,2,1,39.0000,1},
   { 3, "Andersson, Master. Sigvard Harald Elias",Male,4,4,2,31.2750,0},
   { 3, "Rice, Master. Arthur",Male,4,4,1,29.1250,0},
   { 3, "Skoog, Master. Harald",Male,4,3,2,27.9000,0},
   { 2, "Wells, Miss. Joan",Female,4,1,1,23.0000,1},
   { 3, "Kink-Heilmann, Miss. Luise Gretchen",Female,4,0,2,22.0250,1},
   { 3, "Sandstrom, Miss. Marguerite Rut",Female,4,1,1,16.7000,1},
   { 3, "Karun, Miss. Manca",Female,4,0,1,13.4167,1},
   { 3, "Johnson, Master. Harold Theodor",Male,4,1,1,11.1333,1},
   { 2, "Laroche, Miss. Simonne Marie Anne Andree",Female,3,1,2,41.5792,1},
   { 3, "Asplund, Master. Edvin Rojj Felix",Male,3,4,2,31.3875,1},
   { 2, "Navratil, Master. Michel M",Male,3,1,1,26.0000,1},
   { 3, "Palsson, Miss. Stina Viola",Female,3,3,1,21.0750,0},
   { 2, "Richards, Master. William Rowe",Male,3,1,1,18.7500,1},
   { 3, "Coutts, Master. William Loch William",Male,3,1,1,15.9000,1},
   { 3, "Peacock, Miss. Treasteall",Female,3,1,1,13.7750,0},
   { 1, "Allison, Miss. Helen Loraine",Female,2,1,2,151.5500,0},
   { 3, "Panula, Master. Urho Abraham",Male,2,4,1,39.6875,0},
   { 3, "Andersson, Miss. Ellis Anna Maria",Female,2,4,2,31.2750,0},
   { 3, "Rice, Master. Eugene",Male,2,4,1,29.1250,0},
   { 3, "Skoog, Miss. Margit Elizabeth",Female,2,3,2,27.9000,0},
   { 2, "Navratil, Master. Edmond Roger",Male,2,1,1,26.0000,1},
   { 2, "Quick, Miss. Phyllis May",Female,2,1,1,26.0000,1},
   { 2, "Wells, Master. Ralph Lester",Male,2,1,1,23.0000,1},
   { 3, "Palsson, Master. Gosta Leonard",Male,2,3,1,21.0750,0},
   { 3, "Rosblom, Miss. Salli Helena",Female,2,1,1,20.2125,0},
   { 3, "Hirvonen, Miss. Hildur E",Female,2,0,1,12.2875,1},
   { 3, "Strom, Miss. Telma Matilda",Female,2,0,1,10.4625,0},
   { 3, "Goodwin, Master. Sidney Leonard",Male,1,5,2,46.9000,0},
   { 2, "Laroche, Miss. Louise",Female,1,1,2,41.5792,1},
   { 3, "Panula, Master. Eino Viljami",Male,1,4,1,39.6875,0},
   { 2, "Becker, Master. Richard F",Male,1,2,1,39.0000,1},
   { 2, "Mallet, Master. Andre",Male,1,0,2,37.0042,1},
   { 3, "Dean, Master. Bertram Vere",Male,1,1,2,20.5750,1},
   { 3, "Sandstrom, Miss. Beatrice Irene",Female,1,1,1,16.7000,1},
   { 3, "Nakid, Miss. Maria (Mary)",Female,1,0,2,15.7417,1},
   { 3, "Klasen, Miss. Gertrud Emilia",Female,1,1,1,12.1833,0},
   { 3, "Johnson, Miss. Eleanor Ileen",Female,1,1,1,11.1333,1},
   { 1, "Allison, Master. Hudson Trevor",Male,0.9167,1,2,151.5500,1},
   { 2, "West, Miss. Barbara J",Female,0.9167,1,2,27.7500,1},
   { 2, "Caldwell, Master. Alden Gates",Male,0.8333,0,2,29.0000,1},
   { 2, "Richards, Master. George Sibley",Male,0.8333,1,1,18.7500,1},
   { 3, "Aks, Master. Philip Frank",Male,0.8333,0,1,9.3500,1},
   { 3, "Baclini, Miss. Eugenie",Female,0.75,2,1,19.2583,1},
   { 3, "Baclini, Miss. Helene Barbara",Female,0.75,2,1,19.2583,1},
   { 3, "Peacock, Master. Alfred Edward",Male,0.75,1,1,13.7750,0},
   { 2, "Hamalainen, Master. Viljo",Male,0.6667,1,1,14.5000,1},
   { 3, "Thomas, Master. Assad Alexander",Male,0.4167,0,1,8.5167,1},
   { 3, "Danbom, Master. Gilbert Sigvard Emanuel",Male,0.3333,0,2,14.4000,0},

   { 0 } // End Of Database
};

