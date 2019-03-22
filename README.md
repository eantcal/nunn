# nunn
Nunn Library is a Free Open Source Machine Learning Library distributed under MIT License and written in C++11/C++14

## Features
- Supports fully connected multi-layers neural networks and other ML algorithms
- Easy to use and understand
- Easy to save and load entire states
- Multi-platform

The library package includes the following samples and tools.

## Nunn Topology -> Graphviz format converter (nunn_topo)
Using this tool you can export neural network topologies and draw them using Graphviz dot.
dot draws directed graphs. It reads attributed graph text files and writes drawings,
either as graph files or in a graphics format such as GIF, PNG, SVG or PostScript
(which can be converted to PDF).

## MNIST Test Demo (mnist_test)
This demo trains and tests (R)MLP neural network against the MNIST.
The test is performed by using the MNIST data set, which contains 60K + 10K 
scanned images of handwritten digits with their correct classifications
The images are greyscale and 28 by 28 pixels in size.
The first part of 60,000 images were used as training data.
The second part of 10,000 images were used as test data.
The test data was taken from a different set of people than the original training data.
The training input is treated as a 28×28=784-dimensional vector.
Each entry in the vector represents the grey value for a single pixel in the image.
The corresponding desired output is a 10-dimensional vector.
You may obtain info about MNIST at link http://yann.lecun.com/exdb/mnist/

### Handwritten Digit OCR Demo (ocr_test)
This is an interactive demo which uses MNIST trained neural network created 
by using nunn library.
nunn status files (.net) have been created by mnist_test application

### [![Click here for watching the video](https://youtu.be/ereeEG_1lmY)](https://youtu.be/ereeEG_1lmY)
![ocr_test](https://37bdcab3-a-62cb3a1a-s-sites.googlegroups.com/site/nunnlibrary/samples/ocr_test2..jpg)

## TicTacToe Demo (tictactoe)
Basic Tic Tac Toe game which uses neural networks. 

### TicTacToe Demo for Windows (winttt)
Winttt is an interactive Tic Tac Toe version for Windows which may be dynamically 
trained or may use trained neural networks, including those nets created by
using tictactoe program.

![tictactoe](https://37bdcab3-a-62cb3a1a-s-sites.googlegroups.com/site/nunnlibrary/samples/winttt2.jpg)


## XOR Problem sample (xor_test)
A typical example of non-linealy separable function is the XOR.
Implementing the XOR function is a classic problem in neural networks.
 
### Solving the XOR Problem with Nunn Library
XOR function takes two input arguments with values in [0,1] and returns one output in [0,1], as specified in the following table:

```
 x1|x2 |  y
 --+---+----
 0 | 0 |  0
 0 | 1 |  1
 1 | 0 |  1
 1 | 1 |  0
```

Which means that XOR computes the logical exclusive-or, which yields 1 if and only if the two inputs have different values.
So, this classification can not be solved with linear separation, but is very easy for an MLP to generate a non-linear solution to.

### Xor function implementation step by step

Test has been performed training an MLP network. 
During training you can give the algorithm examples of what you want the network to do and it changes the network’s weights. 
When training is finished, it will give you the required output for a particular input.

- Step 1: include MLP NN header
```
#include "nu_mlpnn.h"
#include <iostream>
#include <map>
```
- Step 2: Define net topology

![topology](http://www.nunnlib.eu/_/rsrc/1468877158765/samples/xor-sample/xortopo.jpg?height=122&width=200)

```
int main(int argc, char* argv[])
{
  using vect_t = nu::mlp_neural_net_t::rvector_t;
  nu::mlp_neural_net_t::topology_t topology = {
      2, // input layer takes a two dimensional vector
      2, // hidden layer size
      1  // output
  };
```

- Step 3: Construct the network object specifying topology, learning rate and momentum
```
  try 
  {
      nu::mlp_neural_net_t nn
      {
         topology,
         0.4, // learning rate
         0.9, // momentum
      };
```

- Step 4: Create a training set needed to train the net.
Training set must be a collection of <input-vector, output-vector> pairs. 

```
// Create a training set
      using training_set_t = std::map< std::vector<double>, std::vector<double> >;
      training_set_t traing_set = {
         { { 0, 0 },{ 0 } },
         { { 0, 1 },{ 1 } },
         { { 1, 0 },{ 1 } },
         { { 1, 1 },{ 0 } }
      };
```
- Step 5: Train the net using a trainer object. 
Trainer object iterates for each element of training set until the max number of epochs (20000) is reached or error computed by function passed as second parameter to train() method is less than min error (0.01).

```
        nu::mlp_nn_trainer_t trainer(
         nn,
         20000, // Max number of epochs
         0.01   // Min error
      );
      std::cout
         << "XOR training start ( Max epochs count=" << trainer.get_epochs()
         << " Minimum error=" << trainer.get_min_err() << " )"
         << std::endl;
      trainer.train<training_set_t>(
         traing_set,
         [](
            nu::mlp_neural_net_t& net,
            const nu::mlp_neural_net_t::rvector_t & target) -> double
         {
            static size_t i = 0;
            if (i++ % 200 == 0)
               std::cout << ">";
             return net.mean_squared_error(target);
         }
      );
```

- Step 6: Test if net learnt XOR-function

```
auto step_f = [](double x) { return x < 0.5 ? 0 : 1; };
     std::cout <<  std::endl << " XOR Test " << std::endl;
     for (int a = 0; a < 2; ++a)
     {
        for (int b = 0; b < 2; ++b)
        {  
            vect_t output_vec{ 0.0 };
            vect_t input_vec{ double(a), double(b) }; 
            nn.set_inputs(input_vec);
            nn.feed_forward();
            nn.get_outputs(output_vec);
            // Dump the network status
            std::cout << nn;
            std::cout << "-------------------------------" << std::endl;
            auto net_res = step_f(output_vec[0]);
            std::cout << a << " xor " << b << " = " << net_res << std::endl;
            auto xor_res = a ^ b;
            if (xor_res != net_res)
            {
               std::cerr
                  << "ERROR!: xor(" << a << "," << b << ") !="
                  << xor_res
                  << std::endl; 
               return 1;
            }
            std::cout << "-------------------------------" << std::endl;
        }
     } 
      std::cout << "Test completed successfully" << std::endl;
   }
   catch (nu::mlp_neural_net_t::exception_t & e)
   {
      std::cerr << "nu::mlp_neural_net_t::exception_t n# " << int(e) << std::endl;
      std::cerr << "Check for configuration parameters and retry" << std::endl; 
      return 1;
   }
   catch (...)
   {
      std::cerr
         << "Fatal error. Check for configuration parameters and retry" << std::endl;
      return 1;
   } 
   return 0;
}

```

### Program output

```
XOR training start ( Max epochs count=20000 Minimum error=0.01) 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

XOR Test
Net Inputs
        [0] = 0
        [1] = 0

Neuron layer 0 Hidden
        Neuron 0
                Input  [0] = 0
                Weight [0] = 0.941384
                Input  [1] = 0
                Weight [1] = 0.94404
                Bias =       0.0307751
                Ouput = 0.507693
                Error = 0.0707432
        Neuron 1
                Input  [0] = 0
                Weight [0] = 6.19317
                Input  [1] = 0
                Weight [1] = 6.49756
                Bias =       -0.0227467
                Ouput = 0.494314
                Error = -0.0568667

Neuron layer 1 Output
        Neuron 0
                Input  [0] = 0.507693
                Weight [0] = -16.4831
                Input  [1] = 0.494314
                Weight [1] = 13.2566
                Bias =       -0.00652012
                Ouput = 0.139202
                Error = -0.0171672
-------------------------------
0 xor 0 = 0
-------------------------------
Net Inputs
        [0] = 0
        [1] = 1

Neuron layer 0 Hidden
        Neuron 0
                Input  [0] = 0
                Weight [0] = 0.941384
                Input  [1] = 1
                Weight [1] = 0.94404
                Bias =       0.0307751
                Ouput = 0.726078
                Error = 0.0707432
        Neuron 1
                Input  [0] = 0
                Weight [0] = 6.19317
                Input  [1] = 1
                Weight [1] = 6.49756
                Bias =       -0.0227467
                Ouput = 0.998461
                Error = -0.0568667

Neuron layer 1 Output
        Neuron 0
                Input  [0] = 0.726078
                Weight [0] = -16.4831
                Input  [1] = 0.998461
                Weight [1] = 13.2566
                Bias =       -0.00652012
                Ouput = 0.779318
                Error = -0.0171672
-------------------------------
0 xor 1 = 1
-------------------------------
Net Inputs
        [0] = 1
        [1] = 0

Neuron layer 0 Hidden
        Neuron 0
                Input  [0] = 1
                Weight [0] = 0.941384
                Input  [1] = 0
                Weight [1] = 0.94404
                Bias =       0.0307751
                Ouput = 0.72555
                Error = 0.0707432
        Neuron 1
                Input  [0] = 1
                Weight [0] = 6.19317
                Input  [1] = 0
                Weight [1] = 6.49756
                Bias =       -0.0227467
                Ouput = 0.997914
                Error = -0.0568667

Neuron layer 1 Output
        Neuron 0
                Input  [0] = 0.72555
                Weight [0] = -16.4831
                Input  [1] = 0.997914
                Weight [1] = 13.2566
                Bias =       -0.00652012
                Ouput = 0.77957
                Error = -0.0171672
-------------------------------
1 xor 0 = 1
-------------------------------
Net Inputs
        [0] = 1
        [1] = 1

Neuron layer 0 Hidden
        Neuron 0
                Input  [0] = 1
                Weight [0] = 0.941384
                Input  [1] = 1
                Weight [1] = 0.94404
                Bias =       0.0307751
                Ouput = 0.871714
                Error = 0.0707432
        Neuron 1
                Input  [0] = 1
                Weight [0] = 6.19317
                Input  [1] = 1
                Weight [1] = 6.49756
                Bias =       -0.0227467
                Ouput = 0.999997
                Error = -0.0568667

Neuron layer 1 Output
        Neuron 0
                Input  [0] = 0.871714
                Weight [0] = -16.4831
                Input  [1] = 0.999997
                Weight [1] = 13.2566
                Bias =       -0.00652012
                Ouput = 0.246297
                Error = -0.0171672
-------------------------------
1 xor 1 = 0
-------------------------------
Test completed successfully
```

Topology is defined via a vector of positive integers, where first one represents the input layer size and last one represents the output layer size.
All other values represent the hidden layers from input to output. 
The topology vector must be at least of 3 items and all of them must be non-zero positive integer values

## Perceptron AND sample (and_test)
AND function implemented using a perceptron
A typical example of linearly separable function is the AND. This type
of function can be learned by a single preceptron neural net
AND computes the logical-AND, which yields 1 if and 
only if the two inputs have 1 values.

## Hopfield Test (hopfield_test)
The Hopfield networks may be used to solve the recall problem of matching cues 
for an input pattern to an associated pre-learned pattern.
They are form of recurrent artificial neural networks, which
serve as content-addressable memory systems with binary threshold nodes.
This test shows an use-case of hopfield net used as auto-associative memory
In this example we recognize 100-pixel picture with the 100-neuron neural 
network.

![hopfield test](https://37bdcab3-a-62cb3a1a-s-sites.googlegroups.com/site/nunnlibrary/samples/hopfield.jpg)
