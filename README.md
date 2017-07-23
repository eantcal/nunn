# nunn
Nunn Library is a Free Open Source Machine Learning Library distributed under GPLv2 License and written in C++11/C++14

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


## TicTacToe Demo (tictactoe)
Basic Tic Tac Toe game which uses neural networks. 

### TicTacToe Demo for Windows (winttt)
Winttt is an interactive Tic Tac Toe version for Windows which may be dynamically 
trained or may use trained neural networks, including those nets created by
using tictactoe program.

## XOR Problem sample (xor_test)
A typical example of non-linealy separable function is the XOR.
Implementing the XOR function is a classic problem in neural networks.
XOR computes the logical exclusive-or, which yields 1 if and 
only if the two inputs have different values.
So, this classification can not be solved with linear separation, 
but is very easy for a neural network to generate a non-linear solution to.
 
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
