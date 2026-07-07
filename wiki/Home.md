# nuNN Library Wiki

nuNN is a compact C++20 machine learning library built for study, experimentation, and small practical demos. The project intentionally keeps the algorithms visible: forward passes, gradients, training loops, serialization, and example applications are implemented in readable C++ instead of being hidden behind a large framework.

This wiki complements the README:

- the README is the quick project overview;
- NEWS is the release history;
- the wiki is the guided documentation layer, with theory notes, diagrams, and links back to the implementation.

## Quick Links

- [Getting Started](Getting-Started)
- [Neural Networks](Neural-Networks)
- [MNIST and OCR](MNIST-and-OCR)
- [Recurrent Networks](Recurrent-Networks)
- [Convolutional Networks and Transformer](Convolutional-and-Transformer)
- [Reinforcement Learning](Reinforcement-Learning)
- [Examples Gallery](Examples-Gallery)
- [Theory Notes](Theory-Notes)

## What nuNN Includes

nuNN includes feedforward neural networks, recurrent networks, unsupervised models, convolutional components, a small decoder-only transformer, tabular reinforcement learning, DQN, classical ML helpers, model serialization, Graphviz topology export, MNIST parsing, OCR demos, and package/install helpers.

The current implementation includes:

- `Perceptron`
- `MlpNN`
- `MlpMatrixNN`
- `VanillaRnn`
- `Gru`
- `Lstm`
- `HopfieldNN`
- `Autoencoder`
- `Rbf`
- `Rbm`
- `Vae`
- `Som`
- `Conv1DLayer`, `MaxPool1DLayer`, `ConvNet`
- `MiniTransformer`
- `LinearRegression`
- `KMeans`
- `Pca`
- tabular `QLearn` / `Sarsa`
- `Dqn` with replay buffer and target network

## Using Book Material

Some diagrams in this wiki are adapted from the accompanying book material and stored as PNG assets under `assets/`. They are used here to connect the theory to the corresponding nuNN classes and demo programs.

