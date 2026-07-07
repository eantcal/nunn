# MNIST and OCR

nuNN includes both a command-line MNIST demo and a Windows OCR GUI.

## MNIST Data Representation

MNIST digits are 28 by 28 grayscale images. In the MLP demos, each image is normalized and flattened into a vector of 784 values.

![MNIST digit](assets/mnist-digit.png)

![MNIST vectorization](assets/mnist-flatten-vector.png)

The output layer has 10 neurons, one per digit class.

![MNIST MLP OCR](assets/mnist-mlp-ocr.png)

## `mnist_test`

`mnist_test` trains and evaluates a network on MNIST.

Default training mode:

```sh
mnist_test -p /path/to/mnist
```

Defaults:

- `MlpMatrixNN`
- backend `auto`
- batch size `100`

Useful alternatives:

```sh
mnist_test -p /path/to/mnist --mlp
mnist_test -p /path/to/mnist --backend cpu
mnist_test -p /path/to/mnist --backend opencl
mnist_test -p /path/to/mnist --batch 1
```

`--mlp` selects the classic `MlpNN` path.

## `ocr_test`

`ocr_test` is the interactive handwritten digit demo. It can:

- load `.net` and JSON models;
- recognize a digit drawn by the user;
- train a new MNIST model from the Train menu;
- persist MNIST and save paths in the Windows registry;
- show a cost-convergence chart during training;
- select `Auto`, `CPU`, or `OpenCL` backend.

The installed package places OCR models under:

```text
bin/models
```

`ocr_test` also searches:

```text
bin
../share/nunn/nets
../share/nunn/models
```

This keeps the GUI usable both from the build tree and from an installed package.

