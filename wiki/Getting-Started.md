# Getting Started

This page takes the shortest reliable path from a fresh checkout to a trained model. Commands match the current [top-level CMake configuration](https://github.com/eantcal/nunn/blob/main/CMakeLists.txt), and the first program is reduced from the checked-in [`and_test` example](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc).

## Requirements

You need:

- CMake 3.14 or newer;
- a compiler with C++20 support;
- Git and network access during the first configure, because CMake fetches Eigen 3.4 and nlohmann/json, plus GoogleTest when tests are enabled.

Visual Studio 2022 is the normal Windows toolchain. Recent GCC or Clang works on Linux and macOS.

Optional components:

| Component | Needed for | Behavior when absent |
| --- | --- | --- |
| ArrayFire with OpenCL | GPU path in `MlpMatrixNN` | `ComputeBackend::Auto` falls back to Eigen/CPU |
| Graphviz `dot` | SVG, PNG, and PDF output from `nunn_topo` | DOT text output still works |
| MNIST IDX files | `mnist_test` and OCR training | other examples and all model tests remain available |

## Configure, build, and test

A CPU-only configuration is the most predictable first build:

```sh
git clone https://github.com/eantcal/nunn.git
cd nunn
cmake -S . -B build -DNUNN_ENABLE_OPENCL=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The important options are defined in [`CMakeLists.txt`](https://github.com/eantcal/nunn/blob/main/CMakeLists.txt):

| Option | Default | Meaning |
| --- | --- | --- |
| `NUNN_BUILD_TESTS` | `ON` | Build the GoogleTest suite |
| `NUNN_ENABLE_OPENCL` | `ON` | Look for ArrayFire/OpenCL; absence is not fatal |
| `NUNN_BUILD_OCR_RUNTIME_FALLBACK` | `ON` | On eligible Windows builds, provide OpenCL and CPU OCR executables behind a launcher |

To skip the test target:

```sh
cmake -S . -B build -DNUNN_BUILD_TESTS=OFF
```

To build just one executable after configuration:

```sh
cmake --build build --config Release --target xor_test
```

### Where the executables are

CMake output depends on the generator:

| Generator | Example path |
| --- | --- |
| Ninja, Unix Makefiles, other single-config generators | `build/examples/xor_test/xor_test` |
| Visual Studio Release | `build\examples\xor_test\Release\xor_test.exe` |
| Visual Studio Debug | `build\examples\xor_test\Debug\xor_test.exe` |

Run the smallest examples first:

```powershell
.\build\examples\and_test\Release\and_test.exe
.\build\examples\xor_test\Release\xor_test.exe
.\build\tests\Release\nunn_tests.exe
```

On a single-config build, use:

```sh
./build/examples/and_test/and_test
./build/examples/xor_test/xor_test
./build/tests/nunn_tests
```

## First source-backed program: learn AND

The perceptron needs a threshold for the final hard decision, two inputs, and a learning rate. The core of [`and_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc) is:

```cpp
#include "nu_perceptron.h"

nu::StepFunction step(0.5, 0.0, 1.0);
nu::Perceptron net(2, 0.2, step);

for (size_t epoch = 0; epoch < 2000; ++epoch) {
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            net.setInputVector({double(a), double(b)});
            net.backPropagate(double(a & b));
        }
    }
}

net.setInputVector({1.0, 1.0});
net.feedForward();
const double answer = net.getSharpOutput();
```

The public declaration is in [`nu_perceptron.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_perceptron.h); the weight update is in [`nu_perceptron.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_perceptron.cc); assertions for learning, loss, momentum, and persistence are in [`test_perceptron.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_perceptron.cc).

Two details are worth following in the source:

1. `feedForward()` produces the continuous sigmoid output; `getSharpOutput()` applies the configured `StepFunction`.
2. `backPropagate(target)` performs the training update. Calling only `feedForward()` never changes the parameters.

## From AND to XOR

A perceptron cannot separate XOR with one line. The next example introduces a hidden layer:

```cpp
#include "nu_mlpnn.h"

nu::MlpNN net(
    {2, 2, 1},  // input -> hidden -> output
    0.4,        // learning rate
    0.9         // momentum
);
```

Continue with the complete [`xor_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc), then read [Neural Networks](Neural-Networks) for the training loop, activations, mini-batches, and JSON persistence.

## Optional OpenCL backend

The default configuration attempts to find ArrayFire/OpenCL:

```sh
cmake -S . -B build -DNUNN_ENABLE_OPENCL=ON
```

`MlpMatrixNN::ComputeBackend::Auto` tries OpenCL when the build and runtime support it, then falls back to Eigen/CPU. `ComputeBackend::OpenCL` is strict and throws when GPU support cannot be initialized.

On Windows the helper wraps configuration and runtime deployment:

```powershell
.\build-opencl.ps1
.\build-opencl.ps1 -Target ocr_test
.\build-opencl.ps1 -ArrayFireRoot "C:\Program Files\ArrayFire\v3" -CleanCache
```

Backend selection changes where matrix operations run, not the network definition or model format.

## Install and package layout

`cmake --install` places:

- executables under `bin`;
- the static library under `lib`;
- public headers under `include/nunn`;
- README and NEWS under `share/nunn/doc`;
- bundled model files under `share/nunn/nets` and `share/nunn/models` where applicable.

Example:

```sh
cmake --install build --config Release --prefix ./stage
```

The project does not currently install a CMake package configuration for `find_package(nunn)`. Consumers should either build nuNN in their source tree or explicitly add the installed include and library paths.

## Common first-build failures

| Symptom | Check |
| --- | --- |
| Dependency fetch fails | Git/network access and proxy settings during the first configure |
| Compiler rejects the project | C++20 mode and compiler version |
| `ctest` finds no tests | configure without `-DNUNN_BUILD_TESTS=OFF` and build `nunn_tests` |
| `OpenCL` was requested but is unavailable | use `Auto` or configure with `NUNN_ENABLE_OPENCL=OFF` |
| Windows executable is not under `build/examples/...` | include the `Release` or `Debug` configuration directory |
| `nunn_topo` cannot produce an image | install Graphviz or request DOT output |

## Repository orientation

| Path | Contents |
| --- | --- |
| [`nunn/common`](https://github.com/eantcal/nunn/tree/main/nunn/common) | vectors, activations, costs, neurons, generic trainer |
| [`nunn/neural_networks`](https://github.com/eantcal/nunn/tree/main/nunn/neural_networks) | supervised, sequence, convolutional, classical, and generative models |
| [`nunn/reinforcement`](https://github.com/eantcal/nunn/tree/main/nunn/reinforcement) | Q-learning, SARSA, DQN, policies, replay buffer |
| [`examples`](https://github.com/eantcal/nunn/tree/main/examples) | complete runnable programs |
| [`tests`](https://github.com/eantcal/nunn/tree/main/tests) | focused behavior and persistence checks |
| [`mnist`](https://github.com/eantcal/nunn/tree/main/mnist) | IDX loading and digit conversion |
| [`nunn_topo`](https://github.com/eantcal/nunn/tree/main/nunn_topo) | model topology visualizer |

## Keep reading

Use [Examples Gallery](Examples-Gallery) to choose the next executable, or go directly to [Neural Networks](Neural-Networks) to trace a complete forward and backward pass.
