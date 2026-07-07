# Getting Started

## Requirements

nuNN requires:

- CMake 3.14 or newer
- a C++20 compiler
- Git access during first configure, because Eigen, nlohmann/json, and GoogleTest are fetched with CMake `FetchContent`

On Windows, Visual Studio 2022 or newer is recommended.

## Basic Build

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

To skip tests:

```sh
cmake -S . -B build -DNUNN_BUILD_TESTS=OFF
```

## OpenCL / ArrayFire

OpenCL support is enabled by default at configure time:

```sh
cmake -S . -B build -DNUNN_ENABLE_OPENCL=ON
```

If ArrayFire/OpenCL is found, `MlpMatrixNN` can use the OpenCL backend. If it is not found, `ComputeBackend::Auto` falls back to Eigen/CPU.

On Windows, the helper script assumes the standard ArrayFire install location:

```powershell
.\build-opencl.ps1
.\build-opencl.ps1 -Target ocr_test
.\build-opencl.ps1 -ArrayFireRoot "C:\Program Files\ArrayFire\v3"
```

## Installed Packages

Windows packages install command-line tools and demos under:

```text
<install-root>\bin
```

The installer also provides Start Menu shortcuts for:

- Nunn OCR Test
- Nunn Tic Tac Toe
- Nunn Developer Command Prompt
- Run Nunn Unit Tests
- README and NEWS

On Linux/macOS packages install:

- `nunn-dev-shell.sh`
- `nunn-run-tests.sh`

These scripts place the package `bin` directory on `PATH`, making examples such as `mnist_test`, `xor_test`, `nunn_tests`, and `net2json` easy to launch.

