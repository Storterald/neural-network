# Neural Network

A simple neural network implementation in `C++` and `CUDA`. With both supervised
and unsupervised training.<br>*No external libraries required*.

# Requirements

 - CMake `>= 3.13.0`
 - Visual Studio `2022`
 - Nvidia CUDA Toolkit
 - git

# How To Build

> CMake **3.13.0** or later is required to build the project, you can download it
> [here](https://cmake.org/download/).

The CMakeLists.txt script automatically **downloads** and builds the [dependencies](#Dependencies)
needed by the project, these **do not** include the [requirements](#Requirements).
This means that the **first** configuration will be **slower** then the other ones.

To **configure and build** the project run in a terminal:

```shell
cmake . --preset msvc-release
cmake --build . --preset msvc-release-build
```

The configure command will **fail** if the build is done **without** a selected
**build** preset, this is to ensure no unknown errors occur.

When configuring with `cmake . --preset`, the **configure** preset should be
used, eg: **msvc-release**, when building with `cmake . --build --preset`, the
corresponding **build** preset should be used, eg: **msvc-release-build**.

> A debug (`-debug`) version of the preset can also be used.

# Dependencies

 - [googletest](https://github.com/google/googletest)
