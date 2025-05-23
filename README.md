# Neural Network

A simple neural network implementation in `C++` and `CUDA`. With both supervised
and unsupervised training.<br>*No external libraries required*.

# Requirements

 - **CMake 3.13.0+**
 - **Visual Studio 2022** *or* **Ninja**, **Mingw** and **NASM**
 - **git**
 - **Nvidia CUDA Toolkit** *(optional)*

# How To Build

The CMakeLists.txt script automatically **downloads** and builds the [dependencies](#Dependencies)
needed by the project.
This means that the **first** configuration will be **slower** than the other ones.

To **configure and build** the project run in a terminal:

```shell
cmake --preset msvc-release
cmake --build --preset msvc-release-build
```

When configuring with `cmake --preset`, the **configure** preset should be
used, e.g.: **msvc-release**, when building with `cmake --build --preset`, the
corresponding **build** preset should be used, e.g.: **msvc-release-build**.

> A debug (`-debug`) version of the preset can also be used.

# Dependencies

 - [googletest](https://github.com/google/googletest) *optional*
 - [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h) *optional*
