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
cmake --preset msvc-release-x86_64
cmake --build --preset msvc-release-build-x86_64
```

When configuring with `cmake --preset`, the **configure** preset should be
used, e.g.: **msvc-release-x86_64**, when building with `cmake --build --preset`, the
corresponding **build** preset should be used, e.g.: **msvc-release-build-x86_64**.

> Debug (`-debug`), CUDA (`-cuda`) and arm64 presets (`-arm64`) are available. The
> presets name follow the general rule: `<name>`-`<cuda?>`-`<type>`-`<arch>`.

# Dependencies

 - [googletest](https://github.com/google/googletest) *optional*
 - [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h) *optional*
