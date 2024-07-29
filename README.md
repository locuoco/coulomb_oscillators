# N-body Coulomb oscillators

This repository contains two folders:
- 'Simulation' contains a code that uses a Fast Multipole Method (FMM) in 2D and 3D cartesian coordinates and symplectic integrators for the simulation of a N-body system of charged particles with an external elastic potential (i.e. Coulomb oscillators). It is written in C++20, with multithreading and atomics, and CUDA 12 (any later version can also be used). The minimum compute capability required for the program is 5.0.
- 'Graphics' contains a code that graphically visualizes the evolution of the system. It takes the output of 'Simulation' as input. It is written in C++/OpenGL. It uses OpenGL Mathematics, FreeType, FreeImage, GLEW and GLFW.

The code is currently mainly developed by Alessandro Lo Cuoco. `parasort` is a parallel sorting algorithm by [Amir Baserinia](https://github.com/baserinia/parallel-sort) based on the C++ standard algorithm `std::sort` (modified by Alessandro Lo Cuoco and redistributed under the original GPL v3 license). This software contains source code provided by NVIDIA Corporation.

External download links:
- [GLFW](https://www.glfw.org/) (for Graphics)
- [GLEW](http://glew.sourceforge.net/) (for Graphics)
- [FreeType](https://www.freetype.org/) (for Graphics)
- [FreeImage](https://freeimage.sourceforge.io/) (for Graphics)
- [OpenGL Mathematics](https://github.com/g-truc/glm) for Graphics)

## Simulation

### Compilation and debugging
  `nvcc main3.cu -o nbco3 -O3 --expt-relaxed-constexpr -arch=sm_<xy> <std=c++20>`

`<xy>` is the compute capability of the GPU (usually given in the form `x.y`), for example `sm_50` corresponds to a compute capability of `5.0`.
`<std=c++20>` is the required standard, depending on the compiler used for host code. Use `--std c++20` for Windows (Visual Studio 2022) or `-std=c++20` for Linux (GCC 10-12).
	
The resulting program will be called `nbco3`.

For device-code debugging, first compile with the `-lineinfo` option:

  `nvcc -lineinfo main3.cu -o nbco3 --expt-relaxed-constexpr <std=c++20>`

and then, run the CUDA Compute Sanitizer tool:

  `compute-sanitizer nbco3 [arguments]`

  For host code debugging, follow the instructions for the host compiler (`cl` for Windows or `g++` for Linux, respectively).

### Usage
  `nbco3 [options] [input]`

  `[input]` is the path to a file that contains a state of the system. The format is that of a binary file that contains the positions of all particles and then their respective velocities in the same order. This state is used for initialising the system to be simulated. If not specified, the system will be initalised by sampling from a known distribution. Write `nbco3 -h` for more information on further options.
