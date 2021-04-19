# N-body Coulomb oscillators

This repository contains two folders:
- 'Simulation' contains a code that uses a Fast Multipole Method (FMM) in 2D cartesian coordinates and symplectic integrators for the simulation of a N-body system of charged particles with an external elastic potential (i.e. Coulomb oscillators). It is written in C++11 with multithreading and CUDA 8.0 GA2 (any later version can also be used). It also uses the CUDA UnBound (CUB) library 1.6.4 (again, a later version can be used as long as it is supported by the CUDA compiler). More info on usage by specifying '-h' argument.
- 'Graphics' contains a code that graphically visualizes the evolution of the system. It takes the output of 'Simulation' as input. It is written in C++/OpenGL. It uses GL Mathematics, Freetype, FreeImage, GLEW and GLFW.

The code is currently mainly developed by Alessandro Lo Cuoco. `parasort` is a parallel sorting algorithm by Amir Baserinia based on the C++ standard algorithm `std::sort` (modified by Alessandro Lo Cuoco and redistributed under the original GPL v3 license). This software contains source code provided by NVIDIA Corporation.
