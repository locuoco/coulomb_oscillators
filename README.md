# N-body coulomb oscillators

This repository contains two folders:
- ProgettoCUDA3 contains a code that uses a Fast Multipole Method (FMM) in 2D cartesian coordinates and symplectic integrators for the simulation of a N-body system of charged particles with an external elastic potential (i.e. Coulomb oscillators). It is written in C++11 with multithreading and CUDA 8.0 GA2. It also uses the CUDA UnBound (CUB) library.
- ProgettoGrafica contains a code that graphically visualizes the evolution of the system. It takes the output of ProgettoCUDA3 as input. It is written in C++/OpenGL. It uses GL Mathematics, Freetype, GLEW and GLFW.
