# N-body Coulomb oscillators

This repository contains two folders:
- 'Simulation' contains a code that uses a Fast Multipole Method (FMM) in 2D cartesian coordinates and symplectic integrators for the simulation of a N-body system of charged particles with an external elastic potential (i.e. Coulomb oscillators). It is written in C++11 with multithreading and CUDA 8.0 GA2 (any later version can also be used). It also uses the CUDA UnBound (CUB) library 1.6.4 (again, a later version can be used as long as it is supported by the CUDA compiler). More info on usage later.
- 'Graphics' contains a code that graphically visualizes the evolution of the system. It takes the output of 'Simulation' as input. It is written in C++/OpenGL. It uses GL Mathematics, Freetype, FreeImage, GLEW and GLFW.

The code is currently mainly developed by Alessandro Lo Cuoco. `parasort` is a parallel sorting algorithm by Amir Baserinia based on the C++ standard algorithm `std::sort` (modified by Alessandro Lo Cuoco and redistributed under the original GPL v3 license). This software contains source code provided by NVIDIA Corporation.+

## Simulation

### Compilation
  `nvcc main.cu -o nbco -O3 -std=c++11 -arch=sm_<xy> -I <includes>`

`<xy>` is the compute capability of the GPU (usually given in the form `x.y`),
for example `sm_21` corresponds to a compute capability of `2.1`.
`<includes>` is the folder which contains the CUB library.

Note: some CUDA versions may require the C++14 standard or later.
	
The resulting program will be called `nbco`. A compatible C++ compilator must be
available.

### Usage
  `nbco [options] [input]`

  `[input]` is the path to a file that contains a state of the system. The format
  is that of a binary file that contains the positions of all particles and
  then their respective velocities in the same order. This state is used for
  initialising the system to be simulated. If not specified, the system will be
  initalised by sampling from a known distribution (KV or gaussian).

Other options:
- `-h` or `-help`       Display this information.
- `-o <output>`       Specify the folder where the output will be written.
                    Default is `'./out'`. This folder must already exist. The
                    format of output files is the same as the input file format
                    described before.
- `-n <npart>`        Number of particles to simulate. Default is `30001`. Will be
                    ignored if `[input]` is specified.
- `-ds <v>`           Time step. Default is `5e-4`.
- `-iters <n>`        Number of total simulation iterations. Default is `30000`.
- `-steps <n>`        Number of steps to simulate before saving to file. Default
                    is `200`.
- `-integ <name>`     Set the symplectic integrator to be used instead of the
                    default one (2nd order). `<name>` must be chosen from the
                    list {`eu`, `fr`, `pefrl`}, respectively the semi-implicit Euler
                    (1st order), Forest-Ruth (4th order) and PEFRL (4th order).
- `-p <order>`        FMM expansion order. Default is `5`.
- `-r <radius>`       Interaction radius. Must be 1 or greater. Default is `1`.
- `-eps <v>`          Smoothing factor. Must be greater than `0`. Default is `1e-9`.
- `-i <v>`            A factor so that max FMM level is `round(log(n*i/p^(3/2)))`.
                    Default is `1`.
- `-ncoll`            P2P pass will not be calculated, effectively ignoring
                    collisional effects. Note however that the result will
                    depend on the max FMM level, and the simulation may be
                    highly unreliable at certain conditions.
- `-cpu`              Use CPU with multithreading (default is GPU).
- `-cpu-threads <n>`  Number of CPU threads. Must be `1` or greater (default is `8`).
                    Implies `-cpu`.
- `-cacheline <n>`    CPU cache line size in bytes. Defaut is `64`. Will be ignored
                    if `-cpu` is NOT specified.
- `-gpu <blocksize>`  Specify the number of threads in a GPU block. It must be
                    chosen from the list {`1`, `32`, `64`, `128`, `256`, `512`, `1024`} (`1024`
                    threads is available for compute capability 2.0 and above).
                    Default is `128`. Will be ignored if `-cpu` is specified.
- `-gridsize <n>`     Specify the maximum number of blocks in a GPU grid. Must be
                    `1` or greater. Default is `128`. Will be ignored if `-cpu` is
                    specified.
- `-test`             Show relative errors and execution times of a single
                    iteration instead of doing the simulation.
- `-ga`               Initialise the system with a gaussian distribution instead
                    of a KV distribution (which is the default).
- `-xi <v>`           Set the perveance.
- `-omega0 <vx> <vy>` Set the phase advances.
- `-x <vx> <vy>`      Set the std.dev. of positions. Will be ignored if `[input]`
                    is specified.
- `-u <vx> <vy>`      Set the std.dev. of velocities. Will be ignored if `[input]`
                    is specified.
- `-A <vx> <vy>`      Set the system semi-axes. Will be ignored if `[input]` is
                    specified.
- `-omega <vx> <vy>`  Set the depressed phase advances. Will be ignored if
                    `[input]` is specified.

Note: `x = A / 2` and `u = omega * A / 2`.
