//  Constants
//  Copyright (C) 2024 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef CONSTANTS_CUDA_H

// Important defines
#ifndef SCAL
#define SCAL double // scalar 
#endif

#ifndef DIM
#define DIM 3 // dimensions
#endif
// note: most functions do not depend on the number of dimensions, but others
// will need to be rewritten for DIM != 2

#if (DIM < 2) || (DIM > 4)
#error "DIM cannot be greater than 4 or smaller than 2"
#endif

int BLOCK_SIZE = 128; // number of threads in a GPU block
int MAX_GRID_SIZE = 10; // number of blocks in a GPU grid
int CACHE_LINE_SIZE = 64; // CPU cache line size (in bytes)
SCAL EPS2 = (SCAL)1.e-18; // softening parameter squared

int CPU_THREADS = 8; // number of concurrent threads in CPU
int fmm_order = 2; // fast multipole method order
int tree_radius = 1;

bool coll = true;

SCAL dens_inhom = .1;

template <typename T>
inline __device__ T myAtomicAdd(T* address, T val)
{
	return atomicAdd(address, val);
}

#if __CUDA_ARCH__ < 600

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif // __CUDA_ARCH__ < 600

#endif // !CONSTANTS_CUDA_H














