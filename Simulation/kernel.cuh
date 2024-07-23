//  Some basic code for CPU and GPU
//  Copyright (C) 2021 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

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

#ifndef KERNEL_CUDA_H
#define KERNEL_CUDA_H

#include "constants.cuh"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//#ifndef __CUDACC_RTC__
//#define __CUDACC_RTC__
//#endif

#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "helper_math.h"

// IMPORTANT: set WDDM TDR Delay to 60 sec (default is 2) from NVIDIA Nsight Monitor Options
// otherwise, GPU functions (kernels) that take more than 2 seconds will fail

//#include <device_functions.h>
//#include <cuda_fp16.h> // half precision floating point

#include <cmath>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#define VEC_PASTE(U, n) U##n
#define VEC_T(U, n) VEC_PASTE(U, n) /* vector type */
#define VEC VEC_T(SCAL, DIM)
#define IVEC VEC_T(int, DIM)

#include "mymath.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUassert: " << cudaGetErrorString(code) << ' ' << file << ' ' << line << std::endl;
		if (abort)
		{
			cudaDeviceReset();
			exit(code);
		}
	}
}

struct ParticleSystem { VEC *__restrict__ pos, *__restrict__ vel, *__restrict__ acc; };

std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 2) a)
{
	os << a.x << ", " << a.y;
	return os;
}
std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 3) a)
{
	os << a.x << ", " << a.y << ", " << a.z;
	return os;
}
std::ostream& operator<<(std::ostream& os, VEC_T(SCAL, 4) a)
{
	os << a.x << ", " << a.y << ", " << a.z << ", " << a.w;
	return os;
}

__global__ void step_krnl(VEC *__restrict__ b, const VEC *__restrict__ a, SCAL ds, int n)
// multiply-addition kernel
// b += a * ds
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < n)
	{
		VEC t = a[i];
		VEC s = b[i];

		b[i] = fma(ds, t, s);
		i += blockDim.x * gridDim.x;
	}
}

void step(VEC *b, const VEC *a, SCAL ds, int n)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	step_krnl <<< nBlocks, BLOCK_SIZE >>> (b, a, ds, n);
}

void step_cpu(VEC *__restrict__ b, const VEC *__restrict__ a, SCAL ds, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				b[j] += a[j] * ds;
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void add_elastic_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
                                 const VEC *__restrict__ param)
// elastic force computation kernel with elastic costants defined in "param" pointer
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		VEC t = p[i];
		VEC s = a[i];

		a[i] = fma(-param[0], t, s);
	}
}

__global__ void add_elastic_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n)
// elastic force computation kernel
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] -= p[i];
	}
}

void add_elastic(VEC *p, VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (param != nullptr)
		add_elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, (const VEC*)param);
	else
		add_elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n);
}

void add_elastic_cpu(VEC *__restrict__ p, VEC *__restrict__ a, int n, const SCAL* param)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	if (param != nullptr)
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				const VEC *k = (const VEC*)param;
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] -= p[j] * k[0];
			});
	else
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] -= p[j];
			});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void elastic_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
                             const VEC *__restrict__ param)
// elastic force computation kernel with elastic costants defined in "param" pointer
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] = -param[0]*p[i];
	}
}

__global__ void elastic_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n)
// elastic force computation kernel
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		a[i] = -p[i];
	}
}

void elastic(VEC *p, VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	if (param != nullptr)
		elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, (const VEC*)param);
	else
		elastic_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n);
}

void elastic_cpu(VEC *__restrict__ p, VEC *__restrict__ a, int n, const SCAL* param)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	if (param != nullptr)
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				const VEC *k = (const VEC*)param;
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] = p[j] * -k[0];
			});
	else
		for (int i = 0; i < CPU_THREADS; ++i)
			threads[i] = std::thread([=]{
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] = -p[j];
			});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void gather_krnl(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
// dst array is built from src array through a permutation map pointer
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[map[i]];
	}
}

template<typename T>
void gather_cpu(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[j] = src[map[j]];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void gather_inverse_krnl(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
// dst array is built from src array through the inverse permutation of map pointer
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[map[i]] = src[i];
	}
}

template<typename T>
void gather_inverse_cpu(T *__restrict__ dst, const T *__restrict__ src, const int *__restrict__ map, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[map[j]] = src[j];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

template<typename T>
__global__ void copy_krnl(T *__restrict__ dst, const T *__restrict__ src, int n)
// copy content from src to dst
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[i];
	}
}

template<typename T>
void copy_gpu(T *dst, const T *src, int n)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (dst, src, n);
}

template<typename T>
void copy_cpu(T *__restrict__ dst, const T *__restrict__ src, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=]{
			for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
				dst[j] = src[j];
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

#endif // !KERNEL_CUDA_H