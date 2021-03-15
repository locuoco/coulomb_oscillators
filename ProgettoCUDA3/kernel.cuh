#ifndef KERNEL_CUDA_H
#define KERNEL_CUDA_H

#ifndef SCAL
#pragma message ("warning: scalar type SCAL macro not defined. Defaulting to float.")
#define SCAL float
#endif

#ifndef DIM
#pragma message ("warning: number of dimensions DIM macro not defined. Defaulting to 2.")
#define DIM 2
#endif

#if (DIM < 2) || (DIM > 4)
#error "DIM cannot be greater than 4 or smaller than 2"
#endif

#ifndef BLOCK_SIZE
#pragma message ("warning: BLOCK_SIZE macro not defined. Defaulting to 256.")
#define BLOCK_SIZE 256
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifndef MAX_GRID_SIZE
// so that shared memory (in a grid) is not greater than 48 KB
#pragma message ("warning: MAX_GRID_SIZE macro not defined. Defaulting to 256.")
#define MAX_GRID_SIZE 256
#endif

#ifndef EPS
#pragma message ("warning: softening factor EPS macro not defined. Defaulting to 1.e-15.")
#define EPS (SCAL)1.e-15
#endif

#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

// IMPORTANT: set WDDM TDR Delay to 60 sec (default is 2) from NVIDIA Nsight Monitor Options

#include <device_functions.h>
//#include <cuda_fp16.h> // half precision floating point

#include <cmath>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#define EXPAND(x) x
#define VEC_T(U,n) EXPAND(U)##EXPAND(n) // vector type
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
				VEC k{param[0], param[1]};
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] -= p[j] * k;
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
				VEC k{-param[0], -param[1]};
				for (int j = niter*i; j < std::min(niter*(i+1), n); ++j)
					a[j] = p[j] * k;
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


__global__ void gather_krnl(VEC *__restrict__ dst, const VEC *__restrict__ src, const int *map, int n)
// dst array is built from src array through the map pointer
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[map[i]];
	}
}

template<typename T>
void gather_cpu(T *__restrict__ dst, const T *__restrict__ src, const int *map, int n)
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

__global__ void copy_krnl(VEC *__restrict__ dst, const VEC *__restrict__ src, int n)
// copy content from src to dst
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
        dst[i] = src[i];
	}
}

void copy_gpu(VEC *dst, const VEC *src, int n)
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