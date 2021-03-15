#ifndef DIRECT_CUDA_H
#define DIRECT_CUDA_H

#include <cassert>
#include "kernel.cuh"

inline __host__ __device__ VEC_T(SCAL, 2) kernel(VEC_T(SCAL, 2) a, VEC_T(SCAL, 2) d, SCAL invDist2)
{
	return fma(invDist2, d, a);
}

inline __host__ __device__ VEC_T(SCAL, 3) kernel(VEC_T(SCAL, 3) a, VEC_T(SCAL, 3) d, SCAL invDist2)
{
	SCAL invDist = sqrt(invDist2);
	SCAL invDist3 = invDist2 * invDist;
	return fma(invDist3, d, a);
}

inline __host__ __device__ VEC_T(SCAL, 4) kernel(VEC_T(SCAL, 4) a, VEC_T(SCAL, 4) d, SCAL invDist2)
{
	SCAL invDist4 = invDist2 * invDist2;
	return fma(invDist4, d, a);
}

inline __host__ __device__ VEC_T(SCAL, 2) kernel(VEC_T(SCAL, 2) a, VEC_T(SCAL, 2) d, SCAL invDist2, SCAL c)
{
	return fma(invDist2*c, d, a);
}

inline __host__ __device__ VEC_T(SCAL, 3) kernel(VEC_T(SCAL, 3) a, VEC_T(SCAL, 3) d, SCAL invDist2, SCAL c)
{
	SCAL invDist = sqrt(invDist2);
	SCAL invDist3 = invDist2 * invDist;
	return fma(invDist3*c, d, a);
}

inline __host__ __device__ VEC_T(SCAL, 4) kernel(VEC_T(SCAL, 4) a, VEC_T(SCAL, 4) d, SCAL invDist2, SCAL c)
{
	SCAL invDist4 = invDist2 * invDist2;
	return fma(invDist4*c, d, a);
}

__global__ void direct_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n, const SCAL* param)
// direct force computation kernel
// does not work properly for some values of n (boh?)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	SCAL k = (SCAL)1;
	if (param != nullptr)
		k = param[0];
	while (i < n)
	{
		VEC atmp{};

		int Tiles = n / blockDim.x;

		for (int tile = 0; tile < Tiles; ++tile)
		{
			extern __shared__ VEC spos[]; // shared memory
			spos[threadIdx.x] = p[tile * blockDim.x + threadIdx.x]; // read from global and write to shared mem
			__syncthreads(); // wait that all threads in the current block are ready

#pragma unroll
			for (int j = 0; j < BLOCK_SIZE; ++j)
			{
				VEC d = p[i] - spos[j];
				SCAL dist2 = dot(d, d) + EPS;
				SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

				atmp = kernel(atmp, d, invDist2);
			}
			__syncthreads(); // wait that all threads in the current block have finished before writing
							 // in shared memory again
		}
		for (int j = Tiles * blockDim.x; j < n; ++j)
		{
			VEC d = p[i] - p[j];
			SCAL dist2 = dot(d, d) + EPS;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

			atmp = kernel(atmp, d, invDist2);
		}
		a[i] = k*atmp;
		i += gridDim.x * blockDim.x;
	}
}

void direct(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	direct_krnl <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

inline __host__ __device__ void direct2_core(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
											 const SCAL* param, int begi, int endi, int stride)
// direct force computation kernel without optimizations
// it works properly
{
	int i = begi;
	SCAL k = (SCAL)1;
	if (param != nullptr)
		k = param[0];
	while (i < endi)
	{
		VEC atmp{};

		for (int j = 0; j < n; ++j)
		{
			VEC d = p[i] - p[j];
			SCAL dist2 = dot(d, d) + EPS;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

			atmp = kernel(atmp, d, invDist2);
		}
		a[i] = k*atmp;
		i += stride;
	}
}

__global__ void direct2_krnl(const VEC *p, VEC *a, int n, const SCAL* param)
{
	direct2_core(p, a, n, param, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void direct2(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	direct2_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, param);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void direct2_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(direct2_core, p, a, n, param, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void direct3_core(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
											 const SCAL* param, int begi, int endi, int stride)
// direct force computation kernel without optimizations
// uses Kahan summation
// it works properly
{
	int i = begi;
	SCAL k = (SCAL)1;
	if (param != nullptr)
		k = param[0];
	while (i < endi)
	{
		VEC atmp{};
		VEC c{};

		for (int j = 0; j < n; ++j)
		{
			VEC d = p[i] - p[j];
			SCAL dist2 = dot(d, d) + EPS;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

			VEC y = d * invDist2 - c;
			VEC t = atmp + y;
			c = (t - atmp) - y;
			atmp = t;
		}
		a[i] = k*atmp;
		i += stride;
	}
}

__global__ void direct3_krnl(const VEC *p, VEC *a, int n, const SCAL* param)
{
	direct3_core(p, a, n, param, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void direct3(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	direct3_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, param);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void direct3_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(direct3_core, p, a, n, param, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

#endif // !DIRECT_CUDA_H