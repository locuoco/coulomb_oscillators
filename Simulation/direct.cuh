//  Direct force computation
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

template<int BlockSize>
__global__ void direct_krnl(const VEC *__restrict__ p, VEC *__restrict__ a, int n, const SCAL* param, SCAL d_EPS2)
// direct force computation kernel
// does not work properly for some values of n (boh?)
{
	extern __shared__ VEC spos[]; // shared memory
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int Tiles = n / blockDim.x;
	SCAL k = (SCAL)1;
	if (param != nullptr)
		k = param[0];
	while (i < n)
	{
		VEC atmp{};

		for (int tile = 0; tile < Tiles; ++tile)
		{
			spos[tid] = p[tile * blockDim.x + tid]; // read from global and write to shared mem
			__syncthreads(); // wait that all threads in the current block are ready

#pragma unroll
			for (int j = 0; j < BlockSize; ++j)
			{
				VEC d = p[i] - spos[j];
				SCAL dist2 = dot(d, d) + d_EPS2;
				SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

				atmp = kernel(atmp, d, invDist2);
			}
			__syncthreads(); // wait that all threads in the current block have finished before writing
			                 // in shared memory again
		}
		if (tid < n - Tiles * blockDim.x)
			spos[tid] = p[Tiles * blockDim.x + tid];
		__syncthreads();
		
		for (int j = 0; j < n - Tiles * blockDim.x; ++j)
		{
			VEC d = p[i] - spos[j];
			SCAL dist2 = dot(d, d) + d_EPS2;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

			atmp = kernel(atmp, d, invDist2);
		}
		__syncthreads();
		
		a[i] = k*atmp;
		i += gridDim.x * blockDim.x;
	}
}

void direct(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	switch (BLOCK_SIZE)
	{
		case 1:
			direct_krnl<1> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 32:
			direct_krnl<32> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 64:
			direct_krnl<64> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 128:
			direct_krnl<128> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 256:
			direct_krnl<256> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 512:
			direct_krnl<512> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		case 1024:
			direct_krnl<1024> <<< nBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(VEC) >>> (p, a, n, param, EPS2);
			break;
		default:
			gpuErrchk((cudaError_t)!cudaSuccess);
			break;
	}
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

inline __host__ __device__ void direct2_core(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
											 const SCAL* param, SCAL d_EPS2, int begi, int endi, int stride)
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
			SCAL dist2 = dot(d, d) + d_EPS2;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

			atmp = kernel(atmp, d, invDist2);
		}
		a[i] = k*atmp;
		i += stride;
	}
}

__global__ void direct2_krnl(const VEC *p, VEC *a, int n, const SCAL* param, SCAL d_EPS2)
{
	direct2_core(p, a, n, param, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void direct2(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	direct2_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, param, EPS2);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void direct2_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(direct2_core, p, a, n, param, EPS2, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void direct3_core(const VEC *__restrict__ p, VEC *__restrict__ a, int n,
											 const SCAL* param, SCAL d_EPS2, int begi, int endi, int stride)
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
			SCAL dist2 = dot(d, d) + d_EPS2;
			SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest
#if DIM == 2
			VEC y = d * invDist2 - c;
#elif DIM == 3
			VEC y = d * invDist2 * sqrt(invDist2) - c;
#else // DIM == 4
			VEC y = d * invDist2 * invDist2 - c;
#endif
			VEC t = atmp + y;
			c = (t - atmp) - y;
			atmp = t;
		}
		a[i] = k*atmp;
		i += stride;
	}
}

__global__ void direct3_krnl(const VEC *p, VEC *a, int n, const SCAL* param, SCAL d_EPS2)
{
	direct3_core(p, a, n, param, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void direct3(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	direct3_krnl <<< nBlocks, BLOCK_SIZE >>> (p, a, n, param, EPS2);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void direct3_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	assert(n > 0);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(direct3_core, p, a, n, param, EPS2, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

#endif // !DIRECT_CUDA_H