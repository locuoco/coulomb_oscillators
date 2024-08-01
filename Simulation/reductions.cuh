//  CUDA reductions, based on code by NVIDIA Corporation
//  Copyright (C) 2021-24 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

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

#ifndef REDUCTIONS_CUDA_H
#define REDUCTIONS_CUDA_H

#include "kernel.cuh"

#include <cub/cub.cuh>

#if DIM == 2
#define ONES_VEC VEC{1,1}
#elif DIM == 3
#define ONES_VEC VEC{1,1,1}
#elif DIM == 4
#define ONES_VEC VEC{1,1,1,1}
#endif

bool isPow2(unsigned int x)
{
    return (x&(x-1))==0;
}

inline __device__ __host__ SCAL rel_diff1(VEC x, VEC ref)
{
	VEC d = x - ref;
	SCAL dist2 = dot(d,d), ref2 = dot(ref,ref) + SCAL(1.e-18);
	return sqrt(max(dist2/ref2, SCAL(0)));
}

inline __device__ __host__ SCAL rel_diff2(VEC x, VEC ref)
{
	VEC d = x - ref;
	VEC s = x + ref;
	SCAL dist2 = dot(d,d), div2 = dot(s,s) + SCAL(1.e-18);
	return 2*sqrt(dist2/div2);
}

struct MinMaxVec
{
	SCAL xmin, ymin, zmin, xmax, ymax, zmax;
};

struct MinMax
{
	__host__ __device__ __forceinline__ MinMaxVec operator()(const MinMaxVec& a, const MinMaxVec& b) const
	{
		return MinMaxVec{
			min(a.xmin, b.xmin), min(a.ymin, b.ymin), min(a.zmin, b.zmin),
			max(a.xmax, b.xmax), max(a.ymax, b.ymax), max(a.zmax, b.zmax)
		};
	}
};

__global__ void minmaxReduce2Init_krnl(VEC *minmax)
{
	minmax[0] = {FLT_MAX, FLT_MAX, FLT_MAX};
	minmax[1] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
}

template <int blockSize>
__global__ void minmaxReduce2_krnl(VEC *__restrict__ minmax_, const VEC *__restrict__ x, int n)
// not working properly, this function needs a review
{
	using BlockReduceT = cub::BlockReduce<MinMaxVec, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	MinMaxVec result, data;
	if (gid < n)
		data = {x[gid].x, x[gid].y, x[gid].z, x[gid].x, x[gid].y, x[gid].z};
	else
		data = {FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

	result = BlockReduceT(temp_storage).Reduce(data, MinMax());

	if (tid == 0)
	{
		myAtomicMin(&minmax_[0].x, result.xmin);
		myAtomicMin(&minmax_[0].y, result.ymin);
		myAtomicMin(&minmax_[0].z, result.zmin);

		myAtomicMax(&minmax_[1].x, result.xmax);
		myAtomicMax(&minmax_[1].y, result.ymax);
		myAtomicMax(&minmax_[1].z, result.zmax);
	}
}

void minmaxReduce2(VEC *minmax, const VEC *src, unsigned int n)
{
	int nBlocks = (n-1)/1024 + 1;
	minmaxReduce2Init_krnl <<< 1, 1 >>> (minmax);
	minmaxReduce2_krnl<1024> <<< nBlocks, 1024 >>> (minmax, src, n);
}

template <int blockSize>
__global__ void relerrReduce2_krnl(SCAL *relerr, const VEC *__restrict__ x, const VEC *__restrict__ xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
		result = BlockReduceT(temp_storage).Sum(rel_diff1(x[gid], xref[gid])/n);

	if (tid == 0)
		myAtomicAdd(relerr, result);
}

void relerrReduce2(SCAL *relerr, const VEC *x, const VEC *xref, unsigned int n)
{
	int nBlocks = (n-1)/1024 + 1;
	cudaMemset(relerr, 0, sizeof(SCAL));
	relerrReduce2_krnl<1024> <<< nBlocks, 1024 >>> (relerr, x, xref, n);
}

template <int blockSize>
__global__ void relerrReduce3Num_krnl(SCAL *relerr, const VEC *__restrict__ x, const VEC *__restrict__ xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
	{
		VEC d = x[gid] - xref[gid];
		result = BlockReduceT(temp_storage).Sum(dot(d, d));
	}

	if (tid == 0)
		myAtomicAdd(relerr, result);
}
template <int blockSize>
__global__ void relerrReduce3Den_krnl(SCAL *relerr, const VEC *xref, int n)
{
	using BlockReduceT = cub::BlockReduce<SCAL, blockSize>;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	int tid = threadIdx.x;
	int gid = gridDim.x * blockIdx.x + tid;

	SCAL result;
	if (gid < n)
		result = BlockReduceT(temp_storage).Sum(dot(xref[gid], xref[gid]));

	if (tid == 0)
		myAtomicAdd(relerr+1, result);
}
__global__ void relerrReduce3Res_krnl(SCAL *relerr)
{
	relerr[0] = sqrt(relerr[0] / relerr[1]);
}

void relerrReduce3(SCAL *relerr, const VEC *x, const VEC *xref, unsigned int n)
{
	int nBlocks = (n-1)/1024 + 1;
	cudaMemset(relerr, 0, 2*sizeof(SCAL));
	relerrReduce3Num_krnl<1024> <<< nBlocks, 1024 >>> (relerr, x, xref, n);
	relerrReduce3Den_krnl<1024> <<< nBlocks, 1024 >>> (relerr, xref, n);
	relerrReduce3Res_krnl <<< 1, 1 >>> (relerr);
}

template <int blockSize, bool nIsPow2>
__global__ void minmaxReduce_krnl(VEC *minmax_, const VEC *x, unsigned int n)
{
	extern __shared__ VEC sminmax[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	unsigned int sid = 2*threadIdx.x;

	if (i >= n)
	{
		if (tid == 0)
		{
			minmax_[blockIdx.x*2] = 99999999999 * ONES_VEC;
			minmax_[blockIdx.x*2+1] = -99999999999 * ONES_VEC;
		}
		return;
	}

	VEC val_min(x[i]), val_max(x[i]);

	if (nIsPow2 || i + blockSize < n)
	{
		val_min = fmin(val_min, x[i+blockSize]);
		val_max = fmax(val_max, x[i+blockSize]);
	}

	i += gridSize;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		val_min = fmin(val_min, x[i]);
		val_max = fmax(val_max, x[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
		{
			val_min = fmin(val_min, x[i+blockSize]);
			val_max = fmax(val_max, x[i+blockSize]);
		}

		i += gridSize;
    }

	// each thread puts its local reduction into shared memory
	sminmax[sid] = val_min;
	sminmax[sid+1] = val_max;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 1024) && (tid < 512))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 1024]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 1025]);
	}
	__syncthreads();
	
	if ((blockSize >= 512) && (tid < 256))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 512]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 513]);
	}
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 256]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 257]);
	}
    __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 128]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 129]);
	}
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 64]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 65]);
	}
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 32]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 33]);
	}
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 16]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 17]);
	}
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 8]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 9]);
	}
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 4]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 5]);
	}
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
	{
		sminmax[0] = fmin(sminmax[0], sminmax[2]);
		sminmax[1] = fmax(sminmax[1], sminmax[3]);
	}
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
	{
		minmax_[blockIdx.x*2] = sminmax[0];
		minmax_[blockIdx.x*2+1] = sminmax[1];
	}
}

void minmaxReduce(VEC *minmax, const VEC *src, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(VEC)*2 : BLOCK_SIZE * sizeof(VEC)*2;
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				minmaxReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 32:
				minmaxReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 64:
				minmaxReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 128:
				minmaxReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 256:
				minmaxReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 512:
				minmaxReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 1024:
				minmaxReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				minmaxReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 32:
				minmaxReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 64:
				minmaxReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 128:
				minmaxReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 256:
				minmaxReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 512:
				minmaxReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			case 1024:
				minmaxReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (minmax, src, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

template <int blockSize, bool nIsPow2>
__global__ void relerrReduce_krnl(SCAL *relerr, const VEC *x, const VEC *xref, unsigned int n)
{
	extern __shared__ SCAL srelerr[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	if (i >= n)
	{
		if (tid == 0)
			relerr[blockIdx.x] = (SCAL)0;
		return;
	}

	SCAL val{};

	while (i < n)
	{
		val += rel_diff1(x[i], xref[i]);

		if (nIsPow2 || i + blockSize < n)
			val += rel_diff1(x[i+blockSize], xref[i+blockSize]);

		i += gridSize;
    }

	srelerr[tid] = val;
	__syncthreads();
	
	if ((blockSize >= 1024) && (tid < 512))
		srelerr[tid] += srelerr[tid + 512];
	__syncthreads();

	if ((blockSize >= 512) && (tid < 256))
		srelerr[tid] += srelerr[tid + 256];
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
		srelerr[tid] += srelerr[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
		srelerr[tid] += srelerr[tid +  64];
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
		srelerr[tid] += srelerr[tid + 32];
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
		srelerr[tid] += srelerr[tid + 16];
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
		srelerr[tid] += srelerr[tid +  8];
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
		srelerr[tid] += srelerr[tid +  4];
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
		srelerr[tid] += srelerr[tid +  2];
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
		srelerr[tid] += srelerr[tid +  1];
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
		relerr[blockIdx.x] = srelerr[0];
}

void relerrReduce(SCAL *relerr, const VEC *x, const VEC *xref, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(SCAL) : BLOCK_SIZE * sizeof(SCAL);
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				relerrReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 32:
				relerrReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 64:
				relerrReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 128:
				relerrReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 256:
				relerrReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 512:
				relerrReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 1024:
				relerrReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				relerrReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 32:
				relerrReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 64:
				relerrReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 128:
				relerrReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 256:
				relerrReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 512:
				relerrReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			case 1024:
				relerrReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

inline __host__ __device__ VEC binarypow(VEC x, int n)
// calculates x^n with O(log(n)) multiplications
// assumes n >= 1
{
	VEC y = ONES_VEC;
	while (n > 1)
	{
		y *= ((n & 1) ? x : ONES_VEC);
		x *= x;
		n /= 2;
	}
	return x * y;
}

template <int blockSize, bool nIsPow2>
__global__ void powReduce_krnl(VEC *power, const VEC *x, int expo, unsigned int n)
// sum the powers of vectors x:
// power = sum_i x_i ^ expo
{
	extern __shared__ VEC spow[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	if (i >= n)
	{
		if (tid == 0)
			power[blockIdx.x] = VEC{};
		return;
	}

	VEC v{};

	while (i < n)
	{
		v += binarypow(x[i], expo);

		if (nIsPow2 || i + blockSize < n)
			v += binarypow(x[i+blockSize], expo);

		i += gridSize;
    }

	spow[tid] = v;
	__syncthreads();
	
	if ((blockSize >= 1024) && (tid < 512))
		spow[tid] += spow[tid + 512];
	__syncthreads();

	if ((blockSize >= 512) && (tid < 256))
		spow[tid] += spow[tid + 256];
	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
		spow[tid] += spow[tid + 128];
     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
		spow[tid] += spow[tid +  64];
    __syncthreads();

	if ((blockSize >= 64) && (tid <  32))
		spow[tid] += spow[tid + 32];
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((blockSize >= 32) && (tid <  16))
		spow[tid] += spow[tid + 16];
	__syncthreads();

	if ((blockSize >= 16) && (tid <   8))
		spow[tid] += spow[tid +  8];
	__syncthreads();

	if ((blockSize >=  8) && (tid <   4))
		spow[tid] += spow[tid +  4];
	__syncthreads();

	if ((blockSize >=  4) && (tid <   2))
		spow[tid] += spow[tid +  2];
	__syncthreads();

	if ((blockSize >=  2) && (tid == 0))
		spow[tid] += spow[tid +  1];
	__syncthreads();

    // write result for this block to global mem
	if (tid == 0)
		power[blockIdx.x] = spow[0];
}

void powReduce(VEC *power, const VEC *x, int expo, unsigned int n, int nBlocksRed = 1)
{
	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(VEC) : BLOCK_SIZE * sizeof(VEC);
	if (isPow2(n))
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				powReduce_krnl<1, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 32:
				powReduce_krnl<32, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 64:
				powReduce_krnl<64, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 128:
				powReduce_krnl<128, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 256:
				powReduce_krnl<256, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 512:
				powReduce_krnl<512, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 1024:
				powReduce_krnl<1024, true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
	else
	{
		switch (BLOCK_SIZE)
		{
			case 1:
				powReduce_krnl<1, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 32:
				powReduce_krnl<32, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 64:
				powReduce_krnl<64, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 128:
				powReduce_krnl<128, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 256:
				powReduce_krnl<256, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 512:
				powReduce_krnl<512, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			case 1024:
				powReduce_krnl<1024, false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
				break;
			default:
				gpuErrchk((cudaError_t)!cudaSuccess);
				break;
		}
	}
}

#endif // !REDUCTIONS_CUDA_H