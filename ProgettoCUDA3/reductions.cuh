#ifndef REDUCTIONS_CUDA_H
#define REDUCTIONS_CUDA_H

#include "kernel.cuh"

#define ONES_VEC VEC{1,1}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

template <bool nIsPow2>
__global__ void minmaxReduce(VEC *minmax_, const VEC *x, unsigned int n)
{
	extern __shared__ VEC sminmax[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*BLOCK_SIZE*2 + threadIdx.x;
	unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;
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

	if (nIsPow2 || i + BLOCK_SIZE < n)
	{
		val_min = fmin(val_min, x[i+BLOCK_SIZE]);
		val_max = fmax(val_max, x[i+BLOCK_SIZE]);
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
		if (nIsPow2 || i + BLOCK_SIZE < n)
		{
			val_min = fmin(val_min, x[i+BLOCK_SIZE]);
			val_max = fmax(val_max, x[i+BLOCK_SIZE]);
		}

		i += gridSize;
    }

	// each thread puts its local reduction into shared memory
	sminmax[sid] = val_min;
	sminmax[sid+1] = val_max;
	__syncthreads();

	// do reduction in shared mem
	if ((BLOCK_SIZE >= 512) && (tid < 256))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 512]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 513]);
	}
	__syncthreads();

	if ((BLOCK_SIZE >= 256) && (tid < 128))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 256]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 257]);
	}
     __syncthreads();

    if ((BLOCK_SIZE >= 128) && (tid <  64))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 128]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 129]);
	}
    __syncthreads();

	if ((BLOCK_SIZE >= 64) && (tid <  32))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 64]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 65]);
	}
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((BLOCK_SIZE >= 32) && (tid <  16))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 32]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 33]);
	}
	__syncthreads();

	if ((BLOCK_SIZE >= 16) && (tid <   8))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 16]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 17]);
	}
	__syncthreads();

	if ((BLOCK_SIZE >=  8) && (tid <   4))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 8]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 9]);
	}
	__syncthreads();

	if ((BLOCK_SIZE >=  4) && (tid <   2))
	{
		sminmax[sid] = fmin(sminmax[sid], sminmax[sid + 4]);
		sminmax[sid+1] = fmax(sminmax[sid+1], sminmax[sid + 5]);
	}
	__syncthreads();

	if ((BLOCK_SIZE >=  2) && (tid == 0))
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

inline __device__ __host__ SCAL rel_diff1(VEC x, VEC ref)
{
	VEC d = x - ref;
	SCAL dist2 = dot(d,d), ref2 = dot(ref,ref);
	return sqrt(dist2/ref2);
}

inline __device__ __host__ SCAL rel_diff2(VEC x, VEC ref)
{
	VEC d = x - ref;
	VEC s = x + ref;
	SCAL dist2 = dot(d,d), div2 = dot(s,s);
	return 2*sqrt(dist2/div2);
}

template <bool nIsPow2>
__global__ void relerrReduce_krnl(SCAL *relerr, const VEC *x, const VEC *xref, unsigned int n)
{
	extern __shared__ SCAL srelerr[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*BLOCK_SIZE*2 + threadIdx.x;
	unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;

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

		if (nIsPow2 || i + BLOCK_SIZE < n)
			val += rel_diff1(x[i+BLOCK_SIZE], xref[i+BLOCK_SIZE]);

		i += gridSize;
    }

	srelerr[tid] = val;
	__syncthreads();

	if ((BLOCK_SIZE >= 512) && (tid < 256))
		srelerr[tid] += srelerr[tid + 256];
	__syncthreads();

	if ((BLOCK_SIZE >= 256) && (tid < 128))
		srelerr[tid] += srelerr[tid + 128];
     __syncthreads();

    if ((BLOCK_SIZE >= 128) && (tid <  64))
		srelerr[tid] += srelerr[tid +  64];
    __syncthreads();

	if ((BLOCK_SIZE >= 64) && (tid <  32))
		srelerr[tid] += srelerr[tid + 32];
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((BLOCK_SIZE >= 32) && (tid <  16))
		srelerr[tid] += srelerr[tid + 16];
	__syncthreads();

	if ((BLOCK_SIZE >= 16) && (tid <   8))
		srelerr[tid] += srelerr[tid +  8];
	__syncthreads();

	if ((BLOCK_SIZE >=  8) && (tid <   4))
		srelerr[tid] += srelerr[tid +  4];
	__syncthreads();

	if ((BLOCK_SIZE >=  4) && (tid <   2))
		srelerr[tid] += srelerr[tid +  2];
	__syncthreads();

	if ((BLOCK_SIZE >=  2) && (tid == 0))
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
		relerrReduce_krnl<true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
	else
		relerrReduce_krnl<false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (relerr, x, xref, n);
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

template <bool nIsPow2>
__global__ void powReduce_krnl(VEC *power, const VEC *x, int expo, unsigned int n)
// sum the powers of vectors x:
// power = sum_i x_i ^ expo
{
	extern __shared__ VEC spow[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*BLOCK_SIZE*2 + threadIdx.x;
	unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;

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

		if (nIsPow2 || i + BLOCK_SIZE < n)
			v += binarypow(x[i+BLOCK_SIZE], expo);

		i += gridSize;
    }

	spow[tid] = v;
	__syncthreads();

	if ((BLOCK_SIZE >= 512) && (tid < 256))
		spow[tid] += spow[tid + 256];
	__syncthreads();

	if ((BLOCK_SIZE >= 256) && (tid < 128))
		spow[tid] += spow[tid + 128];
     __syncthreads();

    if ((BLOCK_SIZE >= 128) && (tid <  64))
		spow[tid] += spow[tid +  64];
    __syncthreads();

	if ((BLOCK_SIZE >= 64) && (tid <  32))
		spow[tid] += spow[tid + 32];
    __syncthreads();

    // fully unroll reduction within a single warp
	if ((BLOCK_SIZE >= 32) && (tid <  16))
		spow[tid] += spow[tid + 16];
	__syncthreads();

	if ((BLOCK_SIZE >= 16) && (tid <   8))
		spow[tid] += spow[tid +  8];
	__syncthreads();

	if ((BLOCK_SIZE >=  8) && (tid <   4))
		spow[tid] += spow[tid +  4];
	__syncthreads();

	if ((BLOCK_SIZE >=  4) && (tid <   2))
		spow[tid] += spow[tid +  2];
	__syncthreads();

	if ((BLOCK_SIZE >=  2) && (tid == 0))
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
		powReduce_krnl<true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
	else
		powReduce_krnl<false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (power, x, expo, n);
}

#endif // !REDUCTIONS_CUDA_H