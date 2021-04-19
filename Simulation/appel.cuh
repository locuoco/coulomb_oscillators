//  Appel's method
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

#ifndef APPEL_CUDA_H
#define APPEL_CUDA_H

#include "direct.cuh"
#include "reductions.cuh"
#include <cub/cub.cuh>

struct Tree
{
	VEC *__restrict__ center, *__restrict__ field;
	SCAL *mpole;
	int *__restrict__ mult, *__restrict__ index;
};

// "index" will be the index of the first element that belongs to the node
// Sorting the particle array will be needed in order to avoid slow random access.

__global__ void printp(const VEC *p, int n)
// print positions of all particles, for debug purposes
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n;
		 i += gridDim.x * blockDim.x)
	{
		printf("p%d: (%e, %e)\n", i, p[i].x, p[i].y);
	}
}

inline __host__ __device__ void evalKeys_krnl(int *keys, const VEC *p, const VEC *min, SCAL rdelta, int side,
											  int begi, int endi, int stride)
// put particle i in box keys[i]
{
	for (int i = begi; i < endi; i += stride)
	{
		IVEC iD = to_ivec((p[i] - min[0]) * rdelta);
		iD = clip(iD, 0, side-1); // mymath.cuh
		
		keys[i] = flatten(iD, side); // mymath.cuh
	}
}

__global__ void evalKeys(int *keys, const VEC *p, const VEC *min, int n, SCAL rdelta, int side)
{
	evalKeys_krnl(keys, p, min, rdelta, side, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void evalKeys_cpu(int *keys, const VEC *p, const VEC *min, int n, SCAL rdelta, int side)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalKeys_krnl, keys, p, min, rdelta, side, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void evalIndices_krnl(int *ind, int begi, int endi, int stride)
// will be used to know the index after the particles are sorted,
// so that also p, v, a can be sorted accordingly
{
	for (int i = begi; i < endi; i += stride)
		ind[i] = i;
}

__global__ void evalIndices(int *ind, int n)
{
	evalIndices_krnl(ind, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void evalIndices_cpu(int *ind, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalIndices_krnl, ind, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void check_krnl(const int *keys, int n)
// check that particles are sorted correctly, for debug purposes
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < n-1;
		 i += gridDim.x * blockDim.x)
	{
        if (keys[i] > keys[i+1])
			printf("!");
		else
			printf("_");
	}
}

inline __device__ __host__ int tree_beg(int l)
// return the position of the first element of the l-th level
{
	return ((1 << (l*DIM)) - 1) / ((1 << DIM) - 1);
}
inline __device__ __host__ int tree_end(int l)
// return the position after the last element of the l-th level
{
	return ((1 << ((l+1)*DIM)) - 1) / ((1 << DIM) - 1);
}
inline __device__ __host__ int tree_n(int l)
// return the number of elements in the l-th level
{
	return 1 << (l*DIM);
}
inline __device__ __host__ int tree_side(int l)
// return the side size of the l-th level (square root of n when DIM=2, cube root of n when DIM=3)
{
	return 1 << l;
}

__global__ void initLeaves(VEC *center, int m)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < m;
		 i += gridDim.x * blockDim.x)
	{
		center[i] = VEC{};
	}
}

inline __host__ __device__ void indexLeaves_krnl(int *__restrict__ index, const int *__restrict__ keys,
												 int m, int n, int begi, int endi, int stride)
// the index of first the particle contained in a non-empty cell will be associated to each cell
// if the cell is empty, the index of first particle contained in a successive cell will be associated to it
// if there are no non-empty successive cells, "n" will be associated to it
// m: number of cells (size of index)
// n: number of particles (size of keys)
{
	int i = begi;
	if (i == 0)
	{
		int k = keys[0];
		for (int j = 0; j <= k; ++j)
			index[j] = 0;
		i += stride;
	}
	while (i < endi)
	{
		int k = keys[i];
		for (int j = keys[i-1]+1; j <= k; ++j)
			index[j] = i;
		if (i == n-1)
			for (int j = k+1; j < m; ++j)
				index[j] = n;
		i += stride;
	}
}

__global__ void indexLeaves(int *index, const int *keys, int m, int n)
{
	indexLeaves_krnl(index, keys, m, n, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void indexLeaves_cpu(int *index, const int *keys, int m, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(indexLeaves_krnl, index, keys, m, n, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void multLeaves_krnl(int *__restrict__ mult, const int *__restrict__ index, int m, int n,
												int begi, int endi, int stride)
// calculate the number of particles (multiplicity) of each cell
// it is trivial to calculate it after the index array is computed
{
	int i = begi;
	while (i < min(endi, m-1))
	{
		mult[i] = (index[i+1] - index[i]);
		i += stride;
	}
	if (i == m-1)
		mult[i] = (n - index[i]);
}

__global__ void multLeaves(int *mult, const int *index, int m, int n)
{
	multLeaves_krnl(mult, index, m, n, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void multLeaves_cpu(int *mult, const int *index, int m, int n)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(multLeaves_krnl, mult, index, m, n, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void monopoleLeaves(SCAL *monopole, const int *mult, int m)
// calculate monopoles for each cell
// if all particles have the same charge/mass, then they are proportional to multiplicities
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < m;
		 i += gridDim.x * blockDim.x)
	{
		monopole[i] = (SCAL)mult[i];
	}
}

inline __host__ __device__ void centerLeaves_krnl(VEC *__restrict__ center, const int *mult, const int *index,
												  const VEC *__restrict__ p, int begi, int endi, int stride)
// calculate the center of charge/mass for each cell
{
	for (int i = begi; i < endi; i += stride)
	{
		int mlt = mult[i];
		VEC t{};
		if (mlt > 0)
		{
			const VEC *pi = p + index[i];
			for (int j = 0; j < mlt; ++j)
				t += pi[j];
			t /= (SCAL)mlt;
		}
		center[i] = t;
	}
}

__global__ void centerLeaves(VEC *center, const int *mult, const int *index, const VEC *p, int m)
{
	centerLeaves_krnl(center, mult, index, p, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void centerLeaves_cpu(VEC *center, const int *mult, const int *index, const VEC *p, int m)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(centerLeaves_krnl, center, mult, index, p, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void p2p2_krnl(VEC *__restrict__ a, const int *mult, const int *index,
										  const VEC *__restrict__ p, int side, int radius, SCAL d_EPS2,
										  int begi, int endi, int stride)
// particle to particle interaction
// radius: it is the infinity-norm radius (given as number of leaf cells) of the neighborhood area
//		   centered on the leaf cell that contain the particle. Only interactions in this range
//		   are considered here
{
	for (int ij = begi; ij < endi; ij += stride)
	{
		int ind1 = index[ij];
		int i = ij / side, j = ij - i*side;
		int kmin = ((i-radius > 0) ? (i-radius) : 0),
			kmax = ((i+radius < side-1) ? (i+radius) : (side-1));
		int lmin = ((j-radius > 0) ? (j-radius) : 0),
			lmax = ((j+radius < side-1) ? (j+radius) : (side-1));
		int mlt1 = mult[ij];
		const VEC *p1 = p + ind1;
		VEC *a1 = a + ind1;
		for (int h = 0; h < mlt1; ++h)
		{
			VEC atmp{};
			for (int k = kmin; k <= kmax; ++k)
			{
				int klT = k*side+lmin;
				int indT = index[klT];
				int mltT = mult[klT];
				const int *multT = mult + klT;
				const VEC *pT = p + indT;
				for (int l = 1; l <= lmax - lmin; ++l)
					mltT += multT[l];
				for (int g = 0; g < mltT; ++g)
				{
					VEC d = p1[h] - pT[g];
					SCAL dist2 = dot(d, d) + d_EPS2;
					SCAL invDist2 = (SCAL)1 / dist2;

					atmp = kernel(atmp, d, invDist2);
				}
			}
			a1[h] = atmp;
		}
	}
}

__global__ void p2p2(VEC *a, const int *mult, const int *index, const VEC *p, int m, int side, int radius, SCAL d_EPS2)
{
	p2p2_krnl(a, mult, index, p, side, radius, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void p2p2_cpu(VEC *a, const int *mult, const int *index, const VEC *p, int m, int side, int radius, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(p2p2_krnl, a, mult, index, p, side, radius, d_EPS2, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

__global__ void buildTree2(Tree tree, int l) // L-1 -> 0
// build the l-th of cells after the deeper one (l+1)-th
// "tree" contains only pointers to the actual tree in memory
{
	int sidel = tree_side(l);
	int sidelp = tree_side(l+1);
	int beg = tree_beg(l), begp = tree_beg(l+1);
	int n = tree_n(l);
	for (int ij0 = blockDim.x * blockIdx.x + threadIdx.x; ij0 < n; ij0 += gridDim.x * blockDim.x)
	{
		int i = ij0 / sidel, j = ij0 - i*sidel;
		int ij = beg+ij0;
		int ijp = begp+2*(i*sidelp+j);

		int mlt = tree.mult[ijp];
		mlt += tree.mult[ijp + 1];
		mlt += tree.mult[ijp + sidelp];
		mlt += tree.mult[ijp + sidelp + 1];

		SCAL mpole = (SCAL)mlt;

		VEC coord{};

		if (mlt > 0)
		{
			coord += tree.mpole[ijp] * tree.center[ijp];
			coord += tree.mpole[ijp + 1] * tree.center[ijp + 1];
			coord += tree.mpole[ijp + sidelp] * tree.center[ijp + sidelp];
			coord += tree.mpole[ijp + sidelp + 1] * tree.center[ijp + sidelp + 1];
			coord /= mpole;
		}
		tree.center[ij] = coord;
		tree.mpole[ij] = mpole;
		tree.mult[ij] = mlt;
	}
}

__global__ void c2c2(Tree tree, int l, int radius, SCAL d_EPS2) // L -> 1
// cell to cell interaction
{
	int sidel = tree_side(l);
	int beg = tree_beg(l);
	int n = tree_n(l);
	for (int ij = blockDim.x * blockIdx.x + threadIdx.x; ij < n; ij += gridDim.x * blockDim.x)
	{
		int i = ij / sidel, j = ij - i*sidel;
		int im = (i/2)*2;
		int kmin = ((im-2*radius > 0) ? (im-2*radius) : 0),
			kmax = ((im+(2*radius+1) < sidel-1) ? (im+(2*radius+1)) : (sidel-1));
		int ij1 = beg + ij;
			
		VEC atmp{};
		if (tree.mult[ij1] > 0)
		{
			int jm = (j/2)*2;
			int gmin = ((jm-2*radius > 0) ? (jm-2*radius) : 0),
				gmax = ((jm+(2*radius+1) < sidel-1) ? (jm+(2*radius+1)) : (sidel-1));
/*
			 im,jm : the first child of the parent node
	+---+---+---+---+---+---+
	|       | o     |       |
	+       +       +       +
	|       |     x |       |
	+---+---+---+---+---+---+
				  i,j : current node
		So there are always 2*radius nodes before 'o' and (2*radius+1) nodes after 'o' in the
		not-well-separated area of the parent node. The not-well-separated area of the current
		node will be excluded a posteriori
 */
			for (int k = kmin; k <= kmax; ++k)
				for (int g = gmin; g <= gmax; ++g)
				{
					if (!(k > i + radius || k < i - radius || g > j + radius || g < j - radius))
						continue;
					int ij2 = beg + k*sidel + g;
					VEC d = tree.center[ij1] - tree.center[ij2];
					SCAL dist2 = dot(d, d) + d_EPS2;
					SCAL invDist2 = (SCAL)1 / dist2; // __drcp_rn = (double) reciprocal + round to nearest

					atmp = kernel(atmp, d, invDist2, tree.mpole[ij2]);
				}
		}
		tree.field[ij1] = atmp;
	}
}

__global__ void pushl(Tree tree, int l) // 0 -> L-1
// push informations about the field from l-th level to (l+1)-th level
{
	int sidel = tree_side(l);
	int sidelp = tree_side(l+1);
	int beg = tree_beg(l), begp = tree_beg(l+1);
	int n = tree_n(l);
	for (int ij0 = blockDim.x * blockIdx.x + threadIdx.x; ij0 < n; ij0 += gridDim.x * blockDim.x)
	{
		int i = ij0 / sidel, j = ij0 - i*sidel;
		int ij = beg+ij0;
		int ijp = begp+2*(i*sidelp+j);

		VEC fld = tree.field[ij];

		tree.field[ijp] += fld;
		tree.field[ijp + 1] += fld;
		tree.field[ijp + sidelp] += fld;
		tree.field[ijp + sidelp + 1] += fld;
	}
}

__global__ void pushLeaves(VEC *__restrict__ a, const VEC *__restrict__ field,
						   const int *mult, const int *index, int m)
// push informations about the field from leaves to individual particles
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x;
		 i < m;
		 i += gridDim.x * blockDim.x)
	{
		int mlt = mult[i];
		VEC *ai = a + index[i];
		for (int j = 0; j < mlt; ++j)
			ai[j] += field[i];
	}
}

inline __host__ __device__ void rescale_krnl(VEC *a, const SCAL *param, int begi, int endi, int stride)
// rescale accelerations by a factor
{
	SCAL c = param[0];
	for (int i = begi; i < endi; i += stride)
		a[i] *= c;
}

__global__ void rescale(VEC *a, int n, const SCAL *param)
{
	rescale_krnl(a, param, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void rescale_cpu(VEC *a, int n, const SCAL *param)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(rescale_krnl, a, param, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

void appel(VEC *p, VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int radius = tree_radius;

	static int nBlocksRed = 1;
	static int n_prev = 0, n_max = 0, L = 0, nL1 = 0, sideL = 0;
	static int *d_keys = nullptr, *d_ind = nullptr;
	static char *d_tbuf = nullptr;
	static Tree tree;
	static VEC *d_minmax = nullptr;
	static VEC *minmax_ = new VEC[2*nBlocksRed];
	static VEC *d_tmp = nullptr;
	assert(n > BLOCK_SIZE);

	int nZ = 1<<DIM;
	int L_max = 0;

	if (n != n_prev)
	{
		L = (int)std::round(std::log2((SCAL)n)/DIM); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);
		L_max = (int)std::round(std::log2((SCAL)std::max(1, n_max))/DIM);
		L_max = std::max(L_max, 2);

		nL1 = 1 << ((L+1)*DIM);
		sideL = 1 << L;
		int ntot = (nL1 - 1) / (nZ - 1);
		if (L > L_max)
		{
			if (n_max > 0)
			{
				gpuErrchk(cudaFree(d_tbuf));
				gpuErrchk(cudaFree(d_minmax));
			}
			gpuErrchk(cudaMalloc((void**)&d_tbuf, (sizeof(VEC)*2 + sizeof(SCAL) + sizeof(int)*2)*ntot));
			gpuErrchk(cudaMalloc((void**)&d_minmax, sizeof(VEC)*2*nBlocksRed));
		}
		tree.center = (VEC*)d_tbuf;
		tree.field = tree.center + ntot;
		tree.mpole = (SCAL*)(tree.field + ntot);
		tree.mult = (int*)(tree.mpole + ntot);
		tree.index = tree.mult + ntot;
		if (n > n_prev)
		{
			if (n_prev > 0)
			{
				gpuErrchk(cudaFree(d_keys));
				gpuErrchk(cudaFree(d_ind));
				gpuErrchk(cudaFree(d_tmp));
			}
			gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(int)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_ind, sizeof(int)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_tmp, sizeof(VEC)*n));
		}
	}

	minmaxReduce(d_minmax, p, n, nBlocksRed);

	gpuErrchk(cudaMemcpy(minmax_, d_minmax, sizeof(VEC)*2*nBlocksRed, cudaMemcpyDeviceToHost));

	//printp <<< 1, 1 >>> (p, n);

	if (nBlocksRed > 1)
	{
		for (int i = 2; i < 2*nBlocksRed; i += 2)
		{
			minmax_[0] = fmin(minmax_[0], minmax_[i]);
			minmax_[1] = fmax(minmax_[1], minmax_[i+1]);
		}
		gpuErrchk(cudaMemcpy(d_minmax, minmax_, sizeof(VEC)*2, cudaMemcpyHostToDevice));
	}
	//std::cout << ' ' << minmax_[0] << ", " << minmax_[1] << std::endl;

	VEC Delta = minmax_[1] - minmax_[0];
	SCAL delta = fmax(Delta) / (SCAL)sideL, EPS = sqrt(EPS2);
	if (delta < EPS)
		delta = EPS;
	SCAL rdelta = (SCAL)1 / delta;

	evalKeys <<< nBlocks, BLOCK_SIZE >>> (d_keys, p, d_minmax, n, rdelta, sideL);
	evalIndices <<< nBlocks, BLOCK_SIZE >>> (d_ind, n);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cub::DoubleBuffer<int> d_dbuf(d_keys, d_keys + n);
	cub::DoubleBuffer<int> d_values(d_ind, d_ind + n);
	static void *d_tmp_stor = nullptr;
	static size_t stor_bytes = 0;
	if (n != n_prev)
	{
		if (L > L_max)
		{
			if (n_max > 0)
				gpuErrchk(cudaFree(d_tmp_stor));
			d_tmp_stor = nullptr;
			gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dbuf, d_values, n, 0, L*DIM));
			gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
		}
	}
	gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dbuf, d_values, n, 0, L*DIM));

	//check_krnl <<< nBlocks, BLOCK_SIZE >>> (d_dbuf.Current(), n);

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p, d_tmp, n);

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p+n, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p+n, d_tmp, n);

	int beg = tree_beg(L), m = tree_n(L);

	initLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.center + beg, m);
	indexLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.index + beg, d_dbuf.Current(), m, n);

	multLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.mult + beg, tree.index + beg, m, n);

	monopoleLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.mpole + beg, tree.mult + beg, m);

	centerLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	for (int l = L-1; l >= 2; --l)
		buildTree2 <<< nBlocks, BLOCK_SIZE >>> (tree, l);

	p2p2 <<< nBlocks, BLOCK_SIZE >>> (a, tree.mult + beg, tree.index + beg, p, m, tree_side(L), radius, EPS2);

	for (int l = L; l >= 2; --l)
		c2c2 <<< nBlocks, BLOCK_SIZE >>> (tree, l, radius, EPS2);

	for (int l = 2; l <= L-1; ++l)
		pushl <<< nBlocks, BLOCK_SIZE >>> (tree, l);

	pushLeaves <<< nBlocks, BLOCK_SIZE >>> (a, tree.field + beg, tree.mult + beg,
											tree.index + beg, m);
	if (param != nullptr)
		rescale <<< nBlocks, BLOCK_SIZE >>> (a, n, param);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

#endif // !APPEL_CUDA_H
