//  Fast multipole method (FMM) in 3d cartesian coordinates
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

#ifndef FMM_CART3_KDTREE_CUDA_H
#define FMM_CART3_KDTREE_CUDA_H

#include "appel.cuh"
#include "fmm_cart_base3.cuh"
#include "parasort/parasort.h" // parasort
#include "bb_segsort/bb_segsort.cuh" // bb_segsort

struct fmmTree_kd
{
	VEC *__restrict__ center, *__restrict__ lbound, *__restrict__ rbound;
	SCAL *__restrict__ mpole, *__restrict__ local;
	int *__restrict__ mult, *__restrict__ index, *__restrict__ splitdim;
	int p;
};

constexpr __device__ __host__ int kd_beg(int l)
// return the position of the first element of the l-th level (starting from 0)
{
	return (1 << l) - 1;
}
constexpr __device__ __host__ int kd_end(int l)
// return the position after the last element of the l-th level (starting from 0)
{
	return kd_beg(l+1);
}
constexpr __device__ __host__ int kd_n(int l)
// return the number of elements in the l-th level (starting from 0)
{
	return 1 << l;
}
constexpr __device__ __host__ int kd_ntot(int L)
// return the total number of nodes in a binary tree with maximum level L
{
	return (1 << (L+1)) - 1;
}
constexpr __device__ __host__ int kd_parent(int i)
// return the parent node index of node i
// undefined for i = 0 (root node)
{
	return (i-1) >> 1;
}
constexpr __device__ __host__ int kd_lchild(int i)
// return the left-child node index of node i
{
	return 2*i + 1;
}
constexpr __device__ __host__ int kd_rchild(int i)
// return the right-child node index of node i
{
	return 2*i + 2;
}
constexpr __device__ __host__ bool kd_is_lchild(int i)
// return true if i is a left-child node
{
	return i != 0 && i == kd_lchild(kd_parent(i));
}
constexpr __device__ __host__ bool kd_is_rchild(int i)
// return true if i is a right-child node
{
	return i != 0 && i == kd_rchild(kd_parent(i));
}

inline __host__ __device__ SCAL& get_axis(VEC& v, int axis)
{
	return reinterpret_cast<SCAL*>(&v)[axis];
}
inline __host__ __device__ const SCAL& get_axis(const VEC& v, int axis)
{
	return reinterpret_cast<const SCAL*>(&v)[axis];
}

inline __host__ __device__ void evalRootBox_krnl(fmmTree_kd& tree, const VEC *d_minmax)
{
	VEC d = d_minmax[1] - d_minmax[0];
	int arg = (d.x > d.y) ? ((d.x > d.z) ? 0 : 2) : ((d.y > d.z) ? 1 : 2);
	tree.lbound[0] = d_minmax[0];
	tree.rbound[0] = d_minmax[1];
	tree.splitdim[0] = arg;
	tree.index[0] = 0;
}

__global__ void evalRootBox(fmmTree_kd tree, const VEC *d_minmax)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		evalRootBox_krnl(tree, d_minmax);
}
void evalRootBox_cpu(fmmTree_kd& tree, const VEC *d_minmax)
{
	evalRootBox_krnl(tree, d_minmax);
}

template <bool b_last>
inline __host__ __device__ void evalBox_krnl(fmmTree_kd tree, const VEC *p, int n, int l,
                                             int begi, int endi, int stride) // l = 1 -> L
{
	int m = kd_n(l);
	int beg = kd_beg(l);
	for (int i = begi; i < endi; i += stride)
	{
		int start = (i == 0) ? 0 : ((long long)n * i - 1) / m + 1;
		int end = ((long long)n * (i+1) - 1) / m + 1;
		int j = beg + i;
		int parent = kd_parent(j);
		int split = tree.splitdim[parent];
		VEC lb = tree.lbound[parent];
		VEC rb = tree.rbound[parent];
		if (kd_is_rchild(j))
			get_axis(lb, split) = get_axis(p[start], split);
		if (kd_is_lchild(j))
			get_axis(rb, split) = get_axis(p[end-1], split);
		VEC d = rb - lb;
		int arg = (d.x > d.y) ? ((d.x > d.z) ? 0 : 2) : ((d.y > d.z) ? 1 : 2);
		tree.lbound[j] = lb;
		tree.rbound[j] = rb;
		tree.splitdim[j] = arg;
		tree.index[j] = start;
		if (!b_last && i == m-1)
			tree.index[j+1] = n;
	}
}

template <bool b_last = false>
__global__ void evalBox(fmmTree_kd tree, const VEC *p, int n, int l)
{
	int m = kd_n(l);
	evalBox_krnl<b_last>(tree, p, n, l, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

template <bool b_last = false>
void evalBox_cpu(fmmTree_kd tree, const VEC *p, int n, int l)
{
	int m = kd_n(l);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalBox_krnl<b_last>, tree, p, n, l, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __device__ void evalKeys_kdtree_krnl(float *keys, const int *splitdim, const VEC *p, int n,
                                            int l, int begi, int endi, int stride)
// calculate keys for all particles at level l
{
	long long m = kd_n(l);
	for (int i = begi; i < endi; i += stride)
		keys[i] = (float)get_axis(p[i], splitdim[m * i / n]);
}

inline void evalKeys_kdtree_cpu_krnl(unsigned long long *keys, const int *splitdim, const VEC *p, long long n,
                                     int l, int precision, int begi, int endi, int stride)
// calculate keys for all particles at level l
{
	long long m = kd_n(l);
	for (int i = begi; i < endi; i += stride)
	{
		unsigned long long j = m * i / n;
		union
		{
			unsigned u;
			float f;
		} p_;
		p_.f = (float)get_axis(p[i], splitdim[j]);
		if (!signbit(p_.f))
			p_.f = -p_.f;
		else
			p_.u = ~p_.u; // sorry for undefined behavior
		keys[i] = (j << precision) | (unsigned long long)(p_.u >> (32 - precision));
	}
}

__global__ void evalKeys_kdtree(float *keys, const int *splitdim, const VEC *p, int n, int l)
{
	evalKeys_kdtree_krnl(keys, splitdim, p, n, l, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void evalKeys_kdtree_cpu(unsigned long long *keys, const int *splitdim, const VEC *p, int n, int l, int precision = 32)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalKeys_kdtree_cpu_krnl, keys, splitdim, p, n, l, precision, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline void fmm_init3_kdtree_krnl(fmmTree_kd tree, int begi, int endi, int stride)
{
	int offM = symmetricoffset3(tree.p);
	int offL = tracelessoffset3(tree.p+1);
	for (int i = begi; i < endi; i += stride)
	{
		tree.center[i] = VEC{};
		SCAL *multipole = tree.mpole + offM*i;
		for (int j = 0; j < offM; ++j)
			multipole[j] = (SCAL)0;
		SCAL *loc = tree.local + offL*i;
		for (int j = 0; j < offL; ++j)
			loc[j] = (SCAL)0;
	}
}

void fmm_init3_kdtree_cpu(fmmTree_kd& tree, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = kd_ntot(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_init3_kdtree_krnl, tree, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_multipoleLeaves3_kdtree_krnl(fmmTree_kd tree, const VEC *p,
                                                                 int begi, int endi, int stride)
// calculate multipoles for each cell
// assumes all particles have the same charge/mass
{
	int off = symmetricoffset3(tree.p);
	for (int i = begi; i < endi; i += stride)
	{
		SCAL *multipole = tree.mpole + off*i;
		const VEC *pi = p + tree.index[i];
		multipole[0] = (SCAL)tree.mult[i];
		if (tree.p >= 3)
			for (int j = 0; j < tree.mult[i]; ++j)
			{
				VEC d = pi[j] - tree.center[i];
				for (int q = 2; q <= tree.p-1; ++q)
					static_p2m_acc3(multipole + symmetricoffset3(q), q, d);
			}
	}
}

__global__ void fmm_multipoleLeaves3_kdtree(fmmTree_kd tree, const VEC *p, int L)
{
	int beg = kd_beg(L);
	int end = kd_end(L);
	fmm_multipoleLeaves3_kdtree_krnl(tree, p, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x);
}

void fmm_multipoleLeaves3_kdtree_cpu(fmmTree_kd tree, const VEC *p, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int beg = kd_beg(L);
	int end = kd_end(L);
	int niter = (end-beg-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_multipoleLeaves3_kdtree_krnl, tree, p, beg+niter*i, std::min(beg+niter*(i+1), end), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __device__ void fmm_buildTree3_kdtree_krnl(fmmTree_kd tree, int begi, int endi, int stride) // L-1 -> 0
// build the l-th of cells after the deeper one (l+1)-th
// "tree" contains only pointers to the actual tree in memory
{
	int off = symmetricoffset3(tree.p);
	extern __shared__ SCAL smems[];
	SCAL *smin = smems + 2*off*threadIdx.x;
	SCAL *smout = smin + off;
	int inds[2], mlts[2];
	VEC centers[2];
	for (int ijk = begi; ijk < endi; ijk += stride)
	{
		inds[0] = kd_lchild(ijk);
		inds[1] = kd_rchild(ijk);

		for (int ii = 0; ii < 2; ++ii)
			mlts[ii] = tree.mult[inds[ii]];
		for (int ii = 0; ii < 2; ++ii)
			centers[ii] = tree.center[inds[ii]];

		int mlt = 0;
		for (int ii = 0; ii < 2; ++ii)
			mlt += mlts[ii];

		SCAL mpole0 = (SCAL)mlt;

		VEC coord{};
		for (int ii = 0; ii < 2; ++ii)
			coord += (SCAL)mlts[ii] * centers[ii];
		coord /= mpole0;

		SCAL *multipole = tree.mpole + ijk*off;
		if (tree.p >= 3)
		{
			for (int j = symmetricoffset3(2); j < off; ++j)
				smout[j] = 0;
			const SCAL *multipole2;
			VEC d;
			for (int ii = 0; ii < 2; ++ii)
			{
				d = coord - centers[ii];
				multipole2 = tree.mpole + inds[ii]*off;
				for (int j = 0; j < off; ++j)
					smin[j] = multipole2[j];
				for (int q = 2; q <= tree.p-1; ++q)
					static_m2m_acc3(smout + symmetricoffset3(q), smin, q, d);
			}
			for (int j = symmetricoffset3(2); j < off; ++j)
				multipole[j] = smout[j];
		}
		multipole[0] = mpole0;

		tree.center[ijk] = coord;
		tree.mult[ijk] = mlt;
	}
}

inline __host__ __device__ void fmm_buildTree3_kdtree2_krnl(fmmTree_kd tree, int begi, int endi, int stride) // L-1 -> 0
// build the l-th of cells after the deeper one (l+1)-th
// "tree" contains only pointers to the actual tree in memory
{
	int off = symmetricoffset3(tree.p);
	int inds[2];
	for (int ijk = begi; ijk < endi; ijk += stride)
	{
		inds[0] = kd_lchild(ijk);
		inds[1] = kd_rchild(ijk);

		int mlt = 0;
		for (int ii = 0; ii < 2; ++ii)
			mlt += tree.mult[inds[ii]];

		SCAL mpole0 = (SCAL)mlt;

		VEC coord{};
		for (int ii = 0; ii < 2; ++ii)
			coord += (SCAL)tree.mult[inds[ii]] * tree.center[inds[ii]];
		coord /= mpole0;

		SCAL *multipole = tree.mpole + ijk*off;
		if (tree.p >= 3)
		{
			const SCAL *multipole2;
			VEC d;
			for (int ii = 0; ii < 2; ++ii)
			{
				d = coord - tree.center[inds[ii]];
				multipole2 = tree.mpole + inds[ii]*off;
				for (int q = 2; q <= tree.p-1; ++q)
					static_m2m_acc3(multipole + symmetricoffset3(q), multipole2, q, d);
			}
		}
		multipole[0] = mpole0;

		tree.center[ijk] = coord;
		tree.mult[ijk] = mlt;
	}
}

__global__ void fmm_buildTree3_kdtree(fmmTree_kd tree, int l)
{
	int beg = kd_beg(l);
	int end = kd_end(l);
	fmm_buildTree3_kdtree_krnl(tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x);
}
__global__ void fmm_buildTree3_kdtree2(fmmTree_kd tree, int l)
{
	int beg = kd_beg(l);
	int end = kd_end(l);
	fmm_buildTree3_kdtree2_krnl(tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x);
}

void fmm_buildTree3_kdtree_cpu(fmmTree_kd tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int beg = kd_beg(l);
	int end = kd_end(l);
	int niter = (end-beg-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_buildTree3_kdtree2_krnl, tree, beg + niter*i, std::min(beg + niter*(i+1), end), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ SCAL kd_size(const VEC& l, const VEC& r)
{
	VEC diff = r - l;
	return dot(diff, diff);
}

inline __host__ __device__ bool kd_admissible(const fmmTree_kd& tree, int n1, int n2, SCAL par)
{
	VEC d = tree.center[n2] - tree.center[n1];
	SCAL dist2 = dot(d, d);
	SCAL sz1 = kd_size(tree.lbound[n1], tree.rbound[n1]);
	SCAL sz2 = kd_size(tree.lbound[n2], tree.rbound[n2]);
#ifdef __CUDA_ARCH__
	SCAL M = __powf(float(max(tree.mult[n1], tree.mult[n2])) / tree.mult[0], 1.f/(3*tree.p+6));
#else
	SCAL M = pow(SCAL(max(tree.mult[n1], tree.mult[n2])) / tree.mult[0], SCAL(1)/(3*tree.p+6));
#endif
	SCAL parM = par * M;
	return parM*parM*max(sz1, sz2) < dist2;
}

__constant__ const int2 init_stack7[] = {
	{3, 3}, {3, 4}, {4, 4}, {1, 2}, {5, 5}, {5, 6}, {6, 6}
};
__constant__ const int2 init_stack15[] = {
	{7, 7}, {7, 8}, {8, 8}, {3, 4}, {9, 9}, {9, 10}, {10, 10},
	{1, 2}, {11, 11}, {11, 12}, {12, 12}, {5, 6}, {13, 13}, {13, 14}, {14, 14}
};
__constant__ const int2 init_stack18[] = {
	{7, 7}, {7, 8}, {8, 8}, {3, 4}, {9, 9}, {9, 10}, {10, 10},
	{3, 5}, {3, 6}, {4, 5}, {4, 6}, {11, 11}, {11, 12}, {12, 12},
	{5, 6}, {13, 13}, {13, 14}, {14, 14}
};

template <bool b_m2l_first = true>
__global__ void fmm_dualTraversal(fmmTree_kd tree, int2 *p2p_list, int2 *m2l_list, int2 *stack, int *p2p_n, int *m2l_n,
                                  int p2p_max, int m2l_max, int stack_max, SCAL r, int L)
// call with CUDA gridsize = 1, 3, 7, 15 or 18
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bdim = blockDim.x;
	int gdim = gridDim.x;

	__shared__ int top;

	int ntot = kd_ntot(L), max_top = 1;
	int stack_size = stack_max/gdim;
	int2 *block_stack = stack + stack_size*bid;

	int2 np;

	if (tid == 0)
	{
		switch (gdim)
		{
			case 1:
				block_stack[0] = {0, 0};
				break;
			case 3:
				block_stack[0] = (bid == 0) ? int2{1, 1} :
				                 (bid == 1) ? int2{1, 2} :
				                              int2{2, 2};
				break;
			case 7:
				block_stack[0] = init_stack7[bid];
				break;
			case 15:
				if (L >= 3)
				{
					block_stack[0] = init_stack15[bid];
					break;
				}
			case 18:
				if (L >= 3)
				{
					block_stack[0] = init_stack18[bid];
					break;
				}
			default:
				assert(false); // call with CUDA gridsize = 1, 3, 7, 15 or 18
		}
		top = 1;

		if (bid == 0)
		{
			// Initialize counters
			*p2p_n = 0;
			*m2l_n = 0;
		}
	}

	__threadfence(); // Ensure memory writes are visible
	__syncthreads(); // Ensure all threads wait for initialization

	while (top > 0)
	{
		int stack_pos = top - tid - 1;

		if (stack_pos >= 0)
			np = block_stack[stack_pos];

		__syncthreads();
		if (tid == 0)
			top = max(top - bdim, 0);
		__syncthreads();

		if (stack_pos >= 0)
		{
			if (!b_m2l_first && (kd_lchild(np.x) >= ntot & kd_lchild(np.y) >= ntot))
			{
				if (np.x != np.y)
				{
					int pos = atomicAdd(p2p_n, 1);
					if (pos < p2p_max)
						p2p_list[pos] = np;
				}
			}
			else if (np.x == np.y & kd_lchild(np.x) < ntot)
			{
				int pos = atomicAdd(&top, 3);
				block_stack[pos  ] = {kd_lchild(np.x), kd_lchild(np.x)};
				block_stack[pos+1] = {kd_lchild(np.x), kd_rchild(np.x)};
				block_stack[pos+2] = {kd_rchild(np.x), kd_rchild(np.x)};
			}
			else if (kd_admissible(tree, np.x, np.y, r))
			{
				int pos = atomicAdd(m2l_n, 1);
				if (pos < m2l_max)
					m2l_list[pos] = np;
			}
			else if (b_m2l_first && (kd_lchild(np.x) >= ntot & kd_lchild(np.y) >= ntot))
			{
				if (np.x != np.y)
				{
					int pos = atomicAdd(p2p_n, 1);
					if (pos < p2p_max)
						p2p_list[pos] = np;
				}
			}
			else
			{
				bool cond = kd_lchild(np.x) >= ntot | (kd_lchild(np.y) < ntot
					& kd_size(tree.lbound[np.x], tree.rbound[np.x]) <= kd_size(tree.lbound[np.y], tree.rbound[np.y]));
				int pos = atomicAdd(&top, 2);
				block_stack[pos  ] = cond ? int2{np.x, kd_lchild(np.y)} : int2{kd_lchild(np.x), np.y};
				block_stack[pos+1] = cond ? int2{np.x, kd_rchild(np.y)} : int2{kd_rchild(np.x), np.y};
			}
		}
		__syncthreads();
		if (tid == 0 & top > max_top)
			max_top = top;
	}

	__threadfence();
	__syncthreads();

	if (tid == 0)
	{
		if (*p2p_n > p2p_max)
		{
			*p2p_n = p2p_max;
			printf("Error: exceeded p2p allocated memory\n");
		}
		if (*m2l_n > m2l_max)
		{
			*m2l_n = m2l_max;
			printf("Error: exceeded m2l allocated memory\n");
		}
		if (max_top > stack_max)
			printf("Error: insufficient stack size for dual tree traversal, data may be corrupted.\n");
	}
}

void fmm_dualTraversal_cpu(const fmmTree_kd& tree, std::vector<int2>& p2p_list, std::vector<int2>& m2l_list, std::vector<int2>& stack,
                           SCAL r, int L)
{
	p2p_list.clear();
	m2l_list.clear();
	stack.clear();

	int2 np{0, 0};
	int ntot = kd_ntot(L);

	stack.push_back(np);

	while (stack.size() > 0)
	{
		np = stack.back();
		stack.pop_back();

		if (kd_lchild(np.x) >= ntot && kd_lchild(np.y) >= ntot)
		{
			if (np.x != np.y)
				p2p_list.push_back(np);
		}
		else if (np.x == np.y)
		{
			stack.push_back({kd_lchild(np.x), kd_lchild(np.x)});
			stack.push_back({kd_lchild(np.x), kd_rchild(np.x)});
			stack.push_back({kd_rchild(np.x), kd_rchild(np.x)});
		}
		else if (kd_admissible(tree, np.x, np.y, r))
			m2l_list.push_back(np);
		else if (kd_lchild(np.x) >= ntot || (kd_lchild(np.y) < ntot
			&& kd_size(tree.lbound[np.x], tree.rbound[np.x]) <= kd_size(tree.lbound[np.y], tree.rbound[np.y])))
		{
			stack.push_back({np.x, kd_lchild(np.y)});
			stack.push_back({np.x, kd_rchild(np.y)});
		}
		else
		{
			stack.push_back({kd_lchild(np.x), np.y});
			stack.push_back({kd_rchild(np.x), np.y});
		}
	}
}

inline __host__ __device__ void fmm_c2c3_kdtree_krnl(fmmTree_kd tree, const int2 *m2l_list, SCAL d_EPS2,
                                                     int begi, int endi, int stride, SCAL *tempi)
// cell to cell interaction
{
	int offM = symmetricoffset3(tree.p);
	int offL = tracelessoffset3(tree.p+1);
	int offL2 = tracelessoffset3(tree.p-1);
#ifdef __CUDA_ARCH__
	extern __shared__ SCAL smems[];
	SCAL *smp = smems + (tree.p+1)*(tree.p+2)/2*blockDim.x + offM*threadIdx.x;
	SCAL *sloc = smems + ((tree.p+1)*(tree.p+2)/2+offM)*blockDim.x + offL*threadIdx.x;
#endif

	for (int i = begi; i < endi; i += stride)
	{
		int n1 = m2l_list[i].x;
		int n2 = m2l_list[i].y;
		SCAL *loc1 = tree.local + n1*offL;
		SCAL *loc2 = tree.local + n2*offL;
		SCAL *mp1 = tree.mpole + n1*offM;
		SCAL *mp2 = tree.mpole + n2*offM;

		VEC d = tree.center[n1] - tree.center[n2];
		SCAL r = sqrt(dot(d, d) + d_EPS2);
		d /= r;
#ifdef __CUDA_ARCH__
		for (int j = 0; j < offM; ++j)
			smp[j] = mp2[j];
		for (int j = 1; j < offL; ++j)
			sloc[j] = 0;
		SCAL mp = SCAL(1) / smp[0];

		static_m2l_acc3<1, -2, false, false, true>(sloc, tempi, smp, tree.p, d, r);
		for (int j = 0; j < offL; ++j)
			myAtomicAdd(loc1 + j, sloc[j]);

		for (int j = 0; j < offM; ++j)
			smp[j] = mp1[j];
		for (int j = 1; j < offL2; ++j)
			sloc[j] = 0;
		mp *= smp[0];

		if (tree.p >= 3)
			static_m2l_acc3<1, -2, false, false, true, -2>(sloc, tempi, smp, tree.p, -d, r);
		for (int j = 1; j < offL2; ++j)
			myAtomicAdd(loc2 + j, sloc[j]);
		// exploiting symmetry relations
		for (int q = tree.p-1; q <= tree.p; ++q)
		{
			SCAL c = mp*paritysign(q);
			for (int j = tracelessoffset3(q); j < tracelessoffset3(q+1); ++j)
				myAtomicAdd(loc2 + j, c*sloc[j]);
		}
#else
		static_m2l_acc3<1, -2, false, true, true>(loc1, tempi, mp2, tree.p, d, r);
		static_m2l_acc3<1, -2, false, true, true>(loc2, tempi, mp1, tree.p, -d, r);
#endif
	}
}

__global__ void fmm_c2c3_kdtree2(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
// cell to cell interaction
{
	extern __shared__ SCAL smems[]; // (p+1)*(p+2)/2
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	int gdim = gridDim.x;
	SCAL *temp = smems + (tree.p+1)*(tree.p+2)/2*threadIdx.x;

	int offM = symmetricoffset3(tree.p);
	int offL = tracelessoffset3(tree.p+1);

	int end = *m2l_n;

	for (int i = bid*bdim + tid; i < end; i += bdim*gdim)
	{
		int n1 = m2l_list[i].x;
		int n2 = m2l_list[i].y;
		SCAL *loc1 = tree.local + n1*offL;
		SCAL *loc2 = tree.local + n2*offL;
		SCAL *mp1 = tree.mpole + n1*offM;
		SCAL *mp2 = tree.mpole + n2*offM;

		VEC d = tree.center[n1] - tree.center[n2];
		SCAL r = sqrt(dot(d, d) + d_EPS2);
		d /= r;

		static_m2l_acc3<1, -2, false, true, true>(loc1, temp, mp2, tree.p, d, r);
		static_m2l_acc3<1, -2, false, true, true>(loc2, temp, mp1, tree.p, -d, r);
	}
}

__global__ void fmm_c2c3_kdtree_coalesced(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
// cell to cell interaction
{
	int offM = symmetricoffset3(tree.p);
	int offL = tracelessoffset3(tree.p+1);

	extern __shared__ SCAL smems[]; // (p+1)*(p+2)/2 + offM
	SCAL *temp = smems;
	SCAL *smp = smems + (tree.p+1)*(tree.p+2)/2;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	int gdim = gridDim.x;
	int end = *m2l_n;

	for (int i = bid; i < end; i += gdim)
	{
		int n1 = m2l_list[i].x;
		int n2 = m2l_list[i].y;
		SCAL *loc1 = tree.local + n1*offL;
		SCAL *loc2 = tree.local + n2*offL;
		SCAL *mp1 = tree.mpole + n1*offM;
		SCAL *mp2 = tree.mpole + n2*offM;

		VEC d = tree.center[n1] - tree.center[n2];
		SCAL r = sqrt(dot(d, d) + d_EPS2);
		d /= r;

		for (int j = tid; j < offM; j += bdim)
			smp[j] = mp2[j];
		m2l_acc_coalesced3<true, true>(loc1, temp, smp, tree.p, tree.p, d, r, 1, tree.p);
		__syncthreads();
		for (int j = tid; j < offM; j += bdim)
			smp[j] = mp1[j];
		m2l_acc_coalesced3<true, true>(loc2, temp, smp, tree.p, tree.p, -d, r, 1, tree.p);
		__syncthreads();
	}
}

__global__ void fmm_c2c3_kdtree(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
{
	extern __shared__ SCAL smems[]; // size must be at least ((p+1)*(p+2)/2 + offM + offL)*blockDim.x
	SCAL *tempi = smems + (tree.p+1)*(tree.p+2)/2*threadIdx.x;
	fmm_c2c3_kdtree_krnl(tree, m2l_list, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, *m2l_n, gridDim.x * blockDim.x, tempi);
}

void fmm_c2c3_kdtree_cpu(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[(tree.p+1)*(tree.p+2)/2 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int niter = (*m2l_n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_c2c3_kdtree_krnl, tree, m2l_list, d_EPS2, niter*i, std::min(niter*(i+1), *m2l_n), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

inline __host__ __device__ void fmm_p2p_interaction(VEC *__restrict__ a1, const VEC *__restrict__ p1, const VEC *__restrict__ p2,
                                                    int mlt1, int mlt2, SCAL d_EPS2)
{
	for (int h = 0; h < mlt1; ++h)
	{
		VEC atmp{};
		VEC p1h = p1[h];
		for (int g = 0; g < mlt2; ++g)
		{
			VEC d = p1h - p2[g];
			SCAL dist2 = dot(d, d) + d_EPS2;
			SCAL invDist2 = (SCAL)1 / dist2;

			atmp = kernel(atmp, d, invDist2);
		}
#ifdef __CUDA_ARCH__
		myAtomicAdd(&a1[h].x, atmp.x);
		myAtomicAdd(&a1[h].y, atmp.y);
		myAtomicAdd(&a1[h].z, atmp.z);
#else
		std::atomic_ref<SCAL> atomic0(a1[h].x);
		std::atomic_ref<SCAL> atomic1(a1[h].y);
		std::atomic_ref<SCAL> atomic2(a1[h].z);
		atomic0 += atmp.x;
		atomic1 += atmp.y;
		atomic2 += atmp.z;
#endif
	}
}

inline __host__ __device__ void fmm_p2p3_kdtree_krnl(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                                     const int2 *p2p_list, int mlt_max, SCAL d_EPS2,
                                                     int begi, int endi, int stride)
// particle to particle interaction
{
#ifdef __CUDA_ARCH__
	extern __shared__ VEC smem[];
	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	VEC *sp2 = smem + mlt_max*threadIdx.x;
	VEC *sa2 = smem + mlt_max*(blockDim.x + threadIdx.x);

	for (int i = begi; i < endi; i += stride)
	{
		int n1 = p2p_list[i].x;
		int n2 = p2p_list[i].y;

		int ind1 = tree.index[n1];
		int ind2 = tree.index[n2];
		int mlt1 = tree.mult[n1];
		int mlt2 = tree.mult[n2];
		const VEC *p1 = p + ind1;
		const VEC *p2 = p + ind2;
		VEC *a1 = a + ind1;
		VEC *a2 = a + ind2;

		for (int g = 0; g < mlt2; ++g)
			sp2[g] = p2[g];

		for (int g = 0; g < mlt2; ++g)
			sa2[g] = VEC{};

		for (int h = 0; h < mlt1; ++h)
		{
			VEC atmp{};
			VEC p1h = p1[h];
			for (int g = 0; g < mlt2; ++g)
			{
				VEC d = p1h - sp2[g];
				SCAL k = dot(d, d) + d_EPS2;
				k = (SCAL)1 / k;
				k *= sqrt(k);
				d *= k;

				atmp += d;
				sa2[g] -= d;
			}
			myAtomicAdd(&a1[h].x, atmp.x);
			myAtomicAdd(&a1[h].y, atmp.y);
			myAtomicAdd(&a1[h].z, atmp.z);
		}
		for (int g = 0; g < mlt2; ++g)
		{
			myAtomicAdd(&a2[g].x, sa2[g].x);
			myAtomicAdd(&a2[g].y, sa2[g].y);
			myAtomicAdd(&a2[g].z, sa2[g].z);
		}
	}
#else
	for (int i = begi; i < endi; i += stride)
	{
		int n1 = p2p_list[i].x;
		int n2 = p2p_list[i].y;

		int ind1 = tree.index[n1];
		int ind2 = tree.index[n2];
		int mlt1 = tree.mult[n1];
		int mlt2 = tree.mult[n2];
		const VEC *p1 = p + ind1;
		const VEC *p2 = p + ind2;

		fmm_p2p_interaction(a + ind1, p1, p2, mlt1, mlt2, d_EPS2);
		fmm_p2p_interaction(a + ind2, p2, p1, mlt2, mlt1, d_EPS2);
	}
#endif
}

__global__ void fmm_p2p3_kdtree_coalesced(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                          const int2 *p2p_list, const int *p2p_n, int mlt_max, SCAL d_EPS2)
// particle to particle interaction
{
	extern __shared__ VEC smem[]; // 2*mlt_max*sizeof(VEC)*nwarps
	int tid = threadIdx.x;
	int wdim = min(bitceil(mlt_max), 32u);
	int wid = tid/wdim; // (sub)warp id
	int lid = tid%wdim; // lane id
	int bdim = blockDim.x;
	int nwarps = bdim/wdim;
	int bid = blockIdx.x;
	int gdim = gridDim.x;
	int stride = gdim*nwarps;

	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	VEC *sp2 = smem + mlt_max*wid;
	VEC *sa2 = smem + mlt_max*(nwarps + wid);

	int beg = bid*nwarps+wid;
	int end = *p2p_n;
	unsigned loop_mask = __ballot_sync(0xFFFFFFFF, beg < end);

	for (int i = beg; i < end; i += stride)
	{
		int n1 = p2p_list[i].x;
		int n2 = p2p_list[i].y;
		int mlt1 = tree.mult[n1];
		int mlt2 = tree.mult[n2];
		if (mlt1 > mlt2)
		{
			swap(n1, n2);
			swap(mlt1, mlt2);
		}

		int ind1 = tree.index[n1];
		int ind2 = tree.index[n2];
		const VEC *p1 = p + ind1;
		const VEC *p2 = p + ind2;
		VEC *a1 = a + ind1;
		VEC *a2 = a + ind2;

		for (int g = lid; g < mlt2; g += wdim)
			sp2[g] = p2[g];

		for (int g = lid; g < mlt2; g += wdim)
			sa2[g] = VEC{};

		__syncwarp(loop_mask);

		unsigned mask = __ballot_sync(loop_mask, lid < mlt1);
		for (int h = lid; h < mlt1; h += wdim)
		{
			VEC atmp{};
			VEC p1h = p1[h];
			unsigned inner_mask = mask;
			for (int g = 0; g < mlt2; ++g)
			{
				int gg = (g+lid) % mlt2;
				VEC d = p1h - sp2[gg];
				SCAL k = dot(d, d) + d_EPS2;
				k = (SCAL)1 / k;
				k *= sqrt(k);
				d *= k;

				atmp += d;
				sa2[gg] -= d;
				inner_mask = __ballot_sync(inner_mask, g+1 < mlt2);
			}
			myAtomicAdd(&a1[h].x, atmp.x);
			myAtomicAdd(&a1[h].y, atmp.y);
			myAtomicAdd(&a1[h].z, atmp.z);
			mask = __ballot_sync(mask, h+wdim < mlt1);
		}
		__syncwarp(loop_mask);

		for (int g = lid; g < mlt2; g += wdim)
		{
			myAtomicAdd(&a2[g].x, sa2[g].x);
			myAtomicAdd(&a2[g].y, sa2[g].y);
			myAtomicAdd(&a2[g].z, sa2[g].z);
		}
		loop_mask = __ballot_sync(loop_mask, i+stride < end);
	}
}

__global__ void fmm_p2p3_kdtree_coalesced2(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                           const int2 *p2p_list, const int *p2p_n, SCAL d_EPS2)
// particle to particle interaction
{
	extern __shared__ VEC sp[]; // mlt_max*sizeof(VEC)
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	int gdim = gridDim.x;

	for (int i = bid; i < *p2p_n; i += gdim)
	{
		int n1 = p2p_list[i].x;
		int n2 = p2p_list[i].y;

		int ind1 = tree.index[n1];
		int ind2 = tree.index[n2];
		int mlt1 = tree.mult[n1];
		int mlt2 = tree.mult[n2];
		const VEC *p1 = p + ind1;
		const VEC *p2 = p + ind2;
		VEC *a1 = a + ind1;
		VEC *a2 = a + ind2;

		for (int g = tid; g < mlt2; g += bdim)
			sp[g] = p2[g];
		__syncthreads();

		for (int h = tid; h < mlt1; h += bdim)
		{
			VEC atmp{};
			VEC p1h = p1[h];
			for (int g = 0; g < mlt2; ++g)
			{
				VEC d = p1h - sp[g];
				SCAL dist2 = dot(d, d) + d_EPS2;
				SCAL invDist2 = (SCAL)1 / dist2;

				atmp = kernel(atmp, d, invDist2);
			}
			myAtomicAdd(&a1[h].x, atmp.x);
			myAtomicAdd(&a1[h].y, atmp.y);
			myAtomicAdd(&a1[h].z, atmp.z);
		}
		__syncthreads();

		for (int h = tid; h < mlt1; h += bdim)
			sp[h] = p1[h];
		__syncthreads();

		for (int g = tid; g < mlt2; g += bdim)
		{
			VEC atmp{};
			VEC p2g = p2[g];
			for (int h = 0; h < mlt1; ++h)
			{
				VEC d = p2g - sp[h];
				SCAL dist2 = dot(d, d) + d_EPS2;
				SCAL invDist2 = (SCAL)1 / dist2;

				atmp = kernel(atmp, d, invDist2);
			}
			myAtomicAdd(&a2[g].x, atmp.x);
			myAtomicAdd(&a2[g].y, atmp.y);
			myAtomicAdd(&a2[g].z, atmp.z);
		}
		__syncthreads();
	}
}

__global__ void fmm_p2p3_kdtree(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                const int2 *p2p_list, const int *p2p_n, int mlt_max, SCAL d_EPS2)
{
	fmm_p2p3_kdtree_krnl(a, tree, p, p2p_list, mlt_max, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, *p2p_n, gridDim.x * blockDim.x);
}

void fmm_p2p3_kdtree_cpu(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                         const int2 *p2p_list, const int *p2p_n, int mlt_max, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (*p2p_n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_p2p3_kdtree_krnl, a, tree, p, p2p_list, mlt_max, d_EPS2, niter*i, std::min(niter*(i+1), *p2p_n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_p2p3_self_kdtree_krnl(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                                          int mlt_max, SCAL d_EPS2,
                                                          int begi, int endi, int stride)
// particle to particle interaction
{
#ifdef __CUDA_ARCH__
	extern __shared__ VEC smem[]; // mlt_max*blockDim.x*sizeof(VEC)
	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	VEC *sp = smem + mlt_max*threadIdx.x;
#endif
	for (int i = begi; i < endi; i += stride)
	{
		int ind = tree.index[i];
		int mlt = tree.mult[i];
		const VEC *pi = p + ind;
#ifdef __CUDA_ARCH__
		for (int j = 0; j < mlt; ++j)
			sp[j] = pi[j];
		fmm_p2p_interaction(a + ind, sp, sp, mlt, mlt, d_EPS2);
#else
		fmm_p2p_interaction(a + ind, pi, pi, mlt, mlt, d_EPS2);
#endif
	}
}
__global__ void fmm_p2p3_self_kdtree_coalesced(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                               int L, int mlt_max, SCAL d_EPS2)
// particle to particle interaction
{
	extern __shared__ VEC sp[]; // mlt_max*sizeof(VEC)
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int bid = blockIdx.x;
	int gdim = gridDim.x;
	int end = kd_end(L);

	for (int i = kd_beg(L)+bid; i < end; i += gdim)
	{
		int ind = tree.index[i];
		int mlt = tree.mult[i];
		const VEC *pi = p + ind;
		VEC *ai = a + ind;

		for (int g = tid; g < mlt; g += bdim)
			sp[g] = pi[g];
		__syncthreads();

		for (int h = tid; h < mlt; h += bdim)
		{
			VEC atmp{};
			VEC ph = sp[h];
			for (int g = 0; g < mlt; ++g)
			{
				VEC d = ph - sp[g];
				SCAL dist2 = dot(d, d) + d_EPS2;
				SCAL invDist2 = (SCAL)1 / dist2;

				atmp = kernel(atmp, d, invDist2);
			}
			myAtomicAdd(&ai[h].x, atmp.x);
			myAtomicAdd(&ai[h].y, atmp.y);
			myAtomicAdd(&ai[h].z, atmp.z);
		}
		__syncthreads();
	}
}

__global__ void fmm_p2p3_self_kdtree(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p, int L, int mlt_max, SCAL d_EPS2)
{
	int beg = kd_beg(L);
	int end = kd_end(L);
	fmm_p2p3_self_kdtree_krnl(a, tree, p, mlt_max, d_EPS2, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x);
}

void fmm_p2p3_self_kdtree_cpu(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p, int L, int mlt_max, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int beg = kd_beg(L);
	int end = kd_end(L);
	int niter = (end-beg-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_p2p3_self_kdtree_krnl, a, tree, p, mlt_max, d_EPS2, beg+niter*i, std::min(beg+niter*(i+1), end), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __device__ void fmm_pushl3_kdtree_krnl(fmmTree_kd tree, int begi, int endi, int stride, SCAL *tempi) // 0 -> L-1
// push informations about the field from l-th level to (l+1)-th level
{
	int off = tracelessoffset3(tree.p+1);
	SCAL *slin = tempi + tree.p*(tree.p+1)/2;
	SCAL *slout = slin + symmetricoffset3(tree.p+1);
	int inds[2];
	for (int ijk = begi; ijk < endi; ijk += stride)
	{
		inds[0] = kd_lchild(ijk);
		inds[1] = kd_rchild(ijk);

		const SCAL *local = tree.local + ijk*off;
		for (int q = 1; q <= tree.p; ++q)
		{
			int begt = tracelessoffset3(q);
			int begs = symmetricoffset3(q);
			for (int j = 0; j < tracelesselems3(q); ++j)
				slin[begs+j] = local[begt+j];
			traceless_refine3(slin + begs, q);
		}
		VEC d;
		for (int ii = 0; ii < 2; ++ii)
		{
			d = tree.center[inds[ii]] - tree.center[ijk];
			SCAL *local2 = tree.local + inds[ii]*off;
			for (int j = 1; j < off; ++j)
				slout[j] = 0;

			static_l2l_acc3<1, false>(slout, tempi, slin, tree.p, d);

			for (int j = 1; j < off; ++j)
				local2[j] += slout[j];
		}
	}
}

inline __host__ __device__ void fmm_pushl3_kdtree2_krnl(fmmTree_kd tree, int begi, int endi, int stride, SCAL *tempi) // 0 -> L-1
// push informations about the field from l-th level to (l+1)-th level
{
	int off = tracelessoffset3(tree.p+1);
	int inds[2];
	for (int ijk = begi; ijk < endi; ijk += stride)
	{
		inds[0] = kd_lchild(ijk);
		inds[1] = kd_rchild(ijk);

		const SCAL *local = tree.local + ijk*off;
		VEC d;
		SCAL r;
		for (int ii = 0; ii < 2; ++ii)
		{
			d = tree.center[inds[ii]] - tree.center[ijk];
			r = sqrt(dot(d,d));
			d /= r;
			SCAL *local2 = tree.local + inds[ii]*off;

			static_l2l_acc3<1, true>(local2, tempi, local, tree.p, d, r);
		}
	}
}

__global__ void fmm_pushl3_kdtree(fmmTree_kd tree, int l)
{
	extern __shared__ SCAL temp[];
	SCAL *tempi = temp + (tree.p*(tree.p+1)/2 + symmetricoffset3(tree.p+1) + tracelessoffset3(tree.p+1))*threadIdx.x;
	int beg = kd_beg(l), end = kd_end(l);
	fmm_pushl3_kdtree_krnl(tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x, tempi);
}
__global__ void fmm_pushl3_kdtree2(fmmTree_kd tree, int l)
{
	extern __shared__ SCAL temp[];
	SCAL *tempi = temp + (2*tree.p-1)*threadIdx.x;
	int beg = kd_beg(l), end = kd_end(l);
	fmm_pushl3_kdtree2_krnl(tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x, tempi);
}

void fmm_pushl3_kdtree_cpu(fmmTree_kd tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[2*tree.p-1 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int beg = kd_beg(l), end = kd_end(l);
	int niter = (end-beg-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushl3_kdtree2_krnl, tree, beg+niter*i, std::min(beg + niter*(i+1), end), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

inline __device__ void fmm_pushLeaves3_kdtree_krnl(VEC *__restrict__ a, const VEC *__restrict__ p,
                                                   fmmTree_kd tree, int begi, int endi, int stride, SCAL *tempi)
// push informations about the field from leaves to individual particles
{
	SCAL *slin = tempi + tree.p*(tree.p+1)/2+3;
	int off = tracelessoffset3(tree.p+1);
	for (int i = begi; i < endi; i += stride)
	{
		const SCAL *local = tree.local + i*off;
		int mlt = tree.mult[i], ind = tree.index[i];
		VEC *ai = a + ind;
		const VEC *pi = p + ind;
		for (int q = 1; q <= tree.p; ++q)
		{
			int begt = tracelessoffset3(q);
			int begs = symmetricoffset3(q);
			for (int j = 0; j < tracelesselems3(q); ++j)
				slin[begs+j] = local[begt+j];
			traceless_refine3(slin + begs, q);
		}
		for (int j = 0; j < mlt; ++j)
		{
			VEC d = pi[j] - tree.center[i];
			ai[j] += static_l2p_field3<false>(tempi, slin, tree.p, d);
		}
	}
}

inline __host__ __device__ void fmm_pushLeaves3_kdtree2_krnl(VEC *__restrict__ a, const VEC *__restrict__ p,
                                                             fmmTree_kd tree, int begi, int endi, int stride, SCAL *tempi)
// push informations about the field from leaves to individual particles
{
	int off = tracelessoffset3(tree.p+1);
	for (int i = begi; i < endi; i += stride)
	{
		const SCAL *local = tree.local + i*off;
		int mlt = tree.mult[i], ind = tree.index[i];
		VEC *ai = a + ind;
		const VEC *pi = p + ind;
		for (int j = 0; j < mlt; ++j)
		{
			VEC d = pi[j] - tree.center[i];
			SCAL r(sqrt(dot(d,d)));
			if (r != 0)
				d /= r;
			ai[j] += static_l2p_field3(tempi, local, tree.p, d, r);
		}
	}
}

__global__ void fmm_pushLeaves3_kdtree(VEC *a, const VEC *p, fmmTree_kd tree, int L)
{
	extern __shared__ SCAL temp[];
	SCAL *tempi = temp + (tree.p*(tree.p+1)/2+3 + symmetricoffset3(tree.p+1))*threadIdx.x;
	int beg = kd_beg(L);
	int end = kd_end(L);
	fmm_pushLeaves3_kdtree_krnl(a, p, tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x, tempi);
}
__global__ void fmm_pushLeaves3_kdtree2(VEC *a, const VEC *p, fmmTree_kd tree, int L)
{
	extern __shared__ SCAL temp[]; // size must be at least (2*order+2)*blockDim.x
	SCAL *tempi = temp + (2*tree.p+2)*threadIdx.x;
	int beg = kd_beg(L);
	int end = kd_end(L);
	fmm_pushLeaves3_kdtree2_krnl(a, p, tree, beg + blockDim.x * blockIdx.x + threadIdx.x, end, gridDim.x * blockDim.x, tempi);
}

void fmm_pushLeaves3_kdtree_cpu(VEC *a, const VEC *p, fmmTree_kd tree, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[2*tree.p+2 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int beg = kd_beg(L);
	int end = kd_end(L);
	int niter = (end-beg-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushLeaves3_kdtree2_krnl, a, p, tree, beg+niter*i, std::min(beg+niter*(i+1), end), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

template <typename T>
void sort_particle_gpu(VEC *__restrict__ p, int n, T *__restrict__ d_keys, int *__restrict__ d_ind,
	void *& d_tmp_stor, size_t& stor_bytes, int *__restrict__ d_unsort, int *__restrict__ d_offsets = nullptr, int n_segs = 1,
	bool eval = true)
{
	static bool first_time = true;
	static int2 gather_bt, copy_bt, gatherint_bt, copyint_bt;

	cub::DoubleBuffer<T> d_dkeys(d_keys, d_keys + n);
	cub::DoubleBuffer<int> d_values(d_ind, d_ind + n);

	if (first_time)
	{
		first_time = false;
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&gather_bt.x, &gather_bt.y, gather_krnl<VEC>));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&copy_bt.x, &copy_bt.y, copy_krnl<VEC>));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&gatherint_bt.x, &gatherint_bt.y, gather_krnl<int>));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&copyint_bt.x, &copyint_bt.y, copy_krnl<int>));
	}

	if (eval)
	{
		size_t new_stor_bytes = 0;
		if (n_segs == 1)
		{
			gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, new_stor_bytes, d_dkeys, d_values, n));
			if (new_stor_bytes > stor_bytes)
			{
				if (stor_bytes > 0)
					gpuErrchk(cudaFree(d_tmp_stor));
				stor_bytes = new_stor_bytes;
				gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
			}
		}
	}
	if (n_segs == 1)
	{
		gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dkeys, d_values, n));
	}
	else
		bb_segsort(d_keys, d_ind, d_keys+n, d_ind+n, n, d_offsets, d_offsets+1, n_segs);

	int *vals;
	if (n_segs == 1)
		vals = d_values.Current();
	else
		vals = d_ind+n;

	gather_krnl <<< std::min(gather_bt.x, (n-1)/gather_bt.y+1), gather_bt.y >>> ((VEC*)d_tmp_stor, p, vals, n);
	copy_krnl <<< std::min(copy_bt.x, (n-1)/copy_bt.y+1), copy_bt.y >>> (p, (VEC*)d_tmp_stor, n);

	gather_krnl <<< std::min(gatherint_bt.x, (n-1)/gatherint_bt.y+1), gatherint_bt.y >>> ((int*)d_tmp_stor, d_unsort, vals, n);
	copy_krnl <<< std::min(copyint_bt.x, (n-1)/copyint_bt.y+1), copyint_bt.y >>> (d_unsort, (int*)d_tmp_stor, n);
}

template <typename T>
void sort_particle_cpu(VEC *__restrict__ p, char *__restrict__ c_tmp, int n, T *__restrict__ keys,
	int *__restrict__ ind, int *__restrict__ unsort)
{
	if (n > 99999)
		parasort(n, ind, [keys](int i, int j) { return keys[i] < keys[j]; }, CPU_THREADS);
	else
		std::sort(ind, ind + n, [keys](int i, int j) { return keys[i] < keys[j]; });

	gather_cpu((T*)c_tmp, keys, ind, n);
	copy_cpu(keys, (T*)c_tmp, n);

	gather_cpu((VEC*)c_tmp, p, ind, n);
	copy_cpu(p, (VEC*)c_tmp, n);

	gather_cpu((int*)c_tmp, unsort, ind, n);
	copy_cpu(unsort, (int*)c_tmp, n);
}

inline __host__ __device__ int buildTree_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return 2*symmetricoffset3(*::d_fmm_order)*blocksize*sizeof(SCAL);
#else
	return 2*symmetricoffset3(::fmm_order)*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int p2p0_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	int mlt_max = *::d_mlt_max;
#else
	int mlt_max = ::h_mlt_max;
#endif
	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	return 2*mlt_max*blocksize*sizeof(VEC);
}
inline __host__ __device__ int p2p1_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	int mlt_max = *::d_mlt_max;
#else
	int mlt_max = ::h_mlt_max;
#endif
	int wsize = min(bitceil(mlt_max), 32u);
	int nwarps = blocksize/wsize;
	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	return 2*mlt_max*nwarps*sizeof(VEC);
}
inline __host__ __device__ int p2p_self_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	int mlt_max = *::d_mlt_max;
#else
	int mlt_max = ::h_mlt_max;
#endif
	if (mlt_max % 2 == 0)
		++mlt_max; // to reduce bank conflicts
	return mlt_max*blocksize*sizeof(VEC);
}
inline __host__ __device__ int c2c0_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return ((*::d_fmm_order+1)*(*::d_fmm_order+2)/2 + symmetricoffset3(*::d_fmm_order)
		+ tracelessoffset3(*::d_fmm_order+1))*blocksize*sizeof(SCAL);
#else
	return ((::fmm_order+1)*(::fmm_order+2)/2 + symmetricoffset3(::fmm_order) + tracelessoffset3(::fmm_order+1))*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int c2c2_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return (*::d_fmm_order+1)*(*::d_fmm_order+2)/2*blocksize*sizeof(SCAL);
#else
	return (::fmm_order+1)*(::fmm_order+2)/2*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int pushl_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return ((*::d_fmm_order)*(*::d_fmm_order+1)/2 + symmetricoffset3(*::d_fmm_order+1) + tracelessoffset3(*::d_fmm_order+1))*blocksize*sizeof(SCAL);
#else
	return (::fmm_order*(::fmm_order+1)/2 + symmetricoffset3(::fmm_order+1) + tracelessoffset3(::fmm_order+1))*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int pushl2_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return (2*(*::d_fmm_order)-1)*blocksize*sizeof(SCAL);
#else
	return (2*::fmm_order-1)*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int pushLeaves_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return ((*::d_fmm_order)*(*::d_fmm_order+1)/2+3 + symmetricoffset3(*::d_fmm_order+1))*blocksize*sizeof(SCAL);
#else
	return (::fmm_order*(::fmm_order+1)/2+3 + symmetricoffset3(::fmm_order+1))*blocksize*sizeof(SCAL);
#endif
}
inline __host__ __device__ int pushLeaves2_smem(int blocksize)
{
#ifdef __CUDA_ARCH__
	return (2*(*::d_fmm_order)+2)*blocksize*sizeof(SCAL);
#else
	return (2*::fmm_order+2)*blocksize*sizeof(SCAL);
#endif
}

void fmm_cart3_kdtree(VEC *p, VEC *a, int n, const SCAL* param)
{
	static SCAL i_prev = 0;
	static float *d_keys = nullptr;
	static cudaDeviceProp prop;
	static int order = -1, old_size = 0, counter = 0;
	static int n_prev = 0, n_max = 0, L = 0, ntot_max = 0;
	static int *d_ind = nullptr, *d_unsort = nullptr;
	static char *d_tbuf = nullptr;
	static fmmTree_kd tree;
	static VEC *d_minmax = nullptr;
	static void *d_tmp_stor = nullptr;
	static size_t stor_bytes = 0;
	static int2 *d_p2p_list = nullptr, *d_m2l_list = nullptr, *d_stack = nullptr;
	static int *d_p2p_n = nullptr, *d_m2l_n = nullptr;
	static int p2p_max = 0, m2l_max = 0, stack_max = 0, ntot = 0;
	static int2 evalKeys_bt, evalIndices_bt, evalBox_bt, multLeaves_bt,
		centerLeaves_bt, multipoleLeaves_bt, buildTree_bt, buildTree2_bt,
		rescale_bt, p2p0_bt, p2p1_bt, p2p2_bt, p2p_self_bt, p2p_self2_bt,
		c2c0_bt, c2c1_bt, c2c2_bt, pushl_bt, pushl2_bt,
		pushLeaves_bt, pushLeaves2_bt, gather_bt, copy_bt;

	assert(n > BLOCK_SIZE);

	if (n != n_prev || ::fmm_order != order || ::dens_inhom != i_prev || (::tree_L != 0 && clamp(::tree_L, 2, 30) != L))
	{
		counter = 0;
		order = ::fmm_order;
		i_prev = ::dens_inhom;
		SCAL s = order*order;
		if (::tree_L == 0)
			L = (int)std::round(std::log2(::dens_inhom*(SCAL)n/s)); // maximum level, L+1 is the total number of levels
		else
			L = ::tree_L;
		L = clamp(L, 2, 30);

		while (kd_n(L) > n)
			--L;
		ntot = kd_ntot(L);
		int new_size = (3*sizeof(VEC) + sizeof(SCAL)*symmetricoffset3(order) + sizeof(SCAL)*tracelessoffset3(order+1)
					  + sizeof(int)*3)*ntot;
		::h_mlt_max = (n-1) / kd_n(L) + 1;

		if (new_size > old_size)
		{
			if (old_size > 0)
			{
				gpuErrchk(cudaFree(d_tbuf));
			}
			else
			{
				gpuErrchk(cudaGetDeviceProperties(&prop, 0));

				gpuErrchk(cudaMalloc((void**)&d_minmax, sizeof(VEC)*2));
				gpuErrchk(cudaMalloc((void**)&d_p2p_n, sizeof(int)*2));
				gpuErrchk(cudaMalloc((void**)&d_fmm_order, sizeof(int)));
				gpuErrchk(cudaMalloc((void**)&d_mlt_max, sizeof(int)));

				d_m2l_n = d_p2p_n + 1;

				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&evalKeys_bt.x, &evalKeys_bt.y, evalKeys_kdtree));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&evalIndices_bt.x, &evalIndices_bt.y, evalIndices));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&evalBox_bt.x, &evalBox_bt.y, evalBox<false>));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&multLeaves_bt.x, &multLeaves_bt.y, multLeaves));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&centerLeaves_bt.x, &centerLeaves_bt.y, centerLeaves));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&multipoleLeaves_bt.x, &multipoleLeaves_bt.y, fmm_multipoleLeaves3_kdtree));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&buildTree2_bt.x, &buildTree2_bt.y, fmm_buildTree3_kdtree2));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&rescale_bt.x, &rescale_bt.y, rescale));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&gather_bt.x, &gather_bt.y, gather_inverse_krnl<VEC>));
				gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&copy_bt.x, &copy_bt.y, copy_krnl<VEC>));
			}
			gpuErrchk(cudaMalloc((void**)&d_tbuf, new_size));
			old_size = new_size;
		}
		tree.center = (VEC*)d_tbuf;
		tree.lbound = tree.center + ntot;
		tree.rbound = tree.lbound + ntot;
		tree.mpole = (SCAL*)(tree.rbound + ntot);
		tree.local = tree.mpole + ntot*symmetricoffset3(order);
		tree.mult = (int*)(tree.local + ntot*tracelessoffset3(order+1));
		tree.index = tree.mult + ntot;
		tree.splitdim = tree.index + ntot;
		tree.p = order;
		if (n > n_max)
		{
			if (n_max > 0)
			{
				gpuErrchk(cudaFree(d_keys));
				gpuErrchk(cudaFree(d_ind));
				gpuErrchk(cudaFree(d_unsort));
				gpuErrchk(cudaFree(d_tmp_stor));
			}
			gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(float)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_ind, sizeof(int)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_unsort, sizeof(int)*n));
			stor_bytes = sizeof(VEC)*n;
			gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
		}
		if (ntot > ntot_max)
		{
			if (ntot_max > 0)
			{
				gpuErrchk(cudaFree(d_p2p_list));
				gpuErrchk(cudaFree(d_m2l_list));
				gpuErrchk(cudaFree(d_stack));
			}
			p2p_max = std::min(ntot*1000, int(prop.totalGlobalMem/(4*sizeof(int2))));
			m2l_max = std::min(ntot*1000, int(prop.totalGlobalMem/(4*sizeof(int2))));
			stack_max = std::max(ntot*10, 100000);
			gpuErrchk(cudaMalloc((void**)&d_p2p_list, sizeof(int2)*p2p_max));
			gpuErrchk(cudaMalloc((void**)&d_m2l_list, sizeof(int2)*m2l_max));
			gpuErrchk(cudaMalloc((void**)&d_stack, sizeof(int2)*stack_max));
		}

		gpuErrchk(cudaMemcpy(::d_fmm_order, &::fmm_order, sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(::d_mlt_max, &::h_mlt_max, sizeof(int), cudaMemcpyHostToDevice));

		int smoff = ::h_mlt_max;
		if (smoff % 2 == 0)
			++smoff;
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&p2p2_bt.x, &p2p2_bt.y, fmm_p2p3_kdtree_coalesced2, smoff*sizeof(VEC)));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&p2p_self2_bt.x, &p2p_self2_bt.y, fmm_p2p3_self_kdtree_coalesced, smoff*sizeof(VEC)));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&c2c1_bt.x, &c2c1_bt.y, fmm_c2c3_kdtree_coalesced,
			((order+1)*(order+2)/2 + symmetricoffset3(order))*sizeof(SCAL)));

		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&buildTree_bt.x, &buildTree_bt.y, fmm_buildTree3_kdtree, buildTree_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&p2p0_bt.x, &p2p0_bt.y, fmm_p2p3_kdtree, p2p0_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&p2p1_bt.x, &p2p1_bt.y, fmm_p2p3_kdtree_coalesced, p2p1_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&p2p_self_bt.x, &p2p_self_bt.y, fmm_p2p3_self_kdtree, p2p_self_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&c2c0_bt.x, &c2c0_bt.y, fmm_c2c3_kdtree, c2c0_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&c2c2_bt.x, &c2c2_bt.y, fmm_c2c3_kdtree2, c2c2_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&pushl_bt.x, &pushl_bt.y, fmm_pushl3_kdtree, pushl_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&pushl2_bt.x, &pushl2_bt.y, fmm_pushl3_kdtree2, pushl2_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&pushLeaves_bt.x, &pushLeaves_bt.y, fmm_pushLeaves3_kdtree, pushLeaves_smem));
		gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&pushLeaves2_bt.x, &pushLeaves2_bt.y, fmm_pushLeaves3_kdtree2, pushLeaves2_smem));
	}

	int beg = kd_beg(L), m = kd_n(L);
	int smemSize = 49152;
	SCAL radius = ::tree_radius;

	if (::b_unsort || counter % ::tree_steps == 0)
	{
		minmaxReduce2(d_minmax, p, n, d_tmp_stor, stor_bytes);

		evalRootBox <<< 1, 1 >>> (tree, d_minmax);
		evalKeys_kdtree <<< std::min(evalKeys_bt.x, (n-1)/evalKeys_bt.y+1), evalKeys_bt.y >>> (d_keys, tree.splitdim, p, n, 0);
		evalIndices <<< std::min(evalIndices_bt.x, (n-1)/evalIndices_bt.y+1), evalIndices_bt.y >>> (d_ind, n);
		evalIndices <<< std::min(evalIndices_bt.x, (n-1)/evalIndices_bt.y+1), evalIndices_bt.y >>> (d_unsort, n);

		sort_particle_gpu(p, n, d_keys, d_ind, d_tmp_stor, stor_bytes, d_unsort, nullptr, 1, true);

		for (int l = 1; l <= L-1; ++l)
		{
			evalBox <<< std::min(evalBox_bt.x, (kd_n(l)-1)/evalBox_bt.y+1), evalBox_bt.y >>> (tree, p, n, l);
			evalKeys_kdtree <<< std::min(evalKeys_bt.x, (n-1)/evalKeys_bt.y+1), evalKeys_bt.y >>> (d_keys, tree.splitdim + kd_beg(l), p, n, l);
			evalIndices <<< std::min(evalIndices_bt.x, (n-1)/evalIndices_bt.y+1), evalIndices_bt.y >>> (d_ind, n);

			sort_particle_gpu(p, n, d_keys, d_ind, d_tmp_stor, stor_bytes, d_unsort, tree.index + kd_beg(l), kd_n(l), true);
		}

		evalBox<true> <<< std::min(evalBox_bt.x, (m-1)/evalBox_bt.y+1), evalBox_bt.y >>> (tree, p, n, L);

		multLeaves <<< std::min(multLeaves_bt.x, (m-1)/multLeaves_bt.y+1), multLeaves_bt.y >>> (tree.mult + beg, tree.index + beg, m, n);
	}

	centerLeaves <<< std::min(centerLeaves_bt.x, (m-1)/centerLeaves_bt.y+1), centerLeaves_bt.y >>>
		(tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	// zeroing multipole and local expansions
	gpuErrchk(cudaMemset(tree.mpole, 0, ntot*(symmetricoffset3(order)+tracelessoffset3(order+1))*sizeof(SCAL)));

	fmm_multipoleLeaves3_kdtree <<< std::min(multipoleLeaves_bt.x, (ntot-1)/multipoleLeaves_bt.y+1), multipoleLeaves_bt.y >>> (tree, p, L);

	if (symmetricoffset3(order) >= 64)
		for (int l = L-1; l >= 0; --l)
		{
			int maxBlocks_l = (kd_n(l)-1)/buildTree2_bt.y+1;
			fmm_buildTree3_kdtree2 <<< std::min(buildTree2_bt.x, maxBlocks_l), buildTree2_bt.y >>> (tree, l);
		}
	else
	{
		smemSize = buildTree_smem(buildTree_bt.y);
		for (int l = L-1; l >= 0; --l)
		{
			int maxBlocks_l = (kd_n(l)-1)/buildTree_bt.y+1;
			fmm_buildTree3_kdtree <<< std::min(buildTree_bt.x, maxBlocks_l), buildTree_bt.y, smemSize >>> (tree, l);
		}
	}

	fmm_dualTraversal<true> <<< (L >= 3) ? 18 : 7, 1024 >>>
		(tree, d_p2p_list, d_m2l_list, d_stack, d_p2p_n, d_m2l_n, p2p_max, m2l_max, stack_max, radius, L);

	gpuErrchk(cudaMemset(a, 0, n*sizeof(VEC)));

	if (coll)
	{
		int bsize = clamp(::h_mlt_max/32*32, 32, 1024);

		if (::h_mlt_max >= 4)
		{
			smemSize = p2p1_smem(p2p1_bt.y);
			fmm_p2p3_kdtree_coalesced <<< p2p1_bt.x, p2p1_bt.y, smemSize >>> (a, tree, p, d_p2p_list, d_p2p_n, ::h_mlt_max, EPS2);
		}
		else
		{
			smemSize = p2p0_smem(p2p0_bt.y);
			fmm_p2p3_kdtree <<< p2p0_bt.x, p2p0_bt.y, smemSize >>> (a, tree, p, d_p2p_list, d_p2p_n, ::h_mlt_max, EPS2);
		}

		if (::h_mlt_max < 32)
		{
			smemSize = p2p_self_smem(p2p_self_bt.y);
			fmm_p2p3_self_kdtree <<< std::min(p2p_self_bt.x, (m-1)/p2p_self_bt.y+1), p2p_self_bt.y, smemSize >>>
				(a, tree, p, L, ::h_mlt_max, EPS2);
		}
		else
		{
			smemSize = ::h_mlt_max*sizeof(VEC);
			fmm_p2p3_self_kdtree_coalesced <<< std::min(p2p_self2_bt.x, (m-1)/p2p_self2_bt.y+1), p2p_self2_bt.y, smemSize >>>
				(a, tree, p, L, ::h_mlt_max, EPS2);
		}
	}

	if (symmetricoffset3(order) >= 64)
	{
		smemSize = c2c2_smem(c2c2_bt.y);
		fmm_c2c3_kdtree2 <<< c2c2_bt.x, c2c2_bt.y, smemSize >>> (tree, d_m2l_list, d_m2l_n, EPS2);
	}
	else
	{
		smemSize = c2c0_smem(c2c0_bt.y);
		fmm_c2c3_kdtree <<< c2c0_bt.x, c2c0_bt.y, smemSize >>> (tree, d_m2l_list, d_m2l_n, EPS2);
	}

	if (symmetricoffset3(order) >= 64)
	{
		smemSize = pushl2_smem(pushl2_bt.y);
		for (int l = 1; l <= L-1; ++l)
		{
			int maxBlocks_l = (kd_n(l)-1)/pushl2_bt.y+1;
			fmm_pushl3_kdtree2 <<< std::min(pushl2_bt.x, maxBlocks_l), pushl2_bt.y, smemSize >>> (tree, l);
		}
	}
	else
	{
		smemSize = pushl_smem(pushl_bt.y);
		for (int l = 1; l <= L-1; ++l)
		{
			int maxBlocks_l = (kd_n(l)-1)/pushl_bt.y+1;
			fmm_pushl3_kdtree <<< std::min(pushl_bt.x, maxBlocks_l), pushl_bt.y, smemSize >>> (tree, l);
		}
	}

	if (symmetricoffset3(order) >= 64)
	{
		smemSize = pushLeaves2_smem(pushLeaves2_bt.y);
		fmm_pushLeaves3_kdtree2 <<< std::min(pushLeaves2_bt.x, (m-1)/pushLeaves2_bt.y+1), pushLeaves2_bt.y, smemSize >>> (a, p, tree, L);
	}
	else
	{
		smemSize = pushLeaves_smem(pushLeaves_bt.y);
		fmm_pushLeaves3_kdtree <<< std::min(pushLeaves_bt.x, (m-1)/pushLeaves_bt.y+1), pushLeaves_bt.y, smemSize >>> (a, p, tree, L);
	}

	if (param != nullptr)
		rescale <<< std::min(rescale_bt.x, (n-1)/rescale_bt.y+1), rescale_bt.y >>> (a, n, param);

	if (::b_unsort)
	{
		// unsort particle positions and accelerations to their initial places
		gather_inverse_krnl <<< std::min(gather_bt.x, (n-1)/gather_bt.y+1), gather_bt.y >>> ((VEC*)d_tmp_stor, p, d_unsort, n);
		copy_krnl <<< std::min(copy_bt.x, (n-1)/copy_bt.y+1), copy_bt.y >>> (p, (VEC*)d_tmp_stor, n);

		gather_inverse_krnl <<< std::min(gather_bt.x, (n-1)/gather_bt.y+1), gather_bt.y >>> ((VEC*)d_tmp_stor, a, d_unsort, n);
		copy_krnl <<< std::min(copy_bt.x, (n-1)/copy_bt.y+1), copy_bt.y >>> (a, (VEC*)d_tmp_stor, n);
	}
	else if (counter % ::tree_steps == 0)
	{
		// sort velocities in order to match positions and accelerations indices
		gather_krnl <<< std::min(gather_bt.x, (n-1)/gather_bt.y+1), gather_bt.y >>> ((VEC*)d_tmp_stor, p+n, d_unsort, n);
		copy_krnl <<< std::min(copy_bt.x, (n-1)/copy_bt.y+1), copy_bt.y >>> (p+n, (VEC*)d_tmp_stor, n);
	}

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (n > n_max)
		n_max = n;
	if (ntot > ntot_max)
		ntot_max = ntot;
	n_prev = n;
	++counter;
}

void fmm_cart3_kdtree_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	int radius = ::tree_radius;

	static unsigned long long *keys = nullptr;
	static int order = -1, old_size = 0;
	static int n_prev = 0, n_max = 0, L = 0;
	static int *ind = nullptr, *unsort = nullptr;
	static char *tbuf = nullptr, *c_tmp = nullptr;
	static fmmTree_kd tree;
	static std::vector<int2> p2p_list, m2l_list, stack;
	std::vector<VEC> min_(std::max(CPU_THREADS, 2)), max_(CPU_THREADS);
	assert(n > 0);

	if (n != n_prev || ::fmm_order != order)
	{
		order = ::fmm_order;
		SCAL s = order*order;
		L = (int)std::round(std::log2(::dens_inhom*(SCAL)n/s)); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);
		L = std::min(L, 30);

		while (kd_n(L) > n)
			--L;
		int ntot = kd_ntot(L);
		int new_size = (3*sizeof(VEC) + sizeof(SCAL)*symmetricoffset3(order) + sizeof(SCAL)*tracelessoffset3(order+1)
					  + sizeof(int)*3)*ntot;

		if (new_size > old_size)
		{
			if (old_size > 0)
				delete[] tbuf;
			tbuf = new char[new_size];
			old_size = new_size;
		}
		tree.center = (VEC*)tbuf;
		tree.lbound = tree.center + ntot;
		tree.rbound = tree.lbound + ntot;
		tree.mpole = (SCAL*)(tree.rbound + ntot);
		tree.local = tree.mpole + ntot*symmetricoffset3(order);
		tree.mult = (int*)(tree.local + ntot*tracelessoffset3(order+1));
		tree.index = tree.mult + ntot;
		tree.splitdim = tree.index + ntot;
		tree.p = order;
		if (n > n_max)
		{
			if (n_max > 0)
			{
				delete[] keys;
				delete[] ind;
				delete[] unsort;
				delete[] c_tmp;
			}
			keys = new unsigned long long[n];
			ind = new int[n];
			unsort = new int[n];
			c_tmp = new char[n*sizeof(VEC)];
		}
	}

	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=, &min_, &max_]{
			VEC mn = p[niter*i];
			VEC mx = p[niter*i];
			for (int j = niter*i+1; j < std::min(n, niter*(i+1)); ++j)
			{
				mn = fmin(mn, p[j]);
				mx = fmax(mx, p[j]);
			}
			min_[i] = mn;
			max_[i] = mx;
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();

	for (int i = 1; i < CPU_THREADS; ++i)
	{
		min_[0] = fmin(min_[0], min_[i]);
		max_[0] = fmax(max_[0], max_[i]);
	}
	min_[1] = max_[0];

	evalRootBox_cpu(tree, min_.data());
	evalKeys_kdtree_cpu(keys, tree.splitdim, p, n, 0);
	evalIndices_cpu(ind, n);
	evalIndices_cpu(unsort, n);

	sort_particle_cpu(p, c_tmp, n, keys, ind, unsort);

	for (int l = 1; l <= L-1; ++l)
	{
		evalBox_cpu(tree, p, n, l);
		evalKeys_kdtree_cpu(keys, tree.splitdim + kd_beg(l), p, n, l);
		evalIndices_cpu(ind, n);

		sort_particle_cpu(p, c_tmp, n, keys, ind, unsort);
	}

	evalBox_cpu<true>(tree, p, n, L);

	fmm_init3_kdtree_cpu(tree, L);

	int beg = kd_beg(L), m = kd_n(L);

	multLeaves_cpu(tree.mult + beg, tree.index + beg, m, n);

	centerLeaves_cpu(tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	fmm_multipoleLeaves3_kdtree_cpu(tree, p, L);

	for (int l = L-1; l >= 0; --l)
		fmm_buildTree3_kdtree_cpu(tree, l);

	fmm_dualTraversal_cpu(tree, p2p_list, m2l_list, stack, radius, L);

	rescale_cpu(a, n, param+1);

	int list_n = p2p_list.size();
	if (coll)
	{
		int max_mlt = (n-1) / m + 1;
		fmm_p2p3_kdtree_cpu(a, tree, p, p2p_list.data(), &list_n, max_mlt, EPS2);
		fmm_p2p3_self_kdtree_cpu(a, tree, p, L, max_mlt, EPS2);
	}

	list_n = m2l_list.size();
	fmm_c2c3_kdtree_cpu(tree, m2l_list.data(), &list_n, EPS2);

	for (int l = 1; l <= L-1; ++l)
		fmm_pushl3_kdtree_cpu(tree, l);

	fmm_pushLeaves3_kdtree_cpu(a, p, tree, L);
	if (param != nullptr)
		rescale_cpu(a, n, param);

	if (::b_unsort)
	{
		// unsort particle positions and accelerations to their initial places
		gather_inverse_cpu((VEC*)c_tmp, p, unsort, n);
		copy_cpu(p, (VEC*)c_tmp, n);

		gather_inverse_cpu((VEC*)c_tmp, a, unsort, n);
		copy_cpu(a, (VEC*)c_tmp, n);
	}
	else
	{
		// sort velocities in order to match positions and accelerations indices
		gather_cpu((VEC*)c_tmp, p+n, unsort, n);
		copy_cpu(p+n, (VEC*)c_tmp, n);
	}

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

#endif // !FMM_CART3_KDTREE_CUDA_H
