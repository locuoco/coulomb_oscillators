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
#include "parasort.h"

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

__host__ __device__ void evalRootBox_krnl(fmmTree_kd& tree, const VEC *d_minmax)
{
	VEC d = d_minmax[1] - d_minmax[0];
	int arg = (d.x > d.y) ? ((d.x > d.z) ? 0 : 2) : ((d.y > d.z) ? 1 : 2);
	tree.lbound[0] = d_minmax[0];
	tree.rbound[0] = d_minmax[1];
	tree.splitdim[0] = arg;
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

inline __host__ __device__ void evalBox_krnl(fmmTree_kd tree, const VEC *p, int n, int l,
                                             int begi, int endi, int stride)
{
	int m = kd_n(l);
	int beg = kd_beg(l);
	for (int i = begi; i < endi; i += stride)
	{
		int start = (long long)n * i / m;
		int end = (long long)n * (i+1) / m;
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
	}
}

__global__ void evalBox(fmmTree_kd tree, const VEC *p, int n, int l)
{
	int m = kd_n(l);
	evalBox_krnl(tree, p, n, l, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void evalBox_cpu(fmmTree_kd tree, const VEC *p, int n, int l)
{
	int m = kd_n(l);
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalBox_krnl, tree, p, n, l, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void evalKeys_kdtree_krnl(unsigned long long *keys, const int *splitdim, const VEC *p, int n, int l,
                                                     int begi, int endi, int stride)
// calculate keys for all particles at level l
{
	unsigned long long m = kd_n(l);
	for (int i = begi; i < endi; i += stride)
	{
		unsigned long long j = m * i / n;
		union
		{
			unsigned u;
			float f;
		} p_;
		p_.f = (float)get_axis(p[i], splitdim[j]);
		if (p_.f > 0)
			p_.f = -p_.f;
		else
			p_.u = ~p_.u; // sorry for undefined behavior
		keys[i] = (j << 32) | (unsigned long long)p_.u;
	}
}

__global__ void evalKeys_kdtree(unsigned long long *keys, const int *splitdim, const VEC *p, int n, int l)
{
	evalKeys_kdtree_krnl(keys, splitdim, p, n, l, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void evalKeys_kdtree_cpu(unsigned long long *keys, const int *splitdim, const VEC *p, int n, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalKeys_kdtree_krnl, keys, splitdim, p, n, l, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void evalKeysLeaves_kdtree_krnl(int *keys, int n, int L,
                                                           int begi, int endi, int stride)
{
	long long m = kd_n(L);
	for (int i = begi; i < endi; i += stride)
		keys[i] = m * i / n;
}

__global__ void evalKeysLeaves_kdtree(int *keys, int n, int L)
{
	evalKeysLeaves_kdtree_krnl(keys, n, L, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void evalKeysLeaves_kdtree_cpu(int *keys, int n, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(evalKeysLeaves_kdtree_krnl, keys, n, L, niter*i, std::min(niter*(i+1), n), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_init3_kdtree_krnl(fmmTree_kd tree, int begi, int endi, int stride)
{
	int off = tracelessoffset3(tree.p+1);
	for (int i = begi; i < endi; i += stride)
	{
		tree.center[i] = VEC{};
		SCAL *multipole = tree.mpole + off*i;
		for (int j = 0; j < off; ++j)
			multipole[j] = (SCAL)0;
		SCAL *loc = tree.local + off*i;
		for (int j = 0; j < off; ++j)
			loc[j] = (SCAL)0;
	}
}

__global__ void fmm_init3_kdtree(fmmTree_kd tree, int L)
{
	int m = kd_ntot(L);
	fmm_init3_kdtree_krnl(tree, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
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

inline __host__ __device__ void fmm_multipoleLeaves3_kdtree_krnl(fmmTree_kd tree, const VEC *p, int L,
                                                                 int begi, int endi, int stride)
// calculate multipoles for each cell
// assumes all particles have the same charge/mass
{
	int off = tracelessoffset3(tree.p+1);
	int beg = kd_beg(L);
	const VEC *center = tree.center + beg;
	const int *index = tree.index + beg, *mult = tree.mult + beg;
	SCAL *mpole = tree.mpole + beg*off;
	for (int i = begi; i < endi; i += stride)
	{
		SCAL *multipole = mpole + off*i;
		const VEC *pi = p + index[i];
		multipole[0] = (SCAL)mult[i];
		if (tree.p >= 2)
			for (int j = 0; j < mult[i]; ++j)
			{
				VEC d = pi[j] - center[i];
				SCAL r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;
				for (int q = 2; q <= tree.p; ++q)
					p2m_traceless_acc3(multipole + tracelessoffset3(q), q, d, r);
			}
	}
}

__global__ void fmm_multipoleLeaves3_kdtree(fmmTree_kd tree, const VEC *p, int L)
{
	int m = kd_n(L);
	fmm_multipoleLeaves3_kdtree_krnl(tree, p, L, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void fmm_multipoleLeaves3_kdtree_cpu(fmmTree_kd tree, const VEC *p, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = kd_n(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_multipoleLeaves3_kdtree_krnl, tree, p, L, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_buildTree3_kdtree_krnl(fmmTree_kd tree, int l, int begi, int endi, int stride, SCAL *tempi) // L-1 -> 0
// build the l-th of cells after the deeper one (l+1)-th
// "tree" contains only pointers to the actual tree in memory
{
	int off = tracelessoffset3(tree.p+1);
	int beg = kd_beg(l);
	int inds[2];
	for (int ijk0 = begi; ijk0 < endi; ijk0 += stride)
	{
		int ijk = beg+ijk0;

		inds[0] = kd_lchild(ijk);
		inds[1] = kd_rchild(ijk);

		int mlt = 0;
		for (int ii = 0; ii < 2; ++ii)
			mlt += tree.mult[inds[ii]];

		SCAL mpole0 = (SCAL)mlt;

		if (mlt > 0)
		{
			VEC coord{};
			for (int ii = 0; ii < 2; ++ii)
				coord += (SCAL)tree.mult[inds[ii]] * tree.center[inds[ii]];
			coord /= mpole0;

			SCAL *multipole = tree.mpole + ijk*off;
			const SCAL *multipole2;
			VEC d;
			SCAL r;
			if (tree.p >= 2)
				for (int ii = 0; ii < 2; ++ii)
				{
					d = coord - tree.center[inds[ii]];
					r = sqrt(dot(d,d));
					if (r != 0)
						d /= r;
					multipole2 = tree.mpole + inds[ii]*off;
					for (int q = 2; q <= tree.p; ++q)
						m2m_traceless_acc3(multipole + tracelessoffset3(q), tempi, multipole2, q, d, r);
				}
			multipole[0] = mpole0;

			tree.center[ijk] = coord;
		}
		tree.mult[ijk] = mlt;
	}
}

__global__ void fmm_buildTree3_kdtree(fmmTree_kd tree, int l)
{
	extern __shared__ SCAL temp[]; // size must be at least (2*order+1)*blockDim.x
	SCAL *tempi = temp + (2*tree.p+1)*threadIdx.x;
	int m = kd_n(l);
	fmm_buildTree3_kdtree_krnl(tree, l, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x, tempi);
}

void fmm_buildTree3_kdtree_cpu(fmmTree_kd tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[2*tree.p+1 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int m = kd_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_buildTree3_kdtree_krnl, tree, l, niter*i, std::min(niter*(i+1), m), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

inline __host__ __device__ SCAL kd_size(const VEC& l, const VEC& r)
{
	VEC diff = r - l;
	return dot(diff, diff);
}

inline __host__ __device__ bool kd_admissible(const fmmTree_kd& tree, int n1, int n2, int par)
{
	VEC d = tree.center[n2] - tree.center[n1];
	SCAL dist2 = dot(d, d);
	SCAL sz1 = kd_size(tree.lbound[n1], tree.rbound[n1]);
	SCAL sz2 = kd_size(tree.lbound[n2], tree.rbound[n2]);
	return par*par*max(sz1, sz2) < dist2;
}

__global__ void fmm_dualTraversal(fmmTree_kd tree, int2 *p2p_list, int2 *m2l_list, int2 *stack, int *p2p_n, int *m2l_n,
                                  int p2p_max, int m2l_max, int r, int L)
// call with CUDA gridsize = 1
{
	__shared__ int top;

	int2 np{0, 0};
	int ntot = kd_ntot(L);

	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		stack[0] = np;
		top = 1;
		*p2p_n = 0;
		*m2l_n = 0;
	}

	__syncthreads();

	while (top > 0 && blockIdx.x == 0)
	{
		int stack_pos = top - threadIdx.x - 1;

		if (stack_pos >= 0)
			np = stack[stack_pos];

		__syncthreads();
		if (threadIdx.x == 0)
			top = max(top - (int)blockDim.x, 0);
		__syncthreads();

		if (stack_pos >= 0)
		{
			if (kd_lchild(np.x) >= ntot & kd_lchild(np.y) >= ntot)
			{
				if (np.x != np.y)
				{
					int pos = atomicAdd(p2p_n, 1);
					if (pos < p2p_max)
						p2p_list[pos] = np;
				}
			}
			else if (np.x == np.y)
			{
				int pos = atomicAdd(&top, 3);
				stack[pos] = {kd_lchild(np.x), kd_lchild(np.x)};
				stack[pos+1] = {kd_lchild(np.x), kd_rchild(np.x)};
				stack[pos+2] = {kd_rchild(np.x), kd_rchild(np.x)};
			}
			else if (kd_admissible(tree, np.x, np.y, r))
			{
				int pos = atomicAdd(m2l_n, 1);
				if (pos < m2l_max)
					m2l_list[pos] = np;
			}
			else
			{
				bool cond = kd_lchild(np.x) >= ntot | (kd_lchild(np.y) < ntot
					& kd_size(tree.lbound[np.x], tree.rbound[np.x]) <= kd_size(tree.lbound[np.y], tree.rbound[np.y]));
				int pos = atomicAdd(&top, 2);
				stack[pos] = cond ? int2{np.x, kd_lchild(np.y)} : int2{kd_lchild(np.x), np.y};
				stack[pos+1] = cond ? int2{np.x, kd_rchild(np.y)} : int2{kd_rchild(np.x), np.y};
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		if (*p2p_n > p2p_max)
		{
			*p2p_n = p2p_max;
			printf("\nexceeded p2p allocated memory\n");
		}
		if (*m2l_n > m2l_max)
		{
			*m2l_n = m2l_max;
			printf("\nexceeded m2l allocated memory\n");
		}
		//printf("i: %d\n", i);
		//printf("p2p_n: %d\n", *p2p_n);
		//printf("m2l_n: %d\n", *m2l_n);
	}
}

void fmm_dualTraversal_cpu(const fmmTree_kd& tree, std::vector<int2>& p2p_list, std::vector<int2>& m2l_list, std::vector<int2>& stack,
                           int r, int L)
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
	int off = tracelessoffset3(tree.p+1);
	for (int i = begi; i < endi; i += stride)
	{
		int n1 = m2l_list[i].x;
		int n2 = m2l_list[i].y;

		VEC d = tree.center[n1] - tree.center[n2];
		SCAL r2 = dot(d, d) + d_EPS2;

		static_m2l_acc3<1, true>(tree.local + n1*off, tempi, tree.mpole + n2*off, tree.p, d, r2);
		static_m2l_acc3<1, true>(tree.local + n2*off, tempi, tree.mpole + n1*off, tree.p, -d, r2);
	}
}

__global__ void fmm_c2c3_kdtree(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
{
	extern __shared__ SCAL temp[]; // size must be at least (4*p+1)*blockDim.x
	SCAL *tempi = temp + (4*tree.p+1)*threadIdx.x;
	fmm_c2c3_kdtree_krnl(tree, m2l_list, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, *m2l_n, gridDim.x * blockDim.x, tempi);
}

void fmm_c2c3_kdtree_cpu(fmmTree_kd tree, const int2 *m2l_list, const int *m2l_n, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[(4*tree.p+1) + CACHE_LINE_SIZE/sizeof(SCAL)];
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
	SCAL *a1h;

	for (int h = 0; h < mlt1; ++h)
	{
		VEC atmp{};
		for (int g = 0; g < mlt2; ++g)
		{
			VEC d = p1[h] - p2[g];
			SCAL dist2 = dot(d, d) + d_EPS2;
			SCAL invDist2 = (SCAL)1 / dist2;

			atmp = kernel(atmp, d, invDist2);
		}
		a1h = (SCAL*)(a1 + h);
#ifdef __CUDA_ARCH__
		myAtomicAdd(a1h + 0, atmp.x);
		myAtomicAdd(a1h + 1, atmp.y);
		myAtomicAdd(a1h + 2, atmp.z);
#else
		std::atomic_ref<SCAL> atomic0(a1h[0]);
		std::atomic_ref<SCAL> atomic1(a1h[1]);
		std::atomic_ref<SCAL> atomic2(a1h[2]);
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
	VEC *s_at = (VEC*)alloca(mlt_max*sizeof(VEC));
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
		SCAL *a1h;
		SCAL *a2g;

		for (int g = 0; g < mlt2; ++g)
			s_at[g] = VEC{};

		for (int h = 0; h < mlt1; ++h)
		{
			VEC atmp{};
			for (int g = 0; g < mlt2; ++g)
			{
				VEC d = p1[h] - p2[g];
				SCAL k = dot(d, d) + d_EPS2;
				k = (SCAL)1 / k;
				k *= sqrt(k);
				d *= k;

				atmp += d;
				s_at[g] -= d;
			}
			a1h = (SCAL*)(a1 + h);

			myAtomicAdd(a1h + 0, atmp.x);
			myAtomicAdd(a1h + 1, atmp.y);
			myAtomicAdd(a1h + 2, atmp.z);
		}
		for (int g = 0; g < mlt2; ++g)
		{
			a2g = (SCAL*)(a2 + g);

			myAtomicAdd(a2g + 0, s_at[g].x);
			myAtomicAdd(a2g + 1, s_at[g].y);
			myAtomicAdd(a2g + 2, s_at[g].z);
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

inline __device__ void fmm_p2p3_kdtree_coalesced_krnl(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                                      const int2 *p2p_list, const int *p2p_n, int mlt_max, SCAL d_EPS2)
// particle to particle interaction
{
	extern __shared__ VEC smem[];
	VEC *sp2 = smem;
	VEC *sa2 = smem + mlt_max;

	for (int i = blockIdx.x; i < *p2p_n; i += gridDim.x)
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

		for (int g = threadIdx.x; g < mlt2; g += blockDim.x)
			sp2[g] = p2[g];

		for (int g = threadIdx.x; g < mlt2; g += blockDim.x)
			sa2[g] = VEC{};

		__syncthreads();

		for (int h = threadIdx.x; h < mlt1; h += blockDim.x)
		{
			VEC atmp{};
			for (int g = 0; g < mlt2; ++g)
			{
				int gg = (g+threadIdx.x) % mlt2;
				VEC d = p1[h] - sp2[gg];
				SCAL k = dot(d, d) + d_EPS2;
				k = (SCAL)1 / k;
				k *= sqrt(k);
				d *= k;

				atmp += d;
				sa2[gg] -= d;
			}
			myAtomicAdd(&a1[h].x, atmp.x);
			myAtomicAdd(&a1[h].y, atmp.y);
			myAtomicAdd(&a1[h].z, atmp.z);
		}
		__syncthreads();

		for (int g = threadIdx.x; g < mlt2; g += blockDim.x)
		{
			myAtomicAdd(&a2[g].x, sa2[g].x);
			myAtomicAdd(&a2[g].y, sa2[g].y);
			myAtomicAdd(&a2[g].z, sa2[g].z);
		}
		__syncthreads();
	}
}

__global__ void fmm_p2p3_kdtree(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p,
                                const int2 *p2p_list, const int *p2p_n, int mlt_max, SCAL d_EPS2)
{
	if (mlt_max < blockDim.x)
		fmm_p2p3_kdtree_krnl(a, tree, p, p2p_list, mlt_max, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, *p2p_n, gridDim.x * blockDim.x);
	else
		fmm_p2p3_kdtree_coalesced_krnl(a, tree, p, p2p_list, p2p_n, mlt_max, d_EPS2);
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
                                                          int L, SCAL d_EPS2,
                                                          int begi, int endi, int stride)
// particle to particle interaction
{
	int beg = kd_beg(L);
	int *index = tree.index + beg;
	int *mult = tree.mult + beg;
	for (int i = begi; i < endi; i += stride)
	{
		int ind = index[i];
		int mlt = mult[i];
		const VEC *pi = p + ind;

		fmm_p2p_interaction(a + ind, pi, pi, mlt, mlt, d_EPS2);
	}
}

__global__ void fmm_p2p3_self_kdtree(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p, int L, SCAL d_EPS2)
{
	int m = kd_n(L);
	fmm_p2p3_self_kdtree_krnl(a, tree, p, L, d_EPS2, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void fmm_p2p3_self_kdtree_cpu(VEC *__restrict__ a, const fmmTree_kd tree, const VEC *__restrict__ p, int L, SCAL d_EPS2)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = kd_n(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_p2p3_self_kdtree_krnl, a, tree, p, L, d_EPS2, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_pushl3_kdtree_krnl(fmmTree_kd tree, int l, int begi, int endi, int stride, SCAL *tempi) // 0 -> L-1
// push informations about the field from l-th level to (l+1)-th level
{
	int off = tracelessoffset3(tree.p+1);
	int beg = kd_beg(l);
	int inds[2];
	for (int ijk0 = begi; ijk0 < endi; ijk0 += stride)
	{
		int ijk = beg+ijk0;

		if (tree.mult[ijk] > 0)
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
				if (r != 0)
					d /= r;
				SCAL *local2 = tree.local + inds[ii]*off;
				for (int q = 1; q <= tree.p; ++q)
					l2l_traceless_acc3(local2 + tracelessoffset3(q), tempi, local, q, tree.p, d, r);
			}
		}
	}
}

__global__ void fmm_pushl3_kdtree(fmmTree_kd tree, int l)
{
	extern __shared__ SCAL temp[]; // size must be at least (2*order+1)*blockDim.x
	SCAL *tempi = temp + (2*tree.p+1)*threadIdx.x;
	int n = kd_n(l);
	fmm_pushl3_kdtree_krnl(tree, l, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x, tempi);
}

void fmm_pushl3_kdtree_cpu(fmmTree_kd tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[2*tree.p+1 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int m = kd_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushl3_kdtree_krnl, tree, l, niter*i, std::min(niter*(i+1), m), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

inline __host__ __device__ void fmm_pushLeaves3_kdtree_krnl(VEC *__restrict__ a, const VEC *__restrict__ p,
                                                            fmmTree_kd tree, int L, int begi, int endi, int stride, SCAL *tempi)
// push informations about the field from leaves to individual particles
{
	int off = tracelessoffset3(tree.p+1);
	int beg = kd_beg(L);
	const int *mult = tree.mult + beg, *index = tree.index + beg;
	const SCAL *local = tree.local + beg*off;
	const VEC *center = tree.center + beg;
	for (int i = begi; i < endi; i += stride)
	{
		int mlt = mult[i], ind = index[i];
		VEC *ai = a + ind;
		const VEC *pi = p + ind;
		for (int j = 0; j < mlt; ++j)
		{
			VEC d = pi[j] - center[i];
			SCAL r(sqrt(dot(d,d)));
			if (r != 0)
				d /= r;
			ai[j] += l2p_traceless_field3(tempi, local + off*i, tree.p, d, r);
		}
	}
}

__global__ void fmm_pushLeaves3_kdtree(VEC *a, const VEC *p, fmmTree_kd tree, int L)
{
	extern __shared__ SCAL temp[]; // size must be at least (2*order+4)*blockDim.x
	SCAL *tempi = temp + (2*tree.p+2)*threadIdx.x;
	int m = kd_n(L);
	fmm_pushLeaves3_kdtree_krnl(a, p, tree, L, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x, tempi);
}

void fmm_pushLeaves3_kdtree_cpu(VEC *a, const VEC *p, fmmTree_kd tree, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[2*tree.p+2 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int m = kd_n(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushLeaves3_kdtree_krnl, a, p, tree, L, niter*i, std::min(niter*(i+1), m), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

template <typename T>
void sort_particle_gpu(VEC *p, VEC *d_tmp, int n, cub::DoubleBuffer<T>& d_dkeys, cub::DoubleBuffer<int>& d_values,
	void *& d_tmp_stor, size_t& stor_bytes, bool eval = true, int end_bit = sizeof(T)*8)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

	if (eval)
	{
		size_t new_stor_bytes = 0;
		gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, new_stor_bytes, d_dkeys, d_values, n, 0, end_bit));
		if (new_stor_bytes > stor_bytes)
		{
			if (stor_bytes > 0)
				gpuErrchk(cudaFree(d_tmp_stor));
			stor_bytes = new_stor_bytes;
			gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
		}
	}
	gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dkeys, d_values, n, 0, end_bit));

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p, d_tmp, n);

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p+n, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p+n, d_tmp, n);
}

template <typename T>
void sort_particle_cpu(VEC *p, char *c_tmp, int n, T* keys, int* ind)
{
	if (n > 99999)
		parasort(n, ind, [keys](int i, int j) { return keys[i] < keys[j]; }, CPU_THREADS);
	else
		std::sort(ind, ind + n, [keys](int i, int j) { return keys[i] < keys[j]; });

	gather_cpu((T*)c_tmp, keys, ind, n);
	copy_cpu(keys, (T*)c_tmp, n);

	gather_cpu((VEC*)c_tmp, p, ind, n);
	copy_cpu(p, (VEC*)c_tmp, n);

	gather_cpu((VEC*)c_tmp, p+n, ind, n);
	copy_cpu(p+n, (VEC*)c_tmp, n);
}

void fmm_cart3_kdtree(VEC *p, VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int radius = tree_radius;

	static unsigned long long *d_keys = nullptr;
	static int order = -1, old_size = 0;
	static int smemSize = 48000;
	static int n_prev = 0, n_max = 0, L = 0, ntot_max = 0;
	static int *d_ind = nullptr;
	static char *d_tbuf = nullptr;
	static fmmTree_kd tree;
	static VEC *d_minmax = nullptr;
	static VEC *d_tmp = nullptr;
	static int2 *d_p2p_list = nullptr, *d_m2l_list = nullptr, *d_stack = nullptr;
	static int *d_p2p_n = nullptr, *d_m2l_n = nullptr;
	static int p2p_max = 0, m2l_max = 0, ntot = 0;
	assert(n > BLOCK_SIZE);

	if (n != n_prev || fmm_order != order)
	{
		order = fmm_order;
		SCAL s = order*order;
		L = (int)std::round(std::log2(dens_inhom*(SCAL)n/s)); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);
		L = std::min(L, 30);

		while (kd_n(L) > n)
			--L;
		ntot = kd_ntot(L);
		std::clog << "L: " << L << std::endl;
		std::clog << "ntot: " << ntot << std::endl;
		int new_size = (3*sizeof(VEC) + sizeof(SCAL)*2*tracelessoffset3(order+1)
					  + sizeof(int)*3)*ntot;

		if (new_size > old_size)
		{
			if (old_size > 0)
			{
				gpuErrchk(cudaFree(d_tbuf));
			}
			else
			{
				gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 262144));
				size_t max_stack;
				gpuErrchk(cudaDeviceGetLimit(&max_stack, cudaLimitStackSize));
				std::cout << "max_stack: " << max_stack << std::endl;
				gpuErrchk(cudaMalloc((void**)&d_minmax, sizeof(VEC)*2));
				gpuErrchk(cudaMalloc((void**)&d_p2p_n, sizeof(int)*2));
				d_m2l_n = d_p2p_n + 1;
			}
			gpuErrchk(cudaMalloc((void**)&d_tbuf, new_size));
			old_size = new_size;
		}
		tree.center = (VEC*)d_tbuf;
		tree.lbound = tree.center + ntot;
		tree.rbound = tree.lbound + ntot;
		tree.mpole = (SCAL*)(tree.rbound + ntot);
		tree.local = tree.mpole + ntot*tracelessoffset3(order+1);
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
				gpuErrchk(cudaFree(d_tmp));
			}
			gpuErrchk(cudaMalloc((void**)&d_keys, sizeof(unsigned long long)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_ind, sizeof(int)*n*2));
			gpuErrchk(cudaMalloc((void**)&d_tmp, sizeof(VEC)*n));
		}
		if (ntot > ntot_max)
		{
			if (ntot_max > 0)
			{
				gpuErrchk(cudaFree(d_p2p_list));
				gpuErrchk(cudaFree(d_m2l_list));
				gpuErrchk(cudaFree(d_stack));
			}
			p2p_max = kd_n(L)*200;
			m2l_max = kd_n(L)*200;
			gpuErrchk(cudaMalloc((void**)&d_p2p_list, sizeof(int2)*p2p_max));
			gpuErrchk(cudaMalloc((void**)&d_m2l_list, sizeof(int2)*m2l_max));
			gpuErrchk(cudaMalloc((void**)&d_stack, sizeof(int2)*ntot*10));
		}
	}

	minmaxReduce(d_minmax, p, n, 1);

	static void *d_tmp_stor = nullptr;
	static size_t stor_bytes = 0;

	cub::DoubleBuffer<unsigned long long> d_dbuf(d_keys, d_keys + n);
	cub::DoubleBuffer<int> d_values(d_ind, d_ind + n);

	evalRootBox <<< 1, 1 >>> (tree, d_minmax);
	evalKeys_kdtree <<< nBlocks, BLOCK_SIZE >>> (d_dbuf.Current(), tree.splitdim, p, n, 0);
	evalIndices <<< nBlocks, BLOCK_SIZE >>> (d_values.Current(), n);

	sort_particle_gpu(p, d_tmp, n, d_dbuf, d_values, d_tmp_stor, stor_bytes, true, 32);

	for (int l = 1; l <= L-1; ++l)
	{
		evalBox <<< nBlocks, BLOCK_SIZE >>> (tree, p, n, l);
		evalKeys_kdtree <<< nBlocks, BLOCK_SIZE >>> (d_dbuf.Current(), tree.splitdim + kd_beg(l), p, n, l);
		evalIndices <<< nBlocks, BLOCK_SIZE >>> (d_values.Current(), n);

		sort_particle_gpu(p, d_tmp, n, d_dbuf, d_values, d_tmp_stor, stor_bytes, true, 32 + l + 1);
	}

	evalBox <<< nBlocks, BLOCK_SIZE >>> (tree, p, n, L);
	evalKeysLeaves_kdtree <<< nBlocks, BLOCK_SIZE >>> ((int*)d_keys, n, L);

	fmm_init3_kdtree <<< nBlocks, BLOCK_SIZE >>> (tree, L);

	int beg = kd_beg(L), m = kd_n(L);

	indexLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.index + beg, (int*)d_keys, m, n);

	multLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.mult + beg, tree.index + beg, m, n);

	centerLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	fmm_multipoleLeaves3_kdtree <<< nBlocks, BLOCK_SIZE >>> (tree, p, L);

	smemSize = (2*tree.p+1)*BLOCK_SIZE*sizeof(SCAL);
	for (int l = L-1; l >= 0; --l)
		fmm_buildTree3_kdtree <<< nBlocks, BLOCK_SIZE, smemSize >>> (tree, l);

	fmm_dualTraversal <<< 1, 1024 >>> (tree, d_p2p_list, d_m2l_list, d_stack, d_p2p_n, d_m2l_n, p2p_max, m2l_max, radius, L);

	rescale <<< nBlocks, BLOCK_SIZE >>> (a, n, param+1);

	if (coll)
	{
		int max_mlt = (n-1) / m + 1;
		smemSize = 2*max_mlt*sizeof(VEC);

		fmm_p2p3_kdtree <<< nBlocks, BLOCK_SIZE, smemSize >>> (a, tree, p, d_p2p_list, d_p2p_n, max_mlt, EPS2);
		fmm_p2p3_self_kdtree <<< nBlocks, BLOCK_SIZE >>> (a, tree, p, L, EPS2);
	}

	smemSize = (4*tree.p+1)*BLOCK_SIZE*sizeof(SCAL);
	fmm_c2c3_kdtree <<< nBlocks, BLOCK_SIZE, smemSize >>> (tree, d_m2l_list, d_m2l_n, EPS2);

	smemSize = (2*tree.p+1)*BLOCK_SIZE*sizeof(SCAL);
	for (int l = 1; l <= L-1; ++l)
		fmm_pushl3_kdtree <<< nBlocks, BLOCK_SIZE, smemSize >>> (tree, l);

	smemSize = (2*tree.p+2)*BLOCK_SIZE*sizeof(SCAL);
	fmm_pushLeaves3_kdtree <<< nBlocks, BLOCK_SIZE, smemSize >>> (a, p, tree, L);
	if (param != nullptr)
		rescale <<< nBlocks, BLOCK_SIZE >>> (a, n, param);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (n > n_max)
		n_max = n;
	if (ntot > ntot_max)
		ntot_max = ntot;
	n_prev = n;
}

void fmm_cart3_kdtree_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	int radius = tree_radius;

	static unsigned long long *keys = nullptr;
	static int order = -1, old_size = 0;
	static int n_prev = 0, n_max = 0, L = 0;
	static int *ind = nullptr;
	static char *tbuf = nullptr, *c_tmp = nullptr;
	static fmmTree_kd tree;
	static std::vector<int2> p2p_list, m2l_list, stack;
	std::vector<VEC> min_(std::max(CPU_THREADS, 2)), max_(CPU_THREADS);
	assert(n > 0);

	if (n != n_prev || fmm_order != order)
	{
		order = fmm_order;
		SCAL s = order*order;
		L = (int)std::round(std::log2(dens_inhom*(SCAL)n/s)); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);
		L = std::min(L, 30);

		while (kd_n(L) > n)
			--L;
		int ntot = kd_ntot(L);
		std::clog << "L: " << L << std::endl;
		std::clog << "ntot: " << ntot << std::endl;
		int new_size = (3*sizeof(VEC) + sizeof(SCAL)*2*tracelessoffset3(order+1)
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
		tree.local = tree.mpole + ntot*tracelessoffset3(order+1);
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
				delete[] c_tmp;
			}
			keys = new unsigned long long[n];
			ind = new int[n];
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

	sort_particle_cpu(p, c_tmp, n, keys, ind);

	for (int l = 1; l <= L-1; ++l)
	{
		evalBox_cpu(tree, p, n, l);
		evalKeys_kdtree_cpu(keys, tree.splitdim + kd_beg(l), p, n, l);
		evalIndices_cpu(ind, n);

		sort_particle_cpu(p, c_tmp, n, keys, ind);
	}

	evalBox_cpu(tree, p, n, L);
	evalKeysLeaves_kdtree_cpu((int*)keys, n, L);

	fmm_init3_kdtree_cpu(tree, L);

	int beg = kd_beg(L), m = kd_n(L);

	indexLeaves_cpu(tree.index + beg, (int*)keys, m, n);

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
		std::cout << "max_mlt: " << max_mlt << std::endl;
		fmm_p2p3_kdtree_cpu(a, tree, p, p2p_list.data(), &list_n, max_mlt, EPS2);
	}

	list_n = m2l_list.size();
	fmm_c2c3_kdtree_cpu(tree, m2l_list.data(), &list_n, EPS2);

	for (int l = 1; l <= L-1; ++l)
		fmm_pushl3_kdtree_cpu(tree, l);

	fmm_pushLeaves3_kdtree_cpu(a, p, tree, L);
	if (param != nullptr)
		rescale_cpu(a, n, param);

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

#endif // !FMM_CART3_KDTREE_CUDA_H