#ifndef FMM_CART_CUDA_H
#define FMM_CART_CUDA_H

#include "appel.cuh"
#include "fmm_cart_base.cuh"
#include "parasort.h"

struct fmmTree
{
	VEC *center;
	SCAL *__restrict__ mpole, *__restrict__ local;
	int *__restrict__ mult, *__restrict__ index;
	int p;
};

inline __host__ __device__ void fmm_init_krnl(fmmTree tree, int l, int begi, int endi, int stride)
{
	int offL = tracelessoffset(tree.p+1), offM = tensortupleoffset(tree.p+1);
	int beg = tree_beg(l);
	VEC *center = tree.center + beg;
	SCAL *mpole = tree.mpole + beg*offM,
		 *local = tree.local + beg*offL;
	for (int i = begi; i < endi; i += stride)
	{
		center[i] = VEC{};
		SCAL *multipole = mpole + offM*i;
		for (int j = 0; j < offM; ++j)
			multipole[j] = (SCAL)0;
		SCAL *loc = local + offL*i;
		for (int j = 0; j < offL; ++j)
			loc[j] = (SCAL)0;
	}
}

__global__ void fmm_init(fmmTree tree, int l)
{
	int m = tree_n(l);
	fmm_init_krnl(tree, l, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void fmm_init_cpu(fmmTree tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = tree_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_init_krnl, tree, l, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_multipoleLeaves_krnl(fmmTree tree, const VEC *p, int L,
														 int begi, int endi, int stride)
// calculate multipoles for each cell
// assumes all particles have the same charge/mass
{
	int off = tensortupleoffset(tree.p+1);
	int beg = tree_beg(L);
	const VEC *center = tree.center + beg;
	const int *index = tree.index + beg, *mult = tree.mult + beg;
	SCAL *mpole = tree.mpole + beg*off;
	for (int i = begi; i < endi; i += stride)
	{
		SCAL *multipole = mpole + off*i;
		const VEC *pi = p + index[i];
		multipole[0] = (SCAL)mult[i];
		if (tree.p >= 2)
		{
			for (int j = 0; j < mult[i]; ++j)
			{
				VEC d = pi[j] - center[i];
				/*SCAL r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;*/
				for (int q = 2; q <= tree.p; ++q)
					p2m_acc(multipole + tensortupleoffset(q), q, d);
			}
		}
	}
}

__global__ void fmm_multipoleLeaves(fmmTree tree, const VEC *p, int L)
{
	int m = tree_n(L);
	fmm_multipoleLeaves_krnl(tree, p, L, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void fmm_multipoleLeaves_cpu(fmmTree tree, const VEC *p, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = tree_n(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_multipoleLeaves_krnl, tree, p, L, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}


inline __host__ __device__ void fmm_buildTree2_krnl(fmmTree tree, int l, int begi, int endi, int stride) // L-1 -> 0
// build the l-th of cells after the deeper one (l+1)-th
// "tree" contains only pointers to the actual tree in memory
{
	int off = tensortupleoffset(tree.p+1);
	int sidel = tree_side(l);
	int sidelp = tree_side(l+1);
	int beg = tree_beg(l), begp = tree_beg(l+1);
	for (int ij0 = begi; ij0 < endi; ij0 += stride)
	{
		int i = ij0 / sidel, j = ij0 - i*sidel;
		int ij = beg+ij0;
		int ijp = begp+2*(i*sidelp+j);

		int mlt = tree.mult[ijp];
		mlt += tree.mult[ijp + 1];
		mlt += tree.mult[ijp + sidelp];
		mlt += tree.mult[ijp + sidelp + 1];

		SCAL mpole0 = (SCAL)mlt;

		VEC coord{};

		if (mlt > 0)
		{
			coord += (SCAL)tree.mult[ijp] * tree.center[ijp];
			coord += (SCAL)tree.mult[ijp + 1] * tree.center[ijp + 1];
			coord += (SCAL)tree.mult[ijp + sidelp] * tree.center[ijp + sidelp];
			coord += (SCAL)tree.mult[ijp + sidelp + 1] * tree.center[ijp + sidelp + 1];
			coord /= mpole0;
			
			SCAL *multipole = tree.mpole + ij*off;
			const SCAL *multipole2 = tree.mpole + ijp*off;
			if (tree.p >= 2)
			{
				VEC d = coord - tree.center[ijp];
				/*SCAL r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;*/
				for (int q = 2; q <= tree.p; ++q)
					m2m_acc(multipole + tensortupleoffset(q), multipole2, q, d);

				d = coord - tree.center[ijp + 1];
				/*r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;*/
				multipole2 = tree.mpole + (ijp + 1)*off;
				for (int q = 2; q <= tree.p; ++q)
					m2m_acc(multipole + tensortupleoffset(q), multipole2, q, d);

				d = coord - tree.center[ijp + sidelp];
				/*r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;*/
				multipole2 = tree.mpole + (ijp + sidelp)*off;
				for (int q = 2; q <= tree.p; ++q)
					m2m_acc(multipole + tensortupleoffset(q), multipole2, q, d);

				d = coord - tree.center[ijp + sidelp + 1];
				/*r = sqrt(dot(d,d));
				if (r != 0)
					d /= r;*/
				multipole2 = tree.mpole + (ijp + sidelp + 1)*off;
				for (int q = 2; q <= tree.p; ++q)
					m2m_acc(multipole + tensortupleoffset(q), multipole2, q, d);
			}
			multipole[0] = mpole0;
		}
		
		tree.center[ij] = coord;
		tree.mult[ij] = mlt;
	}
}

__global__ void fmm_buildTree2(fmmTree tree, int l)
{
	//extern __shared__ SCAL temp[]; // size must be at least (order+1)*blockDim.x
	//SCAL *tempi = temp + (tree.p+1)*threadIdx.x;
	int n = tree_n(l);
	fmm_buildTree2_krnl(tree, l, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void fmm_buildTree2_cpu(fmmTree tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	/*std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[tree.p+1 + CACHE_LINE_SIZE/sizeof(SCAL)];*/
	int m = tree_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_buildTree2_krnl, tree, l, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	//for (int i = 0; i < CPU_THREADS; ++i)
	//	delete[] temp[i];
}

inline __host__ __device__ void fmm_c2c2_krnl(fmmTree tree, int l, int radius,
											  int begi, int endi, int stride, SCAL *tempi)
// L -> 1
// cell to cell interaction
{
	int offM = tensortupleoffset(tree.p+1), offL = tracelessoffset(tree.p+1);
	int sidel = tree_side(l);
	int beg = tree_beg(l);
	for (int ij = begi; ij < endi; ij += stride)
	{
		int i = ij / sidel, j = ij - i*sidel;
		int im = (i/2)*2;
		int kmin = ((im-2*radius > 0) ? (im-2*radius) : 0),
			kmax = ((im+(2*radius+1) < sidel-1) ? (im+(2*radius+1)) : (sidel-1));
		int ij1 = beg + ij;

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
					//if (max(abs(k-i), abs(g-j)) <= radius)
					if (!(k > i + radius || k < i - radius || g > j + radius || g < j - radius))
						continue;
					int ij2 = beg + k*sidel + g;
					VEC d = tree.center[ij1] - tree.center[ij2];
					SCAL r2 = dot(d, d) + EPS;

					//m2l_acc(tree.local + ij1*off, tempi, tree.mpole + ij2*off, tree.p, tree.p, d, r2, 1);
					static_m2l_acc<1>(tree.local + ij1*offL, tempi, tree.mpole + ij2*offM, tree.p, d, r2);
				}
		}
	}
}

__global__ void fmm_c2c2(fmmTree tree, int l, int radius)
{
	extern __shared__ SCAL temp[]; // size must be at least (order+2)*blockDim.x
	SCAL *tempi = temp + (tree.p+2)*threadIdx.x;
	int n = tree_n(l);
	fmm_c2c2_krnl(tree, l, radius, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x, tempi);
}

void fmm_c2c2_cpu(fmmTree tree, int l, int radius)
{
	std::vector<std::thread> threads(CPU_THREADS);
	std::vector<SCAL*> temp(CPU_THREADS);
	for (int i = 0; i < CPU_THREADS; ++i)
		temp[i] = new SCAL[tree.p+2 + CACHE_LINE_SIZE/sizeof(SCAL)];
	int m = tree_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_c2c2_krnl, tree, l, radius, niter*i, std::min(niter*(i+1), m), 1, temp[i]);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
	for (int i = 0; i < CPU_THREADS; ++i)
		delete[] temp[i];
}

inline __host__ __device__ void fmm_pushl_krnl(fmmTree tree, int l, int begi, int endi, int stride) // 0 -> L-1
// push informations about the field from l-th level to (l+1)-th level
{
	int off = tracelessoffset(tree.p+1);
	int sidel = tree_side(l);
	int sidelp = tree_side(l+1);
	int beg = tree_beg(l), begp = tree_beg(l+1);
	for (int ij0 = begi; ij0 < endi; ij0 += stride)
	{
		int i = ij0 / sidel, j = ij0 - i*sidel;
		int ij = beg+ij0;
		int ijp = begp+2*(i*sidelp+j);

		VEC d = tree.center[ijp] - tree.center[ij];
		SCAL r = sqrt(dot(d,d));
		if (r != 0)
			d /= r;
		const SCAL *local = tree.local + ij*off;
		SCAL *local2 = tree.local + ijp*off;
		for (int q = 0; q <= tree.p; ++q)
			l2l_traceless_acc(local2 + tracelessoffset(q), local, q, tree.p, d, r);

		d = tree.center[ijp + 1] - tree.center[ij];
		r = sqrt(dot(d,d));
		if (r != 0)
			d /= r;
		local2 = tree.local + (ijp + 1)*off;
		for (int q = 0; q <= tree.p; ++q)
			l2l_traceless_acc(local2 + tracelessoffset(q), local, q, tree.p, d, r);

		d = tree.center[ijp + sidelp] - tree.center[ij];
		r = sqrt(dot(d,d));
		if (r != 0)
			d /= r;
		local2 = tree.local + (ijp + sidelp)*off;
		for (int q = 0; q <= tree.p; ++q)
			l2l_traceless_acc(local2 + tracelessoffset(q), local, q, tree.p, d, r);

		d = tree.center[ijp + sidelp + 1] - tree.center[ij];
		r = sqrt(dot(d,d));
		if (r != 0)
			d /= r;
		local2 = tree.local + (ijp + sidelp + 1)*off;
		for (int q = 0; q <= tree.p; ++q)
			l2l_traceless_acc(local2 + tracelessoffset(q), local, q, tree.p, d, r);
	}
}

__global__ void fmm_pushl(fmmTree tree, int l)
{
	int n = tree_n(l);
	fmm_pushl_krnl(tree, l, blockDim.x * blockIdx.x + threadIdx.x, n, gridDim.x * blockDim.x);
}

void fmm_pushl_cpu(fmmTree tree, int l)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = tree_n(l);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushl_krnl, tree, l, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

inline __host__ __device__ void fmm_pushLeaves_krnl(VEC *__restrict__ a, const VEC *__restrict__ p,
													fmmTree tree, int L, int begi, int endi, int stride)
// push informations about the field from leaves to individual particles
{
	int off = tracelessoffset(tree.p+1);
	int beg = tree_beg(L);
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
			ai[j] += l2p_traceless_field(local + off*i, tree.p, d, r);
		}
	}
}

__global__ void fmm_pushLeaves(VEC *a, const VEC *p, fmmTree tree, int L)
{
	int m = tree_n(L);
	fmm_pushLeaves_krnl(a, p, tree, L, blockDim.x * blockIdx.x + threadIdx.x, m, gridDim.x * blockDim.x);
}

void fmm_pushLeaves_cpu(VEC *a, const VEC *p, fmmTree tree, int L)
{
	std::vector<std::thread> threads(CPU_THREADS);
	int m = tree_n(L);
	int niter = (m-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread(fmm_pushLeaves_krnl, a, p, tree, L, niter*i, std::min(niter*(i+1), m), 1);
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();
}

void fmm_cart(VEC *p, VEC *a, int n, const SCAL* param)
{
	int nBlocks = std::min(MAX_GRID_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int radius = tree_radius;

	static int order = -1, old_size = 0;
	static int nBlocksRed = 1;
	static int n_prev = 0, n_max = 0, L = 0, nL1 = 0, sideL = 0;
	static int *d_keys = nullptr, *d_ind = nullptr;
	static char *d_tbuf = nullptr;
	static fmmTree tree;
	static VEC *d_minmax = nullptr;
	static VEC *minmax_ = new VEC[2*nBlocksRed];
	static VEC *d_tmp = nullptr;
	assert(n > BLOCK_SIZE);

	int nZ = 1<<DIM;

	if (n != n_prev || fmm_order != order)
	{
		order = fmm_order;
		SCAL s = order*sqrt(order);
		L = (int)std::round(std::log2(dens_inhom*(SCAL)n/s)/DIM); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);
		
		nL1 = 1 << ((L+1)*DIM);
		sideL = 1 << L;
		int ntot = (nL1 - 1) / (nZ - 1);
		int new_size = (sizeof(VEC) + sizeof(SCAL)*(tensortupleoffset(order+1) + tracelessoffset(order+1))
					  + sizeof(int)*2)*ntot;

		if (new_size > old_size)
		{
			if (old_size > 0)
			{
				gpuErrchk(cudaFree(d_tbuf));
				gpuErrchk(cudaFree(d_minmax));
			}
			gpuErrchk(cudaMalloc((void**)&d_tbuf, new_size));
			gpuErrchk(cudaMalloc((void**)&d_minmax, sizeof(VEC)*2*nBlocksRed));
			old_size = new_size;
		}
		tree.center = (VEC*)d_tbuf;
		tree.mpole = (SCAL*)(tree.center + ntot);
		tree.local = tree.mpole + ntot*tensortupleoffset(order+1);
		tree.mult = (int*)(tree.local + ntot*tracelessoffset(order+1));
		tree.index = tree.mult + ntot;
		tree.p = order;
		if (n > n_max)
		{
			if (n_max > 0)
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

	int smemSize = (BLOCK_SIZE <= 32) ? 2 * BLOCK_SIZE * sizeof(VEC)*2 : BLOCK_SIZE * sizeof(VEC)*2;
	if (isPow2(n))
		minmaxReduce<true> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (d_minmax, p, n);
	else
		minmaxReduce<false> <<< nBlocksRed, BLOCK_SIZE, smemSize >>> (d_minmax, p, n);

	gpuErrchk(cudaMemcpy(minmax_, d_minmax, sizeof(VEC)*2*nBlocksRed, cudaMemcpyDeviceToHost));

	if (nBlocksRed > 1)
	{
		for (int i = 2; i < 2*nBlocksRed; i += 2)
		{
			minmax_[0] = fmin(minmax_[0], minmax_[i]);
			minmax_[1] = fmax(minmax_[1], minmax_[i+1]);
		}
		gpuErrchk(cudaMemcpy(d_minmax, minmax_, sizeof(VEC)*2, cudaMemcpyHostToDevice));
	}

	VEC Delta = minmax_[1] - minmax_[0];
	SCAL delta = fmax(Delta) / (SCAL)sideL, sqrtESP = sqrt(EPS);
	if (delta < sqrtESP)
		delta = sqrtESP;
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
		if (n > n_max)
		{
			if (n_max > 0)
				gpuErrchk(cudaFree(d_tmp_stor));
			d_tmp_stor = nullptr;
			gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dbuf, d_values, n, 0, L*DIM));
			gpuErrchk(cudaMalloc(&d_tmp_stor, stor_bytes));
		}
	}
	gpuErrchk(cub::DeviceRadixSort::SortPairs(d_tmp_stor, stor_bytes, d_dbuf, d_values, n, 0, L*DIM));

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p, d_tmp, n);

	gather_krnl <<< nBlocks, BLOCK_SIZE >>> (d_tmp, p+n, d_values.Current(), n);
	copy_krnl <<< nBlocks, BLOCK_SIZE >>> (p+n, d_tmp, n);

	int beg = tree_beg(L), m = tree_n(L);

	for (int l = L; l >= 2; --l)
		fmm_init <<< nBlocks, BLOCK_SIZE >>> (tree, l);
	indexLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.index + beg, d_dbuf.Current(), m, n);

	multLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.mult + beg, tree.index + beg, m, n);

	centerLeaves <<< nBlocks, BLOCK_SIZE >>> (tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	fmm_multipoleLeaves <<< nBlocks, BLOCK_SIZE >>> (tree, p, L);

	//smemSize = (tree.p+1)*BLOCK_SIZE*sizeof(SCAL);
	for (int l = L-1; l >= 2; --l)
		fmm_buildTree2 <<< nBlocks, BLOCK_SIZE >>> (tree, l);

	if (coll)
		p2p2 <<< nBlocks, BLOCK_SIZE >>> (a, tree.mult + beg, tree.index + beg, p, m, tree_side(L), radius);
	else
		rescale <<< nBlocks, BLOCK_SIZE >>> (a, n, param+1);

	smemSize = (tree.p+2)*BLOCK_SIZE*sizeof(SCAL);
	for (int l = L; l >= 2; --l)
		fmm_c2c2 <<< nBlocks, BLOCK_SIZE, smemSize >>> (tree, l, radius);

	for (int l = 2; l <= L-1; ++l)
		fmm_pushl <<< nBlocks, BLOCK_SIZE >>> (tree, l);

	fmm_pushLeaves <<< nBlocks, BLOCK_SIZE >>> (a, p, tree, L);
	if (param != nullptr)
		rescale <<< nBlocks, BLOCK_SIZE >>> (a, n, param);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

void fmm_cart_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	int radius = tree_radius;

	static char* c_tmp = nullptr;
	static int order = -1, old_size = 0;
	static int n_prev = 0, n_max = 0, L = 0, nL1 = 0, sideL = 0;
	static int *keys = nullptr, *ind = nullptr;
	static char *tbuf = nullptr;
	static fmmTree tree;
	std::vector<VEC> min_(CPU_THREADS), max_(CPU_THREADS);
	assert(n > 0);

	int nZ = 1<<DIM;

	if (n != n_prev || fmm_order != order)
	{
		order = fmm_order;
		SCAL s = order*sqrt(order);
		L = (int)std::round(std::log2(dens_inhom*(SCAL)n/s)/DIM); // maximum level, L+1 is the number of levels
		L = std::max(L, 2);

		nL1 = 1 << ((L+1)*DIM);
		sideL = 1 << L;
		int ntot = (nL1 - 1) / (nZ - 1);
		int new_size = (sizeof(VEC) + sizeof(SCAL)*(tensortupleoffset(order+1) + tracelessoffset(order+1))
					  + sizeof(int)*2)*ntot;

		if (new_size > old_size)
		{
			if (old_size > 0)
				delete[] tbuf;
			tbuf = new char[new_size];
			old_size = new_size;
		}
		tree.center = (VEC*)tbuf;
		tree.mpole = (SCAL*)(tree.center + ntot);
		tree.local = tree.mpole + ntot*tensortupleoffset(order+1);
		tree.mult = (int*)(tree.local + ntot*tracelessoffset(order+1));
		tree.index = tree.mult + ntot;
		tree.p = order;
		if (n > n_max)
		{
			if (n_max > 0)
			{
				delete[] keys;
				delete[] ind;
				delete[] c_tmp;
			}
			keys = new int[n];
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

	VEC Delta = max_[0] - min_[0];
	SCAL delta = fmax(Delta) / (SCAL)sideL, sqrtESP = sqrt(EPS);
	if (delta < sqrtESP)
		delta = sqrtESP;
	SCAL rdelta = (SCAL)1 / delta;

	evalKeys_cpu(keys, p, min_.data(), n, rdelta, sideL);
	evalIndices_cpu(ind, n);

	if (n > 99999)
		parasort(n, ind, [](int i, int j) { return keys[i] < keys[j]; }, CPU_THREADS);
	else
		std::sort(ind, ind + n, [](int i, int j) { return keys[i] < keys[j]; });

	gather_cpu((int*)c_tmp, keys, ind, n);
	copy_cpu(keys, (int*)c_tmp, n);

	gather_cpu((VEC*)c_tmp, p, ind, n);
	copy_cpu(p, (VEC*)c_tmp, n);

	gather_cpu((VEC*)c_tmp, p+n, ind, n);
	copy_cpu(p+n, (VEC*)c_tmp, n);

	int beg = tree_beg(L), m = tree_n(L);

	for (int l = L; l >= 2; --l)
		fmm_init_cpu(tree, l);
	indexLeaves_cpu(tree.index + beg, keys, m, n);

	multLeaves_cpu(tree.mult + beg, tree.index + beg, m, n);

	centerLeaves_cpu(tree.center + beg, tree.mult + beg, tree.index + beg, p, m);

	fmm_multipoleLeaves_cpu(tree, p, L);

	for (int l = L-1; l >= 2; --l)
		fmm_buildTree2_cpu(tree, l);

	if (coll)
		p2p2_cpu(a, tree.mult + beg, tree.index + beg, p, m, tree_side(L), radius);
	else
		rescale_cpu(a, n, param+1);

	for (int l = L; l >= 2; --l)
		fmm_c2c2_cpu(tree, l, radius);

	for (int l = 2; l <= L-1; ++l)
		fmm_pushl_cpu(tree, l);

	fmm_pushLeaves_cpu(a, p, tree, L);
	if (param != nullptr)
		rescale_cpu(a, n, param);

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

#endif // !FMM_CART_CUDA_H
