//  Base functions used in Fast Multipole Method (FMM) in 3d
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

#ifndef FMM_CART_BASE3_CUDA_H
#define FMM_CART_BASE3_CUDA_H

#include <atomic> // atomic_ref

#include "fmm_cart_base.cuh"


inline __host__ __device__ SCAL coeff13(int n, int m)
// a coefficient used in the calculation of the gradient
// returns (-1)^m * (2n - 2m - 1)!!
// assumes m <= n/2 and n >= 1
// 2n - 2m - 1 must be smaller than 79 (otherwise bad things will occur)
{
	return (SCAL)paritysign(m) * static_odfactorial(2 * (n - m) - 1);
	// for DIM > 3, see Shanker, Huang, 2007
}

/**

Let's define a (totally) symmetric tensor in 3 dimensions with order p. Then it has (p+1)(p+2)/2 independent
coefficients.
Usually a tensor A is denoted with the notation:
	A_a,b,c,d...
where a,b,c,d... are indices that can take only three values since we are in 3D. However,
it would be pointless to store all 3^p elements. Since the tensor is symmetric, swapping
any two indices won't change the result. So, an element of a symmetric tensor is completely
determined by the multiplicities of the values that indices take. Then we use a new set of indices x,y,z
to denote the elements of a symmetric tensor A:
	A_x,y,z
where p = x + y + z. With just 3 indices we can notate symmetric tensors of any order in 3 dimensions.
It is usually convenient in a program to have just one index for an array. We use an index i that
corresponds to the multiplicities x,y,z through an 1-1 relationship:
(we consider all elements where z >= 2 to be dependent in traceless symmetric tensors)
x,y,z | i | j | p
0,0,0 | 0 | 0 | 0
1,0,0 | 0 | 1 | 1
0,1,0 | 1 | 2 | 1
0,0,1 | 2 | 3 | 1
2,0,0 | 0 | 4 | 2
1,1,0 | 1 | 5 | 2
0,2,0 | 2 | 6 | 2
1,0,1 | 3 | 7 | 2
0,1,1 | 4 | 8 | 2 
0,0,2 | 5 | 9 | 2
3,0,0 | 0 |10 | 3 
2,1,0 | 1 |11 | 3
1,2,0 | 2 |12 | 3
0,3,0 | 3 |13 | 3
2,0,1 | 4 |14 | 3
1,1,1 | 5 |15 | 3
0,2,1 | 6 |16 | 3
1,0,2 | 7 |17 | 3
0,1,2 | 8 |18 | 3
0,0,3 | 9 |19 | 3
4,0,0 | 0 |20 | 4
3,1,0 | 1 |21 | 4
2,2,0 | 2 |22 | 4
1,3,0 | 3 |23 | 4
0,4,0 | 4 |24 | 4
3,0,1 | 5 |25 | 4
2,1,1 | 6 |26 | 4
1,2,1 | 7 |27 | 4
0,3,1 | 8 |28 | 4
2,0,2 | 9 |29 | 4
1,1,2 |10 |30 | 4
0,2,2 |11 |31 | 4
1,0,3 |12 |32 | 4
0,1,3 |13 |33 | 4
0,0,4 |14 |34 | 4
...
where 0 <= i <= p is an index for a p-order symmetric tensor, and j is a cumulative index, useful if we
merge a set of tensors of increasing order in a single array. With this notation there is a specific
relationship between x,y,z and i, which is:
i(x, z) = p * (p + 1) / 2 - (p - z) * (p - z + 1) / 2 + p - x
When converting from i to j or viceversa one has to take into account an offset that depends on p:
j = i + p*(p+1)*(p+2)/6
Note that p=0 (j=0) are the monopoles, p=1 (j=1,2,3) are dipoles, p=2 (j=4,5,6,7,8,9) quadrupoles etc...
Note also that, once p, i are fixed, one can uniquely find x, y and z from:
z(i) = k(i) = floor((2*p+3 - sqrt((2*p+3)^2 - 8*i)) / 2)
where k(i) is the smallest integer such that k(i)(2*p+3 - k(i)) > 2*i.
x(i) = p * (p + 1) / 2 - (p - z(i)) * (p - z(i) + 1) / 2 + p - i
y(i) = p - x(i) - z(i)

Traceless symmetric tensors indices:
x,y,z | i | j | p
0,0,0 | 0 | 0 | 0
1,0,0 | 0 | 1 | 1
0,1,0 | 1 | 2 | 1
0,0,1 | 2 | 3 | 1
2,0,0 | 0 | 4 | 2
1,1,0 | 1 | 5 | 2
0,2,0 | 2 | 6 | 2
1,0,1 | 3 | 7 | 2
0,1,1 | 4 | 8 | 2 
3,0,0 | 0 | 9 | 3 
2,1,0 | 1 |10 | 3
1,2,0 | 2 |11 | 3
0,3,0 | 3 |12 | 3
2,0,1 | 4 |13 | 3
1,1,1 | 5 |14 | 3
0,2,1 | 6 |15 | 3
4,0,0 | 0 |16 | 4
3,1,0 | 1 |17 | 4
2,2,0 | 2 |18 | 4
1,3,0 | 3 |19 | 4
0,4,0 | 4 |20 | 4
3,0,1 | 5 |21 | 4
2,1,1 | 6 |22 | 4
1,2,1 | 7 |23 | 4
0,3,1 | 8 |24 | 4

j = i + p^2
i(x, 0) = p - x
i(x, 1) = 2 * p - x
Thus i(x, z) can be simplified for z in {0, 1}:
i(x, z) = (z + 1) * p - x = z * p + p - x
Inverse relations:
z(i) = i div (p + 1) = 1_{j | j >= p + 1} (i)
x(i) = (z(i) + 1) * p - i
y(i) = p - x(i) - z(i)

Difference between symmetric and traceless symmetric tensor in the number of independent components
p | (p+1)*(p+2)/2 | 2*p+1
0 |       1       |   1
1 |       3       |   3
2 |       6       |   5
3 |      10       |   7
4 |      15       |   9
5 |      21       |  11
6 |      28       |  13
7 |      36       |  15
8 |      45       |  17
9 |      55       |  19
10|      66       |  21

Using totally traceless symmetric tensors drastically decreases the number of indipendent elements, and
in 3D we would have just 2*p + 1 elements. In this case we will have:
A_(x+2),y,z + A_x,(y+2),z + A_x,y,(z+2) = 0
where A is a tensor of order x + y + z + 2. This can also be written as a recurrence relation:
A_x,y,z = -A_(x+2),y,(z-2) - A_x,(y+2),(z-2)
where A is a tensor of order x + y + z. This can be written with the index notation as:
A_i = -A_(i-2*p+2*k(i)-5) - A_(i-2*p+2*k(i)-3)

An important property is that if we contract a symmetric tensor with a traceless symmetric tensor, the
result will be a symmetric traceless tensor. So we can save computation time by using the previous relation.

A tuple of 3D symmetric tensors with order from 0 to P (inclusive) will have (P+1)*(P+2)*(P+3)/6 elements.
A tuple of 3D traceless symmetric tensors with order from 0 to P (inclusive) will have (P+1)^2 elements.
In the following calculations we will use cartesian coordinates.

*/

constexpr __host__ __device__ int symmetricelems3(int n)
{
	return (n + 1) * (n + 2) / 2;
}

constexpr __host__ __device__ int tracelesselems3(int n)
{
	return 2 * n + 1;
}

constexpr __host__ __device__ int symmetricoffset3(int p)
{
	return p * (p + 1) * (p + 2) / 6;
}

constexpr __host__ __device__ int tracelessoffset3(int p)
{
	return p * p;
}

inline __host__ __device__ int k_coeff(int index, int n)
{
	int b = 2*n+3;
	return (b - sqrt(SCAL(b*b - 8*index))) / 2;
}

inline __host__ __device__ int symmetric_z_i(int i, int n)
{
	return k_coeff(i, n);
}

constexpr __host__ __device__ int symmetric_x_i(int i, int n, int z)
{
	return (n * (n + 1) - (n - z) * (n - z + 1)) / 2 + n - i;
}
inline __host__ __device__ int symmetric_x_i(int i, int n)
{
	return symmetric_x_i(i, n, symmetric_z_i(i, n));
}

constexpr __host__ __device__ int symmetric_i_x_z(int x, int z, int n)
{
	return (n * (n + 1) - (n - z) * (n - z + 1)) / 2 + n - x;
}

constexpr __host__ __device__ int traceless_z_i(int i, int n)
{
	return (i >= n+1); // i div (p + 1) = 1_{j | j >= p + 1} (i)
}

constexpr __host__ __device__ int traceless_x_i(int i, int n, int z)
{
	return (z + 1) * n - i;
}
constexpr __host__ __device__ int traceless_x_i(int i, int n)
{
	return traceless_x_i(i, n, traceless_z_i(i, n));
}

constexpr __host__ __device__ int traceless_i_x_z(int x, int z, int n)
{
	return (z + 1) * n - x;
}

constexpr __host__ __device__ SCAL traceless_A_x_z(const SCAL *A, int x, int z, int n)
// extract element from traceless tensor A. It's valid also for z >= 2
{
	if (z >= 2)
		return -traceless_A_x_z(A, x+2, z-2, n) - traceless_A_x_z(A, x, z-2, n);
	else
		return A[traceless_i_x_z(x, z, n)];
}

inline __host__ __device__ void trace3(SCAL *__restrict__ out, const SCAL *__restrict__ in, int n, int m)
// contract 2m indices in n-order symmetric tensor "in" giving (n-2m)-order tensor "out"
// assumes n >= 2m
// if n = 2m the result is a scalar
{
	int elems = symmetricelems3(n);
	int n_out = n - 2 * m;
	int elems_out = symmetricelems3(n_out);
	int i = 0;
	for (int z = 0; z <= n_out; ++z)
		for (int x = n_out-z; x >= 0; --x)
		{
			SCAL t(0);
			int j = symmetric_i_x_z(x, z, n);
			int k_j = k_coeff(j, n);
			const SCAL *inj = in + j;
			const SCAL *injk;
			for (int kz = 0; kz <= m; ++kz)
			{
				injk = inj + 2*kz*(n - k_j - kz) + kz;
				for (int kx = 0; kx <= m-kz; ++kx)
					t += (SCAL)trinomial(m, kx, kz) * injk[-2*kx];
			}
			out[i++] = t;
		}
}

inline __host__ __device__ void contract3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                const SCAL *__restrict__ B,
                                          int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor C of order |nA-nB|
// if nA = nB the result is a scalar
// O(|nA-nB|^2 * min(nA,nB)^2)
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // at this point we assume nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *Ajk, *Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] = t;
		}
}

inline __host__ __device__ void contract_acc3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                    const SCAL *__restrict__ B,
                                              int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// and sum (accumulate) the result to C
// O(|nA-nB|^2 * min(nA,nB)^2)
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *Ajk, *Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] += t;
		}
}

inline __host__ __device__ void contract_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                   const SCAL *__restrict__ B,
                                             SCAL c, int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// O(|nA-nB|^2 * min(nA,nB)^2)
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *Ajk, *Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] += c*t;
		}
}

inline __host__ __device__ void contract_traceless_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                             const SCAL *__restrict__ B,
                                                       SCAL c, int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= min(1, nC); ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *Ajk, *Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] += c*t;
		}
}

template <bool b_atomic = false>
inline __host__ __device__ void contract_traceless2_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                              const SCAL *__restrict__ B,
                                                        SCAL c, int nA, int nB)
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int i = 0;
	for (int z = 0; z <= min(1, nC); ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			for (int kz = 0; kz <= nB; ++kz)
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * traceless_A_x_z(A, x+kx, z+kz, nA) * traceless_A_x_z(B, kx, kz, nB);
			if (b_atomic)
			{
#ifdef __CUDA_ARCH__
				myAtomicAdd(C + (i++), c*t);
#else
				std::atomic_ref<SCAL> atomicC(C[i++]);
				atomicC += c*t;
#endif
			}
			else
				C[i++] += c*t;
		}
}

template<int nA, int nB, bool b_atomic = false>
inline __host__ __device__ void static_contract_traceless_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A,
                                                                                    const SCAL *__restrict__ B,
                                                              SCAL c)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
{
	if (nA < nB)
		static_contract_traceless_ma3<nB, nA>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		constexpr int jb = symmetric_i_x_z(0, 0, nB);
		int kjb = k_coeff(jb, nB);
		const SCAL *Bj = B + jb;
		int i = 0;
		for (int z = 0; z <= static_min(1, nC); ++z)
			for (int x = nC-z; x >= 0; --x)
			{
				SCAL t(0);
				int ja = symmetric_i_x_z(x, z, nA);
				int kja = k_coeff(ja, nA);
				const SCAL *Aj = A + ja;
				const SCAL *Ajk, *Bjk;
				for (int kz = 0; kz <= nB; ++kz)
				{
					Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
					Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
					for (int kx = 0; kx <= nB-kz; ++kx)
						t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
				}
				if (b_atomic)
				{
#ifdef __CUDA_ARCH__
					myAtomicAdd(C + (i++), c*t);
#else
					std::atomic_ref<SCAL> atomicC(C[i++]);
					atomicC += c*t;
#endif
				}
				else
					C[i++] += c*t;
			}
	}
}

template<int nA, int nB, bool b_atomic = false>
inline __host__ __device__ void static_contract_traceless2_ma3(SCAL *C, const SCAL *A, const SCAL *B, SCAL c)
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result
// O(|nA-nB| * min(nA,nB)^2)
{
	if (nA < nB)
		static_contract_traceless2_ma3<nB, nA, b_atomic>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		int i = 0;
		for (int z = 0; z <= static_min(1, nC); ++z)
			for (int x = nC-z; x >= 0; --x)
			{
				SCAL t(0);
				for (int kz = 0; kz <= nB; ++kz)
					for (int kx = 0; kx <= nB-kz; ++kx)
						t += (SCAL)trinomial(nB, kx, kz) * traceless_A_x_z(A, x+kx, z+kz, nA) * traceless_A_x_z(B, kx, kz, nB);
				if (b_atomic)
				{
#ifdef __CUDA_ARCH__
					myAtomicAdd(C + (i++), c*t);
#else
					std::atomic_ref<SCAL> atomicC(C[i++]);
					atomicC += c*t;
#endif
				}
				else
					C[i++] += c*t;
			}
	}
}

inline __host__ __device__ void traceless_refine3(SCAL *A, int n)
// build a traceless tensor using symmetric index notation
// O(n^2)
{
	int i = symmetric_i_x_z(n-2, 2, n);
	for (int z = 2; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int jsx = symmetric_i_x_z(x+2, z-2, n);
			int jsy = symmetric_i_x_z(x, z-2, n);
			A[i++] = -A[jsx] - A[jsy];
		}
}

template<int n>
inline __host__ __device__ void static_traceless_refine3(SCAL *A)
// build a traceless tensor using symmetric index notation
// O(n^2)
{
	int i = symmetric_i_x_z(n-2, 2, n);
	for (int z = 2; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int jsx = symmetric_i_x_z(x+2, z-2, n);
			int jsy = symmetric_i_x_z(x, z-2, n);
			A[i++] = -A[jsx] - A[jsy];
		}
}

inline __host__ __device__ void gradient_exact3(SCAL *grad, int n, VEC d, SCAL r)
// calculate the gradient of 1/r of order n i.e. nabla^n 1/r, which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// NOTE that if r2 >> EPS2, it is also totally traceless
// O(n^5)
{
	if (n == 0)
		grad[0] = 1/r;
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1);
		int i = 0;
		for (int z = 0; z <= n; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0), t3(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						t3 = 0;
						for (int k3 = 0; k3 <= z/2; ++k3)
						{
							int m = k1+k2+k3;
							t3 += (SCAL)coeff13(n, m) * coeff2(z, k3) * binarypow(d.z, z - 2*k3);
						}
						t2 += t3 * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				grad[i++] = C * t1;
			}
	}
}

inline __host__ __device__ void gradient3(SCAL *grad, int n, VEC d, SCAL r, SCAL c = 1)
// calculate the gradient of -log(r) of order n i.e. -nabla^n log(r), which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
{
	if (n == 0)
		grad[0] = c/r;
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2;
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				grad[i++] = C * t1 * binarypow(d.z, z);
			}
	}
}

template<int n>
inline __host__ __device__ void static_gradient3(SCAL *grad, VEC d, SCAL r, SCAL c = 1)
// calculate the gradient of -log(r) of order n i.e. -nabla^n log(r), which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
{
	if (n == 0)
		grad[0] = c/r;
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				grad[i++] = C * t1 * binarypow(d.z, z);
			}
	}
}

inline __host__ __device__ void tensorpow3(SCAL *power, int n, VEC d)
// calculate the powers tensor of d of order n, e.g.:
// r^(0) = 1
// r^(1) = (x, y, z)
// r^(2) = (x^2, xy, xz, y^2, yz, z^2)
// r^(3) = (x^3, x^2 y, x^2 z, xy^2, xyz, xz^2, y^3, y^2 z, yz^2, z^3)
// etc...
// O(n^2)
{
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			power[i++] = binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

inline __host__ __device__ void tracelesspow3(SCAL *power, int n, VEC d, SCAL r)
// O(n^3)
{
	if (n == 0)
		power[0] = (SCAL)1;
	else
	{
		SCAL C = binarypow(r, n) / static_odfactorial(2*n-1);
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				power[i++] = C * t1 * binarypow(d.z, z);
			}
	}
}

inline __host__ __device__ void p2m3(SCAL *M, int n, VEC d)
// particle to multipole expansion of order n
// d is the coordinate of the particle from the (near) expansion center
// O(n^2)
{
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			M[i++] = C * binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

inline __host__ __device__ void p2m_acc3(SCAL *M, int n, VEC d, SCAL q = 1)
// particle to multipole expansion of order n + accumulate to M
// d is the coordinate of the particle from the (near) expansion center
// O(n^2)
{
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * q;
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			M[i++] += C * binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

inline __host__ __device__ void p2m_traceless_acc3(SCAL *M, int n, VEC d, SCAL r)
// particle to traceless multipole expansion of order n + accumulate to M
// d is the unit vector of the coordinate of the particle from the (near) expansion center
// O(n^3)
{
	if (n == 0)
		M[0] += (SCAL)1;
	else
	{
		SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * binarypow(r, n) / static_odfactorial(2*n-1);
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				M[i++] += C * t1 * binarypow(d.z, z);
			}
	}
}

inline __host__ __device__ void p2l3(SCAL *L, int n, VEC d, SCAL r)
// particle to local expansion of order n
// d is the unit vector of the coordinate of the particle from the (far) expansion center
// O(n^3)
{
	SCAL c = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	gradient3(L, n, d, r, c);
	traceless_refine3(L, n);
}

inline __host__ __device__ void m2m3(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
// multipole to multipole expansion
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
// O(n^5)
{
	int i = 0;
	SCAL C = inv_factorial(n);
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int y = n-x-z;
			SCAL t{};
			for (int m = 0; m <= n; ++m)
			{
				const SCAL *Mo = Mtuple + symmetricoffset3(n - m);
				SCAL c(0);
				for (int k1 = 0; k1 <= min(x, m); ++k1)
					for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
					{
						int k2 = m-k1-k3; // k2 < y => m-k1-k3 < y => k3 > m-k1-y
						int index = symmetric_i_x_z(x-k1, z-k3, n-m);
						c += binomial(x, k1) * binomial(y, k2) * binomial(z, k3)
						   * binarypow(d.x, k1) * binarypow(d.y, k2) * binarypow(d.z, k3) * Mo[index];
					}
				t += c * static_factorial(n-m);
			}
			Mout[i++] = C * t;
		}
}

inline __host__ __device__ void m2m_acc3(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
// O(n^5)
{
	int i = 0;
	SCAL C = inv_factorial(n);
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int y = n-x-z;
			SCAL t{};
			for (int m = 0; m <= n; ++m)
			{
				const SCAL *Mo = Mtuple + symmetricoffset3(n - m);
				SCAL c(0);
				for (int k1 = 0; k1 <= min(x, m); ++k1)
					for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
					{
						int k2 = m-k1-k3; // k2 < y => m-k1-k3 < y => k3 > m-k1-y
						int index = symmetric_i_x_z(x-k1, z-k3, n-m);
						c += binomial(x, k1) * binomial(y, k2) * binomial(z, k3)
						   * binarypow(d.x, k1) * binarypow(d.y, k2) * binarypow(d.z, k3) * Mo[index];
					}
				t += c * static_factorial(n-m);
			}
			Mout[i++] += C * t;
		}
}

inline __host__ __device__ void m2m_traceless_acc3(SCAL *__restrict__ Mout, SCAL *__restrict__ temp,
                                                   const SCAL *__restrict__ Mtuple, int n, VEC d, SCAL r)
// traceless multipole to multipole expansion + accumulate
// Mtuple is a tuple of traceless multipole tensors of orders from 0 to n (inclusive)
// returns a traceless multipole tensor of order n
// d is the (unit vector) shift from the old position to the new position
// temp is a temporary memory that needs at least 2*n+1 elements (independent for each thread)
// O(n^3)
{
	if (n == 0)
		Mout[0] += Mtuple[0];
	else
	{
		for (int m = 0; m <= n; ++m)
		{
			int i = 0;
			const SCAL *Mo = Mtuple + tracelessoffset3(n - m);
			SCAL C = inv_factorial(m);
			tracelesspow3(temp, m, d, r);
			for (int z = 0; z <= 1; ++z)
				for (int x = n-z; x >= 0; --x)
				{
					int y = n-x-z;
					SCAL t{};
					for (int k1 = 0; k1 <= min(x, m); ++k1)
						for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
							t += traceless_A_x_z(temp, k1, k3, m) * traceless_A_x_z(Mo, x-k1, z-k3, n-m);
					Mout[i++] += C * t;
				}
		}
	}
}

inline __host__ __device__ void m2l_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                         int nM, int nL, VEC d, SCAL r2, int minm = 0, int maxm = -1)
// symmetric multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least (maxm+1)*(maxm+2)/2 elements (independent for each thread)
// maxm = nM+nL by default
// O(maxm*nL^2*nM^2)
{
	r2 = sqrt(r2);
	d /= r2;
	maxm = (maxm == -1) ? nM+nL : maxm;
	for (int m = minm; m <= maxm; ++m)
	{
		gradient3(temp, m, d, r2);
		traceless_refine3(temp, m);
		for (int n = max(0, m-nM); n <= min(nL, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			SCAL C = inv_factorial(n);
			contract_traceless_ma3(Ltuple + tracelessoffset3(n), Mtuple + symmetricoffset3(mn), temp, C, mn, m);
		}
	}
}

template <bool b_atomic = false>
inline __host__ __device__ void m2l_traceless_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                                   int nM, int nL, VEC d, SCAL r2, int minm = 0)
// traceless multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least 2*(nM+nL)+1 elements (independent for each thread)
// O((nM+nL)*nL^2*nM^2)
{
	r2 = sqrt(r2);
	d /= r2;
	for (int m = minm; m <= nM+nL; ++m)
	{
		gradient3(temp, m, d, r2);
		for (int n = max(0, m-nM); n <= min(nL, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			SCAL C = inv_factorial(n);
			contract_traceless2_ma3<b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + tracelessoffset3(mn), temp, C, mn, m);
		}
	}
}

template<int n, int nmax, int m, bool traceless, bool b_atomic>
inline __host__ __device__ typename std::enable_if<(n > nmax), void>::type
	static_m2l_inner2_3(SCAL *, const SCAL *, const SCAL *)
{

}
template<int n, int nmax, int m, bool traceless, bool b_atomic>
inline __host__ __device__ typename std::enable_if<(n <= nmax), void>::type
	static_m2l_inner2_3(SCAL *__restrict__ Ltuple, const SCAL *__restrict__ grad, const SCAL *__restrict__ Mtuple)
// O(n^2 * m * nmax)
{
	constexpr int mn = m-n; // 0 <= mn <= nM
	SCAL C = inv_factorial(n);
	if (traceless)
		static_contract_traceless2_ma3<mn, m, b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + tracelessoffset3(mn), grad, C);
	else
		static_contract_traceless_ma3<mn, m, b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + symmetricoffset3(mn), grad, C);

	static_m2l_inner2_3<n+1, nmax, m, traceless, b_atomic>(Ltuple, grad, Mtuple);
}

template<int m, int N, bool b_atomic>
inline __host__ __device__ void static_m2l_inner_3(SCAL *, SCAL *, const SCAL *, VEC, SCAL)
{
	// TODO
}

template<int m, int N, bool b_atomic>
inline __host__ __device__ typename std::enable_if<(m > 2*N), void>::type
	static_m2l_inner_traceless3(SCAL *, SCAL *, const SCAL *, VEC, SCAL)
{
	
}
template<int m, int N, bool b_atomic>
inline __host__ __device__ typename std::enable_if<(m <= 2*N), void>::type
	static_m2l_inner_traceless3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
                                const SCAL *__restrict__ Mtuple, VEC d, SCAL r)
// O((m^3 + N^3 * m) * N)
{
	static_gradient3<m>(grad, d, r);
	static_m2l_inner2_3<static_max(0, m-N), static_min(N, m), m, true, b_atomic>(Ltuple, grad, Mtuple);

	static_m2l_inner_traceless3<m+1, N, b_atomic>(Ltuple, grad, Mtuple, d, r);
}

template<int n, int N>
inline __host__ __device__ typename std::enable_if<(n > N), void>::type
	static_m2l_refine3(SCAL *)
{
	
}
template<int n, int N>
inline __host__ __device__ typename std::enable_if<(n <= N), void>::type
	static_m2l_refine3(SCAL *Ltuple)
// O(n^2 * N)
{
	static_traceless_refine3<n>(Ltuple + symmetricoffset3(n));

	static_m2l_refine3<n+1, N>(Ltuple);
}

template<int N, int minm, bool traceless, bool b_atomic>
inline __host__ __device__ void static_m2l_acc3_(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
                                                 const SCAL *__restrict__ Mtuple, VEC d, SCAL r2)
// O(N^5) - traceless
{
	r2 = sqrt(r2);
	d /= r2;
	if (traceless)
		static_m2l_inner_traceless3<minm, N, b_atomic>(Ltuple, grad, Mtuple, d, r2);
	else
		static_m2l_inner_3<minm, N, b_atomic>(Ltuple, grad, Mtuple, d, r2);
}

template<int minm = 0, bool b_atomic = false>
inline __host__ __device__ void static_m2l_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp,
											   const SCAL *__restrict__ Mtuple, int N, VEC d, SCAL r2)
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
// O(N^5) - traceless
{
	switch (N)
	{
		case 0:
			static_m2l_acc3_<0, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 1:
			static_m2l_acc3_<1, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 2:
			static_m2l_acc3_<2, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 3:
			static_m2l_acc3_<3, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;
		/*case 4:
			static_m2l_acc3_<4, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 5:
			static_m2l_acc3_<5, minm, true, b_atomic>(Ltuple, temp, Mtuple, d, r2);
			break;*/
		default:
			m2l_traceless_acc3<b_atomic>(Ltuple, temp, Mtuple, N, N, d, r2, minm);
			break;
	}
}

inline __host__ __device__ void l2l_acc3(SCAL *__restrict__ Lout, SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
                                         int n, int nL, VEC d)
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the shift from the old position to the new position
// temp is a temporary memory that needs at least 2*(nL-n)+1 elements (independent for each thread)
{
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tensorpow3(temp, mn, d);
		SCAL C = binomial(m, mn);
		contract_traceless_ma3(Lout, Ltuple + symmetricoffset3(m), temp, C, m, mn);
		traceless_refine3(Lout, n);
	}
}

inline __host__ __device__ void l2l_traceless_acc3(SCAL *__restrict__ Lout, SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
                                                   int n, int nL, VEC d, SCAL r)
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the unit vector from the old position to the new position
// temp is a temporary memory that needs at least 2*(nL-n)+1 elements (independent for each thread)
// O(((nL-n)^3 + n * (nL-n)^2)*nL) ~ O(nL^4) for nL times
{
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tracelesspow3(temp, mn, d, r);
		SCAL C = binomial(m, mn);
		contract_traceless2_ma3(Lout, Ltuple + tracelessoffset3(m), temp, C, m, mn);
	}
}

inline __host__ __device__ SCAL m2p_pot3(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                         int nM, VEC d, SCAL r2)
// multipole to particle expansion (for potential)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nM+1)*(nM+2)/2 + 1 elements (independent for each thread)
// O(nM^4)
{
	SCAL pot(0);
	for (int n = 0; n <= nM; ++n)
	{
		gradient3(temp+1, n, d, r2);
		contract3(temp, Mtuple + symmetricoffset3(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

inline __host__ __device__ SCAL l2p_pot3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
                                         int nL, VEC d)
// local to particle expansion (for potential)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nL+1)*(nL+2)/2 + 1 elements (independent for each thread)
// O(nM^4)
{
	SCAL pot(0);
	for (int n = 0; n <= nL; ++n)
	{
		tensorpow3(temp+1, n, d);
		contract3(temp, Ltuple + symmetricoffset3(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

inline __host__ __device__ VEC m2p_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                          int nM, VEC d, SCAL r2)
// multipole to particle expansion (for field)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nM+2)*(nM+3)/2 + 3 elements (independent for each thread)
{
	VEC field{};
	for (int n = 0; n <= nM; ++n)
	{
		gradient3(temp+3, n+1, d, r2);
		contract3(temp, Mtuple + symmetricoffset3(n), temp+3, n, n+1);
		field.x -= temp[0];
		field.y -= temp[1];
		field.z -= temp[2];
	}
	return field;
}

inline __host__ __device__ VEC l2p_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
                                          int nL, VEC d)
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nL*(nL+1)/2 + 3 elements (independent for each thread)
{
	VEC field{};
	for (int n = 1; n <= nL; ++n)
	{
		tensorpow3(temp+3, n-1, d);
		contract3(temp, Ltuple + symmetricoffset3(n), temp+3, n, n-1);
		SCAL C = (SCAL)n;
		field.x -= C*temp[0];
		field.y -= C*temp[1];
		field.z -= C*temp[2];
	}
	return field;
}

inline __host__ __device__ VEC l2p_traceless_field3(SCAL *__restrict__ temp, const SCAL *Ltuple, int nL, VEC d, SCAL r)
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// d is a unit vector
// temp is a temporary memory that needs at least 2*nL+2 elements (independent for each thread)
// O(nL^4)
{
	for (int i = 0; i < 3; ++i)
		temp[i] = (SCAL)0;
	for (int n = 1; n <= nL; ++n)
	{
		tracelesspow3(temp+3, n-1, d, r);
		SCAL C = (SCAL)n;
		contract_traceless2_ma3(temp, Ltuple + tracelessoffset3(n), temp+3, C, n, n-1);
	}
	return VEC{-temp[0], -temp[1], -temp[2]};
}

#endif // !FMM_CART_BASE3_CUDA_H