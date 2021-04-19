//  Base functions used in Fast Multipole Method (FMM)
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

#ifndef FMM_CART_BASE_CUDA_H
#define FMM_CART_BASE_CUDA_H

#include "kernel.cuh"

inline __host__ __device__ SCAL coeff1(int n, int m)
// a coefficient used in the calculation of the gradient
// returns (-1)^m * (2n - 2m - 2)!!
// assumes m <= n/2 and n >= 1
// 2n - 2m - 2 must be smaller than 80 (otherwise bad things will occur)
{
	return (SCAL)paritysign(m) * static_edfactorial(2 * (n - m) - 2);
	// correct when DIM = 2, for DIM = 3 just add 1 inside dfactorial argument
	// for DIM > 3, see Shanker, Huang, 2007
}

inline __host__ __device__ SCAL coeff2(int n, int m)
// a coefficient used in the calculation of the gradient
// returns n! / (2^m * m! * (n - 2m)!)
// assumes m <= n/2
// n,m,(n - 2m) must be smaller than 40
{
	return static_factorial(n) * inv_power_of_2(m) * inv_factorial(m) * inv_factorial(n - 2*m);
}

inline __host__ __device__ long long coeff2_accurate(int n, int m)
// a coefficient used in the calculation of the gradient
// returns n! / (2^m * m! * (n - 2m)!) = n! / ((2m)!! * (n - 2m)!)
// assumes m <= n/2
{
	long long res = 1;
	for (long long num = n-2*m+1, den = 2; num <= n; den += 2)
	{
		res *= num++;
		res = (res * (num++)) / den;
	}
	return res;
}

/*

Let's define a (totally) symmetric tensor in 2 dimensions with order p. Then it has p+1 independent
coefficients.
Usually a tensor A is denoted with the notation:
	A_a,b,c,d...
where a,b,c,d... are indices that can take only two values since we are in 2D. However,
it would be pointless to store all 2^p elements. Since the tensor is symmetric, swapping
any two indices won't change the result. So, an element of a symmetric tensor is completely
determined by the multiplicities of the values that indices take. Then we use a new set of indices x,y
to denote the elements of a symmetric tensor A:
	A_x,y
where p = x + y. With just 2 indices we can notate symmetric tensors of any order in 2 dimensions.
It is usually convenient in a program to have just one index for an array. We use an index that
corresponds to the multiplicities x,y through an 1-1 relationship:

x,y	| i	| j	| p
0,0	| 0	| 0	| 0
1,0	| 0	| 1	| 1
0,1	| 1	| 2	| 1
2,0	| 0	| 3	| 2
1,1	| 1	| 4	| 2
0,2	| 2	| 5	| 2
3,0	| 0	| 6	| 3
2,1	| 1	| 7	| 3
1,2	| 2	| 8	| 3
0,3	| 3	| 9	| 3
...
where 0 <= i <= p is an index for a p-order symmetric tensor, and j is a cumulative index, useful if we
merge a set of tensors of increasing order in a single array. With this notation there is a specific
relationship between x,y and i, which is:
y = i, x = p - i
When converting from i to j or viceversa one has to take into account an offset that depends on p:
j = i + p*(p+1)/2
Note that p=0 (j=0) are the monopoles, p=1 (j=1,2) are dipoles, p=2 (j=3,4,5) quadrupoles etc...
In higher dimensions the number of indipendent elements increase faster with p (for example in 3D
there are (p+1)(p+2)/2 indipendent elements for a p-order symmetric tensor).

Using totally traceless symmetric tensors drastically decreases the number of indipendent elements, and
in 2D we would have just 2 elements for any p >= 1. In this case we will have:
A_(x+2),y + A_x,(y+2) = 0
where A is a tensor of order x + y + 2. This can be written with the index notation as:
A_(i+2) + A_i = 0      =>   A_i = -A_(i-2)
In practice, if we must build a totally traceless symmetric tensor, we will actually have to compute explicitly
only its first two elements and the other ones will be given by the obtained recursive relations, which are
very fast to calculate.

An important property is that if we contract a symmetric tensor with a traceless symmetric tensor, the
result will be a symmetric traceless tensor. So we can save computation time by using the previous relation.

A tuple of 2D symmetric tensors with order from 0 to P (inclusive) will have (P+1)*(P+2)/2 elements.
In the following calculations we will use cartesian coordinates.

*/

constexpr __host__ __device__ int tensortupleoffset(int p)
{
	return p * (p + 1) / 2;
}

constexpr __host__ __device__ int tracelessoffset(int p)
{
	return (p == 0) ? 0 : (2 * p - 1);
}

inline __host__ __device__ void trace(SCAL *__restrict__ out, const SCAL *__restrict__ in, int n, int m)
// contract 2m indices in n-order tensor "in" giving (n-2m)-order tensor "out"
// assumes n >= 2m
// if n = 2m the result is a scalar
{
	int n_out = n - 2 * m;
	for (int i = 0; i <= n_out; ++i)
	{
		SCAL t(0);
		const SCAL *ini = in + i;
		for (int k = 0; k <= m; ++k)
			t += (SCAL)binomial(m, k) * ini[2*k];
		out[i] = t;
	}
}

template <typename T>
inline __host__ __device__ void swap(T &a, T &b)
{
    T c(a); a = b; b = c;
}

inline __host__ __device__ void contract(SCAL *__restrict__ C, const SCAL *__restrict__ A,
															   const SCAL *__restrict__ B,
										 int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor C of order |nA-nB|
// if nA = nB the result is a scalar
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // at this point we assume nA >= nB
	for (int i = 0; i <= nC; ++i)
	{
		SCAL t(0);
		const SCAL *Ai = A + i;
		for (int k = 0; k <= nB; ++k)
			t += (SCAL)binomial(nB, k) * Ai[k] * B[k];
		C[i] = t;
	}
}

inline __host__ __device__ void contract_acc(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																   const SCAL *__restrict__ B,
											 int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// and sum (accumulate) the result to C
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	for (int i = 0; i <= nC; ++i)
	{
		SCAL t(0);
		const SCAL *Ai = A + i;
		for (int k = 0; k <= nB; ++k)
			t += (SCAL)binomial(nB, k) * Ai[k] * B[k];
		C[i] += t;
	}
}

inline __host__ __device__ void contract_ma(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																  const SCAL *__restrict__ B,
											SCAL c, int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	for (int i = 0; i <= nC; ++i)
	{
		SCAL t(0);
		const SCAL *Ai = A + i;
		for (int k = 0; k <= nB; ++k)
			t += (SCAL)binomial(nB, k) * Ai[k] * B[k];
		C[i] += c*t;
	}
}

inline __host__ __device__ void contract_traceless_ma(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																			const SCAL *__restrict__ B,
													  SCAL c, int nA, int nB)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// will be built a posteriori through the function traceless_refine
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	for (int i = 0; i <= min(1, nC); ++i)
	{
		SCAL t(0);
		const SCAL *Ai = A + i;
		for (int k = 0; k <= nB; ++k)
			t += (SCAL)binomial(nB, k) * Ai[k] * B[k];
		C[i] += c*t;
	}
}

inline __host__ __device__ void contract_traceless2_ma(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																			 const SCAL *__restrict__ B,
													   SCAL c, int nA, int nB)
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result
{
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	if (nB >= 1)
	{
		SCAL t = c * power_of_2(nB - 1);
		C[0] += t * (A[0] * B[0] + A[1] * B[1]);
		if (nC >= 1)
			C[1] += t * (A[1] * B[0] - A[0] * B[1]);
	}
	else
	{
		C[0] += c * A[0] * B[0];
		if (nC >= 1)
			C[1] += c * A[1] * B[0];
	}
}

template<typename T>
constexpr __host__ __device__ T static_min(T a, T b)
{
	return (a < b) ? a : b;
}
template<typename T>
constexpr __host__ __device__ T static_max(T a, T b)
{
	return (a < b) ? b : a;
}

template<int nA, int nB>
inline __host__ __device__ void static_contract_traceless_ma(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																				   const SCAL *__restrict__ B,
															 SCAL c)
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
{
	if (nA < nB)
		static_contract_traceless_ma<nB, nA>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		for (int i = 0; i <= static_min(1, nC); ++i)
		{
			SCAL t(0);
			const SCAL *Ai = A + i;
			for (int k = 0; k <= nB; ++k)
				t += (SCAL)binomial(nB, k) * Ai[k] * B[k];
			C[i] += c*t;
		}
	}
}

template<int nA, int nB>
inline __host__ __device__ void static_contract_traceless2_ma(SCAL *__restrict__ C, const SCAL *__restrict__ A,
																					const SCAL *__restrict__ B,
															  SCAL c)
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result
{
	if (nA < nB)
		static_contract_traceless2_ma<nB, nA>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		if (nB >= 1)
		{
			SCAL t = c * power_of_2(nB - 1);
			C[0] += t * (A[0] * B[0] + A[1] * B[1]);
			if (nC >= 1)
				C[1] += t * (A[1] * B[0] - A[0] * B[1]);
		}
		else
		{
			C[0] += c * A[0] * B[0];
			if (nC >= 1)
				C[1] += c * A[1] * B[0];
		}
	}
}

inline __host__ __device__ void traceless_refine(SCAL *A, int n)
// build a traceless tensor from the first 2 elements
{
	for (int i = 2; i <= n; ++i)
		A[i] = -A[i-2];
}

template<int n>
inline __host__ __device__ void static_traceless_refine(SCAL *A)
// build a traceless tensor from the first 2 elements
{
	for (int i = 2; i <= n; ++i)
		A[i] = -A[i-2];
}

inline __host__ __device__ void gradient_exact(SCAL *grad, int n, VEC d, SCAL r2)
// calculate the gradient of -log(r) of order n i.e. -nabla^n log(r), which is a n-order symmetric tensor.
// NOTE that if r2 >> EPS2, it is also totally traceless, so it has just 2 independent elements
// for n >= 1 (just 1 for n = 0)
// r2 = r^2 = dot(d,d) + EPS2
// this version works even for big EPS2, in which case the gradient is not traceless
{
	if (n == 0)
		grad[0] = -(SCAL)0.5*log(r2); // -log(r) = -log(sqrt(r^2)) = -0.5 * log(r^2)
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r2, -n);
		for (int i = 0; i <= n; ++i)
		{
			int j = n-i;
			SCAL t(0);
			for (int k1 = 0; k1 <= i/2; ++k1)
				for (int k2 = 0; k2 <= j/2; ++k2)
				{
					int m = k1+k2;
					t += (SCAL)coeff1(n, m) * coeff2(i, k1) * coeff2(j, k2) * binarypow(r2, m)
					   * binarypow(d.y, i - 2*k1) * binarypow(d.x, j - 2*k2);
				}
			grad[i] = C * t;
		}
	}
}

inline __host__ __device__ void gradient(SCAL *grad, int n, VEC d, SCAL r)
// calculate the gradient of -log(r) of order n i.e. -nabla^n log(r), which is a n-order symmetric tensor.
// NOTE that if r2 >> EPS2, it is also totally traceless, so it has just 2 independent elements
// for n >= 1 (just 1 for n = 0)
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^3) to O(n))
// if this assumption does not hold, accuracy will be reduced
{
	if (n == 0)
		grad[0] = -log(r);
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n);
		for (int i = 0; i <= 1; ++i)
		{
			int j = n-i;
			SCAL t{};
			for (int m = 0; m <= j/2; ++m)
				t += (SCAL)coeff1(n, m) * coeff2(j, m) * binarypow(d.x, j - 2*m);
			grad[i] = C * t * binarypow(d.y, i);
		}
	}
}

template<int n>
inline __host__ __device__ void static_gradient(SCAL *grad, VEC d, SCAL r)
// calculate the gradient of -log(r) of order n i.e. -nabla^n log(r), which is a n-order symmetric tensor.
// NOTE that if r2 >> EPS2, it is also totally traceless, so it has just 2 independent elements
// for n >= 1 (just 1 for n = 0)
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^3) to O(n))
// if this assumption does not hold, accuracy will be reduced
{
	if (n == 0)
		grad[0] = -log(r);
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n);
		for (int i = 0; i <= 1; ++i)
		{
			int j = n-i;
			SCAL t{};
			for (int m = 0; m <= j/2; ++m)
				t += (SCAL)coeff1(n, m) * coeff2(j, m) * binarypow(d.x, j - 2*m);
			grad[i] = C * t * binarypow(d.y, i);
		}
	}
}

inline __host__ __device__ void tensorpow(SCAL *power, int n, VEC d)
// calculate the powers tensor of d of order n, e.g.:
// r^(0) = 1
// r^(1) = (x, y)
// r^(2) = (x^2, xy, y^2)
// r^(3) = (x^3, x^2 y, xy^2, y^3)
// etc...
{
	for (int i = 0; i <= n; ++i)
		power[i] = binarypow(d.y, i) * binarypow(d.x, n - i);
}

inline __host__ __device__ void tracelesspow(SCAL *power, int n, VEC d, SCAL r)
{
	if (n == 0)
		power[0] = (SCAL)1;
	else
	{
		SCAL C = binarypow(r, n) / static_edfactorial(2*n-2);
		for (int i = 0; i <= 1; ++i)
		{
			SCAL t{};
			int j = n-i;
			for (int m = 0; m <= j/2; ++m)
				t += (SCAL)coeff1(n, m) * coeff2(j, m) * binarypow(d.x, j - 2*m);
			power[i] = C * t * binarypow(d.y, i);
		}
	}
}

inline __host__ __device__ void p2m(SCAL *M, int n, VEC d)
// particle to multipole expansion of order n
// d is the coordinate of the particle from the (near) expansion center
{
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	for (int i = 0; i <= n; ++i)
		M[i] = C * binarypow(d.y, i) * binarypow(d.x, n - i);
}

inline __host__ __device__ void p2m_acc(SCAL *M, int n, VEC d)
// particle to multipole expansion of order n + accumulate to M
// d is the coordinate of the particle from the (near) expansion center
{
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	for (int i = 0; i <= n; ++i)
		M[i] += C * binarypow(d.y, i) * binarypow(d.x, n - i);
}

inline __host__ __device__ void p2m_traceless_acc(SCAL *M, int n, VEC d, SCAL r)
// particle to multipole expansion of order n + accumulate to M
// d is the unit vector of the coordinate of the particle from the (near) expansion center
{
	if (n == 0)
		M[0] += (SCAL)1;
	else
	{
		SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * binarypow(r, n) / static_edfactorial(2*n-2);
		for (int i = 0; i <= 1; ++i)
		{
			SCAL t{};
			int j = n-i;
			for (int m = 0; m <= j/2; ++m)
				t += (SCAL)coeff1(n, m) * coeff2(j, m) * binarypow(d.x, j - 2*m);
			M[i] += C * t * binarypow(d.y, i);
		}
	}
}

inline __host__ __device__ void p2l(SCAL *L, int n, VEC d, SCAL r)
// particle to local expansion of order n
// d is the unit vector of the coordinate of the particle from the (far) expansion center
{
	gradient(L, n, d, r);
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	for (int i = 0; i <= 1; ++i)
		L[i] *= C;
	traceless_refine(L, n);
}

inline __host__ __device__ void m2m(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
// multipole to multipole expansion
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
{
	for (int i = 0; i <= n; ++i)
	{
		int j = n-i;
		SCAL t{};
		for (int m = 0; m <= n; ++m)
		{
			const SCAL *Mo = Mtuple + tensortupleoffset(n - m);
			SCAL c(0);
			for (int k = max(0, m-j); k <= min(i, m); ++k)
			{
				int l = m-k;
				c += binomial(i, k) * binomial(j, l) * binarypow(d.y, k) * binarypow(d.x, l) * Mo[i - k];
			}
			t += c * (static_factorial(n-m) * inv_factorial(n));
		}
		Mout[i] = t;
	}
}

inline __host__ __device__ void m2m_acc(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
{
	for (int i = 0; i <= n; ++i)
	{
		int j = n-i;
		SCAL t{};
		for (int m = 0; m <= n; ++m)
		{
			const SCAL *Mo = Mtuple + tensortupleoffset(n - m);
			SCAL c(0);
			for (int k = max(0, m-j); k <= min(i, m); ++k)
			{
				int l = m-k; // 0 <= l <= min(n-i, m) // l + k = m
				// j >= l  ==>  j >= m-k  ==>  k >= m-j
				c += binomial(i, k) * binomial(j, l) * binarypow(d.y, k) * binarypow(d.x, l) * Mo[i - k];
			}
			t += c * (static_factorial(n-m) * inv_factorial(n));
		}
		Mout[i] += t;
	}
}

inline __host__ __device__ void m2m_traceless_wrong_acc(SCAL *__restrict__ Mout,
												  const SCAL *__restrict__ Mtuple, int n, VEC d, SCAL r)
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the (unit vector) shift from the old position to the new position
{
	const SCAL *Mo = Mtuple + tracelessoffset(n);
	SCAL power[2];
	SCAL t[2]{};
	t[0] += Mo[0];
	if (n >= 1)
	{
		t[1] += Mo[1];
		SCAL u = inv_factorial(n);
		Mo = Mtuple + tracelessoffset(0);
		tracelesspow(power, n, d, r);
		t[0] += power[0] * Mo[0] * u;
		t[1] += power[1] * Mo[0] * u;
	}
	for (int m = 1; m <= n-1; ++m)
	{
		SCAL u = static_factorial(n-m) * inv_factorial(n);
		Mo = Mtuple + tracelessoffset(n - m);
		tracelesspow(power, m, d, r);
		t[0] += binomial(n, m) * power[0] * Mo[0] * u;
		t[1] += (binomial(n-1, m) * power[0] * Mo[1] + binomial(n-1, m-1) * power[1] * Mo[0]) * u;
	}
	for (int i = 0; i <= min(1, n); ++i)
		Mout[i] += t[i];
}

inline __host__ __device__ void m2m_traceless_acc(SCAL *__restrict__ Mout, SCAL *__restrict__ temp,
												  const SCAL *__restrict__ Mtuple, int n, VEC d, SCAL r)
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the (unit vector) shift from the old position to the new position
// temp is a temporary memory that needs at least n+1 elements (independent for each thread)
{
	if (n == 0)
		Mout[0] += Mtuple[0];
	else
	{
		SCAL power[2];
		for (int i = 0; i <= n; ++i)
			temp[i] = (SCAL)0;
		for (int m = 0; m <= n; ++m)
		{
			SCAL C = static_factorial(n-m) * inv_factorial(n);
			const SCAL *Mo = Mtuple + tensortupleoffset(n - m);
			tracelesspow(power, m, d, r);
			for (int i = 0; i <= n; i += 2)
			{
				int j = n-i;
				int sig = i>>1;
				SCAL c{};
				int k0 = max(0, m-j), accum = 0, kmax = min(i, m);
				for (int k = k0; k <= kmax; k += 2)
					accum += binomial(i, k) * binomial(j, m-k);
				accum *= paritysign(sig+k0);
				int elem = k0 & 1;
				c += accum * power[elem] * Mo[elem];
				accum = 0;
				++k0;
				for (int k = k0; k <= kmax; k += 2)
					accum += binomial(i, k) * binomial(j, m-k);
				accum *= paritysign(sig+k0);
				elem = k0 & 1;
				c += accum * power[elem] * Mo[elem];
				temp[i] += c * C;
			}
			for (int i = 1; i <= n; i += 2)
			{
				int j = n-i;
				SCAL c{};
				int k0 = max(0, m-j), accum = 0, kmax = min(i, m);
				for (int k = k0; k <= kmax; k += 2)
					accum += binomial(i, k) * binomial(j, m-k);
				int elem1 = k0 & 1, elem2 = !elem1;
				c += accum * power[elem1] * Mo[elem2];
				accum = 0;
				++k0;
				for (int k = k0; k <= kmax; k += 2)
					accum += binomial(i, k) * binomial(j, m-k);
				elem1 = k0 & 1;
				elem2 = !elem1;
				c += accum * power[elem1] * Mo[elem2];
				temp[i] += c * C * paritysign(i>>1);
			}
		}
		SCAL t{};
		for (int m = 0; m <= n / 2; ++m)
		{
			SCAL c{};
			for (int s = 0; s <= m; ++s)
				c += binomial(m, s) * temp[s*2];
			t += c * coeff1(n, m) * coeff2(n, m);
		}
		Mout[0] += t;
		t = 0;
		int j = n-1;
		for (int m = 0; m <= j / 2; ++m)
		{
			SCAL c{};
			for (int s = 0; s <= m; ++s)
				c += binomial(m, s) * temp[s*2+1];
			t += c * coeff1(n, m) * coeff2(j, m);
		}
		Mout[1] += t;
	}
}

inline __host__ __device__ void m2l(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp,
									const SCAL *__restrict__ Mtuple, int nM, int nL, VEC d, SCAL r2)
// multipole to local expansion
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least nM+2 elements (independent for each thread)
{
	r2 = sqrt(r2);
	d /= r2;
	for (int i = 0; i < tracelessoffset(nL+1); ++i)
		Ltuple[i] = (SCAL)0;
	for (int m = 0; m <= nM+nL; ++m)
	{
		gradient(temp, m, d, r2);
		traceless_refine(temp, min(m, nM+1));
		for (int n = max(0, m-nM); n <= min(nL, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			SCAL C = inv_factorial(n);
			contract_traceless_ma(Ltuple + tracelessoffset(n), Mtuple + tensortupleoffset(mn), temp, C, mn, m);
		}
	}
}

inline __host__ __device__ void m2l_acc(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp,
										const SCAL *__restrict__ Mtuple, int nM, int nL, VEC d, SCAL r2,
										int minm = 0)
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least nM+2 elements (independent for each thread)
{
	r2 = sqrt(r2);
	d /= r2;
	for (int m = minm; m <= nM+nL; ++m)
	{
		gradient(temp, m, d, r2);
		traceless_refine(temp, min(m, nM+1));
		for (int n = max(0, m-nM); n <= min(nL, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			SCAL C = inv_factorial(n);
			contract_traceless_ma(Ltuple + tracelessoffset(n), Mtuple + tensortupleoffset(mn), temp, C, mn, m);
		}
	}
}

inline __host__ __device__ void m2l_traceless_acc(SCAL *__restrict__ Ltuple, const SCAL *__restrict__ Mtuple,
												  int nM, int nL, VEC d, SCAL r2, int minm = 0)
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
{
	SCAL grad[2];
	r2 = sqrt(r2);
	d /= r2;
	for (int m = minm; m <= nM+nL; ++m)
	{
		gradient(grad, m, d, r2);
		for (int n = max(0, m-nM); n <= min(nL, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			SCAL C = inv_factorial(n);
			contract_traceless2_ma(Ltuple + tracelessoffset(n), Mtuple + tracelessoffset(mn), grad, C, mn, m);
		}
	}
}

template<int n, int nmax, int m, bool traceless>
inline __host__ __device__ std::enable_if<(n <= nmax), void>::type
	static_m2l_inner2(SCAL *__restrict__ Ltuple, const SCAL *__restrict__ grad, const SCAL *__restrict__ Mtuple)
{
	constexpr int mn = m-n; // 0 <= mn <= nM
	SCAL C = inv_factorial(n);
	if (traceless)
		static_contract_traceless2_ma<mn, m>(Ltuple + tracelessoffset(n), Mtuple + tracelessoffset(mn), grad, C);
	else
		static_contract_traceless_ma<mn, m>(Ltuple + tracelessoffset(n), Mtuple + tensortupleoffset(mn), grad, C);

	static_m2l_inner2<n+1, nmax, m, traceless>(Ltuple, grad, Mtuple);
}

template<int n, int nmax, int m, bool traceless>
inline __host__ __device__ std::enable_if<(n > nmax), void>::type
	static_m2l_inner2(SCAL *, const SCAL *, const SCAL *)
{

}

template<int m, int N, bool traceless>
inline __host__ __device__ std::enable_if<(m <= 2*N), void>::type
	static_m2l_inner(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
					 const SCAL *__restrict__ Mtuple, VEC d, SCAL r)
{
	static_gradient<m>(grad, d, r);
	if (!traceless)
		static_traceless_refine<static_min(m, N+1)>(grad);
	static_m2l_inner2<static_max(0, m-N), static_min(N, m), m, traceless>(Ltuple, grad, Mtuple);

	static_m2l_inner<m+1, N, traceless>(Ltuple, grad, Mtuple, d, r);
}
template<int m, int N, bool traceless>
inline __host__ __device__ std::enable_if<(m > 2*N), void>::type
	static_m2l_inner(SCAL *, SCAL *, const SCAL *, VEC, SCAL)
{
	
}

template<int n, int N>
inline __host__ __device__ std::enable_if<(n <= N), void>::type
	static_m2l_refine(SCAL *Ltuple)
{
	static_traceless_refine<n>(Ltuple + tensortupleoffset(n));

	static_m2l_refine<n+1, N>(Ltuple);
}
template<int n, int N>
inline __host__ __device__ std::enable_if<(n > N), void>::type
	static_m2l_refine(SCAL *)
{
	
}

template<int N, int minm>
inline __host__ __device__ void static_m2l_acc_(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
												const SCAL *__restrict__ Mtuple, VEC d, SCAL r2)
{
	r2 = sqrt(r2);
	d /= r2;
	static_m2l_inner<minm, N, false>(Ltuple, grad, Mtuple, d, r2);
}

template<int N, int minm>
inline __host__ __device__ void static_m2l_traceless_acc_(SCAL *__restrict__ Ltuple,
														  const SCAL *__restrict__ Mtuple, VEC d, SCAL r2)
{
	SCAL grad[2];
	r2 = sqrt(r2);
	d /= r2;
	static_m2l_inner<minm, N, true>(Ltuple, grad, Mtuple, d, r2);
}

template<int minm = 0>
inline __host__ __device__ void static_m2l_acc(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp,
											   const SCAL *__restrict__ Mtuple, int N, VEC d, SCAL r2)
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the position of the local expansion (L) w.r.t. the multipole (M)
{
	switch (N)
	{
		case 0:
			static_m2l_acc_<0, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 1:
			static_m2l_acc_<1, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 2:
			static_m2l_acc_<2, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 3:
			static_m2l_acc_<3, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 4:
			static_m2l_acc_<4, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 5:
			static_m2l_acc_<5, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		/*case 6:
			static_m2l_acc_<6, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 7:
			static_m2l_acc_<7, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 8:
			static_m2l_acc_<8, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 9:
			static_m2l_acc_<9, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 10:
			static_m2l_acc_<10, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		/*case 11:
			static_m2l_acc_<11, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 12:
			static_m2l_acc_<12, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 13:
			static_m2l_acc_<13, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 14:
			static_m2l_acc_<14, minm>(Ltuple, temp, Mtuple, d, r2);
			break;
		case 15:
			static_m2l_acc_<15, minm>(Ltuple, Mtuple, d, r2);
			break;*/ // uncomment only when needed, because this will rise compilation times significantly
		default:
			m2l_acc(Ltuple, temp, Mtuple, N, N, d, r2, minm);
			break;
	}
}

inline __host__ __device__ void l2l(SCAL *__restrict__ Lout, SCAL *__restrict__ temp,
									const SCAL *__restrict__ Ltuple, int n, int nL, VEC d)
// local to local expansion
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the shift from the old position to the new position
// temp is a temporary memory that needs at least nL-n+1 elements (independent for each thread)
{
	for (int i = 0; i <= n; ++i)
		Lout[i] = (SCAL)0;
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tensorpow(temp, mn, d);
		SCAL C = binomial(m, mn);
		contract_traceless_ma(Lout, Ltuple + tensortupleoffset(m), temp, C, m, mn);
		traceless_refine(Lout, n);
	}
}

inline __host__ __device__ void l2l_acc(SCAL *__restrict__ Lout, SCAL *__restrict__ temp,
										const SCAL *__restrict__ Ltuple, int n, int nL, VEC d)
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the shift from the old position to the new position
// temp is a temporary memory that needs at least nL-n+1 elements (independent for each thread)
{
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tensorpow(temp, mn, d);
		SCAL C = binomial(m, mn);
		contract_traceless_ma(Lout, Ltuple + tensortupleoffset(m), temp, C, m, mn);
		traceless_refine(Lout, n);
	}
}

inline __host__ __device__ void l2l_traceless_acc(SCAL *__restrict__ Lout, const SCAL *__restrict__ Ltuple,
												  int n, int nL, VEC d, SCAL r)
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the shift from the old position to the new position
{
	SCAL power[2];
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tracelesspow(power, mn, d, r);
		SCAL C = binomial(m, mn);
		contract_traceless2_ma(Lout, Ltuple + tracelessoffset(m), power, C, m, mn);
	}
}

inline __host__ __device__ SCAL m2p_pot(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
										int nM, VEC d, SCAL r2)
// multipole to particle expansion (for potential)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nM+2 elements (independent for each thread)
{
	SCAL pot(0);
	for (int n = 0; n <= nM; ++n)
	{
		gradient(temp+1, n, d, r2);
		contract(temp, Mtuple + tensortupleoffset(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

inline __host__ __device__ SCAL l2p_pot(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
										int nL, VEC d)
// local to particle expansion (for potential)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nL+2 elements (independent for each thread)
{
	SCAL pot(0);
	for (int n = 0; n <= nL; ++n)
	{
		tensorpow(temp+1, n, d);
		contract(temp, Ltuple + tensortupleoffset(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

inline __host__ __device__ VEC m2p_field(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
										 int nM, VEC d, SCAL r2)
// multipole to particle expansion (for field)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nM+4 elements (independent for each thread)
{
	VEC field{};
	for (int n = 0; n <= nM; ++n)
	{
		gradient(temp+2, n+1, d, r2);
		contract(temp, Mtuple + tensortupleoffset(n), temp+2, n, n+1);
		field.x -= temp[0];
		field.y -= temp[1];
	}
	return field;
}

inline __host__ __device__ VEC l2p_field(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
										 int nL, VEC d)
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nL+2 elements (independent for each thread)
{
	VEC field{};
	for (int n = 1; n <= nL; ++n)
	{
		tensorpow(temp+2, n-1, d);
		contract(temp, Ltuple + tensortupleoffset(n), temp+2, n, n-1);
		SCAL C = (SCAL)n;
		field.x -= C*temp[0];
		field.y -= C*temp[1];
	}
	return field;
}

inline __host__ __device__ VEC l2p_traceless_field(const SCAL *Ltuple, int nL, VEC d, SCAL r)
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// d is a unit vector
{
	SCAL power[2], temp[2];
	for (int i = 0; i < 2; ++i)
		temp[i] = (SCAL)0;
	for (int n = 1; n <= nL; ++n)
	{
		tracelesspow(power, n-1, d, r);
		SCAL C = (SCAL)n;
		contract_traceless2_ma(temp, Ltuple + tracelessoffset(n), power, C, n, n-1);
	}
	return VEC{-temp[0], -temp[1]};
}

#endif // !FMM_CART_BASE_CUDA_H