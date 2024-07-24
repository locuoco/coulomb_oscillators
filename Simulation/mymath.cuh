//  Some mathematical functions and constant values
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

#ifndef KERNEL_CUDA_H
#error "This CUDA header file can be included only from file kernel.cuh"
#else

// host-side constants

constexpr SCAL pi = (SCAL)3.1415926535897932384626433832795L;
constexpr SCAL twopi = (SCAL)6.283185307179586476925286766559L;

constexpr SCAL h_power_of_2[]{ // from 0 to 40
	1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536, // 2^16
	131072,262144,524288,1048576,2097152,4194304,8388608,16777216, // 2^24
	33554432,67108864,134217728,268435456,536870912,1073741824, // 2^30
	2147483648,4294967296,8589934592,17179869184,34359738368, // 2^35
	68719476736,137438953472,274877906944,549755813888,1099511627776, // 2^40
};
constexpr SCAL h_inv_power_of_2[]{ // from 0 to 40
	1, .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625, // 2^-8
	.001953125, .0009765625, .00048828125, .000244140625, .0001220703125, // 2^-13
	.00006103515625, .000030517578125, .0000152587890625, // 2^-16
	.00000762939453125, .000003814697265625, .0000019073486328125, // 2^-19
	.00000095367431640625, .000000476837158203125, .0000002384185791015625, // 2^-22
	.00000011920928955078125, .000000059604644775390625, // 2^-24
	.0000000298023223876953125, .00000001490116119384765625, // 2^-26
	.000000007450580596923828125, .0000000037252902984619140625, // 2^-28
	.00000000186264514923095703125, .000000000931322574615478515625, // 2^-30
	.0000000004656612873077392578125, .00000000023283064365386962890625, // 2^-32
	1.16415321826934814453125e-10, 5.82076609134674072265625e-11, // 2^-34
	2.910383045673370361328125e-11, 1.4551915228366851806640625e-11, // 2^-36
	7.2759576141834259033203125e-12, 3.63797880709171295166015625e-12, // 2^-38
	1.818989403545856475830078125e-12, 9.094947017729282379150390625e-13, // 2^-40
};
constexpr SCAL h_factorial[]{ // from 0 to 33
	1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, // 10!
	39916800.L, 479001600.L, 6227020800.L, 87178291200.L, 1307674368000.L, // 15!
	20922789888000.L, 355687428096000.L, 6402373705728000.L, // 18!
	121645100408832000.L, 2432902008176640000.L, 51090942171709440000.L, // 21!
	1124000727777607680000.L, 25852016738884976640000.L, 620448401733239439360000.L, // 24!
	15511210043330985984000000.L, 403291461126605635584000000.L, // 26!
	10888869450418352160768000000.L, 304888344611713860501504000000.L, // 28!
	8841761993739701954543616000000.L, 265252859812191058636308480000000.L, // 30!
	8222838654177922817725562880000000.L, 263130836933693530167218012160000000.L, // 32!
	8683317618811886495518194401280000000.L, // 33!
};
constexpr SCAL h_inv_factorial[]{ // from 0 to 33
	1, 1, .5, (long double)1/6, (long double)1/24, (long double)1/120, // 1/5!
	(long double)1/720, (long double)1/5040, (long double)1/40320, // 1/8!
	(long double)1/362880, (long double)1/3628800, (long double)1/39916800, // 1/11!
	(long double)1/479001600, (long double)1/6227020800, // 1/13!
	(long double)1/87178291200, (long double)1/1307674368000, // 1/15!
	(long double)1/20922789888000, (long double)1/355687428096000, // 1/17!
	(long double)1/6402373705728000, (long double)1/121645100408832000, // 1/19!
	(long double)1/2432902008176640000, 1/51090942171709440000.L, // 1/21!
	1/1124000727777607680000.L, 1/25852016738884976640000.L, 1/620448401733239439360000.L, // 1/24!
	1/15511210043330985984000000.L, 1/403291461126605635584000000.L, // 1/26!
	1/10888869450418352160768000000.L, 1/304888344611713860501504000000.L, // 1/28!
	1/8841761993739701954543616000000.L, 1/265252859812191058636308480000000.L, // 1/30!
	1/8222838654177922817725562880000000.L, 1/263130836933693530167218012160000000.L, // 1/32!
	1/8683317618811886495518194401280000000.L, // 1/33!
};
// even double factorials
constexpr SCAL h_edfactorial[]{ // from 0 to 56
	1, 2, 8, 48, 384, 3840, 46080, 645120, 10321920, // 16!!
	185794560.L, 3715891200.L, 81749606400.L, 1961990553600.L, // 24!!
	51011754393600.L, 1428329123020800.L,  42849873690624000.L, // 30!!
	1371195958099968000.L, 46620662575398912000.L, 1678343852714360832000.L, // 36!!
	63777066403145711616000.L, 2551082656125828464640000.L, // 40!!
	107145471557284795514880000.L, 4714400748520531002654720000.L, // 44!!
	216862434431944426122117120000.L, 10409396852733332453861621760000.L, // 48!!
	5.20469842636666622693081088e+32L, 2.7064431817106664380040216576e+34L, // 52!!
	1.461479318123759876522171695104e+36L, 8.1842841814930553085241614925824e+37L, // 56!!
};
// odd double factorials
constexpr SCAL h_odfactorial[]{ // from 1 to 55
	1, 3, 15, 105, 945, 10395, 135135, 2027025, (SCAL)34459425, // 17!!
	654729075.L, 13749310575.L, 316234143225.L, 7905853580625.L, // 25!!
	213458046676875.L, 6190283353629375.L, 191898783962510625.L, // 31!!
	6332659870762850625.L, 221643095476699771875.L, 8200794532637891559375.L, // 37!!
	319830986772877770815625.L, 13113070457687988603440625.L, // 41!!
	563862029680583509947946875.L, 25373791335626257947657609375.L, // 45!!
	1192568192774434123539907640625.L, 58435841445947272053455474390625.L, // 49!!
	2.9802279137433108747262291939219e+33L, 1.5795207942839547636049014727786e+35L, // 53!!
	8.6873643685617511998269581002823e+36L, // 55!!
};

// device-side constants

__device__ const SCAL d_power_of_2[]{ // from 0 to 40
	1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536, // 2^16
	131072,262144,524288,1048576,2097152,4194304,8388608,16777216, // 2^24
	33554432,67108864,134217728,268435456,536870912,1073741824, // 2^30
	2147483648,4294967296,8589934592,17179869184,34359738368, // 2^35
	68719476736,137438953472,274877906944,549755813888,1099511627776, // 2^40
};
__device__ const SCAL d_inv_power_of_2[]{ // from 0 to 40
	1, .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625, // 2^-8
	.001953125, .0009765625, .00048828125, .000244140625, .0001220703125, // 2^-13
	.00006103515625, .000030517578125, .0000152587890625, // 2^-16
	.00000762939453125, .000003814697265625, .0000019073486328125, // 2^-19
	.00000095367431640625, .000000476837158203125, .0000002384185791015625, // 2^-22
	.00000011920928955078125, .000000059604644775390625, // 2^-24
	.0000000298023223876953125, .00000001490116119384765625, // 2^-26
	.000000007450580596923828125, .0000000037252902984619140625, // 2^-28
	.00000000186264514923095703125, .000000000931322574615478515625, // 2^-30
	.0000000004656612873077392578125, .00000000023283064365386962890625, // 2^-32
	1.16415321826934814453125e-10, 5.82076609134674072265625e-11, // 2^-34
	2.910383045673370361328125e-11, 1.4551915228366851806640625e-11, // 2^-36
	7.2759576141834259033203125e-12, 3.63797880709171295166015625e-12, // 2^-38
	1.818989403545856475830078125e-12, 9.094947017729282379150390625e-13, // 2^-40
};
__device__ const SCAL d_factorial[]{ // from 0 to 33
	1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, // 10!
	39916800.L, 479001600.L, 6227020800.L, 87178291200.L, 1307674368000.L, // 15!
	20922789888000.L, 355687428096000.L, 6402373705728000.L, // 18!
	121645100408832000.L, 2432902008176640000.L, 51090942171709440000.L, // 21!
	1124000727777607680000.L, 25852016738884976640000.L, 620448401733239439360000.L, // 24!
	15511210043330985984000000.L, 403291461126605635584000000.L, // 26!
	10888869450418352160768000000.L, 304888344611713860501504000000.L, // 28!
	8841761993739701954543616000000.L, 265252859812191058636308480000000.L, // 30!
	8222838654177922817725562880000000.L, 263130836933693530167218012160000000.L, // 32!
	8683317618811886495518194401280000000.L, // 33!
};
__device__ const SCAL d_inv_factorial[]{ // from 0 to 33
	1, 1, .5, (long double)1/6, (long double)1/24, (long double)1/120, // 1/5!
	(long double)1/720, (long double)1/5040, (long double)1/40320, // 1/8!
	(long double)1/362880, (long double)1/3628800, (long double)1/39916800, // 1/11!
	(long double)1/479001600, (long double)1/6227020800, // 1/13!
	(long double)1/87178291200, (long double)1/1307674368000, // 1/15!
	(long double)1/20922789888000, (long double)1/355687428096000, // 1/17!
	(long double)1/6402373705728000, (long double)1/121645100408832000, // 1/19!
	(long double)1/2432902008176640000, 1/51090942171709440000.L, // 1/21!
	1/1124000727777607680000.L, 1/25852016738884976640000.L, 1/620448401733239439360000.L, // 1/24!
	1/15511210043330985984000000.L, 1/403291461126605635584000000.L, // 1/26!
	1/10888869450418352160768000000.L, 1/304888344611713860501504000000.L, // 1/28!
	1/8841761993739701954543616000000.L, 1/265252859812191058636308480000000.L, // 1/30!
	1/8222838654177922817725562880000000.L, 1/263130836933693530167218012160000000.L, // 1/32!
	1/8683317618811886495518194401280000000.L, // 1/33!
};
// even double factorials
__device__ const SCAL d_edfactorial[]{ // from 0 to 56
	1, 2, 8, 48, 384, 3840, 46080, 645120, 10321920, // 16!!
	185794560.L, 3715891200.L, 81749606400.L, 1961990553600.L, // 24!!
	51011754393600.L, 1428329123020800.L,  42849873690624000.L, // 30!!
	1371195958099968000.L, 46620662575398912000.L, 1678343852714360832000.L, // 36!!
	63777066403145711616000.L, 2551082656125828464640000.L, // 40!!
	107145471557284795514880000.L, 4714400748520531002654720000.L, // 44!!
	216862434431944426122117120000.L, 10409396852733332453861621760000.L, // 48!!
	5.20469842636666622693081088e+32L, 2.7064431817106664380040216576e+34L, // 52!!
	1.461479318123759876522171695104e+36L, 8.1842841814930553085241614925824e+37L, // 56!!
};
// odd double factorials
__device__ const SCAL d_odfactorial[]{ // from 1 to 55
	1, 3, 15, 105, 945, 10395, 135135, 2027025, (SCAL)34459425, // 17!!
	654729075.L, 13749310575.L, 316234143225.L, 7905853580625.L, // 25!!
	213458046676875.L, 6190283353629375.L, 191898783962510625.L, // 31!!
	6332659870762850625.L, 221643095476699771875.L, 8200794532637891559375.L, // 37!!
	319830986772877770815625.L, 13113070457687988603440625.L, // 41!!
	563862029680583509947946875.L, 25373791335626257947657609375.L, // 45!!
	1192568192774434123539907640625.L, 58435841445947272053455474390625.L, // 49!!
	2.9802279137433108747262291939219e+33L, 1.5795207942839547636049014727786e+35L, // 53!!
	8.6873643685617511998269581002823e+36L, // 55!!
};

inline __host__ __device__ long long factorial(int n)
// factorial
// returns n!
// n >= 0 must hold
{
	long long res = 1;
	for (long long i = n; i > 1; --i)
		res *= i;
	return res;
}

inline __host__ __device__ long long dfactorial(int n)
// double factorial
// returns n!!
// assumes n >= -1
{
	long long res = 1;
	for (long long i = n; i > 1; i -= 2)
		res *= i;
	return res;
}

inline __host__ __device__ int binomial(int n, int k)
// binomial coefficient
// returns n! / (k! * (n-k)!)
// it assumes 0 <= k <= n
{
	if (k > n/2)
		k = n - k;
	int res = 1;
	for (int num = n-k+1, den = 1; num <= n; ++num, ++den)
		res = (res * num) / den;
	return res;
}

inline __host__ __device__ int trinomial(int n, int j, int k)
// trinomial coefficient
// returns n! / (j! * k! * (n-j-k)!)
// it assumes 0 <= j,k,j+k <= n
{
	if (j > n-j-k)
		j = n-j-k;
	if (k > n-j-k)
		k = n-j-k;
	int res = 1;
	int num = n-j-k+1;
	for (int den = 1; num <= n-k; ++num, ++den)
		res = (res * num) / den;
	for (int den = 1; num <= n; ++num, ++den)
		res = (res * num) / den;
	return res;
}


inline __host__ __device__ long long factorialfrac(int n, int k)
// division between two factorials
// returns n! / k!
// it assumes 0 <= k <= n
{
	long long res = 1;
	for (long long i = n; i > k; --i)
		res *= i;
	return res;
}

inline __host__ __device__ int paritysign(int n)
// (-1)^n
// returns 1 if n is even and -1 if n is odd
{
	return 1 - 2 * (n & 1);
}

inline __host__ __device__ SCAL binarypow(SCAL x, int n)
// calculates x^n with O(log(n)) multiplications
{
	if (n == 0)
		return (SCAL)1; // note: 0^0 will give 1
	if (n < 0)
	{
		x = 1 / x;
		n = -n;
	}
	SCAL y(1);
	while (n > 1)
	{
		y *= ((n & 1) ? x : (SCAL)1);
		x *= x;
		n /= 2;
	}
	return x * y;
}

inline __host__ __device__ SCAL static_factorial(int n)
{
#ifdef __CUDA_ARCH__
	return d_factorial[n];
#else
	return h_factorial[n];
#endif
}

inline __host__ __device__ SCAL static_edfactorial(int n)
{
#ifdef __CUDA_ARCH__
	return d_edfactorial[n>>1];
#else
	return h_edfactorial[n>>1];
#endif
}

inline __host__ __device__ SCAL static_odfactorial(int n)
{
#ifdef __CUDA_ARCH__
	return d_odfactorial[n>>1];
#else
	return h_odfactorial[n>>1];
#endif
}

inline __host__ __device__ SCAL power_of_2(int n)
{
#ifdef __CUDA_ARCH__
	return d_power_of_2[n];
#else
	return h_power_of_2[n];
#endif
}

inline __host__ __device__ SCAL inv_factorial(int n)
{
#ifdef __CUDA_ARCH__
	return d_inv_factorial[n];
#else
	return h_inv_factorial[n];
#endif
}

inline __host__ __device__ SCAL inv_power_of_2(int n)
{
#ifdef __CUDA_ARCH__
	return d_inv_power_of_2[n];
#else
	return h_inv_power_of_2[n];
#endif
}

inline __device__ __host__ int flatten(int2 v, int s)
{
	return v.x * s + v.y;
}
inline __device__ __host__ int flatten(int3 v, int s)
{
	return (v.x * s + v.y) * s + v.z;
}
inline __device__ __host__ int flatten(int4 v, int s)
{
	return ((v.x * s + v.y) * s + v.z) * s + v.w;
}

inline SCAL fmax(VEC_T(SCAL, 2) a)
{
	return std::fmax(a.x, a.y);
}
inline SCAL fmax(VEC_T(SCAL, 3) a)
{
	return std::fmax(std::fmax(a.x, a.y), a.z);
}
inline SCAL fmax(VEC_T(SCAL, 4) a)
{
	return std::fmax(std::fmax(std::fmax(a.x, a.y), a.z), a.w);
}

inline SCAL fmin(VEC_T(SCAL, 2) a)
{
	return std::fmin(a.x, a.y);
}
inline SCAL fmin(VEC_T(SCAL, 3) a)
{
	return std::fmin(std::fmin(a.x, a.y), a.z);
}
inline SCAL fmin(VEC_T(SCAL, 4) a)
{
	return std::fmin(std::fmin(std::fmin(a.x, a.y), a.z), a.w);
}

inline __device__ __host__ int2 to_ivec(VEC_T(SCAL, 2) a)
{
	return make_int2((int)a.x, (int)a.y);
}
inline __device__ __host__ int3 to_ivec(VEC_T(SCAL, 3) a)
{
	return make_int3((int)a.x, (int)a.y, (int)a.z);
}
inline __device__ __host__ int4 to_ivec(VEC_T(SCAL, 4) a)
{
	return make_int4((int)a.x, (int)a.y, (int)a.z, (int)a.w);
}

inline __device__ __host__ int2 clip(int2 v, int a, int b)
{
	v.x = (v.x < a) ? a : (v.x > b ? b : v.x);
	v.y = (v.y < a) ? a : (v.y > b ? b : v.y);
	return v;
}
inline __device__ __host__ int3 clip(int3 v, int a, int b)
{
	v.x = (v.x < a) ? a : (v.x > b ? b : v.x);
	v.y = (v.y < a) ? a : (v.y > b ? b : v.y);
	v.z = (v.z < a) ? a : (v.z > b ? b : v.z);
	return v;
}
inline __device__ __host__ int4 clip(int4 v, int a, int b)
{
	v.x = (v.x < a) ? a : (v.x > b ? b : v.x);
	v.y = (v.y < a) ? a : (v.y > b ? b : v.y);
	v.z = (v.z < a) ? a : (v.z > b ? b : v.z);
	v.w = (v.w < a) ? a : (v.w > b ? b : v.w);
	return v;
}

inline __device__ __host__ VEC_T(SCAL, 2) sqrt(VEC_T(SCAL, 2) a)
{
	a.x = sqrt(a.x);
	a.y = sqrt(a.y);
	return a;
}
inline __device__ __host__ VEC_T(SCAL, 3) sqrt(VEC_T(SCAL, 3) a)
{
	a.x = sqrt(a.x);
	a.y = sqrt(a.y);
	a.z = sqrt(a.z);
	return a;
}
inline __device__ __host__ VEC_T(SCAL, 4) sqrt(VEC_T(SCAL, 4) a)
{
	a.x = sqrt(a.x);
	a.y = sqrt(a.y);
	a.z = sqrt(a.z);
	a.w = sqrt(a.w);
	return a;
}

inline __device__ __host__ VEC_T(SCAL, 2) fma(SCAL k, VEC_T(SCAL, 2) a, VEC_T(SCAL, 2) b)
{
	VEC_T(SCAL, 2) c;
	c.x = k * a.x + b.x;
	c.y = k * a.y + b.y;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 3) fma(SCAL k, VEC_T(SCAL, 3) a, VEC_T(SCAL, 3) b)
{
	VEC_T(SCAL, 3) c;
	c.x = k * a.x + b.x;
	c.y = k * a.y + b.y;
	c.z = k * a.z + b.z;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 4) fma(SCAL k, VEC_T(SCAL, 4) a, VEC_T(SCAL, 4) b)
{
	VEC_T(SCAL, 4) c;
	c.x = k * a.x + b.x;
	c.y = k * a.y + b.y;
	c.z = k * a.z + b.z;
	c.w = k * a.w + b.w;
	return c;
}

inline __device__ __host__ VEC_T(SCAL, 2) fma(VEC_T(SCAL, 2) k, VEC_T(SCAL, 2) a, VEC_T(SCAL, 2) b)
{
	VEC_T(SCAL, 2) c;
	c.x = k.x * a.x + b.x;
	c.y = k.y * a.y + b.y;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 3) fma(VEC_T(SCAL, 3) k, VEC_T(SCAL, 3) a, VEC_T(SCAL, 3) b)
{
	VEC_T(SCAL, 3) c;
	c.x = k.x * a.x + b.x;
	c.y = k.y * a.y + b.y;
	c.z = k.z * a.z + b.z;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 4) fma(VEC_T(SCAL, 4) k, VEC_T(SCAL, 4) a, VEC_T(SCAL, 4) b)
{
	VEC_T(SCAL, 4) c;
	c.x = k.x * a.x + b.x;
	c.y = k.y * a.y + b.y;
	c.z = k.z * a.z + b.z;
	c.w = k.w * a.w + b.w;
	return c;
}

inline __device__ __host__ VEC_T(SCAL, 2) fma(const SCAL* k, VEC_T(SCAL, 2) a, VEC_T(SCAL, 2) b)
{
	VEC_T(SCAL, 2) c;
	c.x = k[0] * a.x + b.x;
	c.y = k[1] * a.y + b.y;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 3) fma(const SCAL* k, VEC_T(SCAL, 3) a, VEC_T(SCAL, 3) b)
{
	VEC_T(SCAL, 3) c;
	c.x = k[0] * a.x + b.x;
	c.y = k[1] * a.y + b.y;
	c.z = k[2] * a.z + b.z;
	return c;
}
inline __device__ __host__ VEC_T(SCAL, 4) fma(const SCAL* k, VEC_T(SCAL, 4) a, VEC_T(SCAL, 4) b)
{
	VEC_T(SCAL, 4) c;
	c.x = k[0] * a.x + b.x;
	c.y = k[1] * a.y + b.y;
	c.z = k[2] * a.z + b.z;
	c.w = k[3] * a.w + b.w;
	return c;
}

#endif // KERNEL_CUDA_H