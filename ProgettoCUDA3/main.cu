
// Important defines
#define SCAL double // scalar type
#define DIM 2 // dimensions
// note: most functions do not depend on the number of dimensions, but others
// will need to be rewritten for DIM != 2

#define BLOCK_SIZE 128 // number of threads in a GPU block
#define MAX_GRID_SIZE 128 // number of blocks in a GPU grid
#define CACHE_LINE_SIZE 64 // CPU cache line size (in bytes)
#define RESCALE 1
#define EPS ((SCAL)1.e-18*RESCALE*RESCALE) // softening parameter

bool coll = true;

int CPU_THREADS = 8; // number of concurrent threads in CPU default value
int fmm_order = 5; // fast multipole method order default value
int tree_radius = 1;

SCAL dens_inhom = 1;

#include "kernel.cuh"
#include "direct.cuh"
#include "integrator.cuh"
#include "appel.cuh"
#include "fmm_cart.cuh"
#include "reductions.cuh"

#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

using namespace std::chrono;

void coulombOscillatorDirect(VEC *p, VEC *a, int n, const SCAL* param)
{
	direct2(p, a, n, param);
	add_elastic(p, a, n, param+2); // shift pointer
}

void coulombOscillatorDirect_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	direct2_cpu(p, a, n, param);
	add_elastic_cpu(p, a, n, param+2); // shift pointer
}

void coulombOscillatorAppel(VEC *p, VEC *a, int n, const SCAL* param)
{
	appel(p, a, n, param);
	add_elastic(p, a, n, param+2); // shift pointer
}

void coulombOscillatorFMM(VEC *p, VEC *a, int n, const SCAL* param)
{
	fmm_cart(p, a, n, param);
	add_elastic(p, a, n, param+2); // shift pointer
}

void coulombOscillatorFMM_cpu(VEC *p, VEC *a, int n, const SCAL* param)
{
	fmm_cart_cpu(p, a, n, param);
	add_elastic_cpu(p, a, n, param+2); // shift pointer
}

void centerDist(VEC *data, int n)
{
	VEC d{};
	for (int i = 0; i < n; ++i)
		d += data[i];
	d /= (SCAL)n;
	for (int i = 0; i < n; ++i)
		data[i] -= d;
}

void adjustRMS(VEC *data, int n, VEC adj)
{
	VEC d{};
	for (int i = 0; i < n; ++i)
		d += data[i]*data[i];
	d /= (SCAL)n;
	d = sqrt(d);
	for (int i = 0; i < n; ++i)
		data[i] *= adj / d;
}

void initKV(VEC *data, int n, VEC A, VEC omega, std::mt19937_64 &gen)
// Kapchinskij-Vladimirskij distribution
// A is the semiaxis vector
// omega is the depressed phase advance vector
// n is the number of elements (scalars)
{
	assert(DIM == 2); // Assume that we are in 2 dimensions
	std::uniform_real_distribution<SCAL> dist((SCAL)0, (SCAL)1); // uniform in [0,1]
	int nBodies = n / 2;
	for (int i = 0; i < nBodies; ++i)
	{
		SCAL eta = dist(gen), etax = twopi*dist(gen), etay = twopi*dist(gen);
		SCAL rt = sqrt(eta), rt1 = sqrt(1-eta);
		data[i].x = A.x * rt * cos(etax);
		data[i].y = A.y * rt1 * cos(etay);
		data[i+nBodies].x = A.x * omega.x * rt * sin(etax);
		data[i+nBodies].y = A.y * omega.y * rt1 * sin(etay);
	}
	centerDist(data, nBodies);
	adjustRMS(data, nBodies, A / (SCAL)2);

	data += nBodies;

	centerDist(data, nBodies);
	adjustRMS(data, nBodies, omega * A / (SCAL)2);
}

void initGA(VEC *data, int n, VEC x, VEC u, std::mt19937_64 &gen)
// Gaussian distribution
// x is the position std.dev.
// u is the velocity std.dev.
{
	std::normal_distribution<SCAL> dist((SCAL)0, (SCAL)1); // Marsaglia method?
									// mean = 0, std.dev. = 1
	SCAL *s_data = (SCAL*)data;
	for (int i = 0; i < n*DIM; ++i)
		s_data[i] = dist(gen);
	int nBodies = n/2;
	for (int i = 0; i < nBodies; ++i)
		data[i] *= x;
	for (int i = nBodies; i < n; ++i)
		data[i] *= u;

	centerDist(data, nBodies);
	adjustRMS(data, nBodies, x);

	data += nBodies;

	centerDist(data, nBodies);
	adjustRMS(data, nBodies, u);
}

void test_accuracy(void(*test)(VEC*, VEC*, int, const SCAL*), void(*ref)(VEC*, VEC*, int, const SCAL*),
				   SCAL *d_buf, int n, const SCAL* param)
// test the accuracy of "test" function with respect to the reference "ref" function
// print the mean relative error on console window
{
	static int nBlocksRed = 1;
	static int n_prev = 0, n_max = 0;
	static SCAL *d_relerr = nullptr;
	static SCAL *relerr = new SCAL[nBlocksRed];
	static VEC *d_tmp = nullptr;
	if (n != n_prev)
	{
		if (n > n_max)
		{
			if (n_max > 0)
			{
				gpuErrchk(cudaFree(d_tmp));
				gpuErrchk(cudaFree(d_relerr));
			}
			gpuErrchk(cudaMalloc((void**)&d_tmp, sizeof(VEC)*n));
			gpuErrchk(cudaMalloc((void**)&d_relerr, sizeof(SCAL)*nBlocksRed));
		}
	}
	compute_force(test, d_buf, n, param);
	copy_gpu(d_tmp, (VEC*)d_buf + 2 * n, n);
	compute_force(ref, d_buf, n, param);
	relerrReduce(d_relerr, d_tmp, (VEC*)d_buf + 2 * n, n, nBlocksRed);

	gpuErrchk(cudaMemcpy(relerr, d_relerr, sizeof(SCAL)*nBlocksRed, cudaMemcpyDeviceToHost));

	for (int i = 1; i < nBlocksRed; ++i)
		relerr[0] += relerr[i];

	std::cout << "Relative error: " << relerr[0] / (SCAL)n << std::endl;

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

void test_accuracy_cpu(void(*test)(VEC*, VEC*, int, const SCAL*), void(*ref)(VEC*, VEC*, int, const SCAL*),
					   SCAL *buf, int n, const SCAL* param)
// test the accuracy of "test" function with respect to the reference "ref" function
// print the mean relative error on console window
// both functions must run on CPU
{
	static int n_prev = 0, n_max = 0;
	std::vector<SCAL> relerr(CPU_THREADS);
	static VEC *tmp = nullptr;
	if (n != n_prev)
	{
		if (n > n_max)
		{
			if (n_max > 0)
				delete[] tmp;
			tmp = new VEC[n];
		}
	}
	compute_force(test, buf, n, param);
	copy_cpu(tmp, (VEC*)buf + 2 * n, n);
	compute_force(ref, buf, n, param);

	std::vector<std::thread> threads(CPU_THREADS);
	int niter = (n-1)/CPU_THREADS+1;
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i] = std::thread([=, &relerr] {
			SCAL err(0);
			VEC *bufacc = (VEC*)buf + 2*n;
			for (int j = niter*i; j < std::min(n, niter*(i+1)); ++j)
				err += rel_diff1(tmp[j], bufacc[j]);
			relerr[i] = err;
		});
	for (int i = 0; i < CPU_THREADS; ++i)
		threads[i].join();

	for (int i = 1; i < CPU_THREADS; ++i)
		relerr[0] += relerr[i];

	std::cout << "Relative error: " << relerr[0] / (SCAL)n << std::endl;

	if (n > n_max)
		n_max = n;
	n_prev = n;
}

int main(const int argc, const char** argv)
{
	int nBodies = 30001;
	if (argc > 1) nBodies = atoi(argv[1]); // specify N as the first argument
	if (argc > 2) fmm_order = atoi(argv[2]); // specify FMM order as the second argument
	if (argc > 3) CPU_THREADS = atoi(argv[3]); // specify the number of CPU threads as the third argument
	if (argc > 4) tree_radius = atoi(argv[4]); // specify tree interaction radius as the fourth argument

	const SCAL dt = (SCAL)5.e-5*RESCALE; // time step
	const int nIters = 30000001;  // simulation iterations
	const int nSteps = 10000;  // number of steps for every "snapshot" saved to file

	int bytes = 3 * nBodies * sizeof(VEC);
	int cpyBytes = 2 * nBodies * sizeof(VEC);
	char *c_buf = new char[bytes];
	SCAL *buf = (SCAL*)c_buf;

	SCAL xi(2.e-6);
	VEC omega0{5*twopi, 6.21*twopi};
	//VEC omega0{1.095, 1};
	omega0 /= RESCALE;

	// KV parameters
	/*VEC tune_dep{0.8, 0.6}, A, omega, domega, mism{1, 1};
	omega.x = omega0.x*tune_dep.x;
	omega.y = omega0.y*tune_dep.y;
	domega.x = (omega0.x + omega.x) * (omega0.x - omega.x);
	domega.y = (omega0.y + omega.y) * (omega0.y - omega.y);
	A.x = sqrt(2 * xi * domega.y / (domega.x * (domega.x + domega.y) ) ) * mism.x;
	A.y = sqrt(2 * xi * domega.x / (domega.y * (domega.y + domega.x) ) ) * mism.y;
	//A.x = 2.5794e-3 * RESCALE;
	//A.y = 1.5087e-3 * RESCALE * mism.y;
	std::cout << "Ax = " << A.x / RESCALE << "\n"
				 "Ay = " << A.y / RESCALE << std::endl;*/
	
	// GA parameters (MB high temperature)
	//SCAL kT(1.e-6), sqrtkT = sqrt(kT);
	//VEC x{sqrtkT/omega0.x, sqrtkT/omega0.y}, u{sqrtkT, sqrtkT};

	// GA parameters (r.m.s. matched to KV)
	VEC emit{0.03e-3*RESCALE, 0.01e-3*RESCALE}, omega, domega, A, x, u;
	omega.y = 0.6*omega0.y; // tune depression
	A.y = 2*sqrt(emit.y/omega.y);
	SCAL A2 = A.y * A.y;
	domega.y = (omega0.y + omega.y) * (omega0.y - omega.y);
	SCAL om0x2 = omega0.x * omega0.x, om0x4 = om0x2 * om0x2, om0x6 = om0x4 * om0x2;
	SCAL c = -2*om0x2, d = -A2*domega.y*domega.y/(4*emit.x), e = om0x4, p = c, q = d;
	SCAL Delta0 = 16*om0x4, Delta1 = 27*d*d + 128*om0x6;
	SCAL Q = cbrt((Delta1 + sqrt((27*d*d + 256*om0x6) * (27*d*d)))/2);
	SCAL S = sqrt((-2*p + (Q + Delta0/Q))/3)/2;
	SCAL sol[4];
	sol[0] = -S + sqrt(-4*S*S - 2*p + q/S)/2;
	sol[1] = -S - sqrt(-4*S*S - 2*p + q/S)/2;
	sol[2] = S + sqrt(-4*S*S - 2*p - q/S)/2;
	sol[3] = S - sqrt(-4*S*S - 2*p - q/S)/2; // quartic resolution
	omega.x = sol[3]; // 0.986911 // 0.938918
	A.x = 2*sqrt(emit.x/omega.x);
	xi = domega.y * A.y * (A.x + A.y) / 2;
	x = A / 2;
	u = omega * A / 2;

	std::cout << "emittances: " << x * u / RESCALE << std::endl;
	std::cout << "perveance: " << xi << std::endl;
	std::cout << "dep. phase adv.: " << omega * RESCALE << std::endl;
	std::cout << "semi-axes: " << A / RESCALE << std::endl;

	std::mt19937_64 gen(5351550349027530206);
	gen.discard(624*2);
	initKV((VEC*)buf, 2 * nBodies, A, omega, gen);
	//initGA((VEC*)buf, 2 * nBodies, x, u, gen);
	
	SCAL par[]{
		xi/(SCAL)nBodies, // xi/N
		0, // padding, needed for memory alignment
		omega0.x*omega0.x, // omegax0^2 = kx
		omega0.y*omega0.y, // omegay0^2 = ky
	};

	SCAL *d_buf, *d_par;

	 // allocate memory on GPU
	gpuErrchk(cudaMalloc((void**)&d_buf, bytes));
	gpuErrchk(cudaMalloc((void**)&d_par, 4*sizeof(SCAL)));

	// copy data from CPU to GPU
	gpuErrchk(cudaMemcpy(d_buf, buf, cpyBytes, cudaMemcpyHostToDevice)); // acc not copied
	gpuErrchk(cudaMemcpy(d_par, par, 4*sizeof(SCAL), cudaMemcpyHostToDevice));

	/*compute_force(fmm_cart, d_buf, nBodies, d_par); // warming up

	auto begin = steady_clock::now();

	compute_force(fmm_cart, d_buf, nBodies, d_par);

	auto end = steady_clock::now();
	
	std::cout << "Time elapsed: "
			  << duration_cast<microseconds>(end - begin).count() * (SCAL)1.e-6
			  << " [s]" << std::endl;

	for (fmm_order = 1; fmm_order <= 10; ++fmm_order)
	{
		std::cout << fmm_order << ": ";
		test_accuracy(fmm_cart, direct3, d_buf, nBodies, d_par);
	}

	goto jumpjump;*/

	// precompute accelerations
	compute_force(coulombOscillatorFMM, d_buf, nBodies, d_par);
	//compute_force(coulombOscillatorFMM_cpu, buf, nBodies, par);

	for (int iter = 0; iter < nIters; ++iter)
	{
		leapfrog(coulombOscillatorFMM, d_buf, nBodies, d_par, dt, step);
		
		if (iter % nSteps == 0)
		{
			std::cout << iter << ' ' << std::flush;
			// copy data from GPU to CPU
			gpuErrchk(cudaMemcpy(buf, d_buf, cpyBytes, cudaMemcpyDeviceToHost)); // acc not copied

			std::string sout("out/out");
			std::ofstream fout(sout + std::to_string(iter) + '_' + std::to_string(dt/RESCALE)
							 + ".bin", std::ios::out | std::ios::binary);
			if (fout)
				fout.write(c_buf, cpyBytes); // write to file (note that c_buf = (char*)buf)
			else
			{
				std::cerr << "Error: cannot write on output location. "
							 "Check that \"out\" folder exists. Create it if not." << std::endl;

				gpuErrchk(cudaFree(d_buf)); // free memory from GPU
				gpuErrchk(cudaFree(d_par));
				delete[] buf;

				return -1;
			}
			fout.close();
		}
	}

	jumpjump:

	gpuErrchk(cudaFree(d_buf)); // free memory from GPU
	gpuErrchk(cudaFree(d_par));
	delete[] buf;

	return 0;
}