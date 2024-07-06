//  N-body coulomb oscillators
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

/*

Compilation:
	nvcc main.cu -o nbco -O3 -std=c++11 -arch=sm_<xy> -I <includes>
<xy> is the compute capability of the GPU (usually given in the form x.y),
for example sm_21 corresponds to a compute capability of 2.1.
<includes> is the folder which contains the CUB library.
Note: some CUDA versions may require the c++14 standard or later.
	
The resulting program will be called 'nbco'. A compatible C++ compilator must be
available.

nvcc main.cu -o nbco -O3 -std=c++11 -arch=sm_75

*/

// Important defines
#define SCAL double // scalar type
#define DIM 2 // dimensions
// note: most functions do not depend on the number of dimensions, but others
// will need to be rewritten for DIM != 2

int BLOCK_SIZE = 128; // number of threads in a GPU block
int MAX_GRID_SIZE = 128; // number of blocks in a GPU grid
int CACHE_LINE_SIZE = 64; // CPU cache line size (in bytes)
SCAL EPS2 = (SCAL)1.e-18; // softening parameter squared

int CPU_THREADS = 8; // number of concurrent threads in CPU
int fmm_order = 5; // fast multipole method order
int tree_radius = 1;

bool coll = true;

SCAL dens_inhom = 1;

#include "kernel.cuh"
#include "direct.cuh"
#include "integrator.cuh"
#include "appel.cuh"
#include "fmm_cart.cuh"
#include "reductions.cuh"

#include <fstream>
#include <sstream>
#include <limits>
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
// center the sampled distribution
{
	VEC d{};
	for (int i = 0; i < n; ++i)
		d += data[i];
	d /= (SCAL)n;
	for (int i = 0; i < n; ++i)
		data[i] -= d;
}

void adjustRMS(VEC *data, int n, VEC adj)
// adjust the RMS of the sampled distribution such that it's equal to adj
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
	std::cout << "N-body coulomb oscillators, Copyright (C) 2021 Alessandro Lo Cuoco\n\n"
				 "Type 'nbco -h' for a brief documentation.\n";
	
	int nBodies = 30001;
	SCAL dt = (SCAL)5.e-4; // time step
	int nIters = 30001;  // simulation iterations
	int nSteps = 200;  // number of steps for every "snapshot" saved to file
	std::string strout("out"), strin;
	bool in = false, cpu = false, test = false, ga = false, calc_u = false, calc_omega = false;
	
	auto symp_integ = leapfrog;
	
	SCAL xi(2.e-6);
	VEC omega0{6.22*twopi, 6.21*twopi};
	
	//VEC omega0{1.095, 1};

	// KV parameters
	/*VEC tune_dep{0.8, 0.6}, A, omega, domega, mism{1, 1};
	omega.x = omega0.x*tune_dep.x;
	omega.y = omega0.y*tune_dep.y;
	domega.x = (omega0.x + omega.x) * (omega0.x - omega.x);
	domega.y = (omega0.y + omega.y) * (omega0.y - omega.y);
	A.x = sqrt(2 * xi * domega.y / (domega.x * (domega.x + domega.y) ) ) * mism.x;
	A.y = sqrt(2 * xi * domega.x / (domega.y * (domega.y + domega.x) ) ) * mism.y;
	//A.x = 2.5794e-3;
	//A.y = 1.5087e-3 * mism.y;
	std::cout << "Ax = " << A.x << "\n"
				 "Ay = " << A.y << std::endl;*/
	
	// GA parameters (MB high temperature)
	//SCAL kT(1.e-6), sqrtkT = sqrt(kT);
	//VEC x{sqrtkT/omega0.x, sqrtkT/omega0.y}, u{sqrtkT, sqrtkT};

	// GA parameters (r.m.s. matched to KV)
	VEC emit{0.03e-3, 0.01e-3}, omega, domega, A, x, u;
	omega.y = 0.8*omega0.y; // tune depression
	A.y = 2*sqrt(emit.y/omega.y);
	SCAL A2 = A.y * A.y;
	domega.y = (omega0.y + omega.y) * (omega0.y - omega.y);
	SCAL om0x2 = omega0.x * omega0.x, om0x4 = om0x2 * om0x2, om0x6 = om0x4 * om0x2;
	SCAL c = -2*om0x2, d = -A2*domega.y*domega.y/(4*emit.x), /*e = om0x4,*/ p = c, q = d;
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
	
	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] == '-')
		{
			if ((argv[i][1] == 'h' && argv[i][2] == '\0') ||
				(argv[i][1] == 'h' && argv[i][2] == 'e' && argv[i][3] == 'l' && argv[i][4] == 'p' && argv[i][5] == '\0'))
			{
				std::cout << "This program comes with ABSOLUTELY NO WARRANTY.\n"
							 "This is free software, and you can redistribute it under certain conditions.\n"
							 "See the LICENSE file for more details.\n\n"
							 "Usage: nbco [options] [input]\n\n"
							 "  [input] is the path to a file that contains a state of the system. The format\n"
							 "  is that of a binary file that contains the positions of all particles and\n"
							 "  then their respective velocities in the same order. This state is used for\n"
							 "  initialising the system to be simulated. If not specified, the system will be\n"
							 "  initalised by sampling from a known distribution (KV or gaussian).\n\n"
							 "Other options:\n"
							 "  -h or -help       Display this documentation.\n"
							 "  -o <output>       Specify the folder where the output will be written.\n"
							 "                    Default is './out'. This folder must already exist. The\n"
							 "                    format of output files is the same as the input file format\n"
							 "                    described before.\n"
							 "  -n <npart>        Number of particles to simulate. Default is 30001. Will be\n"
							 "                    ignored if [input] is specified.\n"
							 "  -ds <v>           Time step. Default is 5e-4.\n"
							 "  -iters <n>        Number of total simulation iterations. Default is 30000.\n"
							 "  -steps <n>        Number of steps to simulate before saving to file. Default\n"
							 "                    is 200.\n"
							 "  -integ <name>     Set the symplectic integrator to be used instead of the\n"
							 "                    default one (2nd order). <name> must be chosen from the\n"
							 "                    list {eu, fr, pefrl}, respectively the semi-implicit Euler\n"
							 "                    (1st order), Forest-Ruth (4th order) and PEFRL (4th order).\n"
							 "  -p <order>        FMM expansion order. Default is 5.\n"
							 "  -r <radius>       Interaction radius. Must be 1 or greater. Default is 1.\n"
							 "  -eps <v>          Smoothing factor. Must be greater than 0. Default is 1e-9.\n"
							 "  -i <v>            A factor so that max FMM level is round(log(n*i/p^(3/2))).\n"
							 "                    Default is 1.\n" // !
							 "  -ncoll            P2P pass will not be calculated, effectively ignoring\n"
							 "                    collisional effects. Note however that the result will\n"
							 "                    depend on the max FMM level, and the simulation may be\n"
							 "                    highly unreliable at certain conditions.\n"
							 "  -cpu              Use CPU with multithreading (default is GPU).\n"
							 "  -cpu-threads <n>  Number of CPU threads. Must be 1 or greater (default is 8).\n"
							 "                    Implies -cpu.\n"
							 "  -cacheline <n>    CPU cache line size in bytes. Defaut is 64. Will be ignored\n"
							 "                    if -cpu is NOT specified.\n"
							 "  -gpu <blocksize>  Specify the number of threads in a GPU block. It must be\n"
							 "                    chosen from the list {1, 32, 64, 128, 256, 512, 1024} (1024\n"
							 "                    threads is available for compute capability 2.0 and above).\n"
							 "                    Default is 128. Will be ignored if -cpu is specified.\n"
							 "  -gridsize <n>     Specify the maximum number of blocks in a GPU grid. Must be\n"
							 "                    1 or greater. Default is 128. Will be ignored if -cpu is\n"
							 "                    specified.\n"
							 "  -test             Show relative errors and execution times of a single\n"
							 "                    iteration instead of doing the simulation.\n"
							 "  -ga               Initialise the system with a gaussian distribution instead\n"
							 "                    of a KV distribution (which is the default).\n"
							 "  -xi <v>           Set the perveance.\n"
							 "  -omega0 <vx> <vy> Set the phase advances.\n"
							 "  -x <vx> <vy>      Set the std.dev. of positions. Will be ignored if [input]\n"
							 "                    is specified.\n"
							 "  -u <vx> <vy>      Set the std.dev. of velocities. Will be ignored if [input]\n"
							 "                    is specified.\n"
							 "  -A <vx> <vy>      Set the system semi-axes. Will be ignored if [input] is\n"
							 "                    specified.\n"
							 "  -omega <vx> <vy>  Set the depressed phase advances. Will be ignored if\n"
							 "                    [input] is specified.\n"
							 "Note: x = A / 2 and u = omega * A / 2.\n";
				
				return 0;
			}
			else if (argv[i][1] == 'o' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-o'\n";
					return -1;
				}
				strout = argv[i+1];
				++i;
			}
			else if (argv[i][1] == 'n' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-n'\n";
					return -1;
				}
				nBodies = atoi(argv[i+1]);
				if (nBodies <= 0)
				{
					std::cerr << "Error: invalid argument to '-n': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'd' && argv[i][2] == 's' && argv[i][3] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-ds'\n";
					return -1;
				}
				dt = atof(argv[i+1]);
				if (dt <= 0)
				{
					std::cerr << "Error: invalid argument to '-ds': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == 't' && argv[i][3] == 'e' && argv[i][4] == 'r' && argv[i][5] == 's'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-iters'\n";
					return -1;
				}
				nIters = atoi(argv[i+1])+1;
				if (nIters <= 0)
				{
					std::cerr << "Error: invalid argument to '-iters': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 's' && argv[i][2] == 't' && argv[i][3] == 'e' && argv[i][4] == 'p' && argv[i][5] == 's'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-steps'\n";
					return -1;
				}
				nSteps = atoi(argv[i+1]);
				if (nSteps <= 0)
				{
					std::cerr << "Error: invalid argument to '-steps': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == 'n' && argv[i][3] == 't' && argv[i][4] == 'e' && argv[i][5] == 'g'
				  && argv[i][6] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-integ'\n";
					return -1;
				}
				if (argv[i+1][1] == 'e' && argv[i+1][2] == 'u' && argv[i+1][3] == '\0')
					symp_integ = symplectic_euler;
				else if (argv[i+1][1] == 'f' && argv[i+1][2] == 'r' && argv[i+1][3] == '\0')
					symp_integ = forestruth;
				else if (argv[i+1][1] == 'p' && argv[i+1][2] == 'e' && argv[i+1][3] == 'f' && argv[i+1][4] == 'r' && argv[i+1][5] == 'l'
					  && argv[i+1][6] == '\0')
					symp_integ = pefrl;
				else
				{
					std::cerr << "Error: invalid argument to '-integ': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'p' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-p'\n";
					return -1;
				}
				fmm_order = atoi(argv[i+1]);
				if (fmm_order <= 0)
				{
					std::cerr << "Error: invalid argument to '-p': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'r' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-r'\n";
					return -1;
				}
				tree_radius = atoi(argv[i+1]);
				if (tree_radius <= 0)
				{
					std::cerr << "Error: invalid argument to '-r': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'e' && argv[i][2] == 'p' && argv[i][3] == 's' && argv[i][4] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-eps'\n";
					return -1;
				}
				EPS2 = atof(argv[i+1]);
				if (EPS2 <= 0)
				{
					std::cerr << "Error: invalid argument to '-eps': " << argv[i+1] << '\n';
					return -1;
				}
				EPS2 *= EPS2;
				if (EPS2 == 0) // underflow
				{
					std::cerr << "Error: too small argument to '-eps': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'i' && argv[i][2] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-i'\n";
					return -1;
				}
				dens_inhom = atof(argv[i+1]);
				if (dens_inhom <= 0)
				{
					std::cerr << "Error: invalid argument to '-i': " << argv[i+1] << " (should be greater than 0)\n";
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'n' && argv[i][2] == 'c' && argv[i][3] == 'o' && argv[i][4] == 'l' && argv[i][5] == 'l'
				  && argv[i][6] == '\0')
				coll = false;
			else if (argv[i][1] == 'c' && argv[i][2] == 'p' && argv[i][3] == 'u' && argv[i][4] == '\0')
				cpu = true;
			else if (argv[i][1] == 'c' && argv[i][2] == 'p' && argv[i][3] == 'u' && argv[i][4] == '-' && argv[i][5] == 't'
				  && argv[i][6] == 'h' && argv[i][7] == 'r' && argv[i][8] == 'e' && argv[i][9] == 'a' && argv[i][10] == 'd'
				  && argv[i][11] == 's' && argv[i][12] == '\0')
			{
				cpu = true;
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-cpu-threads'\n";
					return -1;
				}
				CPU_THREADS = atoi(argv[i+1]);
				if (CPU_THREADS <= 0)
				{
					std::cerr << "Error: invalid argument to '-cpu-threads': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'c' && argv[i][2] == 'a' && argv[i][3] == 'c' && argv[i][4] == 'h' && argv[i][5] == 'e'
				  && argv[i][6] == 'l' && argv[i][7] == 'i' && argv[i][8] == 'n' && argv[i][9] == 'e' && argv[i][10] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-cacheline'\n";
					return -1;
				}
				CACHE_LINE_SIZE = atoi(argv[i+1]);
				if (CACHE_LINE_SIZE <= 0)
				{
					std::cerr << "Error: invalid argument to '-cacheline': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'g' && argv[i][2] == 'p' && argv[i][3] == 'u' && argv[i][4] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-gpu'\n";
					return -1;
				}
				BLOCK_SIZE = atoi(argv[i+1]);
				if (BLOCK_SIZE <= 0)
				{
					std::cerr << "Error: invalid argument to '-gpu': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'g' && argv[i][2] == 'r' && argv[i][3] == 'i' && argv[i][4] == 'd' && argv[i][5] == 's'
				  && argv[i][6] == 'i' && argv[i][7] == 'z' && argv[i][8] == 'e' && argv[i][9] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-gridsize'\n";
					return -1;
				}
				MAX_GRID_SIZE = atoi(argv[i+1]);
				if (MAX_GRID_SIZE <= 0)
				{
					std::cerr << "Error: invalid argument to '-gridsize': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 't' && argv[i][2] == 'e' && argv[i][3] == 's' && argv[i][4] == 't' && argv[i][5] == '\0')
				test = true;
			else if (argv[i][1] == 'g' && argv[i][2] == 'a' && argv[i][3] == '\0')
				ga = true;
			else if (argv[i][1] == 'x' && argv[i][2] == 'i' && argv[i][3] == '\0')
			{
				if (i+1 >= argc)
				{
					std::cerr << "Error: missing argument to '-xi'\n";
					return -1;
				}
				xi = atof(argv[i+1]);
				if (xi < 0)
				{
					std::cerr << "Error: invalid argument to '-xi': " << argv[i+1] << '\n';
					return -1;
				}
				++i;
			}
			else if (argv[i][1] == 'o' && argv[i][2] == 'm' && argv[i][3] == 'e' && argv[i][4] == 'g' && argv[i][5] == 'a'
				  && argv[i][6] == '0' && argv[i][7] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-omega0'\n";
					return -1;
				}
				omega0.x = atof(argv[i+1]);
				omega0.y = atof(argv[i+2]);
				if (omega0.x < 0 || omega0.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-omega0': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				i += 2;
			}
			else if (argv[i][1] == 'x' && argv[i][2] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-x'\n";
					return -1;
				}
				x.x = atof(argv[i+1]);
				x.y = atof(argv[i+2]);
				if (x.x < 0 || x.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-x': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				A = x * (SCAL)2;
				i += 2;
			}
			else if (argv[i][1] == 'u' && argv[i][2] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-u'\n";
					return -1;
				}
				u.x = atof(argv[i+1]);
				u.y = atof(argv[i+2]);
				if (u.x < 0 || u.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-u': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				calc_omega = true;
				// omega * A / 2 = omega * x = u
				i += 2;
			}
			else if (argv[i][1] == 'A' && argv[i][2] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-A'\n";
					return -1;
				}
				A.x = atof(argv[i+1]);
				A.y = atof(argv[i+2]);
				if (A.x < 0 || A.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-A': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				x = A / (SCAL)2;
				i += 2;
			}
			else if (argv[i][1] == 'o' && argv[i][2] == 'm' && argv[i][3] == 'e' && argv[i][4] == 'g' && argv[i][5] == 'a' && argv[i][6] == '\0')
			{
				if (i+2 >= argc)
				{
					std::cerr << "Error: missing argument(s) to '-omega'\n";
					return -1;
				}
				omega.x = atof(argv[i+1]);
				omega.y = atof(argv[i+2]);
				if (omega.x < 0 || omega.y < 0)
				{
					std::cerr << "Error: invalid argument(s) to '-omega': " << argv[i+1] << ' ' << argv[i+2] << '\n';
					return -1;
				}
				calc_u = true;
				i += 2;
			}
			else
			{
				std::cerr << "Error: unrecognised option '" << argv[i] << "'\n";
				return -1;
			}
		}
		else
		{
			strin = argv[i];
			in = true;
		}
	}
	
	if (calc_omega)
		omega = u / x;
	else if (calc_u)
		u = omega * x;

	int bytes, cpyBytes;
	char *c_buf = nullptr;
	SCAL *buf = nullptr;

	if (in)
	{
		std::ifstream fin(strin, std::ios::in | std::ios::binary);
		if (fin)
		{
			fin.ignore(std::numeric_limits<std::streamsize>::max());
			cpyBytes = (int)fin.gcount();
			nBodies = cpyBytes / 2 / sizeof(VEC);
			bytes = 3 * nBodies * sizeof(VEC);
			c_buf = new char[bytes];
			buf = (SCAL*)c_buf;
			fin.clear();
			fin.seekg(0, std::ios::beg);
			fin.read(c_buf, cpyBytes);
		}
		else
		{
			std::cerr << "Error: cannot read from input location." << std::endl;

			delete[] buf;

			return -1;
		}
	}
	else
	{
		bytes = 3 * nBodies * sizeof(VEC);
		cpyBytes = 2 * nBodies * sizeof(VEC);
		c_buf = new char[bytes];
		buf = (SCAL*)c_buf;
		
		std::cout << "emittances: " << x * u << std::endl;
		std::cout << "perveance: " << xi << std::endl;
		std::cout << "dep. phase adv.: " << omega << std::endl;
		std::cout << "semi-axes: " << A << std::endl;

		std::mt19937_64 gen(5351550349027530206);
		gen.discard(624*2);
		if (ga)
			initGA((VEC*)buf, 2 * nBodies, x, u, gen);
		else
			initKV((VEC*)buf, 2 * nBodies, A, omega, gen);
	}

	if (!test)
	{
		std::ofstream farg(strout + "/args.txt", std::ios::out);
		if (farg)
			for (int i = 0; i < argc; ++i)
				farg << argv[i] << ' ';
		else
		{
			std::cerr << "Error: cannot write on output location. "
						 "Check that \"" << strout << "\" folder exists. Create it if not." << std::endl;
			delete[] buf;

			return -1;
		}
	}
	
	SCAL par[]{
		xi/(SCAL)nBodies, // xi/N
		0, // padding, needed for memory alignment
		omega0.x*omega0.x, // omegax0^2 = kx
		omega0.y*omega0.y, // omegay0^2 = ky
	};

	SCAL *d_buf, *d_par;

	if (!cpu)
	{
		 // allocate memory on GPU
		gpuErrchk(cudaMalloc((void**)&d_buf, bytes));
		gpuErrchk(cudaMalloc((void**)&d_par, 4*sizeof(SCAL)));

		// copy data from CPU to GPU
		gpuErrchk(cudaMemcpy(d_buf, buf, cpyBytes, cudaMemcpyHostToDevice)); // acc not copied
		gpuErrchk(cudaMemcpy(d_par, par, 4*sizeof(SCAL), cudaMemcpyHostToDevice));
	}
	
	if (test)
	{
		// warming up
		if (cpu)
			compute_force(fmm_cart_cpu, buf, nBodies, par);
		else
			compute_force(fmm_cart, d_buf, nBodies, d_par);

		auto begin = steady_clock::now();

		if (cpu)
			compute_force(fmm_cart_cpu, buf, nBodies, par);
		else
			compute_force(fmm_cart, d_buf, nBodies, d_par);

		auto end = steady_clock::now();
		
		std::cout << "Time elapsed: "
				  << duration_cast<microseconds>(end - begin).count() * (SCAL)1.e-6
				  << " [s]" << std::endl;

		for (fmm_order = 1; fmm_order <= 10; ++fmm_order)
		{
			std::cout << fmm_order << ": ";
			if (cpu)
				test_accuracy_cpu(fmm_cart_cpu, direct3_cpu, buf, nBodies, par);
			else
				test_accuracy(fmm_cart, direct3, d_buf, nBodies, d_par);
		}
	}
	else
	{
		// precompute accelerations
		if (cpu)
			compute_force(coulombOscillatorFMM_cpu, buf, nBodies, par);
		else
			compute_force(coulombOscillatorFMM, d_buf, nBodies, d_par);

		for (int iter = 0; iter < nIters; ++iter)
		{
			if (cpu)
				symp_integ(coulombOscillatorFMM_cpu, buf, nBodies, par, dt, step_cpu, 1);
			else
				symp_integ(coulombOscillatorFMM, d_buf, nBodies, d_par, dt, step, 1);
			
			if (iter % nSteps == 0)
			{
				std::cout << iter << ' ' << std::flush;
				// copy data from GPU to CPU
				if (!cpu)
					gpuErrchk(cudaMemcpy(buf, d_buf, cpyBytes, cudaMemcpyDeviceToHost)); // acc not copied

				std::ofstream fout(strout + "/out" + std::to_string(iter) + '_' + std::to_string(dt)
								 + ".bin", std::ios::out | std::ios::binary);
				if (fout)
					fout.write(c_buf, cpyBytes); // write to file (note that c_buf = (char*)buf)
				else
				{
					std::cerr << "Error: cannot write on output location. "
								 "Check that \"" << strout << "\" folder exists. Create it if not." << std::endl;
					if (!cpu)
					{
						gpuErrchk(cudaFree(d_buf)); // free memory from GPU
						gpuErrchk(cudaFree(d_par));
					}
					delete[] buf;

					return -1;
				}
			}
		}
	}
	
	if (!cpu)
	{
		gpuErrchk(cudaFree(d_buf)); // free memory from GPU
		gpuErrchk(cudaFree(d_par));
	}
	delete[] buf;

	return 0;
}