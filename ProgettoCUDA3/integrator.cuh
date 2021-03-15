#ifndef INTEGRATOR_CUDA_H
#define INTEGRATOR_CUDA_H

#include "kernel.cuh"

void compute_force(void(*f)(VEC*, VEC*, int, const SCAL*), SCAL *d_buf, int n, const SCAL* param)
{
	ParticleSystem d_p = { (VEC*)d_buf, ((VEC*)d_buf) + n, ((VEC*)d_buf) + 2 * n };

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);
}

// symplectic methods

void symplectic_euler(void(*f)(VEC*, VEC*, int, const SCAL*), SCAL *d_buf, int n,
					  const SCAL* param, SCAL dt,
					  void(*step_func)(VEC*, const VEC*, SCAL, int) = step,
					  SCAL scale = (SCAL)1)
// 1st order
{
	ParticleSystem d_p = { (VEC*)d_buf, ((VEC*)d_buf) + n, ((VEC*)d_buf) + 2 * n };

	// v += a * dt
	step_func(d_p.vel, d_p.acc, dt * scale, n);

	// x += v * dt
	step_func(d_p.pos, d_p.vel, dt, n);

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);
}

void leapfrog(
	void(*f)(VEC*, VEC*, int, const SCAL*),	// pointer to function for the evaulation of the field f(x)
	SCAL *d_buf,							// pointer to buffer containing x, v, a
	int n,									// number of particles
	const SCAL* param,						// additional parameters
	SCAL dt,								// timestep
	void(*step_func)(VEC*, const VEC*, SCAL, int) = step, // pointer to function for step (multiply/addition)
	SCAL scale = (SCAL)1					// quantity that rescales the field f(x)
)
// 2nd order
{
	ParticleSystem d_p = { (VEC*)d_buf, // positions
						  ((VEC*)d_buf) + n, // velocities
						  ((VEC*)d_buf) + 2 * n // accelerations
	};
	SCAL ds = dt * scale * (SCAL)0.5;

	// v += a * dt / 2
	step_func(d_p.vel, d_p.acc, ds, n);

	// x += v * dt
	step_func(d_p.pos, d_p.vel, dt, n);

	// a = f(x)
	f(d_p.pos, d_p.acc, n, param);

	// v += a * dt / 2
	step_func(d_p.vel, d_p.acc, ds, n);
}

constexpr long double fr_par = 1.3512071919596576340476878089715L; // 1 / (2 - cbrt(2))

void forestruth(void(*f)(VEC*, VEC*, int, const SCAL*), SCAL *d_buf, int n,
				const SCAL* param, SCAL dt, void(*step_func)(VEC*, const VEC*, SCAL, int) = step,
				SCAL scale = (SCAL)1)
// Forest-Ruth method
// 4th order
{
	ParticleSystem d_p = { (VEC*)d_buf, // positions
						  ((VEC*)d_buf) + n, // velocities
						  ((VEC*)d_buf) + 2 * n // accelerations
	};
	SCAL ds = dt * scale;

	step_func(d_p.pos, d_p.vel, SCAL(dt * fr_par / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * fr_par), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - fr_par) / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2*fr_par)), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - fr_par) / 2), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * fr_par), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * fr_par / 2), n);
}

constexpr long double pefrl_parx = +0.1786178958448091E+00L;
constexpr long double pefrl_parl = -0.2123418310626054E+00L;
constexpr long double pefrl_parc = -0.6626458266981849E-01L;

void pefrl(void(*f)(VEC*, VEC*, int, const SCAL*), SCAL *d_buf, int n,
				const SCAL* param, SCAL dt, void(*step_func)(VEC*, const VEC*, SCAL, int) = step,
				SCAL scale = (SCAL)1)
// Position-extended Forest-Ruth-like method
// 4th order, slower but more accurate
{
	ParticleSystem d_p = { (VEC*)d_buf, // positions
						  ((VEC*)d_buf) + n, // velocities
						  ((VEC*)d_buf) + 2 * n // accelerations
	};
	SCAL ds = dt * scale;

	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parx), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2 * pefrl_parl) / 2), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parc), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * pefrl_parl), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * (1 - 2 * (pefrl_parc + pefrl_parx))), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * pefrl_parl), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parc), n);

	f(d_p.pos, d_p.acc, n, param);

	step_func(d_p.vel, d_p.acc, SCAL(ds * (1 - 2 * pefrl_parl) / 2), n);
	step_func(d_p.pos, d_p.vel, SCAL(dt * pefrl_parx), n);
}

#endif // !INTEGRATOR_CUDA_H