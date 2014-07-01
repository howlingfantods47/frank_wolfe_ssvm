#ifndef HGALHALSJGHASASFASFG
#define HGALHALSJGHASASFASFG

#include <stdlib.h>

inline int RandomInteger(int N) // returns random integer in [0,N-1]
{
	while ( 1 )
	{
		int x = (int) ( (((double)N) * rand()) / RAND_MAX );
		if (x<N) return x;
	}
}

inline void generate_permutation(int *buf, int n)
{
	int i, j;

	for (i=0; i<n; i++) buf[i] = i;
	for (i=0; i<n-1; i++)
	{
		j = i + RandomInteger(n-i);
		int tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp;
	}
}


inline void SetZero(double* w, int d)
{
	int k;
	for (k=0; k<d; k++) w[k] = 0;
}

inline double Norm(double* w, int d)
{
	int k;
	double val = 0;
	for (k=0; k<d; k++) val += w[k]*w[k];
	return val;
}

inline double DotProduct(double* u, double* w, int d)
{
	int k;
	double val = 0;
	for (k=0; k<d; k++) val += u[k]*w[k];
	return val;
}

inline void Add(double* u, double* v, int d)
{
	int k;
	for (k=0; k<d; k++) u[k] += v[k];
}

inline void Multiply(double* w, double c, int d)
{
	int k;
	for (k=0; k<d; k++) w[k] *= c;
}

// returns <u-v,u>
inline double Op1(double* u, double* v, int d)
{
	int k;
	double val = 0;
	for (k=0; k<d; k++) val += (u[k]-v[k])*u[k];
	return val;
}

// returns <u-v,w>
inline double Op1(double* u, double* v, double* w, int d)
{
	int k;
	double val = 0;
	for (k=0; k<d; k++) val += (u[k]-v[k])*w[k];
	return val;
}

// returns ||u-v||^2
inline double Op2(double* u, double* v, int d)
{
	int k;
	double val = 0;
	for (k=0; k<d; k++) val += (u[k]-v[k])*(u[k]-v[k]);
	return val;
}

#endif