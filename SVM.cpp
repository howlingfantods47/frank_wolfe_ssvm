#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SVM.h"
#include "SVMutils.h"



//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

SVM::SVM(int _d, int _n, double _lambda, MaxFn _max_fn, void* _user_arg, LowerBoundFn _lower_bound_fn, int _group_size) :
	d(_d), n0(_n), group_size(_group_size), lambda(_lambda), max_fn0(_max_fn), lower_bound_fn0(_lower_bound_fn), user_arg(_user_arg), terms(NULL), current_sum(NULL), buf(1024)
{
	n = ((n0-1)/group_size) + 1;
	w = (double*) buf.Alloc(d*sizeof(double));
	if (group_size > 1)	max_fn_buf = (double*) buf.Alloc((d+1)*sizeof(double));
}

SVM::~SVM()
{
	if (terms)
	{
		int i;
		for (i=0; i<n; i++) delete terms[i];
		delete [] terms;
	}
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SVM::max_fn(int i, double* a)
{
	if (group_size == 1) { (*max_fn0)(i, w, a, user_arg); return; }
	i *= group_size;
	int i_last = i + group_size; if (i_last > n0) i_last = n0;
	(*max_fn0)(i, w, a, user_arg);
	for (i++ ; i<i_last; i++)
	{
		(*max_fn0)(i, w, max_fn_buf, user_arg); 
		int k;
		for (k=0; k<=d; k++) a[k] += max_fn_buf[k];
	}
}

double SVM::lower_bound_fn(int i)
{
	if (group_size == 1) return (*lower_bound_fn0)(i, user_arg);
	i *= group_size;
	int i_last = i + group_size; if (i_last > n0) i_last = n0;
	double val = (*lower_bound_fn0)(i, user_arg);
	for (i++ ; i<i_last; i++) val += (*lower_bound_fn0)(i, user_arg);
	return val;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

double SVM::Evaluate(double* w)
{
	int i;

	double* tmp = new double[d+1];

	double v0 = Norm(w, d), v1 = 0;

	for (i=0; i<n; i++)
	{
		max_fn(i, tmp);
		v1 += DotProduct(w, tmp, d) + tmp[d];
	}

	return v0*lambda/2 + v1/n0;

	delete [] tmp;
}










/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
////////////////////// Implementation of 'Term' /////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////





SVM::Term::Term(int _d, Buffer* _buf, int estimated_num_max, bool maintain_products)
	: d(_d), buf(_buf), num(0), num_max(0)
{
	current = (double*) buf->Alloc((d+1)*sizeof(double));
	a = NULL;
	stats = NULL;
	products = NULL;
	my_buf = NULL;
	Allocate(estimated_num_max, maintain_products);
}

void SVM::Term::Allocate(int num_max_new, bool maintain_products)
{
	int num_max_old = num_max;
	double** a_old = a;
	PlaneStats* stats_old = stats;
	double** products_old = products;
	char* my_buf_old = my_buf;
	num_max = num_max_new;

	int i, my_buf_size = num_max*sizeof(double*) + num_max*sizeof(PlaneStats);
	if (maintain_products) my_buf_size += num_max*sizeof(double*) + num_max*num_max*sizeof(double);
	my_buf = new char[my_buf_size];

	a = (double**)my_buf;
	for (i=0; i<num_max_old; i++) a[i] = a_old[i];
	for ( ; i<num_max; i++) a[i] = NULL;

	stats = ((PlaneStats*)(a+num_max));
	memcpy(stats, stats_old, num_max_old*sizeof(PlaneStats));

	if (maintain_products)
	{
		products = (double**)(stats+num_max);
		for (i=0; i<num_max; i++)
		{
			products[i] = (i==0) ? ((double*)(products+num_max)) : (products[i-1] + num_max);
		}
		int t1, t2;
		for (t1=0; t1<num_max_old; t1++)
		for (t2=0; t2<num_max_old; t2++)
		{
			products[t1][t2] = products_old[t1][t2];
		}
	}

	if (my_buf_old) delete [] my_buf_old;
}


SVM::Term::~Term()
{
	if (my_buf) delete [] my_buf;
}

bool SVM::Term::isDuplicate(double* x)
{
	int t;

	for (t=0; t<num; t++)
	{
		if (!memcmp(x, a[t], (d+1)*sizeof(double))) return true;
	}
	return false;
}

int SVM::Term::AddPlane(double* x, bool replace)
{
	int t, t2;

	if (replace)
	{
		for (t=0, t2=1; t2<num; t2++)
		{
			if (stats[t].counter > stats[t2].counter) t = t2;
		}
	}
	else
	{
		if (num >= num_max) Allocate(2*num_max+1, (products) ? true : false);
		t = num ++;
		if (!a[t]) a[t] = (double*) buf->Alloc((d+1)*sizeof(double));
	}
	memcpy(a[t], x, (d+1)*sizeof(double));
	stats[t].counter = 1;

	if (products)
	{
		for (t2=0; t2<num; t2++)
		{
			products[t][t2] = products[t2][t] = NOT_YET_COMPUTED; // DotProduct(a[t], a[t2], d);
		}
	}

	return t;
}

void SVM::Term::DeletePlane(int t)
{
	num --;
	if (t == num) return;
	double* tmp = a[t]; a[t] = a[num]; a[num] = tmp;

	if (products)
	{
		int t2;
		for (t2=0; t2<num; t2++)
		{
			products[t][t2] = products[t2][t] = products[num][t2];
		}
	}
}

int SVM::Term::Maximize(double* w)
{
	int t_best, t;
	double v_best;
	for (t=0; t<num; t++)
	{
		double v = DotProduct(a[t], w, d) + a[t][d];
		if (t == 0 || v_best <= v) { v_best = v; t_best = t; }
	}
	return t_best;
}

void SVM::Term::UpdateStats(int t_best, int history_size)
{
	int t;
	double q = 1.0 / history_size, p = 1 - q;
	for (t=0; t<num; t++)
	{
		stats[t].counter *= p;
		if (t==t_best) stats[t].counter += q;
	}
}