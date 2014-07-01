#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SVM.h"
#include "SVMutils.h"
#include "timer.h"


void SVM::AddCuttingPlane(int i, double* a)
{
	if (options.cutting_planes_max <= 0) return;
	if (terms[i]->isDuplicate(a)) return;
	bool replace = (terms[i]->num >= options.cutting_planes_max);
	terms[i]->AddPlane(a, replace);
}




void SVM::InitSolver()
{
	int i;

	terms = new Term*[n];
	current_sum = (double*) buf.Alloc((d+1)*sizeof(double));
	SetZero(current_sum, d+1);
	SetZero(w, d);

	for (i=0; i<n; i++)
	{
		terms[i] = new Term(d, &buf, options.cutting_planes_max, (options.kernel_iter_max > 1) ? true : false);
		double* current = terms[i]->current;
		if (lower_bound_fn0)
		{
			SetZero(current, d);
			current[d] = lower_bound_fn(i);
			current_sum[d] += current[d];
		}
		else
		{
			max_fn(i, current);
			Add(current_sum, current, d+1); 
		}
		AddCuttingPlane(i, current);
	}
}


double* SVM::Solve()
{
	if (!terms) InitSolver();
	int _i, i, k, iter, inner_iter;
	
	for (k=0; k<d; k++) w[k] = -current_sum[k] / (lambda*n0);
	// double lower_bound = -Norm(w, d)*lambda/2 + current_sum[d]/n0;
    double t_begin = get_time();
	double* current_new = new double[d+1];
	int* permutation = NULL;
	if (options.randomize_method == 1 || options.randomize_method == 2)	permutation = new int[n];

	for (iter=0; iter<options.iter_max; iter++)
	{
		if (iter > 0) // recompute current_sum every two iterations for numerical stability
		{
			SetZero(current_sum, d+1);
			for (i=0; i<n; i++) Add(current_sum, terms[i]->current, d+1);
		}
		if (permutation && (iter==0 || options.randomize_method == 2)) generate_permutation(permutation, n);

		double t_start = get_time(), t_oracle;
		for (inner_iter=0; inner_iter<options.inner_iter_max; inner_iter++)
		{
			for (_i=0; _i<n; _i++)
			{
				switch (options.randomize_method)
				{
					case 0: i = _i; break;
					case 1:
					case 2: i = permutation[_i]; break;
					case 3: i = RandomInteger(n); break;
				}
			
				double* current = terms[i]->current;

				if (inner_iter == 0)
				{
					max_fn(i, current_new);
					AddCuttingPlane(i, current_new);
				}
				else
				{
					if (options.kernel_iter_max > 1)
					{
						SolveWithKernel(i, options.kernel_iter_max);
						for (k=0; k<d; k++) w[k] = -current_sum[k] / (lambda*n0);
						continue;
					}

					int t = terms[i]->Maximize(w);
					terms[i]->UpdateStats(t, options.cutting_planes_history_size);
					memcpy(current_new, terms[i]->a[t], (d+1)*sizeof(double));
				}

				// min_{gamma \in [0,1]} B*gamma*gamma - 2*A*gamma
				double A = Op1(current, current_new, current_sum, d) + (current_new[d] - current[d])*(lambda*n0); // <current-current_new,current_sum> + (b_new - b) *( lambda*n)
				double B = Op2(current, current_new, d); // ||current-current_new||^2
				double gamma;
				if (B<=0) gamma = (A <= 0) ? 0 : 1;
				else
				{
					gamma = A/B;
					if (gamma < 0) gamma = 0;
					if (gamma > 1) gamma = 1;
				}

				for (k=0; k<=d; k++)
				{
					double old = current[k];
					current[k] = (1-gamma)*current[k] + gamma*current_new[k];
					current_sum[k] += current[k] - old;
				}
				for (k=0; k<d; k++) w[k] = -current_sum[k] / (lambda*n0);
				// lower_bound = -Norm(w, d)*lambda/2 + current_sum[d]/n0;
			}

			if (inner_iter == 0) t_oracle = get_time();
			else
			{
				if (get_time() - t_oracle > options.inner_iter_time_limit*(t_oracle - t_start)) break;
			}
		}
		if (options.callback_fn && ((iter+1) % options.callback_freq) == 0)
		{
            std::cout << get_time() - t_begin << " ";
			if ( ! (*options.callback_fn)(iter, w, this) ) {std::cout << get_time() - t_begin << std::endl; break;}
            else    {std::cout << get_time() - t_begin << std::endl;}
		}
	}

	if (permutation) delete [] permutation;
	delete [] current_new;
	return w;
}

void SVM::GetBounds(double& lower_bound, double& upper_bound, double& dual_gap_bound)
{
	int i;
	double* tmp = new double[d+1];

	double norm = Norm(w, d)*lambda/2;
	lower_bound = -norm + current_sum[d]/n0;
	upper_bound = 0;
	for (i=0; i<n; i++)
	{
		max_fn(i, tmp);
		upper_bound += DotProduct(w, tmp, d) + tmp[d];
	}
	upper_bound /= n0;
	upper_bound += norm;
	dual_gap_bound = upper_bound - lower_bound;

	delete [] tmp;
}

void SVM::SolveWithKernel(int _i, int iter_max)
{
	Term* T = terms[_i];
	int num = T->num, i, t, iter;
	double** kk = T->products;
	double* ck = (double*) rbuf_SolveWithKernel.Alloc(3*num*sizeof(double)); // ck[i] = DotProduct(current, T->a[i], d)
	double* sk = ck + num; // sk[i] = DotProduct(current_sum, T->a[i], d)
	double cc = DotProduct(T->current, T->current, d);
	double cs = DotProduct(T->current, current_sum, d);
	double c_d = T->current[d];

	double* x = sk + num;
	double cx = 1;

	double gamma;

	for (i=0; i<num; i++)
	{
		ck[i] = DotProduct(T->current, T->a[i], d);
		sk[i] = DotProduct(current_sum, T->a[i], d);
		x[i] = 0;
	}

	for (iter=0; iter<iter_max; iter++)
	{
		if (iter > 0)
		{
			// current += gamma*(a[t] - current)
			// sum     += gamma*(a[t] - current)

			c_d += gamma*(T->a[t][d] - c_d);
			double cc_new = cc + 2*gamma*(ck[t] - cc) + gamma*gamma*(kk[t][t] - 2*ck[t] + cc);
			double cs_new = cs+ gamma*(ck[t] + sk[t] - cc - cs) + gamma*gamma*(kk[t][t] - 2*ck[t] + cc);
			cc = cc_new;
			cs = cs_new;

			for (i=0; i<num; i++)
			{
				if (kk[i][t] == NOT_YET_COMPUTED) kk[i][t] = DotProduct(T->a[i], T->a[t], d);
				double delta = gamma*(kk[i][t] - ck[i]);
				ck[i] += delta;
				sk[i] += delta;
			}

/*
double* current_tmp = new double[2*(d+1)];
double* current_sum_tmp = current_tmp + d+1;
for (i=0; i<=d; i++)
{
	current_sum_tmp[i] = current_sum[i] - T->current[i];
	current_tmp[i] = T->current[i] * cx;
}
for (t=0; t<num; t++)
{
	if (x[t] == 0) continue;
	for (i=0; i<=d; i++) current_tmp[i] += x[t]*T->a[t][i];
}
for (i=0; i<=d; i++)
{
	current_sum_tmp[i] += current_tmp[i];
}

printf("cc: %f %f\n", cc, DotProduct(current_tmp, current_tmp, d));
printf("cs: %f %f\n", cs, DotProduct(current_tmp, current_sum_tmp, d));
printf("c_d: %f %f\n", c_d, current_tmp[d]);
for (i=0; i<num; i++)
{
	printf("ck[%d]: %f %f\n", i, ck[i], DotProduct(current_tmp, T->a[i], d));
}
for (i=0; i<num; i++)
{
	printf("sk[%d]: %f %f\n", i, sk[i], DotProduct(current_sum_tmp, T->a[i], d));
}
getchar();
delete [] current_tmp;
*/
		}

		t = 0;
		double v_best;
		for (i=0; i<num; i++)
		{
			double v = -sk[i] / (lambda*n0) + T->a[i][d];
			if (i==0 || v_best < v) { t = i; v_best = v; }
		}
		T->UpdateStats(t, options.cutting_planes_history_size);

		if (kk[t][t] == NOT_YET_COMPUTED) kk[t][t] = DotProduct(T->a[t], T->a[t], d);

		// min_{gamma \in [0,1]} B*gamma*gamma - 2*A*gamma
		double A = cs - sk[t] + (T->a[t][d] - c_d)*(lambda*n0);
		double B = cc + kk[t][t] - 2*ck[t];

		if (B<=0) gamma = (A <= 0) ? 0 : 1;
		else
		{
			gamma = A/B;
			if (gamma < 0) gamma = 0;
			if (gamma > 1) gamma = 1;
		}

		cx *= 1-gamma;
		for (i=0; i<num; i++) x[i] *= 1-gamma;
		x[t] += gamma;
	}

	for (i=0; i<=d; i++)
	{
		current_sum[i] -= T->current[i];
		T->current[i] *= cx;
	}
	for (t=0; t<num; t++)
	{
		if (x[t] == 0) continue;
		for (i=0; i<=d; i++) T->current[i] += x[t]*T->a[t][i];
	}
	for (i=0; i<=d; i++)
	{
		current_sum[i] += T->current[i];
	}
}

