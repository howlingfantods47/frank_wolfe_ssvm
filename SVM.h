#ifndef OAISJNHFOASFASFASFASFNVASF
#define OAISJNHFOASFASFASFASFNVASF

#include "block.h"
#include <iostream>
//////////////////////////////////////////////////////////////////////////
// SVM: class for solving                                               //
//   min  1/2 \lambda ||w||^2 + 1/n \sum_{i=1}^n H_i(w)                 //
// where                                                                //
//   H_i(w) = \max_y <a_{iy},[w 1]>                                     //
// It is assumed H_i(w) can be evaluated efficiently, i.e. the problem  //
//   \max_y <a_{iy},[w 1]>                                              //
// can be solved efficiently for a given i and w.                       //
//////////////////////////////////////////////////////////////////////////

// Type of a function that solves the maximization problem (should be provided by the user).
// INPUT: i, w, user_arg
// OUTPUT: vector a=a_{iy}  where  y \in \argmax_y <a_{iy},[w 1]>
// 
// 'w' and 'a' are arrays of size 'd' and 'd+1' respectively (already allocated). The last element in 'a' corresponds to '1'
typedef void (*MaxFn)(int i, double* w, double* a, void* user_arg);

// This function should return a lower bound on \min_w H_i(w)
typedef double (*LowerBoundFn)(int i, void* user_arg);





class SVM
{
public:
	// d = dimension of w, n = # of examples.
	// Specifying lower_bound_fn might give a better initialization (??). Can be NULL, if unknown.
	// Internally, the terms are partitioned into groups of size 'group_size'. group_size=n corresponds to (non-block coordinate) Frank-Wolfe.
	SVM(int d, int n, double lambda, MaxFn max_fn, void* user_arg, LowerBoundFn lower_bound_fn=NULL, int group_size=1);
	~SVM();

	double* Solve(); // returns a pointer to an array of size 'd' containing solution (vector w).
	                 // For options to Solve(), see SVM::options below

	void GetBounds(double& lower_bound, double& upper_bound, double& dual_gap_bound);

	double Evaluate(double* w); // returns the value of the objective function for given w

	void* GetUserArg() { return user_arg; }


	// cutting_planes_max is perhaps the most important parameter.
	// To get block-coordinate Frank-Wolfe, set cutting_planes_max=0 and inner_iter_max=1.
	struct Options
	{
		Options() :
			randomize_method(0),
			iter_max(1000),
			inner_iter_max(1),
			inner_iter_time_limit(100000),
			cutting_planes_max(0),
			cutting_planes_history_size(20),
			kernel_iter_max(1),

			callback_freq(5),
			callback_fn(default_callback_fn),
			gap_threshold(1e-10), 
			print_bounds(true)
		{
		};

		int randomize_method; // 0: use default order for every iteration (0,1,...,n-1)
		                      // 1: generate a random permutation, use it for every iteration
		                      // 2: generate a new random permutation at every iteration
		                      // 3: for every step sample example in {0,1,n-1} uniformly at random
		int iter_max;
		int inner_iter_max; // >= 1. The first inner iter calls the 'real' oracle, other inner iters query approximate oracles
		double inner_iter_time_limit; // if time for approximate oracles exceeds inner_iter_time_limit * (time for real oracles) then stop
		int cutting_planes_max; // >= 0. 
		int cutting_planes_history_size; // >= 1. parameter of the exponential decay; roughly speaking, how long non-active cutting planes are kept (see implementation).
		int kernel_iter_max; // >= 1. During 'approximate' passes each term is processed 'kernel_iter_max' times. If >1 then a specialized implementation with kernels is used.

		// if callback_fn != NULL then this function will be called after every callback_freq iterations (where iter = n calls to max_fn).
		// If this function returns false then Solve() will terminate.
		// The default function checks the duality gap and prints all bounds.
		int callback_freq;
		bool (*callback_fn)(int iter, double* w, SVM* svm);

		double gap_threshold;
		bool print_bounds;

		static bool default_callback_fn(int iter, double* w, SVM* svm)
		{
			double lower_bound, upper_bound, dual_gap_bound;
			svm->GetBounds(lower_bound, upper_bound, dual_gap_bound);
			//if (svm->options.print_bounds) printf("iter %d: value bounds: [%f %f], gap=%f\n", iter, lower_bound, upper_bound, dual_gap_bound);
			if (svm->options.print_bounds) printf(" %d %.8f %.8f %.8f ", iter, lower_bound, upper_bound, dual_gap_bound);
			if (dual_gap_bound < svm->options.gap_threshold) { return false;}
			return true;
		}
	} options;











//////////////////////////////////////////////////////////////
private:
	class Term // represents term H(w) = \max_y <a,[w 1]> and a 'current' linear lower bound
	{
	public:
	
		Term(int d, Buffer* buf, int estimated_num_max, bool maintain_products);
		~Term();

		int num; // number of planes 'y'
		double* current; // array of size d+1
		double** a; // a[t] points to an array of size d+1, 0<=t<num. a[-1] points to 'current'.
		struct PlaneStats
		{
			double counter;
		}* stats; // of size num
#define NOT_YET_COMPUTED (3e103) // some random number
		double** products; // products[t1][t2]=dot product of vectors a[t1] and a[t2], ignoring the last (d+1)-th coordinate (-1<=t1,t2<num).
						   // valid only if maintain_products was true in the constructor. Computed on demand.

		bool isDuplicate(double* a);
		int AddPlane(double* a, bool replace=false); // if replace=true then the plane with the lowest 'counter' will be deleted
													 // and the new plane 'a' will be inserted instead.
		                                             // returns id of the added plane
		void DeletePlane(int t); // plane 'num-1' is moved to position 't'.

		int Maximize(double* w); // returns id of the cutting plane 'a' that maximizes <[w 1],a>.

		void UpdateStats(int t, int history_size); // increases 'counter' for 't' and decreases it for other planes, with parameter 'history_size' (see implementation)


		////////////////////////////////////
	private:
		int d, num_max;
		Buffer* buf;
		char* my_buf;

		void Allocate(int num_max_new, bool maintain_products);
	};

	int d, n, n0, group_size;
	double lambda;
	MaxFn max_fn0;
	LowerBoundFn lower_bound_fn0;
	void* user_arg;
	Buffer buf;
	ReusableBuffer rbuf_SolveWithKernel;

	double* w; // of size d
	double* current_sum; // of size d+1
	Term** terms; // of size n

	void max_fn(int i, double* a); // calls max_fn0 for current w
	double lower_bound_fn(int i); // calls lower_bound_fn0 for current w
	double* max_fn_buf; // of size n+1

// This function should return a lower bound on \min_w H_i(w)
// where H_i(w) = \max_y <a_{iy},[w 1]>
//typedef double (*LowerBoundFn)(int i, void* user_arg);

	void InitSolver();
	void AddCuttingPlane(int i, double* a);
	void SolveWithKernel(int i, int iter_max);
};



#endif
