/*
 * =====================================================================================
 *
 *       Filename:  LSIS.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/01/2014 19:39:40
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Neel Shah
 *   Organization:  IST Austria
 *
 * =====================================================================================
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

#include "SVM.h"
#include "SVMutils.h"

#include "graph.h"

using namespace std;

struct Edge{
	int s,t;
};

struct Sample{
	int num_sp, num_edges;
	vector<int> labels;
	vector< vector<double> > features;
	vector<Edge> edges;
};

vector<Sample> samples;
int num_train;
int dimension;
int sp_dim = 649;
typedef Graph<double, double, double> GraphType;

void max_function_lsis(int current, double* w, double* a, void* user_arg){
	Sample cur_sample = samples[current];
	int n = cur_sample.num_sp;
	int m = cur_sample.num_edges;
	GraphType *g = new GraphType(n, m);
	for(int i=0; i<n; ++i){
		g -> add_node();
	}
	double E0, E1;
	for(int i=0; i<n; ++i){
		E0 =  -1*DotProduct(&(cur_sample.features[i][0]), w, sp_dim) - 1.0/n*((cur_sample.labels[i]!=0)?1:0);
		E1 = -1*DotProduct(&(cur_sample.features[i][0]), w+sp_dim, sp_dim) - 1.0/n*((cur_sample.labels[i]!=1)?1:0);
		if(E0 < E1){
			g -> add_tweights(i, E1-E0, 0);
		}
		else{
			g -> add_tweights(i, 0, E0-E1);
		}
	}
	int p,q;
	for(int e=0; e<m; ++e){
		p = cur_sample.edges[e].s;
		q = cur_sample.edges[e].t;
		g -> add_tweights(p, 1, 0);
		g -> add_tweights(q, 0, 1);
		g -> add_edge(p, q, 2, 0);
	}
	int flow = g -> maxflow();
	for(int i=0; i<dimension+1; ++i){
		a[i] = 0;
	}
	for(int i=0; i<n; ++i){
		if(cur_sample.labels[i] == 0 && g -> what_segment(i) == 1){
			for(int j=0; j<sp_dim; ++j){
				a[j] -= cur_sample.features[i][j];
				a[sp_dim + j] += cur_sample.features[i][j];
			}
		}
		else if(cur_sample.labels[i] == 1 && g -> what_segment(i) == 0){
			for(int j=0; j<sp_dim; ++j){
				a[j] += cur_sample.features[i][j];
				a[sp_dim + j] -= cur_sample.features[i][j];
			}
		}
	}
	double loss = 0;
	for(int i=0; i<n; ++i){
		if(cur_sample.labels[i] != g -> what_segment(i)){
			loss += 1.0;
		}
	}
	loss = loss/n;
	//absorbing pairwise terms into loss, as they also correspond to a constant term in the weight vector.
	for(int e=0; e<m; ++e){
		p = cur_sample.edges[e].s;
		q = cur_sample.edges[e].t;
		loss += ((cur_sample.labels[p] != cur_sample.labels[q])?1:0) - ((g -> what_segment(p) != g -> what_segment(q))?1:0);
	}
	a[dimension] = loss;
	delete g;
}


void read_lsis(string filename, vector<Sample>& data){
	ifstream infile((filename+"superpixel_count.txt").c_str());
	int nump;
	string line;
	istringstream iss;
	int count = 0;
	if(infile.is_open()){
		while(infile >> nump){
			count++;
			Sample new_sample;
			new_sample.num_sp = nump;
			data.push_back(new_sample);
		}
	}
	infile.close();
	num_train = count;
	infile.open((filename+"edge_count.txt").c_str());
	if(infile.is_open()){
		for(int i=0; i<num_train; ++i){
			infile >> data[i].num_edges;
		}
	}
	infile.close();
	for(int i=0; i<data.size(); ++i){
		data[i].labels.resize(data[i].num_sp);
		data[i].edges.resize(data[i].num_edges);
		data[i].features.resize(data[i].num_sp);
	}
	for(int i=0; i<data.size(); ++i){
		for(int j=0; j<data[i].features.size(); ++j){
			data[i].features[j].resize(sp_dim);
		}
	}
	infile.open((filename+"labels.txt").c_str());
	if(infile.is_open()){
		for(int i=0; i<data.size(); ++i){
			for(int j=0; j<data[i].labels.size(); ++j){
				infile >> data[i].labels[j];
			}
		}
	}
	infile.close();
	infile.open((filename+"superpixel_structure.txt").c_str());
	if(infile.is_open()){
		for(int i=0; i<data.size(); ++i){
			for(int j=0; j<data[i].edges.size(); ++j){
				getline(infile, line);
				iss.str(line);
				iss >> data[i].edges[j].s;
				iss >> data[i].edges[j].t;
				iss.clear();
			}
		}
	}
	infile.close();
	infile.open((filename+"features.txt").c_str());
	if(infile.is_open()){
		for(int i=0; i<data.size(); ++i){
			for(int j=0; j<data[i].features.size(); ++j){
				getline(infile, line);
				iss.str(line);
				for(int k=0; k<data[i].features[j].size(); ++k){
					iss >> data[i].features[j][k];
				}
				iss.clear();
			}
		}
	}
	infile.close();
}

int main(int argc, char* argv[]){
	int group_size;
	int iter_max, cutting_planes_max, inner_iter;
	if(argc != 5){
		cout << "Usage: " << argv[0] << " group_size iter_max cutting_planes_max inner_iter_max (group_size = +ve integer or -1 for simple Frank_Wolfe)" << endl;
		return 1;
	}
	else{
		group_size = atoi(argv[1]);
		iter_max = atoi(argv[2]);
		cutting_planes_max = atoi(argv[3]);
		inner_iter = atoi(argv[4]);

	}
	read_lsis("datasets/LSIS/", samples);
	dimension = 2*sp_dim;
	num_train = samples.size();
	double lambda = 1;
	if(group_size == -1)    group_size = num_train;
	SVM svm_ocr(dimension, num_train, lambda, max_function_lsis, NULL, NULL, group_size);
	svm_ocr.options.iter_max = iter_max;
	svm_ocr.options.cutting_planes_max = cutting_planes_max;
	svm_ocr.options.inner_iter_max = inner_iter ;
	svm_ocr.options.callback_freq = 1;
	double* w_opt;
	cout << "time_before_bound   iteration   lower_bound    upper_bound    duality_gap    time_after_bound" <<endl;
	w_opt = svm_ocr.Solve();
	return 0;
}



