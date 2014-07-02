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

void read_lsis(string filename, vector<Sample>& data){
	ifstream infile(filename+"superpixel_count.txt");
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
	infile.open(filename+"edge_count.txt");
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
	infile.open(filename+"labels.txt");
	if(infile.is_open()){
		for(int i=0; i<data.size(); ++i){
			for(int j=0; j<data[i].labels.size(); ++j){
				infile >> data[i].labels[j];
			}
		}
	}
	infile.close();
	infile.open(filename+"superpixel_structure.txt");
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
	infile.open(filename+"features.txt");
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
	/*      string filename;
			  int group_size;
			  int iter_max, cutting_planes_max, inner_iter;
			  if(argc != 6){
			  cout << "Usage: " << argv[0] << " train_file group_size iter_max cutting_planes_max inner_iter_max (group_size = +ve integer or -1 for simple Frank_Wolfe)" << endl;
			  return 1;
			  }
			  else{
			  filename = argv[1];
			  group_size = atoi(argv[2]);
			  iter_max = atoi(argv[3]);
			  cutting_planes_max = atoi(argv[4]);
			  inner_iter = atoi(argv[5]);

			  }*/
	read_lsis("datasets/LSIS/", samples);
	/*  dimension = num_labels*grid_size + 2*num_labels + num_labels*num_labels;
		 num_train = samples.size();
		 double lambda = 1;
		 if(group_size == -1)    group_size = num_train;
		 SVM svm_ocr(dimension, num_train, lambda, max_function_ocr, NULL, NULL, group_size);
		 svm_ocr.options.iter_max = iter_max;
		 svm_ocr.options.cutting_planes_max = cutting_planes_max;
		 svm_ocr.options.inner_iter_max = inner_iter ;
		 svm_ocr.options.callback_freq = 1;
		 double* w_opt;
		 cout << "time_before_bound   iteration   lower_bound    upper_bound    duality_gap    time_after_bound" <<endl;
		 w_opt = svm_ocr.Solve();*/
	return 0;
}



