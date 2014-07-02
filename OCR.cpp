/*
 * =====================================================================================
 *
 *       Filename:  ocr.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/27/2014 20:41:27
 *
 *         Author:  Neel Shah
 *   *   Organization: IST Austria
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


typedef void (*MaxFn)(int i, double* w, double* a, void* user_arg);

struct Sample{
	int word_length;
	string word;
	vector<int> word_int; //takes values in 0-25 for a-z.
	vector< vector<double> > pixels;
};

// keeping global for now...bad choice maybe
vector<Sample> samples;
int num_labels = 26;
int num_train;
int dimension;
int grid_size = 128;


void max_along_column(vector< vector<double> >& in, vector<double>& out, vector<int>& pos){
	int m = in.size();
	int n = in[0].size();
	out.resize(n);
	pos.resize(n);
	for(int i=0; i<n; ++i){
		double max = in[0][i];
		int arg = 0;
		for(int j=1; j<m; ++j){
			if(in[j][i] > max){
				max = in[j][i];
				arg = j;
			}
		}
		out[i] = max;
		pos[i] = arg;
	}
}

// unary outside index corresponds to num_labels, inside to chain_length, table is other way round
void viterbi_decode(vector< vector<double> >& unary, vector< vector<double> >& pairwise, vector<int>& out){
	int chain_length = unary[0].size();
	vector< vector<double> > table, arg_table;
	table.resize(chain_length);
	arg_table.resize(chain_length);
	for(int i=0; i<chain_length; ++i){
		table[i].resize(num_labels);
		arg_table[i].resize(num_labels);
	}
	vector< vector<double> > temp;
	vector<double> temp_max;
	vector<int> temp_arg;
	temp.resize(num_labels);
	for(int i=0; i<num_labels; ++i) temp[i].resize(num_labels);
	//forward
	for(int t=0; t<num_labels; ++t){
		table[0][t] = unary[t][0];
		arg_table[0][t] = 0;
	}
	for(int t=1; t<chain_length; ++t){
		for(int i=0; i<num_labels; ++i){
			for(int j=0; j<num_labels; ++j){
				temp[i][j] = pairwise[i][j] + table[t-1][i];
			}
		}
		max_along_column(temp, temp_max, temp_arg);
		for(int i=0; i<num_labels; ++i){
			table[t][i] = temp_max[i] + unary[i][t];
			arg_table[t][i] = temp_arg[i];
		}
	}
	// backward
	int argmax_last = 0;
	double max_last = table[chain_length-1][0];
	for(int i=1; i<num_labels; ++i){
		if(table[chain_length-1][i] > max_last){
			max_last = table[chain_length-1][i];
			argmax_last = i;
		}
	}
	out[chain_length-1] = argmax_last;
	for(int t=chain_length-2; t>=0; --t){
		out[t] = arg_table[t+1][out[t+1]];
	}
}

void feature_map(vector< vector<double> >& pixel_grid, vector<int>& labels, vector<double>& phi){
	for(int i=0; i<dimension; ++i)  phi[i] = 0;
	int chain_length = pixel_grid.size();
	for(int i=0; i<chain_length; ++i){
		int cur_label = labels[i];
		for(int j=0; j<grid_size; ++j){
			phi[cur_label*grid_size + j] += pixel_grid[i][j];
		}
	}
	phi[num_labels*grid_size + labels[0]] = 1;
	phi[num_labels*grid_size + num_labels + labels[chain_length-1]] = 1;
	int offset = num_labels*grid_size + 2*num_labels;
	for(int i=0; i<chain_length-1; ++i){
		int id = labels[i] + num_labels*labels[i+1];
		phi[offset + id] += 1;
	}
}

void max_function_ocr(int current, double* w, double* a, void* user_arg){
	vector< vector<double> > theta_unary, theta_pairwise;
	int chain_length = samples[current].word_length;
	theta_unary.resize(num_labels);
	theta_pairwise.resize(num_labels);
	for(int i=0; i<num_labels; ++i){
		theta_unary[i].resize(chain_length);
		theta_pairwise[i].resize(num_labels);
	}
	// construct unary potentials
	for(int i=0; i<num_labels; ++i){
		for(int j=0; j<chain_length; ++j){
			theta_unary[i][j] = DotProduct(&(samples[current].pixels[j][0]), w+i*grid_size, grid_size);
		}
	}
	// add bias to beginning and end unaries
	for(int i=0; i<num_labels; ++i){
		theta_unary[i][0] += w[grid_size*num_labels + i];
		theta_unary[i][chain_length-1] += w[grid_size*num_labels + num_labels + i];
	}
	// construct pairwise terms
	int offset = grid_size*num_labels + 2*num_labels;
	for(int i=0; i<num_labels; ++i){
		for(int j=0; j<num_labels; ++j){
			theta_pairwise[j][i] = w[offset + i*num_labels + j];
		}
	}
	// add normalized Hamming loss terms to unaries
	for(int i=0; i<chain_length; ++i){
		int label = samples[current].word_int[i];
		for(int j=0; j<num_labels; ++j){
			if(j == label)  continue;
			else    theta_unary[j][i] += 1.0/chain_length;
		}
	}
	vector<int> argmax;
	argmax.resize(chain_length);
	viterbi_decode(theta_unary, theta_pairwise, argmax);
	vector<double> phi_i, phi_max;
	phi_i.resize(dimension);
	phi_max.resize(dimension);
	feature_map(samples[current].pixels, samples[current].word_int, phi_i);
	feature_map(samples[current].pixels, argmax, phi_max);
	for(int i=0; i<dimension; ++i){
		a[i] = phi_max[i] - phi_i[i];
	}
	double loss = 0;
	for(int i=0; i<chain_length; ++i){
		if(samples[current].word_int[i] != argmax[i]){
			loss += 1.0/chain_length;
		}
	}
	a[dimension] = loss;
}

void read_ocr(string filename, vector<Sample>& data){
	ifstream infile(filename.c_str());
	string line;
	string cur_word = "";
	char cur_letter;
	string dump;
	int next_id;
	int word_len = 0;
	vector<int> cur_word_int;
	vector< vector<double> > pixel_grid_word;
	vector<double> pixel_grid;
	pixel_grid.resize(grid_size);
	int pixel;
	int count = 0;
	//cout << "Reading features..." << endl;
	//cout << sx << " features of dimension " << sy << " each found..." << endl;
	if(infile.is_open()){
		while(getline(infile, line)){
			count++;
			istringstream iss2(line);
			iss2 >> dump;
			iss2 >> cur_letter;
			cur_word += cur_letter;
			cur_word_int.push_back(cur_letter-'a');
			word_len++;
			iss2 >> next_id;
			for(int i=0; i<3; ++i)  iss2 >> dump;
			for(int i=0; i<grid_size; ++i){
				iss2 >> pixel;
				pixel_grid[i] = pixel;
			}
			pixel_grid_word.push_back(pixel_grid);
			if(next_id == -1){
				Sample new_sample;
				new_sample.word_length = word_len;
				new_sample.word = cur_word;
				new_sample.pixels = pixel_grid_word;
				new_sample.word_int = cur_word_int;
				data.push_back(new_sample);
				word_len = 0;
				cur_word = "";
				cur_word_int.clear();
				pixel_grid_word.clear();
			}
		}
	}
	infile.close();
	//cout << "Features loaded successfully..." << endl;
}


int main(int argc, char* argv[]){
	string filename;
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

	}
	read_ocr(filename, samples);
	dimension = num_labels*grid_size + 2*num_labels + num_labels*num_labels;
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
	w_opt = svm_ocr.Solve();
	return 0;
}


