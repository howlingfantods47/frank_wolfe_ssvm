/*
 * =====================================================================================
 *
 *       Filename:  USPS.cpp
 *
 *    Description: Wrapper for Struct-SVM on USPS dataset using hybrid Frank-Wolfe
 *
 *        Version:  1.0
 *        Created:  05/14/2014 18:56:31
 *       Revision:  none
 *
 *         Author:  Neel Shah
 *   Organization:
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

// keeping global for now...bad choice maybe
vector< vector<double> > features;
vector<int> labels;
int num_labels = 10;
int num_train, dim_feature;
int dimension;  //d+1

// assumes a is already allocated space
void max_function_multiclass(int i, double* w, double*a, void*user_arg){
    int y_i = labels[i];
    vector<double> scores;
    scores.resize(num_labels);
    for(int j=0; j<num_labels; ++j){
        if(j != y_i){
            scores[j] = 1 + DotProduct(w+j*dim_feature, &(features[i][0]), dim_feature);
        }
        else if(j == y_i){
            scores[j] = DotProduct(w+j*dim_feature, &(features[i][0]), dim_feature);
        }
    }
    int argmax;
    vector<double>::iterator argmax_it = std::max_element(scores.begin(), scores.end());
    argmax = std::distance(scores.begin(), argmax_it);
    //Assign exterior product feature representation. [0 0 0 ... i->-phi(x)..0..argmax->phi(x)..0..0..{argmax!=label[i]}]
    for(int j=0; j<dimension; ++j){
        a[j] = 0;
    }
    if(argmax == y_i){
        return;
    }
    else{
        for(int j = 0; j < dim_feature; ++j){
            a[y_i*dim_feature+j] = -features[i][j];
        }
        for(int j = 0; j < dim_feature; ++j){
            a[argmax*dim_feature+j] = features[i][j];
        }
        a[dimension-1] = 1;
    }
    return;
}

void read_feature(string filename, vector< vector<double> >& data){
    ifstream infile(filename.c_str());
    string line;
    double word;
    istringstream iss;
    getline(infile, line);
    iss.str(line);
    //cout << "Reading features..." << endl;
    int sx, sy;
    iss >> sx;
    iss >> sy;
    //cout << sx << " features of dimension " << sy << " each found..." << endl;
    data.resize(sx);
    for(int i=0; i<sx; ++i){
        data[i].resize(sy);
    }
    int i = 0;
    if(infile.is_open()){
        while(getline(infile, line)){
            int j = 0;
            istringstream iss2(line);
            while(iss2 >> word){
                data[i][j] = word;
                ++j;
            }
            ++i;
        }
    }
    //cout << "Features loaded successfully..." << endl;
    num_train = sx;
    dim_feature = sy;
    return;
}

void read_labels(string filename, vector<int>& labels){
    ifstream infile(filename.c_str());
    int sx, label;
    //cout << "Reading labels..." << endl;
    infile >> sx;
    //cout << sx << " labels found..." << endl;
    labels.resize(sx);
    int i = 0;
    while(infile >> label){
        labels[i] = label;
        ++i;
    }
    //cout << "Labels loaded successfully..." << endl;
    return;
}

int main(){
    string file_features = "datasets/USPS-work/usps_train.txt";
    string file_labels = "datasets/USPS-work/usps_train.labels";
    read_feature(file_features, features);
    read_labels(file_labels, labels);
    dimension = num_labels*dim_feature + 1;
    double lambda = 1;
    int group_size = 1;
    //int group_size = num_train;
    SVM svm_usps(dimension-1, num_train, lambda, max_function_multiclass, NULL, NULL, group_size);
    svm_usps.options.iter_max = 50;
    svm_usps.options.cutting_planes_max = 10;
    svm_usps.options.inner_iter_max = svm_usps.options.cutting_planes_max + 1 ;
    svm_usps.options.callback_freq = 1;
    cout << "time_before_bound   iteration   lower_bound    upper_bound    duality_gap    time_after_bound" <<endl;
    double* w_opt;
    w_opt = svm_usps.Solve();
    return 0;
}


