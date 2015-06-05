#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "Parameter.h"
#include "Vec.h"
#include "WordVec.h"
#include "Algorithm.h"

using namespace std;

class Network
{
private:
	Vector**						weights;			//各层权重
	Vector**						weights_b;			//偏置项权重
	Vector*							weights_i;			//输入层权重
	int								amountOfLayer;		//网络层数
	int*							dimOfLayers;		//各层的维数
	vector<pair<Vector*, double> >				v_train_data;
	WordVec*						words;
	double							lambda;
	double							alpha;
	double							threshold;
	bool							is_log;
	ofstream						log_file;
	int								iterTimes;
	vector<vector<string> >			posNote;			//记录输入层的词

public:
	Network(Parameter* para, WordVec* words);

	void train(Parameter* para);

	void get_data(string filename);

	void test(Parameter* para);

	//所有weights的平方和
	double Regularization();
};

#endif
