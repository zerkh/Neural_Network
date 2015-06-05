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
	Vector**						weights;			//����Ȩ��
	Vector**						weights_b;			//ƫ����Ȩ��
	Vector*							weights_i;			//�����Ȩ��
	int								amountOfLayer;		//�������
	int*							dimOfLayers;		//�����ά��
	vector<pair<Vector*, double> >				v_train_data;
	WordVec*						words;
	double							lambda;
	double							alpha;
	double							threshold;
	bool							is_log;
	ofstream						log_file;
	int								iterTimes;
	vector<vector<string> >			posNote;			//��¼�����Ĵ�

public:
	Network(Parameter* para, WordVec* words);

	void train(Parameter* para);

	void get_data(string filename);

	void test(Parameter* para);

	//����weights��ƽ����
	double Regularization();
};

#endif
