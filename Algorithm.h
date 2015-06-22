#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <climits>
#include "Vec.h"

using namespace std;

inline double tanh(double x)
{
	double result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));

	if(result < -1)
	{
		result = -1;
	}
	if(result > 1)
	{
		result = 1;
	}

	return result;

}

inline double tanh(Vector* vec, int i)
{
	double x = vec->getValue(0, i);

	double result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));

	if(result < -1)
	{
		result = -1;
	}
	if(result > 1)
	{
		result = 1;
	}

	return result;

}

inline double sigmoid(Vector* vec, int i)
{
	double x = vec->getValue(0, i);

	double result = 1 / (exp(-x) + 1);

	return result;
}

inline double sigmoid(double x)
{
	double result = 1 / (exp(-x) + 1);

	return result;
}

//获取高斯随机数
/*
inline double gaussRand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if ( phase == 0 ) 
	{
		do 
		{
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} 
	else
	{
		X = V2 * sqrt(-2 * log(S) / S);
	}

	phase = 1 - phase;

	return X/10000;
}
*/

inline double getRand()
{
	double res = (double)rand()/RAND_MAX;

	return res/100;
}

inline double loss(double z1, double z2, int y)
{
	double result = 1-y*(z1-z2);

	result = result;

	return result;
}

inline double loss(double z, double y)
{
	double result = (z-y)*(z-y)/2;

	return result;
}

inline double lossLog(double z, double y)
{
	double result = -(y*log(z) + (1-y)*log(1-z));

	return result;
}

inline double softmax(Vector* layer, int ind)
{
	double result = 0;

	for(int i = 0; i < layer->getCol(); i++)
	{
		result += exp(layer->getValue(0, i));
	}

	return (exp(layer->getValue(0, ind))/result);
}

#endif
