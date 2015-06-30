#include "Network.h"
#include "Vec.h"
#include "WordVec.h"
#include <ctime>

int main()
{
	Parameter* para = new Parameter();

	WordVec* words = new WordVec("vec.ds");


	double start = clock();
	words->readFile();
	double end = clock();
	cout << "The time of read words is " << (end - start)/CLOCKS_PER_SEC << endl << endl;

	Network* network = new Network(para, words);

	start = clock();
	network->train(para);
	end = clock();
	cout << "The time of train network is " << (end - start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	network->test(para);
	end = clock();
	cout << "The time of test network is " << (end- start)/CLOCKS_PER_SEC << endl << endl;

}
