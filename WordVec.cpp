#include "WordVec.h"
#include <fstream>
#include <sstream>

WordVec::WordVec(string filename)
{
	this->filename = filename;
}

int WordVec::getVecSize()
{
	return this->size;
}

void WordVec::readFile()
{
	ifstream fin(filename.c_str(), ios::in);

	if(!fin)
	{
		cerr << "Open " + filename << " fail!" << endl;
		exit(0);
	}

	string line;
	//debug
	//int count = 0;

	getline(fin, line);
	stringstream strin(line);
	strin >> this->amountOfWords;
	strin >> this->size;

	while(getline(fin, line))
	{
		stringstream strin(line);

		string word;
		double* wordvec = new double[this->size];

		strin >> word;
		for(int i = 0; i < this->size; i++)
		{
			strin >> wordvec[i];
		}

		this->m_words.insert(make_pair(word, wordvec));
		//debug
		//cout << count++ << endl;
	}

	//debug
	/*
	for(map<string, double*>::iterator it = this->m_words.begin(); it != m_words.end(); it++)
	{
		cout << it->first << " ";

		for(int i = 0; i < this->size; i++)
		{
			cout << it->second[i] << " ";
		}

		cout << endl;
	}*/
}
