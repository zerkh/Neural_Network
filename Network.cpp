#include "Network.h"

Network::Network(Parameter* para, WordVec* words)
{
	this->amountOfLayer = 3;
	this->dimOfLayers = new int[amountOfLayer];
	this->words = words;
	this->lambda = 0.001;
	this->alpha = 0.6;
	this->iterTimes = 200;
	this->threshold = 1e-3;
	this->is_log = true;
	this->log_file.open("log_file", ios::out);

	this->dimOfLayers[0] = words->getVecSize();
	this->dimOfLayers[1] = 150;
	this->dimOfLayers[2] = 1;

	//initialize weights
	this->weights = new Vector*[this->amountOfLayer-1];
	this->weights_b = new Vector*[this->amountOfLayer-1];
	this->weights_i = new Vector(1, this->dimOfLayers[0]);

	for(int i = 0; i < amountOfLayer-1; i++)
	{
		this->weights[i] = new Vector(this->dimOfLayers[i+1], this->dimOfLayers[i]);
		for(int row = 0; row < dimOfLayers[i+1]; row++)
		{
			for(int col = 0; col < dimOfLayers[i]; col++)
			{
				this->weights[i]->setValue(row, col, getRand());
			}
		}

		this->weights_b[i] = new Vector(this->dimOfLayers[i+1], 1);
		for(int row = 0; row < dimOfLayers[i+1]; row++)
		{
			this->weights_b[i]->setValue(row, 0, getRand());
		}
	}

	for(int col = 0; col < dimOfLayers[0]; col++)
	{
		this->weights_i->setValue(0, col, 0);
	}
}

void Network::get_data(string filename)
{
	ifstream fin(filename.c_str(), ios::in);
	string line;
	string postive = "POS\t";
	string negative = "NEG\t";
	double* x = new double[this->dimOfLayers[0]];


	for(int i = 0; i < v_train_data.size(); i++)
	{
		delete v_train_data[i].first;
	}
	this->v_train_data.clear();
	this->posNote.clear();

	if(!fin)
	{
		cerr << "Open" + filename + "fail!" << endl;
		exit(0);
	}
	//debug
	int line_count = 0;

	while(getline(fin, line))
	{
		int pos = 0;
		double y;
		for(int i = 0; i < this->dimOfLayers[0]; i++)
		{
			x[i] = 0;
		}

		vector<string> tmpVec;
		//extract postive data
		pos = line.find(postive);
		if(pos != -1)
		{
			y = 1;

			Vector* vec = new Vector(1, this->dimOfLayers[0]);
			stringstream strin;
			line = line.erase(pos, postive.length());
			strin.str(line);

			int count = 0;
			string temp;
			while(strin >> temp)
			{
				pos = 0;

				map<string, double*>::iterator m_it = words->m_words.find(temp);

				if(m_it == words->m_words.end())
				{
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						x[i] += 0;
					}
				}
				else
				{
					count++;
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						if(false)
						{	
							log_file << m_it->second[i] << " ";
						}
						tmpVec.push_back(temp);
						x[i] += m_it->second[i];
					}
				}
			}


			//debug
			//			cout << "Valid:" << count << endl;
			for(int i = 0; i < this->dimOfLayers[0]; i++)
			{
				vec->setValue(0, i, x[i]/count);
			}
			this->v_train_data.push_back(make_pair(vec, y));
		}
		//extract negative sample
		else if((pos = line.find(negative)) != -1)
		{
			y = 0;

			Vector* vec = new Vector(1, this->dimOfLayers[0]);
			stringstream strin;
			line = line.erase(pos, negative.length());
			strin.str(line);

			int count = 0;
			string temp;
			while(strin >> temp)
			{
				pos = 0;

				map<string, double*>::iterator m_it = words->m_words.find(temp);

				if(m_it == words->m_words.end())
				{
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						x[i] += 0;
					}
				}
				else
				{
					count++;
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						if(false)
						{
							log_file << m_it->second[i] << " ";
						}
						x[i] += m_it->second[i];
						tmpVec.push_back(temp);
					}
				}
			}

			for(int i = 0; i < this->dimOfLayers[0]; i++)
			{
				vec->setValue(0, i, x[i]/count);
			}
			//debug
			//			cout << "Valid:" << count << endl;
			this->v_train_data.push_back(make_pair(vec, y));
		}
		//test data
		else
		{
			y = -1;

			Vector* vec = new Vector(1, this->dimOfLayers[0]);
			stringstream strin;
			strin.str(line);

			int count = 0;
			string temp;
			while(strin >> temp)
			{
				pos = 0;

				map<string, double*>::iterator m_it = words->m_words.find(temp);
				if(m_it == words->m_words.end())
				{
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						x[i] += 0;
					}
				}
				else
				{
					count++;
					for(int i = 0; i < this->dimOfLayers[0]; i++)
					{
						x[i] += m_it->second[i];
					}
				}
			}

			for(int i = 0; i < this->dimOfLayers[0]; i++)
			{
				vec->setValue(0, i, x[i]/count);
			}
			this->v_train_data.push_back(make_pair(vec, y));
		}

		posNote.push_back(tmpVec);
		tmpVec.clear();
		//		cout << line_count++ << " ";
		//		cout << "y:" << y << endl;
	}
}

void Network::train(Parameter* para)
{
	get_data("train.ds");

	//初始化
	int count = 1;

	Vector** der_Weights = new Vector*[this->amountOfLayer-1];
	Vector** der_Weights_b = new Vector*[this->amountOfLayer-1];
	Vector* der_Input = new Vector(1, this->dimOfLayers[0]);

	for(int i = 0; i < amountOfLayer-1; i++)
	{
		der_Weights[i] = new Vector(weights[i]->getRow(), weights[i]->getCol());
		der_Weights_b[i] = new Vector(weights_b[i]->getRow(), weights_b[i]->getCol());
	}


	//迭代
	while(true)
	{
		if(is_log)
		{
			const time_t t = time(NULL);
			struct tm* cur_time = localtime(&t);

			log_file << cur_time->tm_hour << ":" << cur_time->tm_min << ":" << cur_time->tm_sec << endl;
		}

		for(int i = 0; i < amountOfLayer-1; i++)
		{
			for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
			{
				der_Weights_b[i]->setValue(row, 0, 0);
			}
		}

		for(int row = 0; row < der_Weights[0]->getRow(); row++)
		{
			for(int col = 0; col < der_Weights[0]->getCol(); col++)
			{
				der_Weights[0]->setValue(row, col, 0);
			}
		}

		for(int row = 0; row < der_Weights[1]->getRow(); row++)
		{
			for(int col = 0; col < der_Weights[1]->getCol(); col++)
			{
				der_Weights[1]->setValue(row, col, 0);
			}
		}


		for(int col = 0; col < der_Input->getCol(); col++)
		{
			der_Input->setValue(0, col, 0);
		}

		if(is_log)
		{
			log_file << "Iteration " << count << endl;

			log_file << "Weights 1:" << endl;
			for(int row = 0; row < weights[0]->getRow(); row++)
			{
				for(int col = 0; col < weights[0]->getCol(); col++)
				{
					log_file << weights[0]->getValue(row, col) << " ";
				}
				log_file << endl;
			}
			log_file << "b 1:" << endl;
			for(int row = 0; row < weights_b[0]->getRow(); row++)
			{
				for(int col = 0; col < weights_b[0]->getCol(); col++)
				{
					log_file << weights_b[0]->getValue(row, col) << " ";
				}
				log_file << endl;
			}

			log_file << "Weights 2:" << endl;
			for(int row = 0; row < weights[1]->getRow(); row++)
			{
				for(int col = 0; col < weights[1]->getCol(); col++)
				{
					log_file << weights[1]->getValue(row, col) << " ";
				}
				log_file << endl;
			}
			log_file << "b 2:" << endl;
			for(int row = 0; row < weights_b[1]->getRow(); row++)
			{
				for(int col = 0; col < weights_b[1]->getCol(); col++)
				{
					log_file << weights_b[1]->getValue(row, col) << " ";
				}
				log_file << endl;
			}
		}

		int trueCount = 0;
		int size = v_train_data.size();

		//遍历一遍数据
		for(int j = 0; j < size; j++)
		{
			Vector* input_layer = new Vector(1, this->dimOfLayers[0]);
			Vector* hidden_layer;
			Vector* output_layer;
			Vector* pre_hidden_layer;
			Vector* pre_output_layer;
			double pre_Loss = 0;

			if(is_log)
			{
				log_file << "Data " << j << ":" << endl;
			}

			//feed-forward

			//input layer
			for(int col = 0; col < this->dimOfLayers[0]; col++)
			{
				input_layer->setValue(0, col, v_train_data[j].first->getValue(0, col) - weights_i->getValue(0, col));
			}

			//hidden layer
			pre_hidden_layer = input_layer->Multiply(weights[0], 1);

			for(int i = 0; i < pre_hidden_layer->getCol(); i++)
			{
				pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
			}
			hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
			for(int i = 0; i < hidden_layer->getCol(); i++)
			{
				hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
			}

			//output layer
			pre_output_layer = hidden_layer->Multiply(weights[1], 1);

			for(int i = 0; i < pre_output_layer->getCol(); i++)
			{
				pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
			}
			output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
			for(int i = 0; i < output_layer->getCol(); i++)
			{
				output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
			}

			//pre_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();
			pre_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second);

			if(is_log)
			{
				log_file << "Z of output layer:" << endl;
				for(int i = 0; i < output_layer->getCol(); i++)
				{
					log_file << pre_output_layer->getValue(0, i) << " ";
				}
				log_file << endl;

				log_file << "A of output layer:" << endl;
				for(int i = 0; i < output_layer->getCol(); i++)
				{
					log_file << output_layer->getValue(0, i) << " " << v_train_data[j].second;
				}
				log_file << endl;

				if(j == 0)
				{
					log_file << "Loss Value:" << endl;
					log_file << pre_Loss << endl;
				}
			}

			if(output_layer->getValue(0, 0) > 0.5 && v_train_data[j].second == 1)
			{
				trueCount++;
			}
			else if(output_layer->getValue(0, 0) < 0.5 && v_train_data[j].second == 0)
			{
				trueCount++;
			}
			//feed-backward
			Vector** err_term = new Vector*[this->amountOfLayer];
			for(int i = 0 ; i < amountOfLayer; i++)
			{
				err_term[i] = new Vector(1, this->dimOfLayers[this->amountOfLayer-1-i]);
			}

			//err_term of output layer
			err_term[0]->setValue(0, 0, (output_layer->getValue(0, 0) - v_train_data[j].second) * (sigmoid(pre_output_layer->getValue(0, 0))-pow(sigmoid(pre_output_layer->getValue(0, 0)), 2)));

			//err_term of hidden layer
			for(int i = 0; i < err_term[1]->getCol(); i++)
			{
				double result = sigmoid(pre_hidden_layer->getValue(0, i))-pow(sigmoid(pre_hidden_layer->getValue(0, i)), 2);
				result = result * err_term[0]->getValue(0, 0) * weights[1]->getValue(0, i);
				err_term[1]->setValue(0, i, result);
			}

			//err_term of input layer
			for(int i = 0; i < err_term[2]->getCol(); i++)
			{
				double result = 0;

				for(int k = 0; k < err_term[1]->getCol(); k++)
				{
					result += err_term[1]->getValue(0, k) * weights[0]->getValue(k, i);
				}

				err_term[2]->setValue(0, i, result);

				der_Input->setValue(0, i, result);
			}

			//求导
			for(int i = 0; i < amountOfLayer-1; i++)
			{
				for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
				{
					der_Weights_b[i]->setValue(row, 0, der_Weights_b[i]->getValue(row, 0)+err_term[amountOfLayer-2-i]->getValue(0, row));
				}
			}

			for(int row = 0; row < der_Weights[0]->getRow(); row++)
			{
				for(int col = 0; col < der_Weights[0]->getCol(); col++)
				{
					der_Weights[0]->setValue(row, col, der_Weights[0]->getValue(row,col)+err_term[1]->getValue(0, row) * v_train_data[j].first->getValue(0, col));
				}
			}

			for(int row = 0; row < der_Weights[1]->getRow(); row++)
			{
				for(int col = 0; col < der_Weights[1]->getCol(); col++)
				{
					der_Weights[1]->setValue(row, col, der_Weights[1]->getValue(row,col)+err_term[0]->getValue(0, row) * hidden_layer->getValue(0, col));
				}
			}

			cout << "Iteration:" << count << "\tData: " << j << endl;

			delete hidden_layer;
			delete pre_hidden_layer;
			delete output_layer;
			delete pre_output_layer;
			delete input_layer;
			for(int i = 0; i < amountOfLayer; i++)
			{
				delete err_term[i];
			}
			delete[] err_term;
		}

		//更新参数
		for(int i = 0; i < amountOfLayer-1; i++)
		{
			for(int row = 0; row < der_Weights[i]->getRow(); row++)
			{
				for(int col = 0; col < der_Weights[i]->getCol(); col++)
				{
					weights[i]->setValue(row, col, weights[i]->getValue(row,col) - alpha*(der_Weights[i]->getValue(row, col)/v_train_data.size()));
					//weights[i]->setValue(row, col, weights[i]->getValue(row,col) - alpha*(der_Weights[i]->getValue(row, col)/v_train_data.size() + lambda*weights[i]->getValue(row,col)));
				}
			}

			for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
			{
				weights_b[i]->setValue(row, 0, weights_b[i]->getValue(row, 0) - alpha*(der_Weights_b[i]->getValue(row, 0)/v_train_data.size()));
			}
		}

		//更新输入层权重
		for(int col = 0; col < this->dimOfLayers[0]; col++)
		{
			weights_i->setValue(0, col, weights_i->getValue(0,col) + der_Input->getValue(0, col)/v_train_data.size());
		}

		log_file << "True count: " << trueCount << endl;
		log_file << count << " Accuary:" << (double)trueCount/(double)v_train_data.size() << endl;

		if(count == this->iterTimes)
		{		 
			break;
		}

		count++;
	}

	/*for(int j = 0; j < v_train_data.size(); j++)
	{
		for(int a = 0; a < posNote[j].size(); a++)
		{
			map<string, double*>::iterator m_it = words->m_words.find(posNote[j][a]);

			for(int col = 0; col < this->dimOfLayers[0]; col++)
			{
				log_file << m_it->second[col] << endl;
				m_it->second[col] -= (weights_i->getValue(0, col)/posNote[j].size());
				log_file << m_it->second[col] << endl;
			}
		}
	}*/

	//梯度检验
	if(true)
	{
		//W
		for(int lay = 1; lay >= 0; lay--)
		{
			for(int row = 0; row < weights[lay]->getRow(); row++)
			{
				for(int col = 0; col < weights[lay]->getCol(); col++)
				{
					log_file << "weights[" << lay << "]" << " row: " << row << " col: " << col << endl;

					for(int j = v_train_data.size()-10; j < v_train_data.size(); j++)
					{
						Vector* hidden_layer;
						Vector* output_layer;
						Vector* pre_hidden_layer;
						Vector* pre_output_layer;
						double min_Loss = -1;
						double add_Loss = -1;


						//g(theta+10^5)
						weights[lay]->setValue(row, col, weights[lay]->getValue(row, col)+pow(10, -5));

						pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

						for(int i = 0; i < pre_hidden_layer->getCol(); i++)
						{
							pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
						}
						hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
						for(int i = 0; i < hidden_layer->getCol(); i++)
						{
							hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
						}

						//output layer
						pre_output_layer = hidden_layer->Multiply(weights[1], 1);

						for(int i = 0; i < pre_output_layer->getCol(); i++)
						{
							pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
						}
						output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
						for(int i = 0; i < output_layer->getCol(); i++)
						{
							output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
						}

						//add_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();
						add_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second);

						//g(theta-10^5)
						weights[lay]->setValue(row, col, weights[lay]->getValue(row, col)-2*pow(10, -5));
						pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

						for(int i = 0; i < pre_hidden_layer->getCol(); i++)
						{
							pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
						}
						hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
						for(int i = 0; i < hidden_layer->getCol(); i++)
						{
							hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
						}


						//output layer
						pre_output_layer = hidden_layer->Multiply(weights[1], 1);

						for(int i = 0; i < pre_output_layer->getCol(); i++)
						{
							pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
						}
						output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
						for(int i = 0; i < output_layer->getCol(); i++)
						{
							output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
						}

						//min_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();
						min_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second);
						weights[lay]->setValue(row, col, weights[lay]->getValue(row, col)+pow(10, -5));

						double g = (add_Loss-min_Loss) / (2*pow(10, -5));


						//feed-backward
						Vector** err_term = new Vector*[this->amountOfLayer-1];
						for(int i = 0 ; i < amountOfLayer-1; i++)
						{
							err_term[i] = new Vector(1, this->dimOfLayers[this->amountOfLayer-1-i]);
						}

						err_term[0]->setValue(0, 0, (output_layer->getValue(0, 0) - v_train_data[j].second)*(sigmoid(pre_output_layer->getValue(0, 0))-pow(sigmoid(pre_output_layer->getValue(0, 0)), 2)));

						for(int i = 0; i < err_term[1]->getCol(); i++)
						{
							double result = sigmoid(pre_hidden_layer->getValue(0, i)) - pow(sigmoid(pre_hidden_layer->getValue(0, i)), 2);
							result *= (err_term[0]->getValue(0, 0)*weights[1]->getValue(0, i));
							err_term[1]->setValue(0, i, result);
						}

						//求导
						for(int i = 0; i < amountOfLayer-1; i++)
						{
							for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
							{
								der_Weights_b[i]->setValue(row, 0, err_term[amountOfLayer-2-i]->getValue(0, row));
							}
						}

						for(int row1 = 0; row1 < der_Weights[0]->getRow(); row1++)
						{
							for(int col1 = 0; col1 < der_Weights[0]->getCol(); col1++)
							{
								//der_Weights[0]->setValue(row1, col1, err_term[1]->getValue(0, row1) * v_train_data[j].first->getValue(0, col1) + lambda*weights[0]->getValue(row1,col1));
								der_Weights[0]->setValue(row1, col1, err_term[1]->getValue(0, row1) * v_train_data[j].first->getValue(0, col1));
							}
						}

						for(int row1 = 0; row1 < der_Weights[1]->getRow(); row1++)
						{
							for(int col1 = 0; col1 < der_Weights[1]->getCol(); col1++)
							{
								//der_Weights[1]->setValue(row1, col1, err_term[0]->getValue(0, row1) * hidden_layer->getValue(0, col1) + lambda*weights[1]->getValue(row1,col1));
								der_Weights[1]->setValue(row1, col1, err_term[0]->getValue(0, row1) * hidden_layer->getValue(0, col1));
							}
						}

						log_file << g << "||" << der_Weights[lay]->getValue(row, col) << " " << endl;

						delete hidden_layer;
						delete pre_hidden_layer;
						delete output_layer;
						delete pre_output_layer;
					}

					log_file << endl;
				}
			}
		}

		//b
		for(int lay = 1; lay >= 0; lay--)
		{
			for(int row = 0; row < weights_b[lay]->getRow(); row++)
			{
				for(int col = 0; col < weights_b[lay]->getCol(); col++)
				{
					log_file << "weights_b[" << lay << "]" << " row: " << row << " col: " << col << endl;

					for(int j = v_train_data.size()-10; j < v_train_data.size(); j++)
					{
						Vector* hidden_layer;
						Vector* output_layer;
						Vector* pre_hidden_layer;
						Vector* pre_output_layer;
						double min_Loss = -1;
						double add_Loss = -1;


						//g(theta-10^+4)
						weights_b[lay]->setValue(row, col, weights_b[lay]->getValue(row, col)+pow(10, -5));

						pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

						for(int i = 0; i < pre_hidden_layer->getCol(); i++)
						{
							pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
						}
						hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
						for(int i = 0; i < hidden_layer->getCol(); i++)
						{
							hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
						}

						//output layer
						pre_output_layer = hidden_layer->Multiply(weights[1], 1);

						for(int i = 0; i < pre_output_layer->getCol(); i++)
						{
							pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
						}
						output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
						for(int i = 0; i < output_layer->getCol(); i++)
						{
							output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
						}

						add_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();

						//g(theta-10^-4)
						weights_b[lay]->setValue(row, col, weights_b[lay]->getValue(row, col)-2*pow(10, -5));
						pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

						for(int i = 0; i < pre_hidden_layer->getCol(); i++)
						{
							pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
						}
						hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
						for(int i = 0; i < hidden_layer->getCol(); i++)
						{
							hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
						}


						//output layer
						pre_output_layer = hidden_layer->Multiply(weights[1], 1);

						for(int i = 0; i < pre_output_layer->getCol(); i++)
						{
							pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
						}
						output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
						for(int i = 0; i < output_layer->getCol(); i++)
						{
							output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
						}

						min_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();
						weights_b[lay]->setValue(row, col, weights_b[lay]->getValue(row, col)+pow(10, -5));

						double g = (add_Loss-min_Loss) / (2*pow(10, -5));


						//feed-backward
						Vector** err_term = new Vector*[this->amountOfLayer-1];
						for(int i = 0 ; i < amountOfLayer-1; i++)
						{
							err_term[i] = new Vector(1, this->dimOfLayers[this->amountOfLayer-1-i]);
						}

						err_term[0]->setValue(0, 0, (output_layer->getValue(0, 0) - v_train_data[j].second)*(sigmoid(pre_output_layer->getValue(0, 0))-pow(sigmoid(pre_output_layer->getValue(0, 0)), 2)));

						for(int i = 0; i < err_term[1]->getCol(); i++)
						{
							double result = sigmoid(pre_hidden_layer->getValue(0, i)) - pow(sigmoid(pre_hidden_layer->getValue(0, i)), 2);
							result *= (err_term[0]->getValue(0, 0)*weights[1]->getValue(0, i));
							err_term[1]->setValue(0, i, result);
						}

						//求导
						for(int i = 0; i < amountOfLayer-1; i++)
						{
							for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
							{
								der_Weights_b[i]->setValue(row, 0, err_term[amountOfLayer-2-i]->getValue(0, row));
							}
						}

						for(int row1 = 0; row1 < der_Weights[0]->getRow(); row1++)
						{
							for(int col1 = 0; col1 < der_Weights[0]->getCol(); col1++)
							{
								der_Weights[0]->setValue(row1, col1, err_term[1]->getValue(0, row1) * v_train_data[j].first->getValue(0, col1));
							}
						}

						for(int row1 = 0; row1 < der_Weights[1]->getRow(); row1++)
						{
							for(int col1 = 0; col1 < der_Weights[1]->getCol(); col1++)
							{
								der_Weights[1]->setValue(row1, col1, err_term[0]->getValue(0, row1) * hidden_layer->getValue(0, col1));
							}
						}

						log_file << g << "||" << der_Weights_b[lay]->getValue(row, col) << " " << endl;

						delete hidden_layer;
						delete pre_hidden_layer;
						delete output_layer;
						delete pre_output_layer;
					}

					log_file << endl;
				}
			}
		}

		//Input layer
		for(int col = 0; col < words->getVecSize(); col++)
		{
			log_file << "Input " << " col: " << col << endl;

			for(int j = v_train_data.size()-10; j < v_train_data.size(); j++)
			{
				Vector* hidden_layer;
				Vector* output_layer;
				Vector* pre_hidden_layer;
				Vector* pre_output_layer;
				double min_Loss = -1;
				double add_Loss = -1;


				//g(theta-10^+5)
				v_train_data[j].first->setValue(0, col, v_train_data[j].first->getValue(0, col)+pow(10, -5));

				pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

				for(int i = 0; i < pre_hidden_layer->getCol(); i++)
				{
					pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
				}
				hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
				for(int i = 0; i < hidden_layer->getCol(); i++)
				{
					hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
				}

				//output layer
				pre_output_layer = hidden_layer->Multiply(weights[1], 1);

				for(int i = 0; i < pre_output_layer->getCol(); i++)
				{
					pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
				}
				output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
				for(int i = 0; i < output_layer->getCol(); i++)
				{
					output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
				}

				add_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();

				//g(theta-10^-5)
				v_train_data[j].first->setValue(0, col, v_train_data[j].first->getValue(0, col) - 2*pow(10, -5));
				pre_hidden_layer = v_train_data[j].first->Multiply(weights[0], 1);

				for(int i = 0; i < pre_hidden_layer->getCol(); i++)
				{
					pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
				}
				hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
				for(int i = 0; i < hidden_layer->getCol(); i++)
				{
					hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
				}


				//output layer
				pre_output_layer = hidden_layer->Multiply(weights[1], 1);

				for(int i = 0; i < pre_output_layer->getCol(); i++)
				{
					pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
				}
				output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
				for(int i = 0; i < output_layer->getCol(); i++)
				{
					output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
				}

				min_Loss = loss(output_layer->getValue(0, 0), v_train_data[j].second) + Regularization();
				v_train_data[j].first->setValue(0, col, v_train_data[j].first->getValue(0, col)+pow(10, -5));

				double g = (add_Loss-min_Loss) / (2*pow(10, -5));


				//feed-backward
				Vector** err_term = new Vector*[this->amountOfLayer];
				for(int i = 0 ; i < amountOfLayer; i++)
				{
					err_term[i] = new Vector(1, this->dimOfLayers[this->amountOfLayer-1-i]);
				}

				err_term[0]->setValue(0, 0, (output_layer->getValue(0, 0) - v_train_data[j].second)*(sigmoid(pre_output_layer->getValue(0, 0))-pow(sigmoid(pre_output_layer->getValue(0, 0)), 2)));

				for(int i = 0; i < err_term[1]->getCol(); i++)
				{
					double result = sigmoid(pre_hidden_layer->getValue(0, i)) - pow(sigmoid(pre_hidden_layer->getValue(0, i)), 2);
					result *= (err_term[0]->getValue(0, 0)*weights[1]->getValue(0, i));
					err_term[1]->setValue(0, i, result);
				}

				//err_term of input layer
				for(int i = 0; i < err_term[2]->getCol(); i++)
				{
					double result = 0;

					for(int k = 0; k < err_term[1]->getCol(); k++)
					{
						result += err_term[1]->getValue(0, k) * weights[0]->getValue(k, i);
					}

					err_term[2]->setValue(0, i, result);
				}

				//求导
				for(int i = 0; i < amountOfLayer-1; i++)
				{
					for(int row = 0; row < der_Weights_b[i]->getRow(); row++)
					{
						der_Weights_b[i]->setValue(row, 0, err_term[amountOfLayer-2-i]->getValue(0, row));
					}
				}

				for(int row1 = 0; row1 < der_Weights[0]->getRow(); row1++)
				{
					for(int col1 = 0; col1 < der_Weights[0]->getCol(); col1++)
					{
						der_Weights[0]->setValue(row1, col1, err_term[1]->getValue(0, row1) * v_train_data[j].first->getValue(0, col1));
					}
				}

				for(int row1 = 0; row1 < der_Weights[1]->getRow(); row1++)
				{
					for(int col1 = 0; col1 < der_Weights[1]->getCol(); col1++)
					{
						der_Weights[1]->setValue(row1, col1, err_term[0]->getValue(0, row1) * hidden_layer->getValue(0, col1));
					}
				}

				log_file << g << "||" << err_term[2]->getValue(0, col) << " " << endl;

				delete hidden_layer;
				delete pre_hidden_layer;
				delete output_layer;
				delete pre_output_layer;
			}

			log_file << endl;
		}
	}

	ofstream fout("output.ds");
	fout << "Weights 1:" << endl;
	for(int row = 0; row < weights[0]->getRow(); row++)
	{
		for(int col = 0; col < weights[0]->getCol(); col++)
		{
			fout << weights[0]->getValue(row, col) << " ";
		}
		fout << endl;
	}
	fout << "b 1:" << endl;
	for(int row = 0; row < weights_b[0]->getRow(); row++)
	{
		for(int col = 0; col < weights_b[0]->getCol(); col++)
		{
			fout << weights_b[0]->getValue(row, col) << " ";
		}
		fout << endl;
	}

	fout << "Weights 2:" << endl;
	for(int row = 0; row < weights[1]->getRow(); row++)
	{
		for(int col = 0; col < weights[1]->getCol(); col++)
		{
			fout << weights[1]->getValue(row, col) << " ";
		}
		fout << endl;
	}
	fout << "b 2:" << endl;
	for(int row = 0; row < weights_b[1]->getRow(); row++)
	{
		for(int col = 0; col < weights_b[1]->getCol(); col++)
		{
			fout << weights_b[1]->getValue(row, col) << " ";
		}
		fout << endl;
	}

	fout.close();
}

double Network::Regularization()
{
	double result = 0;

	for(int i = 0 ; i < amountOfLayer-1; i++)
	{
		for(int row = 0; row < weights[i]->getRow(); row++)
		{
			for(int col = 0; col < weights[i]->getCol(); col++)
			{
				result += pow(weights[i]->getValue(row, col), 2);
			}
		}
	}

	return lambda * result / 2;
}

void Network::test(Parameter* para)
{
	ofstream fout("test_res");
	v_train_data.clear();

	get_data("test.ds");
	int true_count = 0;

	for(int j = 0; j < v_train_data.size(); j++)
	{
		Vector* hidden_layer;
		Vector* output_layer;
		Vector* pre_hidden_layer;
		Vector* pre_output_layer;
		Vector* input_layer = new Vector(1, this->dimOfLayers[0]);

		//feed-forward

		for(int col = 0; col < this->dimOfLayers[0]; col++)
		{
			input_layer->setValue(0, col, v_train_data[j].first->getValue(0, col) - weights_i->getValue(0, col));
		}

		//hidden layer
		pre_hidden_layer = input_layer->Multiply(weights[0], 1);
		//transpos delete
		for(int i = 0; i < pre_hidden_layer->getCol(); i++)
		{
			pre_hidden_layer->setValue(0, i, pre_hidden_layer->getValue(0, i) + weights_b[0]->getValue(i, 0));
		}
		hidden_layer = new Vector(pre_hidden_layer->getRow(), pre_hidden_layer->getCol());
		for(int i = 0; i < hidden_layer->getCol(); i++)
		{
			hidden_layer->setValue(0, i, sigmoid(pre_hidden_layer->getValue(0, i)));
		}

		//output layer
		pre_output_layer = hidden_layer->Multiply(weights[1], 1);
		//transpos delete
		for(int i = 0; i < pre_output_layer->getCol(); i++)
		{
			pre_output_layer->setValue(0, i, pre_output_layer->getValue(0, i) + weights_b[1]->getValue(i, 0));
		}
		output_layer = new Vector(pre_output_layer->getRow(), pre_output_layer->getCol());
		for(int i = 0; i < output_layer->getCol(); i++)
		{
			output_layer->setValue(0, i, sigmoid(pre_output_layer, i));
		}

		if(output_layer->getValue(0, 0) > 0.5)
		{
			if(v_train_data[j].second == 1)
			{
				true_count++;
			}
		}
		else
		{
			if(v_train_data[j].second == 0)
			{
				true_count++;
			}
		}

		fout << "Data " << j << ":" << endl;
		fout << output_layer->getValue(0, 0) << "\t\t" << v_train_data[j].second  << "\t\t" << true_count << endl;

		delete hidden_layer;
		delete pre_hidden_layer;
		delete output_layer;
		delete pre_output_layer;
	}

	fout << "Accuracy:\t" << (double)true_count/v_train_data.size() << endl;
}
