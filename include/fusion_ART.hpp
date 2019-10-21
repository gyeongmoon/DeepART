#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>

// Sort Template
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

	return idx;
}

// Sample Input Template
template<typename T>
struct Sample_input{
	std::vector< std::vector<T> > input;
};

// ART Init.
template<typename T>
void initART(Layer<T>& layer)
{
	std::vector< std::vector< std::vector<double> > > _weight(layer.option.numChannel, std::vector< std::vector<double> > (1, std::vector<double> (1, 1)));
	std::vector< std::vector<T> > _x(layer.option.numChannel, std::vector<T> (1));

	layer.weight = _weight;
	layer.x = _x;
}

// ART Activate.
template<typename T>
void activateART(Layer<T>& layer)
{
	double weightLength;
	std::vector<double> matchVector;

	for(unsigned int i=0; i<layer.y.size(); i++)
	{
		layer.y[i] = 0;

		for(auto j=0; j<layer.option.numChannel; j++)
		{
			for(unsigned int k=0; k<layer.x[j].size(); k++)
			{
				if(layer.x[j][k] < layer.weight[j][i][k])
					matchVector.push_back(layer.x[j][k]);
				else
					matchVector.push_back(layer.weight[j][i][k]);
			}

			weightLength = std::accumulate(layer.weight[j][i].begin(), layer.weight[j][i].end(), 0.0);
			layer.y[i] = layer.y[i] + layer.option.contribution[j]*std::accumulate(matchVector.begin(), matchVector.end(), 0.0)/(layer.option.bias[j]+weightLength);

			matchVector.clear();
		}
	}
}

// Complement Coding.
template<typename T>
void complementCoding(std::vector< std::vector<T> >& sampledInput)
{
	std::vector<std::vector<T>> complement = sampledInput;

	for(unsigned int i=0; i<sampledInput.size(); i++)
	{
		std::fill(complement[i].begin(), complement[i].end(), 1);
		std::transform(complement[i].begin(), complement[i].end(), sampledInput[i].begin(), complement[i].begin(), std::minus<double>());
		sampledInput[i].insert(sampledInput[i].end(), complement[i].begin(), complement[i].end());
	}
}

template<typename T, typename U>
void getMaxIndex(std::vector< std::vector<T> > sampledInput, Layer<U> layer, int& category)
{
	complementCoding(sampledInput);

	for(unsigned int i=0; i<layer.x.size(); i++)
		layer.x[i] = sampledInput[i];

	activateART(layer);
	category = std::distance(layer.y.begin(), std::max_element(layer.y.begin(), layer.y.end()));
}

template<typename T>
void makeInput2(int numEpisode, std::vector<Episode>& episode, Layer<T> layer_obj, Layer<T> layer_obj2)
{
	int category = 0;
	Sample_input<int> sampledInput_obj;
	Sample_input<int> sampledInput_plc;

	for(int i=0; i<numEpisode; i++)
	{
		episode[i].nObject.resize(episode[i].numEvent);
		episode[i].nObject2.resize(episode[i].numEvent);

		sampledInput_obj.input.resize(layer_obj.option.numChannel);
		sampledInput_plc.input.resize(layer_obj2.option.numChannel);

		for(int j=0; j<episode[i].numEvent; j++)
		{
			episode[i].nObject[j].resize(layer_obj.y.size(),0);
			episode[i].nObject2[j].resize(layer_obj2.y.size(),0);

			sampledInput_obj.input[0] = episode[i].object[j];
			sampledInput_obj.input[1] = episode[i].object_color[j];
			sampledInput_obj.input[2] = episode[i].object_shape[j];
			sampledInput_obj.input[3] = episode[i].object_type[j];
			sampledInput_obj.input[4] = episode[i].object_preposition[j];

			getMaxIndex<int>(sampledInput_obj.input, layer_obj, category);
			episode[i].nObject[j][category] = 1;

			sampledInput_plc.input[0] = episode[i].object2[j];
			sampledInput_plc.input[1] = episode[i].object2_color[j];
			sampledInput_plc.input[2] = episode[i].object2_shape[j];
			sampledInput_plc.input[3] = episode[i].object2_type[j];
			sampledInput_plc.input[4] = episode[i].object2_preposition[j];

			getMaxIndex<int>(sampledInput_plc.input, layer_obj2, category);
			episode[i].nObject2[j][category] = 1;
		}
	}
}

// Template Matching
template<typename T>
void matchART(std::vector< std::vector<T> > x, std::vector< std::vector<double> > weight, std::vector<double>& match)
{
	std::vector<double> check;

	for(unsigned int i=0; i<weight.size(); i++)
	{
		for(unsigned int j=0; j<x[i].size(); j++)
		{
			if(x[i][j] < weight[i][j])
				check.push_back(x[i][j]);
			else
				check.push_back(weight[i][j]);
		}
		match[i] = std::accumulate(check.begin(), check.end(), 0.0)/std::accumulate(x[i].begin(), x[i].end(), 0.0);
		check.clear();
	}
}

// Weight Update
template<typename T>
void updateART(Layer<T>& layer, int resonanceIdx)
{
	for(int i=0; i<layer.option.numChannel; i++)
	{
		for(unsigned int j=0; j<layer.x[i].size(); j++)
			layer.weight[i][resonanceIdx-1][j] = layer.option.learningRate[i]*layer.x[i][j]+(1-layer.option.learningRate[i])*layer.weight[i][resonanceIdx-1][j];
	}
}

// ART Learn.
template<typename T, typename U>
void learnART(std::vector< std::vector<T> > sampledInput, Layer<U>& layer)
{
	complementCoding(sampledInput);

	int resonanceIdx = 0;
	std::vector<double> match(layer.option.numChannel);
	std::vector<long unsigned int> idx(layer.y.size());
	std::vector< std::vector<double> > weight(layer.option.numChannel);

	for(unsigned int m=0; m<layer.x.size(); m++)
		layer.x[m] = sampledInput[m];

	for(int i=0; i<layer.option.maxIter; i++)
	{
		activateART(layer);

		idx.assign(layer.y.begin(), layer.y.end());
		for(unsigned int h=0; h<layer.y.size(); h++)
			idx[h] = (-1)*layer.y[h];

		idx = sort_indexes(idx);

		for(unsigned int j=0; j<layer.y.size(); j++)
		{
			for(int k=0; k<layer.option.numChannel; k++)
				weight[k] = layer.weight[k][idx[j]];

			match.assign(layer.option.numChannel, 0);
			matchART(layer.x, weight, match);

		// match should be checked for all channels.
		//	for(k=0; k<match.size(); k++)
		//	{
		//		if(match)
		//	}
			if(match > layer.option.vigiliance)
			{
				if(std::accumulate(match.begin(), match.end(), 0.0) == layer.option.numChannel)
				{
					resonanceIdx = idx[j]+1;
					break;
				}
			}
		}

		if(resonanceIdx == 0)
		{
			layer.y.push_back(1);
			resonanceIdx = (int)(layer.y.size());
			for(int l=0; l<layer.option.numChannel; l++)
			{
				layer.weight[l].resize(resonanceIdx);
				layer.weight[l][layer.weight[l].size()-1] = std::vector<double> (layer.x[l].size(),1);
			}
		}

		updateART(layer, resonanceIdx);
	}
}

// Main
template<typename T>
void fusion_ART(std::vector<Episode>& input, Layer<T>& layer)
{
	initART(layer);

	Sample_input<T> sampledInput;

	for(int i=0; i<NUM_EPISODE; i++)
	{
		sampledInput.input.resize(layer.option.numChannel);

		for(int j=0; j<input[i].numEvent; j++)
		{
			if(layer.identity == 0)
			{
				sampledInput.input[0] = input[i].object[j];
				sampledInput.input[1] = input[i].object_color[j];
				sampledInput.input[2] = input[i].object_shape[j];
				sampledInput.input[3] = input[i].object_type[j];
				sampledInput.input[4] = input[i].object_preposition[j];
			}
			else if(layer.identity == 1)
			{
				sampledInput.input[0] = input[i].object2[j];
				sampledInput.input[1] = input[i].object2_color[j];
				sampledInput.input[2] = input[i].object2_shape[j];
				sampledInput.input[3] = input[i].object2_type[j];
				sampledInput.input[4] = input[i].object2_preposition[j];
			}
			else
			{
				sampledInput.input[0] = input[i].action[j];
				sampledInput.input[1] = input[i].nObject[j];
				sampledInput.input[2] = input[i].nObject2[j];
			}

			learnART<int, int>(sampledInput.input, layer);
		}
	}
}
