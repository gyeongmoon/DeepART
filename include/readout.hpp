#include <vector>
#include <iterator>
#include <algorithm>

template<typename T>
void readoutART(Layer<T> layer, int numEpisode, std::vector< std::vector<double> >& output)
{
	output.resize(layer.x.size());

	for(unsigned int i=0; i<layer.x.size();i++)
	{
		output[i].resize(layer.weight[i][numEpisode].size()/2);

		for(unsigned int j=0; j<layer.weight[i][numEpisode].size()/2; j++)
		{
			output[i][j] = layer.weight[i][numEpisode][j];
		}
	}
}

bool near_zero(double i)
{
	return (i < 0.000001);
}

void readout(Layer<int> layer_obj, Layer<int> layer_obj2, Layer<int> layer1, Layer<double> layer2, int numEpisode, std::vector<Episode> episode)
{
	std::vector< std::vector<double> > output1;
	std::vector< std::vector<double> > output_obj;
	std::vector< std::vector<double> > output_obj2;

	std::vector< std::vector<double> > normalized_oy;
	std::vector< std::vector<double> > ix(episode[numEpisode].numEvent);
	std::vector< std::vector<double> > bx(episode[numEpisode].numEvent);
	std::vector< std::vector<double> > oy(episode[numEpisode].numEvent);
	std::vector< std::vector<double> > temp(episode[numEpisode].numEvent);

	int L = 0;
	int E = 0;

	readoutART(layer2, numEpisode, normalized_oy); // Weight readout process..

	for(auto& i:normalized_oy[0])
	{
		if(near_zero(i))
			i = 0;
	}

	for(int i=0; i<episode[numEpisode].numEvent; i++)
	{
		ix[i].resize(normalized_oy[0].size(),0);
		bx[i].resize(normalized_oy[0].size(),0);
		oy[i].resize(normalized_oy[0].size(),0);
		temp[i].resize(normalized_oy[0].size(),0);

		if(i == 0)
		{
			// De-normalization ..
			for(unsigned int j=0; j<normalized_oy[0].size(); j++)
				oy[i][j] = int(normalized_oy[0][j]*std::pow(10,max_position_num[numEpisode]));
		}
		else
			oy[i] = bx[i-1];

		L = std::distance(oy[i].begin(), std::max_element(oy[i].begin(), oy[i].end()));
		for(int k=0; k<episode[numEpisode].numEvent; k++)
		{
			if(std::abs(iw*std::pow((bw*ow),k) - oy[i][L]) < 0.001)
			{
				E = k;
				break;
			}
			else if(iw*std::pow((bw*ow),k) > oy[i][L])
			{
				E = k-1;
				break;
			}
			else if(k == (episode[numEpisode].numEvent-1))
				E = k;
		}
		ix[i][L] = 1;
		bx[i] = oy[i];
		bx[i][L] = oy[i][L] - iw*std::pow((bw*ow),E)*ix[i][L];

		readoutART(layer1, std::distance(ix[i].begin(), std::max_element(ix[i].begin(), ix[i].end())), output1);
		readoutART(layer_obj, std::distance(output1[1].begin(), std::max_element(output1[1].begin(), output1[1].end())), output_obj);
		readoutART(layer_obj2, std::distance(output1[2].begin(), std::max_element(output1[2].begin(), output1[2].end())), output_obj2);

		std::stringstream ss;

		std::cout << "Event " << i << " : " << std::endl;

		ss << ACTION[std::distance(output1[0].begin(), std::max_element(output1[0].begin(), output1[0].end()))] << " ";

		if(!near_zero(*std::max_element(output_obj[4].begin(), output_obj[4].end())))
			ss << OBJECT_PREPOSITION[std::distance(output_obj[4].begin(), std::max_element(output_obj[4].begin(), output_obj[4].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj[1].begin(), output_obj[1].end())))
			ss << OBJECT_COLOR[std::distance(output_obj[1].begin(), std::max_element(output_obj[1].begin(), output_obj[1].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj[2].begin(), output_obj[2].end())))
			ss << OBJECT_SHAPE[std::distance(output_obj[2].begin(), std::max_element(output_obj[2].begin(), output_obj[2].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj[3].begin(), output_obj[3].end())))
			ss << OBJECT_TYPE[std::distance(output_obj[3].begin(), std::max_element(output_obj[3].begin(), output_obj[3].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj[0].begin(), output_obj[0].end())))
			ss << OBJECT[std::distance(output_obj[0].begin(), std::max_element(output_obj[0].begin(), output_obj[0].end()))] << " ";

		if(!near_zero(*std::max_element(output_obj2[4].begin(), output_obj2[4].end())))
			ss << OBJECT_PREPOSITION[std::distance(output_obj2[4].begin(), std::max_element(output_obj2[4].begin(), output_obj2[4].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj2[1].begin(), output_obj2[1].end())))
			ss << OBJECT_COLOR[std::distance(output_obj2[1].begin(), std::max_element(output_obj2[1].begin(), output_obj2[1].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj2[2].begin(), output_obj2[2].end())))
			ss << OBJECT_SHAPE[std::distance(output_obj2[2].begin(), std::max_element(output_obj2[2].begin(), output_obj2[2].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj2[3].begin(), output_obj2[3].end())))
			ss << OBJECT_TYPE[std::distance(output_obj2[3].begin(), std::max_element(output_obj2[3].begin(), output_obj2[3].end()))] << " ";
		if(!near_zero(*std::max_element(output_obj2[0].begin(), output_obj2[0].end())))
			ss << OBJECT[std::distance(output_obj2[0].begin(), std::max_element(output_obj2[0].begin(), output_obj2[0].end()))];

		std::cout << ss.str() << std::endl;
	}
}
