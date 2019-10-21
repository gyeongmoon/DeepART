#include <cmath>
#include <algorithm>
#include <functional>

int positionalNum(int value)
{
	int count = 0;
	do{
		value = int(value/10);
		count++;
	}while(value>0);

	return count;
}

template<typename T, typename U>
void DeepART(std::vector<Episode> episode, Layer<T> layer1, Layer<U>& layer2)
{
	int category = 0;
	Sample_input<int> sampledInput;

	// Episode - Event - Value
	std::vector<std::vector<std::vector<double>>> ix(NUM_EPISODE);
	std::vector<std::vector<std::vector<double>>> bx(NUM_EPISODE);
	std::vector<std::vector<std::vector<double>>> oy(NUM_EPISODE);
	std::vector<std::vector<std::vector<double>>> temp(NUM_EPISODE);

	std::vector< std::vector< std::vector<double> > > normalized_oy(NUM_EPISODE, std::vector< std::vector<double> > (layer2.option.numChannel, std::vector<double> (layer2.option.numInput[0],0.0)));

	initART(layer2);
	max_position_num.clear();
	max_position_num.resize(NUM_EPISODE);

	for(int i=0; i<NUM_EPISODE; i++)
	{
		ix[i].resize(episode[i].numEvent);
		bx[i].resize(episode[i].numEvent);
		oy[i].resize(episode[i].numEvent);
		temp[i].resize(episode[i].numEvent);

		sampledInput.input.resize(layer1.option.numChannel);

		for(int j=0; j<episode[i].numEvent; j++)
		{
			ix[i][j].resize(layer1.y.size(),0);
			bx[i][j].resize(layer1.y.size(),0);
			oy[i][j].resize(layer1.y.size(),0);
			temp[i][j].resize(layer1.y.size(),0);

			sampledInput.input[0] = episode[i].action[j];
			sampledInput.input[1] = episode[i].nObject[j];
			sampledInput.input[2] = episode[i].nObject2[j];

			// bx[i][j] = ow*oy[i][j-1];
			if(j != 0)
				std::transform(oy[i][j-1].begin(), oy[i][j-1].end(), bx[i][j].begin(), std::bind1st(std::multiplies<T>(),ow));

			// ix[i][j]
			getMaxIndex<int>(sampledInput.input, layer1, category);
			ix[i][j][category] = 1;

			// oy[i][j] = iw*ix[i][j];
			std::transform(ix[i][j].begin(), ix[i][j].end(), oy[i][j].begin(), std::bind1st(std::multiplies<T>(),iw));
			// temp[i][j] = bw*bx[i][j];
			std::transform(bx[i][j].begin(), bx[i][j].end(), temp[i][j].begin(), std::bind1st(std::multiplies<T>(),bw));

			// oy[i][j] = oy[i][j] + temp[i][j] = iw*ix[i][j] + bw*bx[i][j];
			std::transform(oy[i][j].begin(), oy[i][j].end(), temp[i][j].begin(), oy[i][j].begin(), std::plus<double>());
		}

		max_position_num[i] = positionalNum(*std::max_element(oy[i][episode[i].numEvent-1].begin(), oy[i][episode[i].numEvent-1].end()));
		normalized_oy[i][0] = oy[i][episode[i].numEvent-1];

		for(auto& u:normalized_oy[i][0])
			u = u/std::pow(10,max_position_num[i]);

		learnART<double, double>(normalized_oy[i], layer2); // Event layer learning..
	}
}
