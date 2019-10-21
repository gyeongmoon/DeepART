template<typename T, typename U>
void inference(Episode input, Layer<T> layer1, Layer<U> layer2, int& numEpisode)
{
	// Event Inference..
	int max_pos_num, category = 0;
	Layer<double> memory(layer2);
	Sample_input<int> sampledInput;
	sampledInput.input.resize(layer1.option.numChannel);

	// Event - Value
	std::vector<std::vector<double>> ix(input.numEvent);
	std::vector<std::vector<double>> bx(input.numEvent);
	std::vector<std::vector<double>> oy(input.numEvent);
	std::vector<std::vector<double>> temp(input.numEvent);

	std::vector<double> normalized_oy(layer2.option.numInput[0],0.0);

	for(int i=0; i<input.numEvent; i++)
	{
		ix[i].resize(layer1.y.size(),0);
		bx[i].resize(layer1.y.size(),0);
		oy[i].resize(layer1.y.size(),0);
		temp[i].resize(layer1.y.size(),0);

		sampledInput.input[0] = input.action[i];
		sampledInput.input[1] = input.nObject[i];
		sampledInput.input[2] = input.nObject2[i];

		// bx[i][j] = ow*oy[i][j-1];
		if(i != 0)
			std::transform(oy[i-1].begin(), oy[i-1].end(), bx[i].begin(), std::bind1st(std::multiplies<T>(),ow));

		// ix[i][j]
		getMaxIndex<int>(sampledInput.input, layer1, category);
		ix[i][category] = 1;

		// oy[i][j] = iw*ix[i][j];
		std::transform(ix[i].begin(), ix[i].end(), oy[i].begin(), std::bind1st(std::multiplies<T>(),iw));
		// temp[i][j] = bw*bx[i][j];
		std::transform(bx[i].begin(), bx[i].end(), temp[i].begin(), std::bind1st(std::multiplies<T>(),bw));

		// oy[i][j] = oy[i][j] + temp[i][j] = iw*ix[i][j] + bw*bx[i][j];
		std::transform(oy[i].begin(), oy[i].end(), temp[i].begin(), oy[i].begin(), std::plus<double>());
	}

	max_pos_num = positionalNum(*std::max_element(oy[input.numEvent-1].begin(), oy[input.numEvent-1].end()));
	normalized_oy = oy[input.numEvent-1];

	for(auto& u:normalized_oy)
		u = u/std::pow(10,max_pos_num);

	memory.x[0].clear(); // For removing the complements..

	for(unsigned int j=0; j<normalized_oy.size(); j++)
		memory.x[0].push_back(normalized_oy[j]);

	//complementCoding(memory.x); // For robust retrieval from partial cues..

	activateART(memory);

	numEpisode = std::distance(memory.y.begin(), std::max_element(memory.y.begin(), memory.y.end()));

	std::cout << "Predicted Episode : " << numEpisode << std::endl;
}
