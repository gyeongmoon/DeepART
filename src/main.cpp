#include <vector>
#include <iostream>

#include "makeInput.hpp"
#include "fusion_ART.hpp"
#include "DeepART.hpp"
#include "inference.hpp"
#include "readout.hpp"

int main()
{
	// Read Files & Encoding..
	readFile();

	// Define Inputs..
	std::vector<Episode> episode(NUM_EPISODE);

	// Make Inputs..
	makeInput(NUM_EPISODE, EVENT, episode);

	// Learning The Input field from the Attribute field..
	Layer<int> layer_obj(0, 0);
	Layer<int> layer_obj2(0, 1);
	fusion_ART(episode, layer_obj);
	fusion_ART(episode, layer_obj2);

	// Learning The Event field..
	Layer<int> layer1(1, 2, std::vector<int> {(int)layer_obj.y.size(),(int)layer_obj2.y.size()});
	makeInput2(NUM_EPISODE, episode, layer_obj, layer_obj2);
	fusion_ART(episode, layer1);

	// Learning The Episode field..
	Layer<double> layer2(2, 3, std::vector<int> ((int)layer1.y.size(), 0));
	DeepART(episode, layer1, layer2);

	// Inference..
	std::vector<Episode> cue(NUM_CUE);
	makeInput(NUM_CUE, CUE, cue);
	makeInput2(NUM_CUE, cue, layer_obj, layer_obj2);

	int numEpisode = 0;
	inference(cue[NUM_CUE-1], layer1, layer2, numEpisode);

	// Readout..
	readout(layer_obj, layer_obj2, layer1, layer2, numEpisode, episode);

	return 0;
}
