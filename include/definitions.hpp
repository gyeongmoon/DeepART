#include <vector>

// Define the number of inputs
int NUM_EPISODE, NUM_CUE;

// Define weights for encoding episodes.
const double iw=1., bw=2., ow=1.;
std::vector<int> max_position_num;

// Define Lists for Readout
std::vector<std::string> ACTION, OBJECT, OBJECT_COLOR, OBJECT_SHAPE, OBJECT_TYPE, OBJECT_PREPOSITION;
std::vector< std::vector<std::string> > CUE, EVENT;
std::vector< std::vector<int> > eAction, eObject, eObject_color, eObject_shape, eObject_type, eObject_preposition; // Encoded Action, Object

// Define Input Structure
struct Episode {
	int numEvent;
	std::vector< std::vector<int> > action;

	std::vector< std::vector<int> > object;
	std::vector< std::vector<int> > object_color;
	std::vector< std::vector<int> > object_shape;
	std::vector< std::vector<int> > object_type;
	std::vector< std::vector<int> > object_preposition;

	std::vector< std::vector<int> > object2;
	std::vector< std::vector<int> > object2_color;
	std::vector< std::vector<int> > object2_shape;
	std::vector< std::vector<int> > object2_type;
	std::vector< std::vector<int> > object2_preposition;

	// For the second layer..
	std::vector< std::vector<int> > nObject;
	std::vector< std::vector<int> > nObject2;
};

// Layer Definition
template<typename T>
class Layer{

	struct Option{
		int maxIter;
		int numChannel; 
		std::vector<int> numInput;
		std::vector<int> numOutput;
		std::vector<double> vigiliance;
		std::vector<double> contribution;
		std::vector<double> bias;
		std::vector<double> learningRate;
	};

public:
	int numLayer;
	std::vector<int> size;

	Option option;

	int identity;
	std::vector< std::vector< std::vector<double> > > weight; //X, Y
	std::vector< std::vector<T> > x; //Action, Object, Place
	std::vector<double> y; // t

	Layer(int arg1, int arg2, std::vector<int> arg3={0,0})
	{
		numLayer = arg1;
		identity = arg2;
		size = arg3;

		// For Layer 0..
		if(numLayer == 0)
		{
			option.maxIter = 10;
			option.numChannel = 5; 
			option.numInput = {(int)OBJECT.size(), (int)OBJECT_COLOR.size(), (int)OBJECT_SHAPE.size(), (int)OBJECT_TYPE.size(), (int)OBJECT_PREPOSITION.size()};
			option.numOutput = {0};
			option.vigiliance = {0.9, 0.9, 0.9, 0.9, 0.9};
			option.contribution = {0.2, 0.2, 0.2, 0.2, 0.2};
			option.bias = {1.0000e-006, 1.0000e-006, 1.0000e-006, 1.0000e-006, 1.0000e-006};
			option.learningRate = {0.8, 0.8, 0.8, 0.8, 0.8};
		}

		// For Layer 1..
		if(numLayer == 1)
		{
			option.maxIter = 10;
			option.numChannel = 3; 
			option.numInput = {(int)ACTION.size(), size[0], size[1]};
			option.numOutput = {0};
			option.vigiliance = {0.9, 0.9, 0.9};
			option.contribution = {0.3, 0.3, 0.3};
			option.bias = {1.0000e-006, 1.0000e-006, 1.0000e-006};
			option.learningRate = {0.8, 0.8, 0.8};
		}

		// For Layer 2..
		if(numLayer == 2)
		{
			option.maxIter = 10; 
			option.numChannel = 1;
			option.numInput.push_back(size[0]);
			option.numOutput = {0}; 
			option.vigiliance = {0.8}; 
			option.contribution = {1}; 
			option.bias = {1.0000e-006}; 
			option.learningRate = {0.8}; 
		}
	}
};
