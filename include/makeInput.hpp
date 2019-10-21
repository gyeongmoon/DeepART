#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <iterator>

#include "pugixml.hpp"
#include "definitions.hpp"

// Load dataset from "dataset.xml" file.
void load_dataset(std::string address)
{
    pugi::xml_document doc;
    if (!doc.load_file(address.c_str())) // const char*
    {
        std::cout << "'dataset.xml' Opening Failed!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Load action attributes..
    pugi::xml_node actions = doc.child("dataset").child("action");
    for (pugi::xml_node action = actions.first_child(); action; action = action.next_sibling())
        ACTION.push_back(action.child_value());

    // Load object attributes..
    pugi::xml_node objects = doc.child("dataset").child("object");
    for (pugi::xml_node object = objects.first_child(); object; object = object.next_sibling("name"))
        OBJECT.push_back(object.child_value());

    for (pugi::xml_node object = objects.child("color"); object; object = object.next_sibling("color"))
        OBJECT_COLOR.push_back(object.child_value());

    for (pugi::xml_node object = objects.child("shape"); object; object = object.next_sibling("shape"))
        OBJECT_SHAPE.push_back(object.child_value());

    for (pugi::xml_node object = objects.child("type"); object; object = object.next_sibling("type"))
        OBJECT_TYPE.push_back(object.child_value());

    for (pugi::xml_node object = objects.child("preposition"); object; object = object.next_sibling("preposition"))
        OBJECT_PREPOSITION.push_back(object.child_value());

    for (pugi::xml_node object = objects.child("preposition"); object; object = object.next_sibling("preposition"))
        OBJECT_PREPOSITION.push_back(object.child_value());

    // Load episode attributes..
	NUM_EPISODE = 0;
    for(pugi::xml_node episodes: doc.child("dataset").children("episode"))
    {
        ++NUM_EPISODE;
        EVENT.resize(NUM_EPISODE);
        for (pugi::xml_node episode = episodes.first_child(); episode; episode = episode.next_sibling())
            EVENT[NUM_EPISODE-1].push_back(std::string(" ") + episode.child_value() + std::string(" "));
    }

    // Load cue attributes..
	NUM_CUE = 1; CUE.resize(NUM_CUE);
    pugi::xml_node cues = doc.child("dataset").child("cue");

    std::cout << "CUE: ";
    for (pugi::xml_node cue = cues.first_child(); cue; cue = cue.next_sibling())
    {
        CUE[NUM_CUE-1].push_back(std::string(" ") + cue.child_value() + std::string(" "));
        std::cout << cue.child_value() << std::endl;
    }
    std::cout << std::endl;
}

// Read input lists (actions, objects, and events) and make inputs..
void readFile()
{
	CUE.clear(); EVENT.clear();

    load_dataset("../dataset.xml");

	eAction.assign((int)ACTION.size(), std::vector<int> ((int)ACTION.size(),0));
	eObject.assign((int)OBJECT.size(), std::vector<int> ((int)OBJECT.size(),0));
	eObject_color.assign((int)OBJECT_COLOR.size(), std::vector<int> ((int)OBJECT_COLOR.size(),0));
	eObject_shape.assign((int)OBJECT_SHAPE.size(), std::vector<int> ((int)OBJECT_SHAPE.size(),0));
	eObject_type.assign((int)OBJECT_TYPE.size(), std::vector<int> ((int)OBJECT_TYPE.size(),0));
	eObject_preposition.assign((int)OBJECT_PREPOSITION.size(), std::vector<int> ((int)OBJECT_PREPOSITION.size(),0));

	for(unsigned int i=0; i<eAction.size(); i++)
		eAction[i][i] = 1;
	for(unsigned int i=0; i<eObject.size(); i++)
		eObject[i][i] = 1;
	for(unsigned int i=0; i<eObject_color.size(); i++)
		eObject_color[i][i] = 1;
	for(unsigned int i=0; i<eObject_shape.size(); i++)
		eObject_shape[i][i] = 1;
	for(unsigned int i=0; i<eObject_type.size(); i++)
		eObject_type[i][i] = 1;
	for(unsigned int i=0; i<eObject_preposition.size(); i++)
		eObject_preposition[i][i] = 1;
}

// makeInput
void makeInput(int numEpisode, std::vector< std::vector<std::string> > eventList, std::vector<Episode>& input){

	std::vector<unsigned int> found_pp; // Found preposition..

	// Object & Object2 init..
	for(int i=0; i<numEpisode; i++)
	{
		input[i].numEvent = (int)eventList[i].size();
		found_pp.assign(input[i].numEvent,0);

		input[i].action.assign(input[i].numEvent, std::vector<int> ((int)ACTION.size()));

		input[i].object.assign(input[i].numEvent, std::vector<int> ((int)OBJECT.size()));
		input[i].object_color.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_COLOR.size()));
		input[i].object_shape.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_SHAPE.size()));
		input[i].object_type.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_TYPE.size()));
		input[i].object_preposition.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_PREPOSITION.size()));

		input[i].object2.assign(input[i].numEvent, std::vector<int> ((int)OBJECT.size()));
		input[i].object2_color.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_COLOR.size()));
		input[i].object2_shape.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_SHAPE.size()));
		input[i].object2_type.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_TYPE.size()));
		input[i].object2_preposition.assign(input[i].numEvent, std::vector<int> ((int)OBJECT_PREPOSITION.size()));

		for(int j=0; j<input[i].numEvent; j++)
		{
			for(int k=0; k<(int)ACTION.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+ACTION[k]+" ");
				if(found!=std::string::npos)
				{
					input[i].action[j] = eAction[k];
					break;
				}
			}

			for(int k=0; k<(int)OBJECT_PREPOSITION.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+OBJECT_PREPOSITION[k]+" ");
				if(found!=std::string::npos)
				{
					found_pp[j] = found;
					input[i].object2_preposition[j] = eObject_preposition[k];
					break;
				}
			}

			for(int k=0; k<(int)OBJECT.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+OBJECT[k]+" ");
				if(found!=std::string::npos)
				{
					if(found_pp[j] > found) // It means there is pp in sentence.
					{
						input[i].object[j] = eObject[k];

						found = eventList[i][j].find(" "+OBJECT[k]+" ",found+1);
						if(found!=std::string::npos) // It means there are two same objects in one sentence.
						{
							input[i].object2[j] = eObject[k];
							break;
						}
					}
					else // It means there is no object2, so found object is object1.
					{
						input[i].object2[j] = eObject[k];
					}
				}
			}

			for(int k=0; k<(int)OBJECT_COLOR.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+OBJECT_COLOR[k]+" ");
				if(found!=std::string::npos)
				{
					if(found_pp[j] > found) // It means there is pp in sentence.
					{
						input[i].object_color[j] = eObject_color[k];

						found = eventList[i][j].find(" "+OBJECT_COLOR[k]+" ",found+1);
						if(found!=std::string::npos) // It means there are two same objects in one sentence.
						{
							input[i].object2_color[j] = eObject_color[k];
							break;
						}
					}
					else // It means there is no object2, so found object is object1.
					{
						input[i].object2_color[j] = eObject_color[k];
					}
				}
			}

			for(int k=0; k<(int)OBJECT_SHAPE.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+OBJECT_SHAPE[k]+" ");
				if(found!=std::string::npos)
				{
					if(found_pp[j] > found) // It means there is pp in sentence.
					{
						input[i].object_shape[j] = eObject_shape[k];

						found = eventList[i][j].find(" "+OBJECT_SHAPE[k]+" ",found+1);
						if(found!=std::string::npos) // It means there are two same objects in one sentence.
						{
							input[i].object2_shape[j] = eObject_shape[k];
							break;
						}
					}
					else // It means there is no object2, so found object is object1.
					{
						input[i].object2_shape[j] = eObject_shape[k];
					}
				}
			}

			for(int k=0; k<(int)OBJECT_TYPE.size(); k++)
			{
				std::size_t found = eventList[i][j].find(" "+OBJECT_TYPE[k]+" ");
				if(found!=std::string::npos)
				{
					if(found_pp[j] > found) // It means there is pp in sentence.
					{
						input[i].object_type[j] = eObject_type[k];

						found = eventList[i][j].find(" "+OBJECT_TYPE[k]+" ",found+1);
						if(found!=std::string::npos) // It means there are two same objects in one sentence.
						{
							input[i].object2_type[j] = eObject_type[k];
							break;
						}
					}
					else // It means there is no object2, so found object is object1.
					{
						input[i].object2_type[j] = eObject_type[k];
					}
				}
			}
		}
		found_pp.clear();
	}
}
