/**
 * File: simple_loops.cpp
 * Date: April 2017
 * Author: Oladapo Afolabi
 * Description: demo application of DBoW2
 * License: No license here folks
 */

#include <iostream>
#include <vector>
#include <stdlib.h>  /** atoi **/


// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// Nicolo Valigi's code
#include "frame_descriptor.h"
#include "utils.h"
#include <utility>
#include <sstream>
#include <fstream>
#include <memory>
#include <cstdlib>


using namespace DBoW2;
using namespace DUtils;
using namespace std;

using namespace slc;



void loadFeatures(vector<vector<vector<float> > > &features,string file_location);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L);
void VocCreation(const vector<vector<vector<float> > > &features);

// number of training images
const int NIMAGES = 2146;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;


int main(int argc, char** argv){

	if(argc < 2){

		cerr<< "Usage " << argv[0] << " [mode] [path to database] [path to images] " << endl;
		exit(1); 

	}

	if(atoi(argv[1]) == 0){

		vector<vector<vector<float> > > features;
		string dataset_folder(argv[1]);
		loadFeatures(features,dataset_folder);
		VocCreation(features);
	}

	else{


		string vocabulary_path(argv[2]);
		FrameDescriptor descriptor(vocabulary_path);

		string dataset_folder(argv[3]);
		auto filenames = load_filenames(dataset_folder);
		std::cout << "Processing " << filenames.size() << " images\n";

		// Will hold BoW representations for each frame
		vector<DBoW2::BowVector> bow_vecs;

		for (unsigned int img_i = 0; img_i < filenames.size(); img_i++) {
			auto img_filename = dataset_folder + "/Images/" + filenames[img_i];
			auto img = cv::imread(img_filename);

			cout << img_filename << endl;

			if (img.empty()) {
				cerr << endl << "Failed to load: " << img_filename << endl;
				exit(1);
			}

			// Get a BoW description of the current image
			DBoW2::BowVector bow_vec;
			descriptor.describe_frame(img, bow_vec);
			bow_vecs.push_back(bow_vec);
		}

		cout << "Writing output..." << endl;

		ofstream of;
		of.open(
			getenv("HOME") + string("/code/loop_closures/out/confusion_matrix.txt"));

		// Compute confusion matrix
		// i.e. the (i, j) element of the matrix contains the distance
		// between the BoW representation of frames i and j
		for (unsigned int i = 0; i < bow_vecs.size(); i++) {
		   for (unsigned int j = 0; j < bow_vecs.size(); j++) {
				of << descriptor.vocab_->score(
				    bow_vecs[i], bow_vecs[j]) << " ";
			}
			of << "\n";
		}

		of.close();
		cout << "Output done" << endl;

		
	}


	

	return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features, string file_location)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

  cout << "Extracting SURF features..." << endl;
  for(int i = 1; i <= NIMAGES; ++i)
  {
    stringstream ss;
	char buffer[50];
	sprintf(buffer,"%04d",i);
    ss << file_location << "/"<< buffer << ".jpg"; 

	std::cout << "file location :" << ss.str() << std::endl;

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    vector<float> descriptors;

    surf->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<vector<float> >());
    changeStructure(descriptors, features.back(), surf->descriptorSize());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L)
{
  out.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    out[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
  }
}

void VocCreation(const vector<vector<vector<float> > > &features)
{
  // branching factor and depth levels 
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Surf64Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("generated_voc.yml.gz");
  cout << "Done" << endl;
}

