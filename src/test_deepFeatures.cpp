#include "DeepFeatures.h"

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char** argv){


	string caffe_model_file = "/home/sastrygrp2/code/3rd_party/caffe/models/PlacesCNN/new_places205CNN_iter_300000.caffemodel";
	//string caffe_model_file = "/home/sastrygrp2/code/3rd_party/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
	string prototxt_file = "/home/sastrygrp2/code/3rd_party/caffe/models/PlacesCNN/new_places205CNN_deploy.prototxt";
	//string prototxt_file = "/home/sastrygrp2/code/3rd_party/caffe/models/bvlc_alexnet/train_val.prototxt";
	int resize_width = 227;
	int resize_height = 227;
	string blob_name = "pool5";
	string compute_mode = "GPU";
	int device_id = 0;
	bool timing_extraction = true; 

	CV_DEEPFEATURES d_feature(caffe_model_file, prototxt_file, blob_name);

 	std::vector<KeyPoint> keypoints_1;

 	cv::Mat img;
	string img_file = "/home/sastrygrp2/code/3rd_party/caffe/examples/images/cat.jpg";

	// read image from file
	img = cv::imread(img_file, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img );                   // Show our image inside it.

    cv::waitKey(0);    

	//d_feature.detect(img,keypoints_1);

	cv::Mat descriptors;

	//d_feature.compute(img,keypoints_1,descriptors);

	d_feature(img,noArray(), keypoints_1,
    		descriptors, false );

	std::cout <<  "descriptor rows: " <<  descriptors.rows << std::endl;
	cv::Mat dst;
    cv::normalize(descriptors, dst, 0, 1, cv::NORM_MINMAX);
    cv::imshow("test", dst);
    cv::waitKey(0);


	return 0;
}