//# include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include "CaffeFeatExtractor.hpp"

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace caffe;
using namespace std;


int main(int argc, char** argv) {

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



	CaffeFeatExtractor<float> netExtractor(caffe_model_file,prototxt_file,resize_height,resize_width,blob_name,compute_mode,device_id,timing_extraction);
	float times[2];
	caffe::Blob<float>* feature_layer = new caffe::Blob<float>() ;
	cv::Mat img;
	string img_file = "/home/sastrygrp2/code/3rd_party/caffe/examples/images/cat.jpg";

	// read image from file
	img = cv::imread(img_file, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window


	bool res;
	res = netExtractor.extract_singleFeat(img, feature_layer,times);
	float start_time = times[0];
	float end_time = times[1];
	float duration = end_time - start_time;
	std::cout << "res: "<< res << std::endl;
	std::cout << "num: "<< feature_layer[0].shape(0) << std::endl;
	std::cout << "channels: "<< feature_layer[0].shape(1) << std::endl;
	std::cout << "height: "<< feature_layer[0].shape(2) << std::endl;
	std::cout << "width: "<< feature_layer[0].shape(3) << std::endl;
	std::cout.precision(17);
	std::cout << start_time << std::endl;
	std::cout << end_time << std::endl;
	std::cout << "time: " << std::fixed <<  (double)(duration) << std::endl;

	// resahape blob 

	int new_height = (int)((feature_layer[0].shape(0)*feature_layer[0].shape(1)*feature_layer[0].shape(2)*feature_layer[0].shape(3))/64);
	vector<int> new_shape(4);
	new_shape[0] = 1;
	new_shape[1] = 1;
	new_shape[2] = new_height;
	new_shape[3] = 64;
	feature_layer[0].Reshape(new_shape);

	// get mutable date

	float* input_data = feature_layer[0].mutable_cpu_data();

	// put in cv Mat
	cv::Mat featImage(new_height, 64, CV_32FC1, input_data);
	cv::Mat dst;
    cv::normalize(featImage, dst, 0, 1, cv::NORM_MINMAX);
    cv::imshow("test", dst);
    cv::waitKey(0);


	// display

return 0;

}
