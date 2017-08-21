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
#include "opencv2/core/cvstd.hpp"

//FLANN
//#include <flann/flann.hpp>

#include <fstream>

using namespace caffe;
using namespace std;


int main(int argc, char** argv) {


	// Set up netExtractor

	string caffe_model_file = "/home/sastrygrp2/code/3rd_party/caffe/models/PlacesCNN/new_places205CNN_iter_300000.caffemodel";
	//string caffe_model_file = "/home/sastrygrp2/code/3rd_party/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
	string prototxt_file = "/home/sastrygrp2/code/3rd_party/caffe/models/PlacesCNN/new_places205CNN_deploy.prototxt";
	//string prototxt_file = "/home/sastrygrp2/code/3rd_party/caffe/models/bvlc_alexnet/train_val.prototxt";
	int resize_width = 227;
	int resize_height = 227;
	string blob_name = "pool5";
	string compute_mode = "GPU";
	int device_id = 0;
	bool timing_extraction = false; 


	CaffeFeatExtractor<float> netExtractor(caffe_model_file,prototxt_file,resize_height,resize_width,blob_name,compute_mode,device_id,timing_extraction);
	float times[2];
	caffe::Blob<float>* feature_layer = new caffe::Blob<float>() ;


	// For each image in directory pass the image through neural network 

	cv::String folder_path = "/home/sastrygrp2/code/loop_closures/data/Lip6IndoorDataSet/Images/*.ppm";
	vector<cv::String> filenames;
	cv::glob(folder_path, filenames);

	cv::Mat database;

	for (size_t i=0; i<filenames.size(); i++)
	{
		std::cout << "reading file from : " << filenames[i] << std::endl;
	    cv::Mat im = cv::imread(filenames[i], CV_LOAD_IMAGE_COLOR);
	    bool res;
		res = netExtractor.extract_singleFeat(im, feature_layer,times);
		float start_time = times[0];
		float end_time = times[1];
		float duration = end_time - start_time;

		/**
		std::cout << "res: "<< res << std::endl;
		std::cout << "num: "<< feature_layer[0].shape(0) << std::endl;
		std::cout << "channels: "<< feature_layer[0].shape(1) << std::endl;
		std::cout << "height: "<< feature_layer[0].shape(2) << std::endl;
		std::cout << "width: "<< feature_layer[0].shape(3) << std::endl;
		std::cout.precision(17);
		std::cout << start_time << std::endl;
		std::cout << end_time << std::endl;
		std::cout << "time: " << std::fixed <<  (double)(duration) << std::endl;
		**/

		// reshape blob 

		int new_height = (int)((feature_layer[0].shape(0)*feature_layer[0].shape(1)*feature_layer[0].shape(2)*feature_layer[0].shape(3)));
		std::cout << "new height" << new_height << std::endl;
		vector<int> new_shape(4);
		new_shape[0] = 1;
		new_shape[1] = 1;
		new_shape[2] = new_height;
		new_shape[3] = 1;
		feature_layer[0].Reshape(new_shape);

		// get mutable date

		float* input_data = feature_layer[0].mutable_cpu_data();

		// put in cv Mat
		cv::Mat featImage(1,new_height, CV_32FC1, input_data);

		database.push_back(featImage);

	
	}

	cvflann::Matrix<float> data( (float*)database.data, database.rows, database.cols );
	cvflann::Index<cvflann::L2<float> > index(data, cvflann::LinearIndexParams());
	index.buildIndex();

	int test_rank = 388;

	// make cv::Mat-header for easier access 
	// (you can do the same with the dists-matrix if you need it)
	cv::Mat ind(data.rows, test_rank, CV_32S);
	CV_Assert(ind.isContinuous());
	cvflann::Matrix<int> indices((int*) ind.data, ind.rows, ind.cols);
	cvflann::Matrix<float> dists(new float[data.rows*test_rank], data.rows, test_rank);
	index.knnSearch(data, indices, dists, test_rank, cvflann::SearchParams(database.cols-1));

	cv::Mat dist_mat(dists.rows,dists.cols, CV_32FC1, dists.data);

	cv::Mat groundtruth_mat(388,388,CV_32F,cvScalar(0.));

	for(int i = 0; i < 388; i++){

		for(int j = 0; j < test_rank; j ++){


			groundtruth_mat.at<float>(i,ind.at<int>(i,j)) = 1.0; 



		}



	}

	std::vector<int> params;
	params.push_back(CV_IMWRITE_PXM_BINARY);

	cv::imwrite("./large_estimate_truth.ppm", groundtruth_mat,params);

	std::ofstream myfile;
	myfile.open("large_distance.csv");
	myfile << format(dist_mat,cv::Formatter::FMT_CSV) << std::endl << std::endl;
	myfile.close();


	std::ofstream myfile2;
	myfile2.open("large_groundtruth.csv");
	myfile2 << format(groundtruth_mat,cv::Formatter::FMT_CSV) << std::endl << std::endl;
	myfile2.close();



	std::cout << "ind = "<< std::endl << " "  << ind << std::endl;
	std::cout << "distance = "<< std::endl << " "  << dist_mat<< std::endl;
	std::cout << "distance = "<< std::endl << " "  << database.rows << std::endl;
	std::cout << "distance = "<< std::endl << " "  << database.cols << std::endl;


	
	/**
	cv::Mat dst;
    cv::normalize(featImage, dst, 0, 1, cv::NORM_MINMAX);
    cv::imshow("test", dst);
    cv::waitKey(0);
    **/


	// display

return 0;

}
