/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>


#include "DeepFeatures.h"

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace caffe;
using namespace std;
using namespace cv;

const int resize_width = 227;
const int resize_height = 227;
const string compute_mode = "GPU";
const int device_id = 0;
const bool timing_extraction = false;


/**TODO
*  Need to add field for dsize (descriptor size)
*  Need to add private field for number of features
*/

/** Constructor
 * @param _caffeModelFile the absolute path to the caffe model file for the deep network
 * @param _caffePrototxtFile the absolute path to the caffe prototxt file for the deep network 
 * @param _layerName the name of the layer whose output is to be used as in image descriptor
 */
CV_DEEPFEATURES::CV_DEEPFEATURES(string _caffeModelFile, string _caffePrototxtFile, string _layerName) :
    caffeModelFile(_caffeModelFile), caffePrototxtFile(_caffePrototxtFile), layerName(_layerName) 
{
	netExtractor = new CaffeFeatExtractor<float> (caffeModelFile,caffePrototxtFile,resize_height,resize_width,layerName,compute_mode,device_id,timing_extraction);
}

CV_DEEPFEATURES::~CV_DEEPFEATURES()
{
	delete netExtractor;
	netExtractor = NULL;
}
 

int CV_DEEPFEATURES::descriptorSize() const
{
    return kBytes;
}

int CV_DEEPFEATURES::descriptorType() const
{
    return CV_32FC1;
}

/** Compute the DEEPFEATURES decriptors
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
 * @param level the scale at which we compute the orientation
 * @param keypoints the keypoints to use
 * @param descriptors the resulting descriptors
 */
static void computeDescriptors(Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors,
                               int dsize, string caffeModelFile, string caffePrototxtFile, string layerName, CaffeFeatExtractor<float> *netExtractor)
{
    //check that the image is color
    CV_Assert(image.type() == CV_8UC3);
   
    /**TODO
    * Need to move construction of netExtractor to constructor
    *
    */
    // forward propagate the network and extract layer
    //CaffeFeatExtractor<float> netExtractor(caffeModelFile,caffePrototxtFile,resize_height,resize_width,layerName,compute_mode,device_id,timing_extraction);
    float times[2];
    caffe::Blob<float>* feature_layer = new caffe::Blob<float>() ;

   
    netExtractor->extract_singleFeat(image, feature_layer,times);

    // resahape blob 

    int new_height = (int)((feature_layer[0].shape(0)*feature_layer[0].shape(1)*feature_layer[0].shape(2)*feature_layer[0].shape(3))/dsize);
    vector<int> new_shape(4);
    new_shape[0] = 1;
    new_shape[1] = 1;
    new_shape[2] = new_height;
    new_shape[3] = 64;
    feature_layer[0].Reshape(new_shape);

    int _nFeatures = new_height;

    //create the descriptor mat, _nFeatures rows, BYTES cols
    descriptors = Mat::zeros(_nFeatures, dsize, CV_32FC1);

    // get mutable date

    float* input_data = feature_layer[0].mutable_cpu_data();

    // put in cv Mat
    cv::Mat featImage(new_height, dsize, CV_32FC1, input_data);
    descriptors = featImage;
		delete feature_layer;
		feature_layer = NULL;


}


/** Compute the FAST features and DEEPFEATURES descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 */
void CV_DEEPFEATURES::operator()(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}


/** Compute the FAST features and DEEPFEATURES descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 */
void CV_DEEPFEATURES::operator()( InputArray _image, InputArray _mask, std::vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors, bool useProvidedKeypoints) const
{


    /** TODO
    *   Need to find a better way to get number of keypoints and move descriptor computation to end
    */
    cv::Mat img = _image.getMat();
    cv::Mat descriptors = _descriptors.getMat();
    computeDescriptors(img, _keypoints, descriptors,64, caffeModelFile, caffePrototxtFile, layerName, netExtractor);
    uint _nKeypoints = descriptors.rows;
    descriptors.copyTo(_descriptors);
    /**TODO:
    * This could crash if we have less than _nKeypoints
    * need to fix this
    */
    if(useProvidedKeypoints){

        if( _keypoints.size() > _nKeypoints){

            _keypoints.resize(_nKeypoints);
        }


    }
    else{

        int _threshold = 40;
        cv::Mat grayImg;
        cv::cvtColor(_image, grayImg, CV_BGR2GRAY);
        FAST(grayImg,_keypoints,_threshold);
        if( _keypoints.size() > _nKeypoints){

            _keypoints.resize(_nKeypoints);
        } 

    }





}

void CV_DEEPFEATURES::detectImpl( const Mat& image, std::vector<KeyPoint>& keypoints, const Mat& mask) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}

void CV_DEEPFEATURES::computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
{
    (*this)(image, Mat(), keypoints, descriptors, true);
}


