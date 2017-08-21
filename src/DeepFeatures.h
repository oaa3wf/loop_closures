/**
 * File from OpenCV (see License below), modified by Mathieu Labbe 2016
 */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef CORELIB_SRC_OPENCV_DEEPFEATURES_H_
#define CORELIB_SRC_OPENCV_DEEPFEATURES_H_

#include <opencv2/features2d/features2d.hpp>
#include <cstring>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include "CaffeFeatExtractor.hpp"

using namespace std;


//namespace rtabmap {

/*!
 DEEPFEATURES implementation.
*/
class CV_DEEPFEATURES : public cv::Feature2D
{
public:
    // the size of the signature in bytes
    enum { kBytes = 64*4};

    CV_WRAP explicit CV_DEEPFEATURES(string caffeModelFile, string caffePrototxtFile, string layerName="pool5");
		~CV_DEEPFEATURES();

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;

    // Compute the ORB features and descriptors on an image
    void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints) const;

    // Compute the ORB features and descriptors on an image
    void operator()( cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
    		cv::OutputArray descriptors, bool useProvidedKeypoints=false ) const;

protected:

    void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors ) const;
    void detectImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat() ) const;

    CV_PROP_RW string caffeModelFile;
    CV_PROP_RW string caffePrototxtFile;
    CV_PROP_RW string layerName;
		CaffeFeatExtractor<float> *netExtractor;


};

//}


#endif /* CORELIB_SRC_OPENCV_DEEPFEATURES_H_ */
