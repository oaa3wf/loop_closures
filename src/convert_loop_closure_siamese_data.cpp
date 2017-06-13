//
// This script converts the loop closure datasets to the leveldb format used
// by caffe to train siamese network.
// Usage:
//    convert_loop_closure_data input_image_folder mat_groundtruth_file output_db_file

#include <fstream>  // NOLINT(readability/streams)
#include <string>

//#include "glog/logging.h"
//#include "google/protobuf/text_format.h"
#include "stdint.h"

//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/format.hpp"
//#include "caffe/util/math_functions.hpp"

//#ifdef USE_LEVELDB
#include "leveldb/db.h"

void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_filename) {


	// create list of filenames
    vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
    String folder = "<some_folder_path>"; // again we are using the Opencv's embedded "String" class

    glob(folder, filenames); // new function that does the job ;-)

	// read Mat file and store as a matrix






}
